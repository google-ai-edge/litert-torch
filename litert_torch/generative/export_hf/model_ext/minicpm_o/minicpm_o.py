# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MiniCPM-o 4.5 (openbmb/MiniCPM-o-4_5) multimodal Any2Any export for LiteRT."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import gc
import json
import logging
import math
import os
import shutil
from typing import Any

import huggingface_hub
import litert_torch
from litert_torch._convert import interface as converter_utils
from litert_torch.generative.export_hf.core import attention as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import cache as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import quantization
from litert_torch.generative.export_hf.core import weights_loader as weights_loader_lib
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
from litert_torch.generative.export_hf.model_ext.qwen3 import patch as _  # pylint: disable=unused-import
import safetensors
import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F
import transformers

try:
  from litert_lm_builder.runtime.proto import any2any_metadata_pb2
  from litert_lm_builder.runtime.proto import any2any_model_type_pb2
except ImportError:
  any2any_metadata_pb2 = None
  any2any_model_type_pb2 = None

_LARGE_TENSOR_NUMEL_THRESHOLD = 1_000_000

_MINICPMO_EXTRA_KWARGS = (
    "generate_audio",
    "include_decode_hidden_states",
    "min_tts_steps",
    "tts_num_layers",
    "vocoder_chunk_frames",
    "vocoder_hidden_channels",
)
exportable_module_config.register_extra_kwargs_keys(_MINICPMO_EXTRA_KWARGS)


def _make_linear(
    in_features: int, out_features: int, bias: bool = True
) -> nn.Linear:
  """Creates an nn.Linear, skipping slow CPU RNG init for large matrices."""
  if in_features * out_features <= _LARGE_TENSOR_NUMEL_THRESHOLD:
    return nn.Linear(in_features, out_features, bias=bias)
  linear = nn.Linear(in_features, out_features, bias=bias, device="meta")
  linear.weight = nn.Parameter(
      torch.zeros(out_features, in_features, dtype=torch.float32)
  )
  if bias:
    linear.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
  return linear


def _make_embedding(num_embeddings: int, embedding_dim: int) -> nn.Embedding:
  """Creates an nn.Embedding, skipping slow CPU RNG init for large tables."""
  if num_embeddings * embedding_dim <= _LARGE_TENSOR_NUMEL_THRESHOLD:
    return nn.Embedding(num_embeddings, embedding_dim)
  return nn.Embedding(
      num_embeddings,
      embedding_dim,
      _weight=torch.zeros(num_embeddings, embedding_dim, dtype=torch.float32),
  )


def _build_1d_sincos_pos_embed(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
  """Builds 1D sin-cos positional embeddings for flattened positions `pos`."""
  half = max(1, embed_dim // 2)
  omega = torch.arange(half, dtype=torch.float32) / float(half)
  omega = 1.0 / (10000.0**omega)
  out = torch.outer(pos.reshape(-1).to(torch.float32), omega)
  emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
  if emb.shape[1] < embed_dim:
    emb = F.pad(emb, (0, embed_dim - emb.shape[1]))
  return emb[:, :embed_dim]


def _build_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int = 70
) -> torch.Tensor:
  """Builds 2D sin-cos positional embeddings matching MiniCPM-o Resampler."""
  grid_h = torch.arange(grid_size, dtype=torch.float32)
  grid_w = torch.arange(grid_size, dtype=torch.float32)
  # Match np.meshgrid(grid_w, grid_h) in MiniCPM-o resampler.py:
  # first half encodes column (w), second half encodes row (h).
  grid_y, grid_x = torch.meshgrid(grid_h, grid_w, indexing="ij")
  half_dim = embed_dim // 2
  emb_x = _build_1d_sincos_pos_embed(half_dim, grid_x)
  emb_y = _build_1d_sincos_pos_embed(embed_dim - half_dim, grid_y)
  return torch.cat([emb_x, emb_y], dim=1)


_DEFAULT_MINICPMO_CONFIG: dict[str, Any] = {
    "model_type": "minicpmo",
    "version": "4.5",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151748,
    "max_position_embeddings": 40960,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "attention_bias": False,
    "scale_resolution": 448,
    "patch_size": 14,
    "max_slice_nums": 9,
    "query_num": 64,
    "audio_pool_step": 5,
    "vision_config": {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "patch_size": 14,
        "image_size": 980,
    },
    "audio_config": {
        "d_model": 1024,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "encoder_layers": 24,
        "num_mel_bins": 80,
    },
    "tts_config": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 20,
        "num_attention_heads": 12,
        "num_audio_tokens": 6562,
        "num_mel_bins": 100,
    },
}


class MiniCpmOGeluTransformerBlock(nn.Module):
  """Self-attention + GELU MLP transformer block for SigLIP and Whisper."""

  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      intermediate_size: int,
      k_bias: bool = True,
      tanh_gelu: bool = False,
      layer_norm_eps: float = 1e-6,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.tanh_gelu = tanh_gelu
    self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    self.q_proj = _make_linear(hidden_size, hidden_size, bias=True)
    self.k_proj = _make_linear(hidden_size, hidden_size, bias=k_bias)
    self.v_proj = _make_linear(hidden_size, hidden_size, bias=True)
    self.o_proj = _make_linear(hidden_size, hidden_size, bias=True)
    self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    self.fc1 = _make_linear(hidden_size, intermediate_size, bias=True)
    self.fc2 = _make_linear(intermediate_size, hidden_size, bias=True)

  def forward(
      self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
  ) -> torch.Tensor:
    b, s, d = x.shape
    h = self.norm1(x)
    q = (
        self.q_proj(h)
        .reshape(b, s, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    k = (
        self.k_proj(h)
        .reshape(b, s, self.num_heads, self.head_dim)
        .permute(0, 2, 3, 1)
    )
    v = (
        self.v_proj(h)
        .reshape(b, s, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    scores = torch.matmul(q, k) * (1.0 / math.sqrt(float(self.head_dim)))
    if attn_mask is not None:
      scores = scores + attn_mask
    probs = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(b, s, d)
    x = x + self.o_proj(attn_out)
    h2 = self.norm2(x)
    act = F.gelu(self.fc1(h2), approximate="tanh" if self.tanh_gelu else "none")
    return x + self.fc2(act)


class MiniCpmOVisionEncoderModule(nn.Module):
  """SigLIP vision encoder + 64-query perceiver resampler for MiniCPM-o 4.5.

  Inputs:
    pixel_values: [1, 3, patch_size, patch_size * num_patches] float32 strip.
    position_ids: [1, num_patches] int32 patch position IDs (-1 for padding;
      for valid patches, encodes `resampler_pos_id * 10000 + vit_pos_id`).
  Output:
    vision_embeds: [1, query_num, llm_hidden_size] float32 embeddings.
  """

  resampler_pos_embed: torch.Tensor

  def __init__(
      self,
      vision_hidden_size: int = 1152,
      vision_intermediate_size: int = 4304,
      vision_num_layers: int = 2,
      vision_num_heads: int = 16,
      patch_size: int = 14,
      max_patches: int = 4900,
      query_num: int = 64,
      llm_hidden_size: int = 4096,
  ):
    super().__init__()
    self.patch_size = patch_size
    self.query_num = query_num
    self.vision_hidden_size = vision_hidden_size
    self.llm_hidden_size = llm_hidden_size
    self.resampler_heads = (
        llm_hidden_size // 128
        if llm_hidden_size >= 128 and llm_hidden_size % 128 == 0
        else vision_num_heads
    )
    self.resampler_head_dim = llm_hidden_size // self.resampler_heads

    self.patch_embedding = nn.Conv2d(
        3,
        vision_hidden_size,
        kernel_size=(patch_size, patch_size),
        stride=(patch_size, patch_size),
        bias=True,
    )
    self.pos_embedding = _make_embedding(max_patches, vision_hidden_size)
    self.blocks = nn.ModuleList([
        MiniCpmOGeluTransformerBlock(
            vision_hidden_size,
            vision_num_heads,
            vision_intermediate_size,
            k_bias=True,
            tanh_gelu=True,
        )
        for _ in range(vision_num_layers)
    ])
    self.post_layernorm = nn.LayerNorm(vision_hidden_size, eps=1e-6)

    # Perceiver resampler: cross-attention from `query_num` learned queries into
    # patch embeddings + 2D sin-cos positional embeddings, followed by
    # projection to `llm_hidden_size`.
    grid_size = max(1, int(math.isqrt(max_patches)))
    self.register_buffer(
        "resampler_pos_embed",
        _build_2d_sincos_pos_embed(llm_hidden_size, grid_size),
        persistent=False,
    )
    self.latents = nn.Parameter(torch.randn(query_num, llm_hidden_size) * 0.02)
    self.kv_proj = _make_linear(vision_hidden_size, llm_hidden_size, bias=False)
    self.ln_q = nn.LayerNorm(llm_hidden_size, eps=1e-6)
    self.ln_kv = nn.LayerNorm(llm_hidden_size, eps=1e-6)
    self.attn_q = _make_linear(llm_hidden_size, llm_hidden_size, bias=True)
    self.attn_k = _make_linear(llm_hidden_size, llm_hidden_size, bias=True)
    self.attn_v = _make_linear(llm_hidden_size, llm_hidden_size, bias=True)
    self.attn_out = _make_linear(llm_hidden_size, llm_hidden_size, bias=True)
    self.ln_post = nn.LayerNorm(llm_hidden_size, eps=1e-6)
    self.proj = nn.Parameter(
        torch.eye(llm_hidden_size, dtype=torch.float32) * 0.02
    )

  def encode_patches(
      self,
      pixel_values: torch.Tensor,
      vit_pos_ids: torch.Tensor,
      attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Runs SigLIP patch + positional embedding, transformer blocks, and post-norm."""
    patches = self.patch_embedding(pixel_values)  # [1, C, 1, N]
    b, c, _, n = patches.shape
    x = patches.reshape(b, c, n).permute(0, 2, 1)  # [1, N, C]
    x = x + self.pos_embedding(vit_pos_ids)
    for blk in self.blocks:
      x = blk(x, attn_mask)
    return self.post_layernorm(x)

  def forward(
      self, pixel_values: torch.Tensor, position_ids: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    # pixel_values: [1, 3, patch_size, patch_size * num_patches]
    n = position_ids.shape[1]
    safe_ids = torch.clamp(position_ids, min=0)
    resampler_raw = torch.div(safe_ids, 10000, rounding_mode="trunc")
    vit_raw = safe_ids - resampler_raw * 10000
    vit_pos_ids = torch.clamp(
        vit_raw, min=0, max=self.pos_embedding.num_embeddings - 1
    )
    resampler_pos_ids = torch.clamp(
        resampler_raw,
        min=0,
        max=self.resampler_pos_embed.shape[0] - 1,
    )

    valid_mask = (
        (position_ids >= 0).to(pixel_values.dtype).unsqueeze(1).unsqueeze(2)
    )
    attn_mask = (1.0 - valid_mask) * (-1e9)
    x = self.encode_patches(pixel_values, vit_pos_ids, attn_mask)

    kv = self.ln_kv(self.kv_proj(x))  # [1, N, llm_hidden_size]
    pos_emb = F.embedding(resampler_pos_ids, self.resampler_pos_embed)
    q_in = self.ln_q(self.latents.unsqueeze(0))  # [1, query_num, llm_hidden]
    # Place dynamic activation `k` on the LHS and constant-folded `q_t` on the
    # RHS of `matmul` so TFLite hybrid `BATCH_MATMUL` receives
    # `lhs=float32, rhs=int8` when dynamic_wi8_afp32 quantization is applied.
    q_t = (
        self.attn_q(q_in)
        .reshape(
            1, self.query_num, self.resampler_heads, self.resampler_head_dim
        )
        .permute(0, 2, 3, 1)
    )
    k = (
        self.attn_k(kv + pos_emb)
        .reshape(1, n, self.resampler_heads, self.resampler_head_dim)
        .permute(0, 2, 1, 3)
    )
    v = (
        self.attn_v(kv)
        .reshape(1, n, self.resampler_heads, self.resampler_head_dim)
        .permute(0, 2, 1, 3)
    )
    scores = torch.matmul(k, q_t).permute(0, 1, 3, 2) * (
        1.0 / math.sqrt(float(self.resampler_head_dim))
    )
    scores = scores + attn_mask
    probs = F.softmax(scores, dim=-1)
    attn_out = (
        torch.matmul(probs, v)
        .permute(0, 2, 1, 3)
        .reshape(1, self.query_num, self.llm_hidden_size)
    )
    resampled = self.attn_out(attn_out)
    resampled = self.ln_post(resampled)
    return {"vision_embeds": torch.matmul(resampled, self.proj)}


class MiniCpmOAudioEncoderModule(nn.Module):
  """Whisper-medium audio encoder + audio projection + 5x temporal pooling.

  Inputs:
    mel_features: [1, num_mel_bins, mel_frames] float32 log-mel spectrogram.
    valid_mel_frames: Optional [1] int32 count of valid mel frames (<=
    mel_frames).
    position_offset: Optional [1] int32 starting post-conv2 position index.
  Output:
    dict with `"audio_embeds"`: [1, mel_frames // (2 * pool_step),
    llm_hidden_size].
  """

  def __init__(
      self,
      num_mel_bins: int = 80,
      audio_hidden_size: int = 1024,
      audio_intermediate_size: int = 4096,
      audio_num_layers: int = 2,
      audio_num_heads: int = 16,
      audio_pool_step: int = 5,
      llm_hidden_size: int = 4096,
      max_source_positions: int = 1500,
  ):
    super().__init__()
    self.conv1 = nn.Conv1d(
        num_mel_bins, audio_hidden_size, kernel_size=3, stride=1, padding=1
    )
    self.conv2 = nn.Conv1d(
        audio_hidden_size, audio_hidden_size, kernel_size=3, stride=2, padding=1
    )
    self.embed_positions = _make_embedding(
        max_source_positions, audio_hidden_size
    )
    self.blocks = nn.ModuleList([
        MiniCpmOGeluTransformerBlock(
            audio_hidden_size,
            audio_num_heads,
            audio_intermediate_size,
            k_bias=False,
            tanh_gelu=False,
            layer_norm_eps=1e-5,
        )
        for _ in range(audio_num_layers)
    ])
    self.layer_norm = nn.LayerNorm(audio_hidden_size, eps=1e-5)
    self.audio_pool_step = audio_pool_step
    self.proj1 = _make_linear(audio_hidden_size, llm_hidden_size, bias=True)
    self.proj2 = _make_linear(llm_hidden_size, llm_hidden_size, bias=True)

  def encode_whisper_features(
      self,
      mel_features: torch.Tensor,
      valid_mel_frames: torch.Tensor | None = None,
      position_offset: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Runs Whisper conv stem, positional embedding, attention blocks, and norm."""
    # mel_features: [B, 80, T]
    mel_len = mel_features.shape[2]
    mel_mask: torch.Tensor | None = None
    if valid_mel_frames is not None:
      mel_idx = torch.arange(
          mel_len, dtype=torch.int32, device=mel_features.device
      ).unsqueeze(0)
      mel_mask = (
          (mel_idx < valid_mel_frames.to(torch.int32).reshape(-1, 1))
          .unsqueeze(1)
          .to(mel_features.dtype)
      )
      mel_features = mel_features * mel_mask

    x = F.gelu(self.conv1(mel_features))
    if mel_mask is not None:
      x = x * mel_mask
    x = F.gelu(self.conv2(x))  # [B, D, T // 2]
    x = x.permute(0, 2, 1)  # [B, T // 2, D]
    seq_len = x.shape[1]

    pos = torch.arange(seq_len, dtype=torch.int32, device=x.device).unsqueeze(0)
    if position_offset is not None:
      pos = pos + position_offset.to(torch.int32).reshape(-1, 1)
    pos_ids = torch.clamp(
        pos,
        min=0,
        max=self.embed_positions.num_embeddings - 1,
    )
    x = x + self.embed_positions(pos_ids)

    attn_mask: torch.Tensor | None = None
    # Match MiniCPMO.get_audio_embedding: 1-second (50 post-conv2 frame)
    # block-causal attention mask `(j < (i // 50 + 1) * 50)` for multi-second
    # segments, combined with key-padding mask `(j < valid_conv2_len)` when
    # `valid_mel_frames` is provided.
    if seq_len > 50 or valid_mel_frames is not None:
      idx = torch.arange(seq_len, dtype=torch.int32, device=x.device)
      block_end = (torch.div(idx, 50, rounding_mode="trunc") + 1) * 50
      allowed = idx.unsqueeze(0) < block_end.unsqueeze(1)  # [S, S]
      allowed = allowed.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
      if valid_mel_frames is not None:
        valid_mel = valid_mel_frames.to(torch.int32).reshape(-1, 1)
        valid_conv2 = torch.where(
            valid_mel > 0,
            torch.div(valid_mel - 1, 2, rounding_mode="trunc") + 1,
            torch.zeros_like(valid_mel),
        )
        key_valid = (
            (idx.unsqueeze(0) < valid_conv2).unsqueeze(1).unsqueeze(2)
        )  # [B, 1, 1, S]
        allowed = allowed & key_valid
      attn_mask = (1.0 - allowed.to(x.dtype)) * (-1e9)

    for blk in self.blocks:
      x = blk(x, attn_mask)
    return self.layer_norm(x)

  def forward(
      self,
      mel_features: torch.Tensor,
      valid_mel_frames: torch.Tensor | None = None,
      position_offset: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    x = self.encode_whisper_features(
        mel_features, valid_mel_frames, position_offset
    )
    x = self.proj2(F.relu(self.proj1(x)))
    if self.audio_pool_step > 1:
      # Pool along time using AvgPool1d over [B, llm_hidden_size, T // 2]
      x = F.avg_pool1d(
          x.permute(0, 2, 1),
          kernel_size=self.audio_pool_step,
          stride=self.audio_pool_step,
      ).permute(0, 2, 1)
    return {"audio_embeds": x}


class MiniCpmOTextEmbedderModule(nn.Module):
  """Token embedding lookup table for MiniCPM-o 4.5 Qwen3 LLM backbone."""

  def __init__(self, vocab_size: int = 151748, llm_hidden_size: int = 4096):
    super().__init__()
    self.embed_tokens = _make_embedding(vocab_size, llm_hidden_size)

  def forward(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"token_embeds": self.embed_tokens(token_ids)}


class MiniCpmORMSNorm(nn.Module):
  """RMSNorm layer matching LlamaRMSNorm used in MiniCPM-o TTS decoder."""

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + self.eps)
    return (self.weight * xf).to(orig_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotates half the hidden dims of `x` for rotary positional embeddings."""
  half = x.shape[-1] // 2
  return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class MiniCpmOTtsTransformerBlock(nn.Module):
  """Single-step LLaMA transformer layer for the MiniCPM-o 4.5 TTS decoder."""

  def __init__(
      self,
      hidden_size: int = 768,
      num_heads: int = 12,
      intermediate_size: int = 3072,
      rms_norm_eps: float = 1e-6,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.input_layernorm = MiniCpmORMSNorm(hidden_size, eps=rms_norm_eps)
    self.q_proj = _make_linear(hidden_size, hidden_size, bias=False)
    self.k_proj = _make_linear(hidden_size, hidden_size, bias=False)
    self.v_proj = _make_linear(hidden_size, hidden_size, bias=False)
    self.o_proj = _make_linear(hidden_size, hidden_size, bias=False)
    self.post_attention_layernorm = MiniCpmORMSNorm(
        hidden_size, eps=rms_norm_eps
    )
    self.gate_proj = _make_linear(hidden_size, intermediate_size, bias=False)
    self.up_proj = _make_linear(hidden_size, intermediate_size, bias=False)
    self.down_proj = _make_linear(intermediate_size, hidden_size, bias=False)

  def forward(
      self,
      x: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
  ) -> torch.Tensor:
    b, s, d = x.shape
    h = self.input_layernorm(x)
    q = (
        self.q_proj(h)
        .reshape(b, s, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    k = (
        self.k_proj(h)
        .reshape(b, s, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    v = (
        self.v_proj(h)
        .reshape(b, s, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    scores = torch.matmul(q, k.permute(0, 1, 3, 2)) * (
        1.0 / math.sqrt(float(self.head_dim))
    )
    probs = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(b, s, d)
    x = x + self.o_proj(attn_out)
    h2 = self.post_attention_layernorm(x)
    mlp_out = self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
    return x + mlp_out


class MiniCpmOTtsProjectorModule(nn.Module):
  """Projects Qwen3 LLM hidden states to TTS conditioning embeddings.

  Matches `tts.projector_semantic` (`Projector` in `modeling_minicpmo.py`):
  `Linear(llm_dim, tts_dim) -> ReLU -> Linear(tts_dim, tts_dim)` followed by
  L2 normalization along the last dimension.

  Input:
    llm_hidden: [1, 1, llm_hidden_size] float32.
  Output:
    dict with `"tts_cond"`: [1, 1, tts_hidden_size] float32.
  """

  def __init__(
      self,
      llm_hidden_size: int = 4096,
      tts_hidden_size: int = 768,
      normalize_projected_hidden: bool = True,
  ):
    super().__init__()
    self.linear1 = _make_linear(llm_hidden_size, tts_hidden_size, bias=True)
    self.linear2 = _make_linear(tts_hidden_size, tts_hidden_size, bias=True)
    self.normalize_projected_hidden = normalize_projected_hidden

  def forward(self, llm_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    x = self.linear2(F.relu(self.linear1(llm_hidden)))
    if self.normalize_projected_hidden:
      norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
      x = x * torch.rsqrt(torch.clamp(norm_sq, min=1e-12))
    return {"tts_cond": x}


class MiniCpmOTtsDecoderModule(nn.Module):
  """Autoregressive TTS decoder step module for MiniCPM-o 4.5.

  Inputs:
    cond_emb: [1, 1, tts_hidden_size] float32 projected LLM conditioning.
    prev_audio_code: [1, 1] int32 previous audio token ID (or TTS BOS token ID
      151703 on step 0).
    step_idx: [1] int32 step index within the current text token's TTS loop.
  Outputs:
    dict with:
      `"code_logits"`: [1, 1, tts_vocab_size] float32 audio token logits.
      `"mel_frame"`: [1, 1, tts_num_mel_bins] float32 predicted mel frame.
  """

  inv_freq: torch.Tensor
  bos_emb: torch.Tensor
  mel_proj: torch.Tensor

  def __init__(
      self,
      tts_hidden_size: int = 768,
      tts_intermediate_size: int = 3072,
      tts_num_layers: int = 20,
      tts_num_heads: int = 12,
      tts_vocab_size: int = 6562,
      tts_num_mel_bins: int = 100,
      rope_theta: float = 10000.0,
      min_tts_steps: int = 64,
  ):
    super().__init__()
    self.tts_hidden_size = tts_hidden_size
    self.tts_vocab_size = tts_vocab_size
    self.tts_num_mel_bins = tts_num_mel_bins
    self.min_tts_steps = min_tts_steps
    head_dim = tts_hidden_size // tts_num_heads
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.register_buffer(
        "bos_emb",
        torch.zeros(1, 1, tts_hidden_size, dtype=torch.float32),
        persistent=False,
    )
    mel_idx = torch.arange(tts_num_mel_bins, dtype=torch.float32).unsqueeze(0)
    hid_idx = torch.arange(tts_hidden_size, dtype=torch.float32).unsqueeze(1)
    default_mel_proj = torch.cos(
        (hid_idx + 0.5)
        * (mel_idx + 1.0)
        * (math.pi / float(max(1, tts_hidden_size)))
    ) * (1.0 / math.sqrt(float(max(1, tts_hidden_size))))
    self.register_buffer("mel_proj", default_mel_proj, persistent=False)

    self.emb_code = _make_embedding(tts_vocab_size, tts_hidden_size)
    self.layers = nn.ModuleList([
        MiniCpmOTtsTransformerBlock(
            hidden_size=tts_hidden_size,
            num_heads=tts_num_heads,
            intermediate_size=tts_intermediate_size,
        )
        for _ in range(tts_num_layers)
    ])
    self.norm = MiniCpmORMSNorm(tts_hidden_size)
    self.head_code = _make_linear(tts_hidden_size, tts_vocab_size, bias=False)

  def forward_hidden(
      self,
      cond_emb: torch.Tensor,
      prev_audio_code: torch.Tensor,
      step_idx: torch.Tensor,
  ) -> torch.Tensor:
    """Runs embedding lookup, LLaMA layers with RoPE at `step_idx`, and RMSNorm."""
    safe_code = torch.clamp(prev_audio_code, min=0, max=self.tts_vocab_size - 1)
    code_emb = self.emb_code(safe_code)
    is_bos = (prev_audio_code < 0) | (prev_audio_code >= self.tts_vocab_size)
    bos = self.bos_emb.to(dtype=code_emb.dtype, device=code_emb.device).reshape(
        1, 1, self.tts_hidden_size
    )
    tok_emb = torch.where(is_bos.unsqueeze(-1), bos, code_emb)
    h = cond_emb + tok_emb

    pos = step_idx.to(torch.float32).reshape(1, 1)
    freqs = pos.unsqueeze(-1) * self.inv_freq.reshape(1, 1, -1)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = torch.cos(emb).unsqueeze(1).to(cond_emb.dtype)
    sin = torch.sin(emb).unsqueeze(1).to(cond_emb.dtype)

    for layer in self.layers:
      h = layer(h, cos, sin)
    return self.norm(h)

  def forward(
      self,
      cond_emb: torch.Tensor,
      prev_audio_code: torch.Tensor,
      step_idx: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    h = self.forward_hidden(cond_emb, prev_audio_code, step_idx)
    code_logits = self.head_code(h)
    if self.min_tts_steps > 0 and self.tts_vocab_size > 1:
      step_f = step_idx.to(code_logits.dtype).reshape(1, 1, 1)
      min_steps_f = float(self.min_tts_steps)
      eos_ramp = (step_f - min_steps_f + 1.0) * 8.0
      eos_bias = torch.where(
          step_f < min_steps_f,
          torch.full_like(step_f, -1e4),
          eos_ramp,
      )
      non_eos = code_logits[..., :-1]
      eos_logit = code_logits[..., -1:] + eos_bias
      code_logits = torch.cat([non_eos, eos_logit], dim=-1)

    mel_frame = torch.matmul(
        h, self.mel_proj.to(dtype=h.dtype, device=h.device)
    )
    return {
        "code_logits": code_logits,
        "mel_frame": mel_frame,
    }


class MiniCpmOAudioVocoderModule(nn.Module):
  """HiFi-GAN / CosyVoice2 24kHz mel-to-waveform vocoder (`256x` upsampling).

  Input:
    mel: [1, num_mel_bins, chunk_frames] float32 mel spectrogram chunk.
  Output:
    dict with `"waveform"`: [1, chunk_frames * 256] float32 PCM waveform in
    [-1, 1] at 24 kHz (`hop_size = 256`).
  """

  def __init__(
      self,
      num_mel_bins: int = 100,
      hidden_channels: int = 256,
      upsample_rates: tuple[int, ...] = (8, 8, 4),
  ):
    super().__init__()
    self.num_mel_bins = num_mel_bins
    self.hidden_channels = hidden_channels
    self.upsample_rates = upsample_rates
    self.input_conv = nn.Conv1d(
        num_mel_bins, hidden_channels, kernel_size=7, padding=3
    )
    up_layers: list[nn.Module] = []
    res_layers: list[nn.Module] = []
    ch = hidden_channels
    for rate in upsample_rates:
      next_ch = max(16, ch // 2)
      kernel_size = rate * 2
      padding = rate // 2
      up_layers.append(
          nn.ConvTranspose1d(
              ch,
              next_ch,
              kernel_size=kernel_size,
              stride=rate,
              padding=padding,
          )
      )
      res_layers.append(nn.Conv1d(next_ch, next_ch, kernel_size=3, padding=1))
      ch = next_ch
    self.upsamples = nn.ModuleList(up_layers)
    self.res_convs = nn.ModuleList(res_layers)
    self.output_conv = nn.Conv1d(ch, 1, kernel_size=7, padding=3)

  def forward(self, mel: torch.Tensor) -> dict[str, torch.Tensor]:
    x = self.input_conv(mel)
    for up, res in zip(self.upsamples, self.res_convs):
      x = F.leaky_relu(x, 0.1)
      x = up(x)
      x = x + res(F.leaky_relu(x, 0.1))
    x = torch.tanh(self.output_conv(F.leaky_relu(x, 0.1)))
    return {"waveform": x.squeeze(1)}


class _MiniCpmOLlmGenerateModule(
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder
):
  """Generate module that concatenates `[logits, last_hidden_state]` in `logits`.

  `MiniCpmOLlmStage::RunDecodeStep` inspects the length of the `decode` output
  buffer (`logits`): when its size equals `vocab_size + llm_hidden_size`, the
  runtime splits the trailing `llm_hidden_size` floats into
  `step_out.hidden_state`
  (required by `MiniCpmOTtsStage` when `generate_audio` is enabled) and resizes
  `logits` to `vocab_size`.
  """

  # pylint: disable=arguments-renamed
  def forward(
      self,
      embeddings,
      input_pos,
      kv_cache,
      mask,
      local_mask=None,
      **kwargs,
  ):
    if self.export_config.apply_gpu_composites:
      kwargs["apply_gpu_composites"] = True
    if self.export_config.use_sdpa_composite:
      kwargs["use_sdpa_composite"] = True
    inputs = self.adapt_inputs(
        None,
        embeddings,
        input_pos,
        kv_cache,
        mask,
        local_mask=local_mask,
        use_bool_mask=self.export_config.extra_kwargs.get(
            "use_bool_mask", False
        ),
        **kwargs,
    )
    inputs |= self.attention_kwargs()
    include_hidden = bool(
        self.export_config.extra_kwargs.get(
            "include_decode_hidden_states", True
        )
    )
    output = self.model(
        **inputs,
        output_hidden_states=include_hidden or bool(self.export_verifier),
    )
    logits = output.logits.to(torch.float32)
    if include_hidden:
      last_hidden = output.hidden_states[-1].to(torch.float32)
      logits = torch.cat([logits, last_hidden], dim=-1)
    extra_outputs = (
        {"activations": output.hidden_states[-1]}
        if self.export_verifier
        else {}
    )
    return {
        "kv_cache": output.past_key_values,
        "logits": logits,
        **extra_outputs,
    }


class _MiniCpmOLlmWeightsLoader(weights_loader_lib.HFCheckpointWeightsLoader):
  """Streams `llm.*`-prefixed Qwen3/Qwen2 weights from a MiniCPM-o checkpoint."""

  @staticmethod
  def _candidates(param_name: str) -> list[str]:
    base = weights_loader_lib.HFCheckpointWeightsLoader._candidates(  # pylint: disable=protected-access
        param_name
    )
    return base + [f"llm.{c}" for c in base]

  def __call__(self, param_name: str) -> torch.Tensor:
    try:
      return super().__call__(param_name)
    except KeyError:
      if self._tie_word_embeddings and param_name.endswith("lm_head.weight"):
        for emb_key in (
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
        ):
          tensor = self._try_load(emb_key)
          if tensor is not None:
            return tensor
      raise


class MiniCpmO:
  """openbmb/MiniCPM-o-4_5 multimodal Any-to-Any model exporter for LiteRT."""

  def __init__(
      self,
      model_path: str,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ):
    self.model_path = model_path
    self.model_dir = (
        model_path
        if os.path.exists(model_path)
        else huggingface_hub.snapshot_download(model_path)
    )
    self.export_config = export_config
    self._config: dict[str, Any] = copy.deepcopy(_DEFAULT_MINICPMO_CONFIG)
    self._load_config()
    self._state_dict: dict[str, torch.Tensor] = {}
    self._weight_map: dict[str, str] = {}
    if export_config is None or not export_config.use_random_weights:
      self._load_weights_if_available()

  def _load_config(self) -> None:
    """Loads and deep-merges config.json from `self.model_dir` if present."""
    cfg_path = os.path.join(self.model_dir, "config.json")
    if os.path.exists(cfg_path):
      with open(cfg_path, "r") as f:
        loaded = json.load(f)
      for k, v in loaded.items():
        if isinstance(v, dict) and isinstance(self._config.get(k), dict):
          self._config[k].update(v)
        else:
          self._config[k] = v
      slice_cfg = loaded.get("slice_config")
      if isinstance(slice_cfg, dict):
        for sk in ("max_slice_nums", "scale_resolution", "patch_size"):
          if sk in slice_cfg and sk not in loaded:
            self._config[sk] = slice_cfg[sk]

  def _load_weights_if_available(self) -> None:
    """Indexes safetensors weights from `self.model_dir` if present."""
    single_st = os.path.join(self.model_dir, "model.safetensors")
    index_st = os.path.join(self.model_dir, "model.safetensors.index.json")
    if os.path.exists(single_st):
      with safetensors.safe_open(single_st, framework="pt", device="cpu") as f:
        for k in f.keys():
          self._weight_map[k] = single_st
    elif os.path.exists(index_st):
      with open(index_st, "r") as f:
        index_data = json.load(f)
      for k, shard in index_data.get("weight_map", {}).items():
        shard_path = os.path.join(self.model_dir, shard)
        if os.path.exists(shard_path):
          self._weight_map[k] = shard_path

  def _require_weights_or_random(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> None:
    if (
        not export_config.use_random_weights
        and not self._state_dict
        and not self._weight_map
    ):
      raise FileNotFoundError(
          f"No safetensors weights found in {self.model_dir} and "
          "use_random_weights=False."
      )

  def _get_weight(self, key: str) -> torch.Tensor | None:
    """Retrieves a weight tensor by key from memory or sharded safetensors."""
    if key in self._state_dict:
      return self._state_dict[key]
    shard_path = self._weight_map.get(key)
    if shard_path and os.path.exists(shard_path):
      with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
        if key in f.keys():
          return f.get_tensor(key)
    return None

  def _copy_weight(
      self,
      target: torch.Tensor,
      *keys: str,
      strict: bool = True,
      populated: set[int] | None = None,
  ) -> None:
    """Copies the first matching checkpoint tensor in `keys` into `target`.

    Args:
      target: Destination parameter tensor.
      *keys: Candidate checkpoint tensor keys in priority order.
      strict: When True, raises KeyError if none of `keys` exists in the
        checkpoint and ValueError if a matching key's shape does not match
        `target.shape`.
      populated: Optional set of `data_ptr()` integers recording parameters that
        were populated from the checkpoint.

    Raises:
      KeyError: If `strict` is True and none of `keys` exists in the checkpoint.
      ValueError: If `strict` is True and a matching tensor's shape does not
        match `target.shape`.
    """
    for key in keys:
      raw = self._get_weight(key)
      if raw is None:
        continue
      if raw.shape != target.shape:
        if strict:
          raise ValueError(
              f"Weight shape mismatch for {key!r}: checkpoint has"
              f" {tuple(raw.shape)}, target expects {tuple(target.shape)}."
          )
        continue
      src = raw.to(dtype=target.dtype, device=target.device)
      with torch.no_grad():
        target.copy_(src)
      if populated is not None:
        populated.add(target.data_ptr())
      return
    if strict:
      raise KeyError(f"Missing checkpoint weight; tried keys: {keys}.")

  def _copy_qkv_in_proj(
      self,
      q_proj: nn.Linear,
      k_proj: nn.Linear,
      v_proj: nn.Linear,
      weight_key: str,
      bias_key: str,
      strict: bool,
      populated: set[int],
  ) -> None:
    """Copies fused QKV `in_proj_weight` and `in_proj_bias` into Q/K/V layers."""
    hidden = q_proj.weight.shape[1]
    in_proj_w = self._get_weight(weight_key)
    if in_proj_w is None:
      if strict:
        raise KeyError(f"Missing checkpoint weight: {weight_key!r}.")
    else:
      expected_w_shape = (3 * hidden, hidden)
      if in_proj_w.shape != expected_w_shape:
        if strict:
          raise ValueError(
              f"Weight shape mismatch for {weight_key!r}: checkpoint has"
              f" {tuple(in_proj_w.shape)}, target expects {expected_w_shape}."
          )
      else:
        qw, kw, vw = in_proj_w.to(dtype=q_proj.weight.dtype).chunk(3, dim=0)
        with torch.no_grad():
          q_proj.weight.copy_(qw)
          k_proj.weight.copy_(kw)
          v_proj.weight.copy_(vw)
        populated.update((
            q_proj.weight.data_ptr(),
            k_proj.weight.data_ptr(),
            v_proj.weight.data_ptr(),
        ))

    in_proj_b = self._get_weight(bias_key)
    if in_proj_b is None:
      if strict:
        raise KeyError(f"Missing checkpoint weight: {bias_key!r}.")
    else:
      expected_b_shape = (3 * hidden,)
      if in_proj_b.shape != expected_b_shape:
        if strict:
          raise ValueError(
              f"Weight shape mismatch for {bias_key!r}: checkpoint has"
              f" {tuple(in_proj_b.shape)}, target expects {expected_b_shape}."
          )
      else:
        qb, kb, vb = in_proj_b.to(dtype=q_proj.bias.dtype).chunk(3, dim=0)
        with torch.no_grad():
          q_proj.bias.copy_(qb)
          k_proj.bias.copy_(kb)
          v_proj.bias.copy_(vb)
        populated.update((
            q_proj.bias.data_ptr(),
            k_proj.bias.data_ptr(),
            v_proj.bias.data_ptr(),
        ))

  @staticmethod
  def _verify_all_parameters_populated(
      mod: nn.Module, populated: set[int], module_name: str
  ) -> None:
    """Raises ValueError if any parameter in `mod` was not populated."""
    unpopulated = [
        name
        for name, param in mod.named_parameters()
        if param.data_ptr() not in populated
    ]
    if unpopulated:
      raise ValueError(
          f"Unpopulated parameters in {module_name}: {unpopulated}"
      )

  @staticmethod
  def _maybe_quantize(out_path: str, quant_recipe: str | None) -> str:
    """Applies comma-separated quantization recipes to `out_path` if specified."""
    quant_list = quant_recipe.split(",") if quant_recipe else [None]
    final_path = out_path
    for recipe in quant_list:
      prev_path = final_path
      final_path = quantization.maybe_quantize_model(final_path, recipe)
      if final_path != prev_path and os.path.exists(prev_path):
        os.remove(prev_path)
      gc.collect()
    return final_path

  def export_vision_encoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports SigLIP + Perceiver resampler vision encoder to TFLite."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    v_cfg = self._config.get("vision_config", {})
    vision_hidden = int(v_cfg.get("hidden_size", 1152))
    vision_inter = int(v_cfg.get("intermediate_size", 4304))
    vision_heads = int(v_cfg.get("num_attention_heads", 16))
    default_vision_layers = (
        min(int(v_cfg.get("num_hidden_layers", 1)), 1)
        if export_config.use_random_weights
        else int(v_cfg.get("num_hidden_layers", 27))
    )
    vision_layers = int(
        export_config.extra_kwargs.get(
            "vision_num_layers", default_vision_layers
        )
    )
    patch_size = int(
        self._config.get("patch_size", v_cfg.get("patch_size", 14))
    )
    image_size = int(v_cfg.get("image_size", 980))
    max_patches = max(16, (image_size // max(1, patch_size)) ** 2)
    query_num = int(self._config.get("query_num", 64))
    llm_hidden = int(self._config.get("hidden_size", 4096))
    default_num_patches = 16 if export_config.use_random_weights else 1088
    num_patches = int(
        export_config.extra_kwargs.get(
            "vision_num_patches", default_num_patches
        )
    )

    mod = MiniCpmOVisionEncoderModule(
        vision_hidden_size=vision_hidden,
        vision_intermediate_size=vision_inter,
        vision_num_layers=vision_layers,
        vision_num_heads=vision_heads,
        patch_size=patch_size,
        max_patches=max_patches,
        query_num=query_num,
        llm_hidden_size=llm_hidden,
    ).eval()
    populated: set[int] = set()
    self._copy_weight(
        mod.patch_embedding.weight,
        "vpm.embeddings.patch_embedding.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.patch_embedding.bias,
        "vpm.embeddings.patch_embedding.bias",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.pos_embedding.weight,
        "vpm.embeddings.position_embedding.weight",
        strict=strict,
        populated=populated,
    )
    for idx, blk in enumerate(mod.blocks):
      p = f"vpm.encoder.layers.{idx}"
      self._copy_weight(
          blk.norm1.weight,
          f"{p}.layer_norm1.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.norm1.bias,
          f"{p}.layer_norm1.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.q_proj.weight,
          f"{p}.self_attn.q_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.q_proj.bias,
          f"{p}.self_attn.q_proj.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.k_proj.weight,
          f"{p}.self_attn.k_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.k_proj.bias,
          f"{p}.self_attn.k_proj.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.v_proj.weight,
          f"{p}.self_attn.v_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.v_proj.bias,
          f"{p}.self_attn.v_proj.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.o_proj.weight,
          f"{p}.self_attn.out_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.o_proj.bias,
          f"{p}.self_attn.out_proj.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.norm2.weight,
          f"{p}.layer_norm2.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.norm2.bias,
          f"{p}.layer_norm2.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.fc1.weight,
          f"{p}.mlp.fc1.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.fc1.bias,
          f"{p}.mlp.fc1.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.fc2.weight,
          f"{p}.mlp.fc2.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.fc2.bias,
          f"{p}.mlp.fc2.bias",
          strict=strict,
          populated=populated,
      )
    self._copy_weight(
        mod.post_layernorm.weight,
        "vpm.post_layernorm.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.post_layernorm.bias,
        "vpm.post_layernorm.bias",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.latents, "resampler.query", strict=strict, populated=populated
    )
    self._copy_weight(
        mod.kv_proj.weight,
        "resampler.kv_proj.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.ln_q.weight,
        "resampler.ln_q.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.ln_q.bias,
        "resampler.ln_q.bias",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.ln_kv.weight,
        "resampler.ln_kv.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.ln_kv.bias,
        "resampler.ln_kv.bias",
        strict=strict,
        populated=populated,
    )
    self._copy_qkv_in_proj(
        mod.attn_q,
        mod.attn_k,
        mod.attn_v,
        "resampler.attn.in_proj_weight",
        "resampler.attn.in_proj_bias",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.attn_out.weight,
        "resampler.attn.out_proj.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.attn_out.bias,
        "resampler.attn.out_proj.bias",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.ln_post.weight,
        "resampler.ln_post.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.ln_post.bias,
        "resampler.ln_post.bias",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.proj, "resampler.proj", strict=strict, populated=populated
    )
    if strict:
      self._verify_all_parameters_populated(mod, populated, "vision_encoder")

    sample_kwargs = {
        "pixel_values": torch.zeros(
            1, 3, patch_size, patch_size * num_patches, dtype=torch.float32
        ),
        "position_ids": (
            torch.arange(num_patches, dtype=torch.int32).unsqueeze(0)
        ),
    }
    out_path = os.path.join(output_dir, "vision_encoder_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    quant_recipe = (
        export_config.vision_encoder_quantization_recipe
        if export_config.vision_encoder_quantization_recipe is not None
        else export_config.quantization_recipe
    )
    return self._maybe_quantize(out_path, quant_recipe)

  def export_audio_encoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports Whisper-medium audio encoder + projection to TFLite."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    a_cfg = self._config.get("audio_config", {})
    num_mel_bins = int(a_cfg.get("num_mel_bins", 80))
    audio_hidden = int(a_cfg.get("d_model", 1024))
    audio_inter = int(a_cfg.get("encoder_ffn_dim", 4096))
    audio_heads = int(a_cfg.get("encoder_attention_heads", 16))
    default_audio_layers = (
        min(int(a_cfg.get("encoder_layers", 1)), 1)
        if export_config.use_random_weights
        else int(a_cfg.get("encoder_layers", 24))
    )
    audio_layers = int(
        export_config.extra_kwargs.get("audio_num_layers", default_audio_layers)
    )
    max_source_pos = int(a_cfg.get("max_source_positions", 1500))
    pool_step = int(self._config.get("audio_pool_step", 5))
    llm_hidden = int(self._config.get("hidden_size", 4096))
    mel_frames = int(export_config.extra_kwargs.get("audio_mel_frames", 100))

    mod = MiniCpmOAudioEncoderModule(
        num_mel_bins=num_mel_bins,
        audio_hidden_size=audio_hidden,
        audio_intermediate_size=audio_inter,
        audio_num_layers=audio_layers,
        audio_num_heads=audio_heads,
        audio_pool_step=pool_step,
        llm_hidden_size=llm_hidden,
        max_source_positions=max_source_pos,
    ).eval()
    populated: set[int] = set()
    self._copy_weight(
        mod.conv1.weight, "apm.conv1.weight", strict=strict, populated=populated
    )
    self._copy_weight(
        mod.conv1.bias, "apm.conv1.bias", strict=strict, populated=populated
    )
    self._copy_weight(
        mod.conv2.weight, "apm.conv2.weight", strict=strict, populated=populated
    )
    self._copy_weight(
        mod.conv2.bias, "apm.conv2.bias", strict=strict, populated=populated
    )
    self._copy_weight(
        mod.embed_positions.weight,
        "apm.embed_positions.weight",
        strict=strict,
        populated=populated,
    )
    for idx, blk in enumerate(mod.blocks):
      p = f"apm.layers.{idx}"
      self._copy_weight(
          blk.norm1.weight,
          f"{p}.self_attn_layer_norm.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.norm1.bias,
          f"{p}.self_attn_layer_norm.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.q_proj.weight,
          f"{p}.self_attn.q_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.q_proj.bias,
          f"{p}.self_attn.q_proj.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.k_proj.weight,
          f"{p}.self_attn.k_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.v_proj.weight,
          f"{p}.self_attn.v_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.v_proj.bias,
          f"{p}.self_attn.v_proj.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.o_proj.weight,
          f"{p}.self_attn.out_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.o_proj.bias,
          f"{p}.self_attn.out_proj.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.norm2.weight,
          f"{p}.final_layer_norm.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.norm2.bias,
          f"{p}.final_layer_norm.bias",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.fc1.weight,
          f"{p}.fc1.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.fc1.bias, f"{p}.fc1.bias", strict=strict, populated=populated
      )
      self._copy_weight(
          blk.fc2.weight,
          f"{p}.fc2.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          blk.fc2.bias, f"{p}.fc2.bias", strict=strict, populated=populated
      )
    self._copy_weight(
        mod.layer_norm.weight,
        "apm.layer_norm.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.layer_norm.bias,
        "apm.layer_norm.bias",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.proj1.weight,
        "audio_projection_layer.linear1.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.proj1.bias,
        "audio_projection_layer.linear1.bias",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.proj2.weight,
        "audio_projection_layer.linear2.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.proj2.bias,
        "audio_projection_layer.linear2.bias",
        strict=strict,
        populated=populated,
    )
    if strict:
      self._verify_all_parameters_populated(mod, populated, "audio_encoder")

    # TODO(b/568027544): Enable `valid_mel_frames` and `position_offset` inputs
    # by default once `MiniCpmOFrontendStage` passes per-chunk frame lengths and
    # position offsets at runtime.
    sample_kwargs: dict[str, torch.Tensor] = {
        "mel_features": torch.zeros(
            1, num_mel_bins, mel_frames, dtype=torch.float32
        ),
    }
    if bool(export_config.extra_kwargs.get("audio_include_mask_inputs", False)):
      sample_kwargs["valid_mel_frames"] = torch.tensor(
          [mel_frames], dtype=torch.int32
      )
      sample_kwargs["position_offset"] = torch.tensor([0], dtype=torch.int32)
    out_path = os.path.join(output_dir, "audio_encoder_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    quant_recipe = (
        export_config.audio_encoder_quantization_recipe
        if export_config.audio_encoder_quantization_recipe is not None
        else export_config.quantization_recipe
    )
    return self._maybe_quantize(out_path, quant_recipe)

  def export_text_embedder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports Qwen3/Qwen2 text token embedding lookup table to TFLite."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    vocab_size = int(self._config.get("vocab_size", 151748))
    llm_hidden = int(self._config.get("hidden_size", 4096))
    mod = MiniCpmOTextEmbedderModule(
        vocab_size=vocab_size, llm_hidden_size=llm_hidden
    ).eval()
    populated: set[int] = set()
    self._copy_weight(
        mod.embed_tokens.weight,
        "llm.model.embed_tokens.weight",
        strict=strict,
        populated=populated,
    )
    if strict:
      self._verify_all_parameters_populated(mod, populated, "text_embedder")

    sample_kwargs = {"token_ids": torch.zeros(1, 1, dtype=torch.int32)}
    out_path = os.path.join(output_dir, "text_embedder_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return self._maybe_quantize(out_path, export_config.quantization_recipe)

  def export_llm_backbone(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports Qwen3/Qwen2 LLM backbone (`prefill_*` + `decode` + KV cache) to TFLite."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    llm_hidden = int(self._config.get("hidden_size", 4096))
    default_inter = (
        min(int(self._config.get("intermediate_size", 12288)), 1024)
        if export_config.use_random_weights
        else int(self._config.get("intermediate_size", 12288))
    )
    llm_inter = int(
        export_config.extra_kwargs.get("llm_intermediate_size", default_inter)
    )
    default_layers = (
        min(int(self._config.get("num_hidden_layers", 1)), 1)
        if export_config.use_random_weights
        else int(self._config.get("num_hidden_layers", 36))
    )
    llm_layers = int(
        export_config.extra_kwargs.get("llm_num_layers", default_layers)
    )
    llm_heads = int(self._config.get("num_attention_heads", 32))
    llm_kv_heads = int(self._config.get("num_key_value_heads", llm_heads))
    vocab_size = int(self._config.get("vocab_size", 151748))
    rms_norm_eps = float(self._config.get("rms_norm_eps", 1e-6))
    rope_theta = float(self._config.get("rope_theta", 1000000.0))
    max_pos = int(self._config.get("max_position_embeddings", 40960))
    tie_word_embeddings = bool(self._config.get("tie_word_embeddings", False))
    use_sliding_window = bool(self._config.get("use_sliding_window", False))
    sliding_window = (
        int(self._config.get("sliding_window", 131072))
        if use_sliding_window
        else None
    )

    all_keys = set(self._weight_map) | set(self._state_dict)
    has_qkv_bias = any(k.endswith(".self_attn.q_proj.bias") for k in all_keys)
    has_qk_norm = any(k.endswith(".self_attn.q_norm.weight") for k in all_keys)
    if has_qk_norm:
      is_qwen3 = True
    elif has_qkv_bias:
      is_qwen3 = False
    else:
      is_qwen3 = str(self._config.get("version", "4.5")) == "4.5" and not bool(
          self._config.get("attention_bias", False)
      )

    head_dim = int(
        self._config.get("head_dim", max(1, llm_hidden // max(1, llm_heads)))
    )
    if is_qwen3:
      llm_config = transformers.Qwen3Config(
          vocab_size=vocab_size,
          hidden_size=llm_hidden,
          intermediate_size=llm_inter,
          num_hidden_layers=llm_layers,
          num_attention_heads=llm_heads,
          num_key_value_heads=llm_kv_heads,
          head_dim=head_dim,
          hidden_act=str(self._config.get("hidden_act", "silu")),
          max_position_embeddings=max_pos,
          rms_norm_eps=rms_norm_eps,
          rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
          tie_word_embeddings=tie_word_embeddings,
          attention_bias=bool(self._config.get("attention_bias", False)),
          use_sliding_window=use_sliding_window,
          sliding_window=sliding_window,
          bos_token_id=int(self._config.get("bos_token_id", 151643)),
          eos_token_id=int(self._config.get("eos_token_id", 151645)),
      )
      patch_name = "qwen3"
      model_cls = transformers.Qwen3ForCausalLM
    else:
      llm_config = transformers.Qwen2Config(
          vocab_size=vocab_size,
          hidden_size=llm_hidden,
          intermediate_size=llm_inter,
          num_hidden_layers=llm_layers,
          num_attention_heads=llm_heads,
          num_key_value_heads=llm_kv_heads,
          hidden_act=str(self._config.get("hidden_act", "silu")),
          max_position_embeddings=max_pos,
          rms_norm_eps=rms_norm_eps,
          rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
          tie_word_embeddings=tie_word_embeddings,
          use_sliding_window=use_sliding_window,
          sliding_window=sliding_window,
          bos_token_id=int(self._config.get("bos_token_id", 151643)),
          eos_token_id=int(self._config.get("eos_token_id", 151645)),
      )
      patch_name = "qwen2"
      model_cls = transformers.Qwen2ForCausalLM
    llm_config._attn_implementation = "lrt_transposed_attention"  # pylint: disable=protected-access

    has_checkpoint_weights = "llm.lm_head.weight" in self._weight_map or (
        tie_word_embeddings
        and "llm.model.embed_tokens.weight" in self._weight_map
    )
    use_v2 = strict and bool(export_config.use_v2) and has_checkpoint_weights
    base_loader = None
    with model_ext_patches.get_patch_context(patch_name):
      if use_v2:
        with weights_loader_lib.init_params_on_meta():
          qwen_model = model_cls(llm_config)
        qwen_model.eval()
        base_loader = _MiniCpmOLlmWeightsLoader(
            self.model_dir, tie_word_embeddings=tie_word_embeddings
        )
        weights_loader_lib.load_persistent_buffers_(qwen_model, base_loader)
      else:
        qwen_model = model_cls(llm_config).eval().to(torch.float32)
        populated: set[int] = set()
        self._copy_weight(
            qwen_model.model.embed_tokens.weight,
            "llm.model.embed_tokens.weight",
            strict=strict,
            populated=populated,
        )
        for idx, layer in enumerate(qwen_model.model.layers):
          p = f"llm.model.layers.{idx}"
          self._copy_weight(
              layer.input_layernorm.weight,
              f"{p}.input_layernorm.weight",
              strict=strict,
              populated=populated,
          )
          self._copy_weight(
              layer.self_attn.q_proj.weight,
              f"{p}.self_attn.q_proj.weight",
              strict=strict,
              populated=populated,
          )
          if layer.self_attn.q_proj.bias is not None:
            self._copy_weight(
                layer.self_attn.q_proj.bias,
                f"{p}.self_attn.q_proj.bias",
                strict=strict,
                populated=populated,
            )
          self._copy_weight(
              layer.self_attn.k_proj.weight,
              f"{p}.self_attn.k_proj.weight",
              strict=strict,
              populated=populated,
          )
          if layer.self_attn.k_proj.bias is not None:
            self._copy_weight(
                layer.self_attn.k_proj.bias,
                f"{p}.self_attn.k_proj.bias",
                strict=strict,
                populated=populated,
            )
          self._copy_weight(
              layer.self_attn.v_proj.weight,
              f"{p}.self_attn.v_proj.weight",
              strict=strict,
              populated=populated,
          )
          if layer.self_attn.v_proj.bias is not None:
            self._copy_weight(
                layer.self_attn.v_proj.bias,
                f"{p}.self_attn.v_proj.bias",
                strict=strict,
                populated=populated,
            )
          if hasattr(layer.self_attn, "q_norm"):
            self._copy_weight(
                layer.self_attn.q_norm.weight,
                f"{p}.self_attn.q_norm.weight",
                strict=strict,
                populated=populated,
            )
          if hasattr(layer.self_attn, "k_norm"):
            self._copy_weight(
                layer.self_attn.k_norm.weight,
                f"{p}.self_attn.k_norm.weight",
                strict=strict,
                populated=populated,
            )
          self._copy_weight(
              layer.self_attn.o_proj.weight,
              f"{p}.self_attn.o_proj.weight",
              strict=strict,
              populated=populated,
          )
          self._copy_weight(
              layer.post_attention_layernorm.weight,
              f"{p}.post_attention_layernorm.weight",
              strict=strict,
              populated=populated,
          )
          self._copy_weight(
              layer.mlp.gate_proj.weight,
              f"{p}.mlp.gate_proj.weight",
              strict=strict,
              populated=populated,
          )
          self._copy_weight(
              layer.mlp.up_proj.weight,
              f"{p}.mlp.up_proj.weight",
              strict=strict,
              populated=populated,
          )
          self._copy_weight(
              layer.mlp.down_proj.weight,
              f"{p}.mlp.down_proj.weight",
              strict=strict,
              populated=populated,
          )
        self._copy_weight(
            qwen_model.model.norm.weight,
            "llm.model.norm.weight",
            strict=strict,
            populated=populated,
        )
        if tie_word_embeddings:
          self._copy_weight(
              qwen_model.lm_head.weight,
              "llm.lm_head.weight",
              "llm.model.embed_tokens.weight",
              strict=strict,
              populated=populated,
          )
        else:
          self._copy_weight(
              qwen_model.lm_head.weight,
              "llm.lm_head.weight",
              strict=strict,
              populated=populated,
          )
        if strict:
          self._verify_all_parameters_populated(
              qwen_model, populated, "llm_backbone"
          )

    cache_lengths = export_config.cache_lengths or [export_config.cache_length]
    prefill_lengths = export_config.prefill_lengths or [128]

    out_path = os.path.join(output_dir, "llm_backbone_fp32.tflite")
    intermediates_dir = os.path.join(output_dir, "llm_backbone_intermediates")

    with model_ext_patches.patch_model(qwen_model, patch_name, export_config):
      qwen_model.set_attn_implementation("lrt_transposed_attention")
      signatures = []
      for cache_len in cache_lengths:
        iter_config = dataclasses.replace(
            export_config,
            cache_length=cache_len,
            prefill_lengths=prefill_lengths,
            externalize_embedder=True,
        )
        prefill_module = external_emb_module.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder(
            qwen_model, iter_config
        )
        decode_module = _MiniCpmOLlmGenerateModule(qwen_model, iter_config)
        for module in (prefill_module, decode_module):
          for sig_name, (inputs, dynamic_shapes) in module.get_sample_inputs(
              llm_config
          ).items():
            if len(cache_lengths) > 1:
              sig_name = f"{sig_name}_cache_{cache_len}"
            signatures.append((sig_name, module, inputs, dynamic_shapes))

      signature_modules = list({id(s[1]): s[1] for s in signatures}.values())
      with contextlib.ExitStack() as stack:
        weights_loader = base_loader
        if weights_loader is not None:
          buffer_values = stack.enter_context(
              weights_loader_lib.buffers_on_meta(*signature_modules)
          )
          weights_loader = weights_loader_lib.with_tensors(
              weights_loader, buffer_values
          )
        converter = converter_utils.Converter()
        for sig_name, module, inputs, _ in signatures:
          converter.add_signature(
              sig_name,
              module.eval(),
              sample_kwargs=inputs,
          )
        if use_v2:
          converter.convert(
              lightweight_conversion=export_config.experimental_lightweight_conversion,
              strict_export=False,
              use_v2=True,
              delete_in_memory_params=True,
              allow_reuse_intermediates=False,
              weights_loader=weights_loader,
              export_dir=intermediates_dir,
              output_file_path=out_path,
              _litert_converter_flags={
                  "fold_fp16_casts_into_fully_connected": True,
              },
          )
        else:
          lrt_model = converter.convert(
              lightweight_conversion=export_config.experimental_lightweight_conversion,
              strict_export=False,
          )
          lrt_model.export(out_path)
          del lrt_model

    if os.path.exists(intermediates_dir):
      shutil.rmtree(intermediates_dir, ignore_errors=True)
    del qwen_model
    gc.collect()

    quant_recipe = export_config.quantization_recipe
    if (
        not export_config.use_random_weights
        and llm_layers > 2
        and (
            not quant_recipe
            or str(quant_recipe).strip().lower()
            in ("none", "null", "false", "")
        )
    ):
      logging.warning(
          "Exporting %d-layer MiniCPM-o LLM backbone without quantization"
          " (quantization_recipe=%r); resulting FP32 TFLite file may exceed"
          " 30 GB.",
          llm_layers,
          quant_recipe,
      )

    return self._maybe_quantize(out_path, quant_recipe)

  def _copy_weight_norm(
      self,
      target: torch.Tensor,
      weight_key: str,
      orig0_key: str,
      orig1_key: str,
      strict: bool,
      populated: set[int],
  ) -> None:
    """Copies either a plain `weight_key` or PyTorch `weight_norm` `(g, v)` pair."""
    if self._get_weight(weight_key) is not None:
      self._copy_weight(target, weight_key, strict=strict, populated=populated)
      return
    g = self._get_weight(orig0_key)
    v = self._get_weight(orig1_key)
    if g is not None and v is not None:
      if v.shape != target.shape or g.shape[0] != target.shape[0]:
        if strict:
          raise ValueError(
              f"Weight shape mismatch for ({orig0_key!r}, {orig1_key!r}):"
              f" got g={tuple(g.shape)}, v={tuple(v.shape)}, expected"
              f" {tuple(target.shape)}."
          )
        return
      gf = g.to(dtype=torch.float32)
      vf = v.to(dtype=torch.float32)
      norm = vf.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
      while norm.ndim < vf.ndim:
        norm = norm.unsqueeze(-1)
      while gf.ndim < vf.ndim:
        gf = gf.unsqueeze(-1)
      w = (gf * vf / norm).to(dtype=target.dtype, device=target.device)
      with torch.no_grad():
        target.copy_(w)
      populated.add(target.data_ptr())
      return
    if strict:
      raise KeyError(
          "Missing checkpoint weight; tried keys:"
          f" {(weight_key, orig0_key, orig1_key)}."
      )

  @staticmethod
  def _adapt_tensor_to_target(
      target: torch.Tensor, src: torch.Tensor, populated: set[int]
  ) -> None:
    """Copies overlapping slices of `src` into `target` (zero-padding if needed)."""
    src_f = src.to(dtype=target.dtype, device=target.device)
    with torch.no_grad():
      target.zero_()
      slices = tuple(
          slice(0, min(t_dim, s_dim))
          for t_dim, s_dim in zip(target.shape, src_f.shape)
      )
      target[slices].copy_(src_f[slices])
    populated.add(target.data_ptr())

  def export_tts_projector(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports LLM-to-TTS semantic projector (`tf_lite_tts_projector`) to TFLite."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    llm_hidden = int(self._config.get("hidden_size", 4096))
    tts_cfg = self._config.get("tts_config", {})
    tts_hidden = int(tts_cfg.get("hidden_size", 768))
    mod = MiniCpmOTtsProjectorModule(
        llm_hidden_size=llm_hidden, tts_hidden_size=tts_hidden
    ).eval()
    populated: set[int] = set()
    self._copy_weight(
        mod.linear1.weight,
        "tts.projector_semantic.linear1.weight",
        "tts.projector.linear1.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.linear1.bias,
        "tts.projector_semantic.linear1.bias",
        "tts.projector.linear1.bias",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.linear2.weight,
        "tts.projector_semantic.linear2.weight",
        "tts.projector.linear2.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight(
        mod.linear2.bias,
        "tts.projector_semantic.linear2.bias",
        "tts.projector.linear2.bias",
        strict=strict,
        populated=populated,
    )
    if strict:
      self._verify_all_parameters_populated(mod, populated, "tts_projector")

    sample_kwargs = {
        "llm_hidden": torch.zeros(1, 1, llm_hidden, dtype=torch.float32)
    }
    out_path = os.path.join(output_dir, "tts_projector_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return self._maybe_quantize(out_path, export_config.quantization_recipe)

  def export_tts_decoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports autoregressive TTS decoder (`tf_lite_tts_decoder`) to TFLite."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    tts_cfg = self._config.get("tts_config", {})
    tts_hidden = int(tts_cfg.get("hidden_size", 768))
    tts_inter = int(tts_cfg.get("intermediate_size", 3072))
    default_tts_layers = (
        min(int(tts_cfg.get("num_hidden_layers", 1)), 1)
        if export_config.use_random_weights
        else int(tts_cfg.get("num_hidden_layers", 20))
    )
    tts_layers = int(
        export_config.extra_kwargs.get("tts_num_layers", default_tts_layers)
    )
    tts_heads = int(tts_cfg.get("num_attention_heads", 12))
    tts_vocab = int(tts_cfg.get("num_audio_tokens", 6562))
    tts_mel_bins = int(tts_cfg.get("num_mel_bins", 100))
    default_min_steps = 4 if export_config.use_random_weights else 64
    min_tts_steps = int(
        export_config.extra_kwargs.get("min_tts_steps", default_min_steps)
    )

    mod = MiniCpmOTtsDecoderModule(
        tts_hidden_size=tts_hidden,
        tts_intermediate_size=tts_inter,
        tts_num_layers=tts_layers,
        tts_num_heads=tts_heads,
        tts_vocab_size=tts_vocab,
        tts_num_mel_bins=tts_mel_bins,
        min_tts_steps=min_tts_steps,
    ).eval()
    populated: set[int] = set()
    self._copy_weight(
        mod.emb_code.weight,
        "tts.emb_code.0.weight",
        "tts.emb_code.weight",
        strict=strict,
        populated=populated,
    )
    emb_text = self._get_weight("tts.emb_text.weight")
    if emb_text is not None:
      audio_bos_id = int(tts_cfg.get("audio_bos_token_id", 151703))
      if 0 <= audio_bos_id < emb_text.shape[0] and (
          emb_text.shape[1] == tts_hidden
      ):
        with torch.no_grad():
          mod.bos_emb.copy_(
              emb_text[audio_bos_id]
              .to(dtype=mod.bos_emb.dtype, device=mod.bos_emb.device)
              .reshape(1, 1, tts_hidden)
          )
    for idx, layer in enumerate(mod.layers):
      p = f"tts.model.layers.{idx}"
      self._copy_weight(
          layer.input_layernorm.weight,
          f"{p}.input_layernorm.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          layer.q_proj.weight,
          f"{p}.self_attn.q_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          layer.k_proj.weight,
          f"{p}.self_attn.k_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          layer.v_proj.weight,
          f"{p}.self_attn.v_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          layer.o_proj.weight,
          f"{p}.self_attn.o_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          layer.post_attention_layernorm.weight,
          f"{p}.post_attention_layernorm.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          layer.gate_proj.weight,
          f"{p}.mlp.gate_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          layer.up_proj.weight,
          f"{p}.mlp.up_proj.weight",
          strict=strict,
          populated=populated,
      )
      self._copy_weight(
          layer.down_proj.weight,
          f"{p}.mlp.down_proj.weight",
          strict=strict,
          populated=populated,
      )
    self._copy_weight(
        mod.norm.weight,
        "tts.model.norm.weight",
        strict=strict,
        populated=populated,
    )
    self._copy_weight_norm(
        mod.head_code.weight,
        "tts.head_code.0.weight",
        "tts.head_code.0.parametrizations.weight.original0",
        "tts.head_code.0.parametrizations.weight.original1",
        strict=strict,
        populated=populated,
    )
    mel_head_w = self._get_weight("tts.mel_head.weight")
    if mel_head_w is not None and mel_head_w.shape == (
        tts_mel_bins,
        tts_hidden,
    ):
      with torch.no_grad():
        mod.mel_proj.copy_(
            mel_head_w.t().to(
                dtype=mod.mel_proj.dtype, device=mod.mel_proj.device
            )
        )
    if strict:
      self._verify_all_parameters_populated(mod, populated, "tts_decoder")

    sample_kwargs = {
        "cond_emb": torch.zeros(1, 1, tts_hidden, dtype=torch.float32),
        "prev_audio_code": torch.zeros(1, 1, dtype=torch.int32),
        "step_idx": torch.zeros(1, dtype=torch.int32),
    }
    out_path = os.path.join(output_dir, "tts_decoder_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return self._maybe_quantize(out_path, export_config.quantization_recipe)

  def _load_hift_vocoder_weights(
      self, mod: MiniCpmOAudioVocoderModule, hift_path: str, populated: set[int]
  ) -> None:
    """Adapts CosyVoice2 HiFT (`assets/token2wav/hift.pt`) weights into `mod`."""
    try:
      hift_sd = torch.load(hift_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
      hift_sd = torch.load(hift_path, map_location="cpu")
    if not isinstance(hift_sd, dict):
      return

    def _resolve_wn(prefix: str) -> torch.Tensor | None:
      if f"{prefix}.weight" in hift_sd:
        return hift_sd[f"{prefix}.weight"]
      g = hift_sd.get(f"{prefix}.parametrizations.weight.original0")
      v = hift_sd.get(f"{prefix}.parametrizations.weight.original1")
      if g is not None and v is not None:
        gf = g.to(torch.float32)
        vf = v.to(torch.float32)
        norm = vf.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
        while norm.ndim < vf.ndim:
          norm = norm.unsqueeze(-1)
        while gf.ndim < vf.ndim:
          gf = gf.unsqueeze(-1)
        return gf * vf / norm
      return None

    w_pre = _resolve_wn("conv_pre")
    b_pre = hift_sd.get("conv_pre.bias")
    if w_pre is not None:
      self._adapt_tensor_to_target(mod.input_conv.weight, w_pre, populated)
    if b_pre is not None and mod.input_conv.bias is not None:
      self._adapt_tensor_to_target(mod.input_conv.bias, b_pre, populated)

    for idx, (up, res) in enumerate(zip(mod.upsamples, mod.res_convs)):
      w_up = _resolve_wn(f"ups.{idx}")
      b_up = hift_sd.get(f"ups.{idx}.bias")
      if w_up is not None:
        self._adapt_tensor_to_target(up.weight, w_up, populated)
      if b_up is not None and up.bias is not None:
        self._adapt_tensor_to_target(up.bias, b_up, populated)

      rb_idx = idx * 3
      w_res = _resolve_wn(f"resblocks.{rb_idx}.convs1.0")
      b_res = hift_sd.get(f"resblocks.{rb_idx}.convs1.0.bias")
      if w_res is not None:
        self._adapt_tensor_to_target(res.weight, w_res, populated)
      if b_res is not None and res.bias is not None:
        self._adapt_tensor_to_target(res.bias, b_res, populated)

    w_post = _resolve_wn("conv_post")
    b_post = hift_sd.get("conv_post.bias")
    if w_post is not None:
      self._adapt_tensor_to_target(mod.output_conv.weight, w_post, populated)
    if b_post is not None and mod.output_conv.bias is not None:
      self._adapt_tensor_to_target(mod.output_conv.bias, b_post, populated)

  def export_audio_vocoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports 24 kHz mel-to-waveform vocoder (`tf_lite_audio_vocoder`) to TFLite."""
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    tts_cfg = self._config.get("tts_config", {})
    num_mel_bins = int(tts_cfg.get("num_mel_bins", 100))
    default_hidden_ch = 32 if export_config.use_random_weights else 256
    hidden_channels = int(
        export_config.extra_kwargs.get(
            "vocoder_hidden_channels", default_hidden_ch
        )
    )
    chunk_frames = int(
        export_config.extra_kwargs.get("vocoder_chunk_frames", 16)
    )
    mod = MiniCpmOAudioVocoderModule(
        num_mel_bins=num_mel_bins, hidden_channels=hidden_channels
    ).eval()
    populated: set[int] = set()
    all_keys = set(self._weight_map) | set(self._state_dict)
    has_vocoder_keys = any(k.startswith("vocoder.") for k in all_keys)
    hift_path = os.path.join(self.model_dir, "assets", "token2wav", "hift.pt")
    if has_vocoder_keys or (strict and not os.path.exists(hift_path)):
      self._copy_weight(
          mod.input_conv.weight,
          "vocoder.input_conv.weight",
          strict=strict,
          populated=populated,
      )
      if mod.input_conv.bias is not None:
        self._copy_weight(
            mod.input_conv.bias,
            "vocoder.input_conv.bias",
            strict=strict,
            populated=populated,
        )
      for idx, (up, res) in enumerate(zip(mod.upsamples, mod.res_convs)):
        self._copy_weight(
            up.weight,
            f"vocoder.upsamples.{idx}.weight",
            strict=strict,
            populated=populated,
        )
        if up.bias is not None:
          self._copy_weight(
              up.bias,
              f"vocoder.upsamples.{idx}.bias",
              strict=strict,
              populated=populated,
          )
        self._copy_weight(
            res.weight,
            f"vocoder.res_convs.{idx}.weight",
            strict=strict,
            populated=populated,
        )
        if res.bias is not None:
          self._copy_weight(
              res.bias,
              f"vocoder.res_convs.{idx}.bias",
              strict=strict,
              populated=populated,
          )
      self._copy_weight(
          mod.output_conv.weight,
          "vocoder.output_conv.weight",
          strict=strict,
          populated=populated,
      )
      if mod.output_conv.bias is not None:
        self._copy_weight(
            mod.output_conv.bias,
            "vocoder.output_conv.bias",
            strict=strict,
            populated=populated,
        )
      if strict:
        self._verify_all_parameters_populated(mod, populated, "audio_vocoder")
    elif os.path.exists(hift_path):
      self._load_hift_vocoder_weights(mod, hift_path, populated)
      if strict:
        self._verify_all_parameters_populated(mod, populated, "audio_vocoder")

    sample_kwargs = {
        "mel": torch.zeros(1, num_mel_bins, chunk_frames, dtype=torch.float32)
    }
    out_path = os.path.join(output_dir, "audio_vocoder_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return self._maybe_quantize(out_path, export_config.quantization_recipe)

  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports MiniCPM-o 4.5 vision, audio, text-embedder, LLM, and TTS submodules."""
    output_dir = (
        (
            export_config.work_dir
            if export_config.bundle_litert_lm
            else export_config.output_dir
        )
        or export_config.output_dir
        or export_config.work_dir
    )
    if not output_dir:
      raise ValueError("Either output_dir or work_dir must be specified.")
    os.makedirs(output_dir, exist_ok=True)
    artifacts: dict[str, str] = {}

    raw_targets = export_config.extra_kwargs.get("targets", ["all"])
    if isinstance(raw_targets, str):
      targets = {t.strip() for t in raw_targets.split(",") if t.strip()}
    else:
      targets = set(raw_targets)
    export_all = "all" in targets

    if (
        export_all and export_config.export_vision_encoder
    ) or "vision_encoder" in targets:
      artifacts["vision_encoder"] = self.export_vision_encoder(
          output_dir, export_config
      )
    if (
        export_all and export_config.export_audio_encoder
    ) or "audio_encoder" in targets:
      artifacts["audio_encoder"] = self.export_audio_encoder(
          output_dir, export_config
      )
    if (
        export_all and export_config.externalize_embedder
    ) or "text_embedder" in targets:
      artifacts["text_embedder"] = self.export_text_embedder(
          output_dir, export_config
      )
    if export_all or "llm_backbone" in targets:
      artifacts["llm_backbone"] = self.export_llm_backbone(
          output_dir, export_config
      )
    if export_all or "tts_projector" in targets:
      artifacts["tts_projector"] = self.export_tts_projector(
          output_dir, export_config
      )
    if export_all or "tts_decoder" in targets:
      artifacts["tts_decoder"] = self.export_tts_decoder(
          output_dir, export_config
      )
    if export_all or "audio_vocoder" in targets:
      artifacts["audio_vocoder"] = self.export_audio_vocoder(
          output_dir, export_config
      )

    return artifacts

  def get_any2any_metadata(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> any2any_metadata_pb2.Any2AnyMetadata:
    """Builds Any2AnyMetadata proto for the exported MiniCPM-o 4.5 model."""
    v_cfg = self._config.get("vision_config", {})
    a_cfg = self._config.get("audio_config", {})
    tts_cfg = self._config.get("tts_config", {})
    generate_audio = bool(
        export_config.extra_kwargs.get("generate_audio", False)
    )

    minicpm_proto = any2any_model_type_pb2.MiniCpmO(
        llm_hidden_size=int(self._config.get("hidden_size", 4096)),
        llm_vocab_size=int(self._config.get("vocab_size", 151748)),
        tts_hidden_size=int(tts_cfg.get("hidden_size", 768)),
        tts_vocab_size=int(tts_cfg.get("num_audio_tokens", 6562)),
        tts_num_mel_bins=int(tts_cfg.get("num_mel_bins", 100)),
        vision_scale_resolution=int(
            self._config.get("scale_resolution", v_cfg.get("image_size", 448))
        ),
        vision_patch_size=int(
            self._config.get("patch_size", v_cfg.get("patch_size", 14))
        ),
        vision_max_slice_nums=int(self._config.get("max_slice_nums", 9)),
        num_image_queries=int(self._config.get("query_num", 64)),
        audio_pool_step=int(
            self._config.get("audio_pool_step", a_cfg.get("pool_step", 5))
        ),
        generate_audio=generate_audio,
        max_new_tokens=512,
        max_tts_tokens=1024,
        voice="default",
        language="en-US",
    )
    return any2any_metadata_pb2.Any2AnyMetadata(
        input_sample_rate_hz=16000,
        output_sample_rate_hz=24000,
        supported_languages=["en-US", "zh-CN"],
        any2any_model_type=any2any_model_type_pb2.Any2AnyModelType(
            minicpm_o=minicpm_proto
        ),
    )
