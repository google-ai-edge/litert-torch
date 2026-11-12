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
"""Vision, audio, and text-embedder modules for MiniCPM-o 4.5 LiteRT export."""

from __future__ import annotations

import math

from litert_torch.generative.export_hf.model_ext.minicpm_o import weights_utils
import torch
from torch import nn
import torch.nn.functional as F

_make_linear = weights_utils.make_linear
_make_embedding = weights_utils.make_embedding


def build_1d_sincos_pos_embed(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
  """Builds 1D sin-cos positional embeddings for flattened positions `pos`.

  Args:
    embed_dim: Embedding dimension.
    pos: 1D or N-D tensor of position coordinates.

  Returns:
    Float32 positional embedding tensor of shape `[pos.numel(), embed_dim]`.
  """
  half = max(1, embed_dim // 2)
  omega = torch.arange(half, dtype=torch.float32) / float(half)
  omega = 1.0 / (10000.0**omega)
  out = torch.outer(pos.reshape(-1).to(torch.float32), omega)
  emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
  if emb.shape[1] < embed_dim:
    emb = F.pad(emb, (0, embed_dim - emb.shape[1]))
  return emb[:, :embed_dim]


def build_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int = 70
) -> torch.Tensor:
  """Builds 2D sin-cos positional embeddings matching MiniCPM-o Resampler.

  Args:
    embed_dim: Total embedding dimension across `(w, h)`.
    grid_size: Height and width of the 2D patch grid.

  Returns:
    Float32 positional embedding tensor of shape `[grid_size**2, embed_dim]`.
  """
  grid_h = torch.arange(grid_size, dtype=torch.float32)
  grid_w = torch.arange(grid_size, dtype=torch.float32)
  # Match np.meshgrid(grid_w, grid_h) in MiniCPM-o resampler.py:
  # first half encodes column (w), second half encodes row (h).
  grid_y, grid_x = torch.meshgrid(grid_h, grid_w, indexing="ij")
  half_dim = embed_dim // 2
  emb_x = build_1d_sincos_pos_embed(half_dim, grid_x)
  emb_y = build_1d_sincos_pos_embed(embed_dim - half_dim, grid_y)
  return torch.cat([emb_x, emb_y], dim=1)


_build_1d_sincos_pos_embed = build_1d_sincos_pos_embed
_build_2d_sincos_pos_embed = build_2d_sincos_pos_embed


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
        build_2d_sincos_pos_embed(llm_hidden_size, grid_size),
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
    """Runs SigLIP patch/pos embeddings, transformer blocks, and post-norm."""
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
    """Runs Whisper conv stem, pos embedding, attention blocks, and norm."""
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


SIGLIP_BLOCK_PARAM_MAP = {
    "norm1.weight": "layer_norm1.weight",
    "norm1.bias": "layer_norm1.bias",
    "q_proj.weight": "self_attn.q_proj.weight",
    "q_proj.bias": "self_attn.q_proj.bias",
    "k_proj.weight": "self_attn.k_proj.weight",
    "k_proj.bias": "self_attn.k_proj.bias",
    "v_proj.weight": "self_attn.v_proj.weight",
    "v_proj.bias": "self_attn.v_proj.bias",
    "o_proj.weight": "self_attn.out_proj.weight",
    "o_proj.bias": "self_attn.out_proj.bias",
    "norm2.weight": "layer_norm2.weight",
    "norm2.bias": "layer_norm2.bias",
    "fc1.weight": "mlp.fc1.weight",
    "fc1.bias": "mlp.fc1.bias",
    "fc2.weight": "mlp.fc2.weight",
    "fc2.bias": "mlp.fc2.bias",
}

WHISPER_BLOCK_PARAM_MAP = {
    "norm1.weight": "self_attn_layer_norm.weight",
    "norm1.bias": "self_attn_layer_norm.bias",
    "q_proj.weight": "self_attn.q_proj.weight",
    "q_proj.bias": "self_attn.q_proj.bias",
    "k_proj.weight": "self_attn.k_proj.weight",
    "v_proj.weight": "self_attn.v_proj.weight",
    "v_proj.bias": "self_attn.v_proj.bias",
    "o_proj.weight": "self_attn.out_proj.weight",
    "o_proj.bias": "self_attn.out_proj.bias",
    "norm2.weight": "final_layer_norm.weight",
    "norm2.bias": "final_layer_norm.bias",
}


def load_vision_encoder_weights(
    mod: MiniCpmOVisionEncoderModule,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool,
) -> None:
  """Loads SigLIP and Perceiver resampler weights into `mod`.

  Args:
    mod: Target `MiniCpmOVisionEncoderModule`.
    state_dict: In-memory checkpoint tensor dictionary.
    weight_map: Mapping of checkpoint keys to safetensors shard paths.
    strict: Whether missing or shape-mismatched tensors raise errors.
  """
  populated: set[int] = set()
  weights_utils.copy_parameter_dict(
      {
          mod.patch_embedding.weight: "vpm.embeddings.patch_embedding.weight",
          mod.patch_embedding.bias: "vpm.embeddings.patch_embedding.bias",
          mod.pos_embedding.weight: "vpm.embeddings.position_embedding.weight",
          mod.post_layernorm.weight: "vpm.post_layernorm.weight",
          mod.post_layernorm.bias: "vpm.post_layernorm.bias",
          mod.latents: "resampler.query",
          mod.kv_proj.weight: "resampler.kv_proj.weight",
          mod.ln_q.weight: "resampler.ln_q.weight",
          mod.ln_q.bias: "resampler.ln_q.bias",
          mod.ln_kv.weight: "resampler.ln_kv.weight",
          mod.ln_kv.bias: "resampler.ln_kv.bias",
          mod.attn_out.weight: "resampler.attn.out_proj.weight",
          mod.attn_out.bias: "resampler.attn.out_proj.bias",
          mod.ln_post.weight: "resampler.ln_post.weight",
          mod.ln_post.bias: "resampler.ln_post.bias",
          mod.proj: "resampler.proj",
      },
      state_dict,
      weight_map,
      strict=strict,
      populated=populated,
  )
  for idx, blk in enumerate(mod.blocks):
    weights_utils.copy_module_parameters(
        blk,
        state_dict,
        weight_map,
        f"vpm.encoder.layers.{idx}",
        param_map=SIGLIP_BLOCK_PARAM_MAP,
        strict=strict,
        populated=populated,
    )
  weights_utils.copy_qkv_in_proj(
      mod.attn_q,
      mod.attn_k,
      mod.attn_v,
      state_dict,
      weight_map,
      "resampler.attn.in_proj_weight",
      "resampler.attn.in_proj_bias",
      strict=strict,
      populated=populated,
  )
  if strict:
    weights_utils.verify_all_parameters_populated(
        mod, populated, "vision_encoder"
    )


def load_audio_encoder_weights(
    mod: MiniCpmOAudioEncoderModule,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool,
) -> None:
  """Loads Whisper encoder and audio projection weights into `mod`.

  Args:
    mod: Target `MiniCpmOAudioEncoderModule`.
    state_dict: In-memory checkpoint tensor dictionary.
    weight_map: Mapping of checkpoint keys to safetensors shard paths.
    strict: Whether missing or shape-mismatched tensors raise errors.
  """
  populated: set[int] = set()
  weights_utils.copy_parameter_dict(
      {
          mod.conv1.weight: "apm.conv1.weight",
          mod.conv1.bias: "apm.conv1.bias",
          mod.conv2.weight: "apm.conv2.weight",
          mod.conv2.bias: "apm.conv2.bias",
          mod.embed_positions.weight: "apm.embed_positions.weight",
          mod.layer_norm.weight: "apm.layer_norm.weight",
          mod.layer_norm.bias: "apm.layer_norm.bias",
          mod.proj1.weight: "audio_projection_layer.linear1.weight",
          mod.proj1.bias: "audio_projection_layer.linear1.bias",
          mod.proj2.weight: "audio_projection_layer.linear2.weight",
          mod.proj2.bias: "audio_projection_layer.linear2.bias",
      },
      state_dict,
      weight_map,
      strict=strict,
      populated=populated,
  )
  for idx, blk in enumerate(mod.blocks):
    weights_utils.copy_module_parameters(
        blk,
        state_dict,
        weight_map,
        f"apm.layers.{idx}",
        param_map=WHISPER_BLOCK_PARAM_MAP,
        strict=strict,
        populated=populated,
    )
  if strict:
    weights_utils.verify_all_parameters_populated(
        mod, populated, "audio_encoder"
    )


def load_text_embedder_weights(
    mod: MiniCpmOTextEmbedderModule,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool,
) -> None:
  """Loads Qwen3/Qwen2 token embedding weights into `mod`.

  Args:
    mod: Target `MiniCpmOTextEmbedderModule`.
    state_dict: In-memory checkpoint tensor dictionary.
    weight_map: Mapping of checkpoint keys to safetensors shard paths.
    strict: Whether missing or shape-mismatched tensors raise errors.
  """
  populated: set[int] = set()
  weights_utils.copy_weight(
      mod.embed_tokens.weight,
      state_dict,
      weight_map,
      "llm.model.embed_tokens.weight",
      strict=strict,
      populated=populated,
  )
  if strict:
    weights_utils.verify_all_parameters_populated(
        mod, populated, "text_embedder"
    )
