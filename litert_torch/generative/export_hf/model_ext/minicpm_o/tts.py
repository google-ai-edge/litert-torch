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
"""TTS semantic projector and KV-cached LLaMA decoder for MiniCPM-o 4.5."""

from __future__ import annotations

import math
from typing import Any

from litert_torch.generative.export_hf.model_ext.minicpm_o import weights_utils
import torch
from torch import nn
import torch.nn.functional as F

_make_linear = weights_utils.make_linear
_make_embedding = weights_utils.make_embedding


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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotates half the hidden dims of `x` for rotary positional embeddings."""
  half = x.shape[-1] // 2
  return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


_rotate_half = rotate_half


class MiniCpmOTtsTransformerBlock(nn.Module):
  """Single-step LLaMA transformer layer with KV cache for the TTS decoder."""

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
      k_cache: torch.Tensor | None = None,
      v_cache: torch.Tensor | None = None,
      write_mask: torch.Tensor | None = None,
      attn_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    v_step = self.v_proj(h)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    k_step = k.permute(0, 2, 1, 3).reshape(b, s, d)

    if (
        k_cache is not None
        and v_cache is not None
        and write_mask is not None
        and attn_mask is not None
    ):
      keep_mask = 1.0 - write_mask
      k_updated = k_cache * keep_mask + k_step * write_mask
      v_updated = v_cache * keep_mask + v_step * write_mask
      cache_len = k_updated.shape[1]
      k_all = k_updated.reshape(
          b, cache_len, self.num_heads, self.head_dim
      ).permute(0, 2, 1, 3)
      v_all = v_updated.reshape(
          b, cache_len, self.num_heads, self.head_dim
      ).permute(0, 2, 1, 3)
      scores = (
          torch.matmul(q, k_all.permute(0, 1, 3, 2))
          * (1.0 / math.sqrt(float(self.head_dim)))
          + attn_mask
      )
      probs = F.softmax(scores, dim=-1)
      attn_out = torch.matmul(probs, v_all).permute(0, 2, 1, 3).reshape(b, s, d)
    else:
      k_updated = k_step
      v_updated = v_step
      v = v_step.reshape(b, s, self.num_heads, self.head_dim).permute(
          0, 2, 1, 3
      )
      scores = torch.matmul(q, k.permute(0, 1, 3, 2)) * (
          1.0 / math.sqrt(float(self.head_dim))
      )
      probs = F.softmax(scores, dim=-1)
      attn_out = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(b, s, d)

    x = x + self.o_proj(attn_out)
    h2 = self.post_attention_layernorm(x)
    mlp_out = self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
    return x + mlp_out, k_updated, v_updated


class MiniCpmOTtsProjectorModule(nn.Module):
  """Combines `tts.emb_text` and L2-normalized `tts.projector_semantic`.

  Matches `MiniCPMTTS` conditioning in `modeling_minicpmo.py`:
  - Text token steps inside `<|tts_bos|> ... <|tts_eos|>` sum
    `tts.emb_text(token_id) + L2Normalize(tts.projector_semantic(llm_hidden))`.
  - Special TTS control tokens (`tts_bos=151664`, `text_eos=151692`,
    `audio_bos=151687`) pass `llm_hidden = 0` and `token_id >= 0`, yielding
    `tts.emb_text(token_id)`.
  - Legacy callers passing only `llm_hidden` (`token_id=None` or `token_id < 0`)
    receive `L2Normalize(tts.projector_semantic(llm_hidden))`.

  Inputs:
    llm_hidden: [1, 1, llm_hidden_size] float32.
    token_id: optional [1, 1] int32 token ID in `tts.emb_text` vocabulary.
  Output:
    dict with `"tts_cond"`: [1, 1, tts_hidden_size] float32.
  """

  def __init__(
      self,
      llm_hidden_size: int = 4096,
      tts_hidden_size: int = 768,
      num_text_tokens: int = 152064,
      normalize_projected_hidden: bool = True,
  ):
    super().__init__()
    self.num_text_tokens = num_text_tokens
    self.emb_text = _make_embedding(num_text_tokens, tts_hidden_size)
    with torch.no_grad():
      self.emb_text.weight.zero_()
    self.linear1 = _make_linear(llm_hidden_size, tts_hidden_size, bias=True)
    self.linear2 = _make_linear(tts_hidden_size, tts_hidden_size, bias=True)
    self.normalize_projected_hidden = normalize_projected_hidden

  def forward(
      self,
      llm_hidden: torch.Tensor,
      token_id: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    proj = self.linear2(F.relu(self.linear1(llm_hidden)))
    if self.normalize_projected_hidden:
      norm_sq = torch.sum(proj * proj, dim=-1, keepdim=True)
      proj = proj * torch.rsqrt(torch.clamp(norm_sq, min=1e-12))
    if token_id is None:
      return {"tts_cond": proj}
    has_hidden = (
        torch.sum(llm_hidden * llm_hidden, dim=-1, keepdim=True) > 1e-12
    ).to(proj.dtype)
    safe_id = torch.clamp(token_id, min=0, max=self.num_text_tokens - 1)
    has_token = (
        ((token_id >= 0) & (token_id < self.num_text_tokens))
        .unsqueeze(-1)
        .to(proj.dtype)
    )
    text_emb = self.emb_text(safe_id) * has_token
    return {"tts_cond": text_emb + proj * has_hidden}


class MiniCpmOTtsDecoderModule(nn.Module):
  """Autoregressive KV-cached TTS decoder step module for MiniCPM-o 4.5.

  Inputs:
    cond_emb: [1, 1, tts_hidden_size] float32 conditioning embedding (zero on
      autoregressive audio decode steps).
    prev_audio_code: [1, 1] int32 previous audio token ID (`0..tts_vocab_size-1`
      on audio decode steps, or `-1` on conditioning prefill steps).
    step_idx: [1] int32 0-based position in the TTS KV cache.
    kv_cache: optional [1, 2 * tts_num_layers, tts_max_cache_len,
      tts_hidden_size] float32 causal KV cache state.
  Outputs:
    dict with:
      `"code_logits"`: [1, 1, tts_vocab_size] float32 audio token logits.
      `"new_kv_cache"`: [1, 2 * tts_num_layers, tts_max_cache_len,
        tts_hidden_size] float32 updated KV cache state.
  """

  inv_freq: torch.Tensor
  cache_positions: torch.Tensor

  def __init__(
      self,
      tts_hidden_size: int = 768,
      tts_intermediate_size: int = 3072,
      tts_num_layers: int = 20,
      tts_num_heads: int = 12,
      tts_vocab_size: int = 6562,
      tts_num_mel_bins: int = 100,
      tts_max_cache_len: int = 256,
      rope_theta: float = 10000.0,
      min_tts_steps: int = 0,
  ):
    super().__init__()
    self.tts_hidden_size = tts_hidden_size
    self.tts_num_layers = tts_num_layers
    self.tts_vocab_size = tts_vocab_size
    self.tts_num_mel_bins = tts_num_mel_bins
    self.tts_max_cache_len = tts_max_cache_len
    self.min_tts_steps = min_tts_steps
    head_dim = tts_hidden_size // tts_num_heads
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.register_buffer(
        "cache_positions",
        torch.arange(tts_max_cache_len, dtype=torch.int32).reshape(
            1, tts_max_cache_len, 1
        ),
        persistent=False,
    )

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
      kv_cache: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs embedding lookup, KV-cached LLaMA layers, and final RMSNorm."""
    safe_code = torch.clamp(prev_audio_code, min=0, max=self.tts_vocab_size - 1)
    has_code = (
        ((prev_audio_code >= 0) & (prev_audio_code < self.tts_vocab_size))
        .unsqueeze(-1)
        .to(cond_emb.dtype)
    )
    h = cond_emb + self.emb_code(safe_code) * has_code

    pos = step_idx.to(torch.float32).reshape(1, 1)
    freqs = pos.unsqueeze(-1) * self.inv_freq.reshape(1, 1, -1)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = torch.cos(emb).unsqueeze(1).to(cond_emb.dtype)
    sin = torch.sin(emb).unsqueeze(1).to(cond_emb.dtype)

    if kv_cache is None:
      kv_cache = torch.zeros(
          1,
          2 * self.tts_num_layers,
          self.tts_max_cache_len,
          self.tts_hidden_size,
          dtype=cond_emb.dtype,
          device=cond_emb.device,
      )

    step_pos = torch.clamp(
        step_idx.to(torch.int32), min=0, max=self.tts_max_cache_len - 1
    ).reshape(1, 1, 1)
    write_mask = (self.cache_positions == step_pos).to(cond_emb.dtype)
    valid_mask = (
        (self.cache_positions <= step_pos)
        .reshape(1, 1, 1, self.tts_max_cache_len)
        .to(cond_emb.dtype)
    )
    attn_mask = (1.0 - valid_mask) * (-1e9)

    new_kvs: list[torch.Tensor] = []
    for idx, layer in enumerate(self.layers):
      k_c = kv_cache[:, 2 * idx]
      v_c = kv_cache[:, 2 * idx + 1]
      h, k_u, v_u = layer(h, cos, sin, k_c, v_c, write_mask, attn_mask)
      new_kvs.append(k_u)
      new_kvs.append(v_u)
    new_kv_cache = torch.stack(new_kvs, dim=1)
    return self.norm(h), new_kv_cache

  def forward(
      self,
      cond_emb: torch.Tensor,
      prev_audio_code: torch.Tensor,
      step_idx: torch.Tensor,
      kv_cache: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    h, new_kv_cache = self.forward_hidden(
        cond_emb, prev_audio_code, step_idx, kv_cache
    )
    code_logits = self.head_code(h)
    if self.min_tts_steps > 0 and self.tts_vocab_size > 1:
      step_f = step_idx.to(code_logits.dtype).reshape(1, 1, 1)
      min_steps_f = float(self.min_tts_steps)
      eos_bias = torch.where(
          step_f < min_steps_f,
          torch.full_like(step_f, -1e4),
          torch.zeros_like(step_f),
      )
      non_eos = code_logits[..., :-1]
      eos_logit = code_logits[..., -1:] + eos_bias
      code_logits = torch.cat([non_eos, eos_logit], dim=-1)

    return {
        "code_logits": code_logits,
        "new_kv_cache": new_kv_cache,
    }


def load_tts_projector_weights(
    mod: MiniCpmOTtsProjectorModule,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    tts_cfg: dict[str, Any],
    strict: bool,
) -> None:
  """Loads `tts.emb_text` and `tts.projector_semantic` weights into `mod`.

  Args:
    mod: Target `MiniCpmOTtsProjectorModule`.
    state_dict: In-memory checkpoint tensor dictionary.
    weight_map: Mapping of checkpoint keys to safetensors shard paths.
    tts_cfg: `tts_config` dictionary from `config.json`.
    strict: Whether missing or shape-mismatched tensors raise errors.

  Raises:
    ValueError: If `strict` is True and `text_eos_token_id` or
      `audio_bos_token_id` is outside `[0, mod.num_text_tokens)`, or if any
      parameter is unpopulated.
  """
  populated: set[int] = set()
  cw = weights_utils.copy_weight
  cw(
      mod.emb_text.weight,
      state_dict,
      weight_map,
      "tts.emb_text.weight",
      strict=strict,
      populated=populated,
  )
  num_text_tokens = mod.num_text_tokens
  emb_text_w = weights_utils.get_weight(
      state_dict, weight_map, "tts.emb_text.weight"
  )
  if emb_text_w is not None or strict:
    default_text_eos = (
        151692 if num_text_tokens > 151692 else max(0, num_text_tokens - 2)
    )
    default_audio_bos = (
        151687 if num_text_tokens > 151687 else max(0, num_text_tokens - 1)
    )
    text_eos_id = int(tts_cfg.get("text_eos_token_id", default_text_eos))
    audio_bos_id = int(tts_cfg.get("audio_bos_token_id", default_audio_bos))
    if strict and not 0 <= text_eos_id < num_text_tokens:
      raise ValueError(
          f"text_eos_token_id={text_eos_id} out of range"
          f" [0, {num_text_tokens})."
      )
    if strict and not 0 <= audio_bos_id < num_text_tokens:
      raise ValueError(
          f"audio_bos_token_id={audio_bos_id} out of range"
          f" [0, {num_text_tokens})."
      )
    with torch.no_grad():
      if 0 <= text_eos_id < num_text_tokens and 151672 < num_text_tokens:
        mod.emb_text.weight[151672].copy_(mod.emb_text.weight[text_eos_id])
      if 0 <= audio_bos_id < num_text_tokens and 151676 < num_text_tokens:
        mod.emb_text.weight[151676].copy_(mod.emb_text.weight[audio_bos_id])

  weights_utils.copy_parameter_dict(
      {
          mod.linear1.weight: (
              "tts.projector_semantic.linear1.weight",
              "tts.projector.linear1.weight",
          ),
          mod.linear1.bias: (
              "tts.projector_semantic.linear1.bias",
              "tts.projector.linear1.bias",
          ),
          mod.linear2.weight: (
              "tts.projector_semantic.linear2.weight",
              "tts.projector.linear2.weight",
          ),
          mod.linear2.bias: (
              "tts.projector_semantic.linear2.bias",
              "tts.projector.linear2.bias",
          ),
      },
      state_dict,
      weight_map,
      strict=strict,
      populated=populated,
  )
  if strict:
    weights_utils.verify_all_parameters_populated(
        mod, populated, "tts_projector"
    )


_TTS_LAYER_PARAM_MAP = {
    "q_proj.weight": "self_attn.q_proj.weight",
    "k_proj.weight": "self_attn.k_proj.weight",
    "v_proj.weight": "self_attn.v_proj.weight",
    "o_proj.weight": "self_attn.o_proj.weight",
    "gate_proj.weight": "mlp.gate_proj.weight",
    "up_proj.weight": "mlp.up_proj.weight",
    "down_proj.weight": "mlp.down_proj.weight",
}


def load_tts_decoder_weights(
    mod: MiniCpmOTtsDecoderModule,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool,
) -> None:
  """Loads LLaMA TTS decoder and `head_code` weights into `mod`.

  Args:
    mod: Target `MiniCpmOTtsDecoderModule`.
    state_dict: In-memory checkpoint tensor dictionary.
    weight_map: Mapping of checkpoint keys to safetensors shard paths.
    strict: Whether missing or shape-mismatched tensors raise errors.
  """
  populated: set[int] = set()
  cw = weights_utils.copy_weight
  cw(
      mod.emb_code.weight,
      state_dict,
      weight_map,
      "tts.emb_code.0.weight",
      "tts.emb_code.weight",
      strict=strict,
      populated=populated,
  )
  for idx, layer in enumerate(mod.layers):
    weights_utils.copy_module_parameters(
        layer,
        state_dict,
        weight_map,
        f"tts.model.layers.{idx}",
        param_map=_TTS_LAYER_PARAM_MAP,
        strict=strict,
        populated=populated,
    )
  cw(
      mod.norm.weight,
      state_dict,
      weight_map,
      "tts.model.norm.weight",
      strict=strict,
      populated=populated,
  )
  weights_utils.copy_weight_norm(
      mod.head_code.weight,
      state_dict,
      weight_map,
      "tts.head_code.0.weight",
      "tts.head_code.0.parametrizations.weight.original0",
      "tts.head_code.0.parametrizations.weight.original1",
      strict=strict,
      populated=populated,
  )
  if strict:
    weights_utils.verify_all_parameters_populated(mod, populated, "tts_decoder")
