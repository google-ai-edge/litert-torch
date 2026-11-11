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
"""Single-step KV-cached Talker decoder wrapper for Qwen2.5-Omni."""

from __future__ import annotations

from typing import Any

from litert_torch.generative.custom_ops import dynamic_update_slice as tfl_dus
from litert_torch.generative.export_hf.model_ext.qwen_omni import llm
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as hf_qwen25


class _TalkerFlatKVCache:
  """Adapts C++'s flat `[1, 2*L, max_cache_len, kv_dim]` buffer to HF Talker.

  `QwenOmniTtsStage` passes a single flat 4D KV-cache tensor rather than `2*L`
  per-layer tensors (`export_hf/core/cache.py`). This adapter implements HF's
  `past_key_values.update(key_states, value_states, layer_idx)` protocol via
  `tfl.dynamic_update_slice` so `talker.model` runs unmodified.
  """

  def __init__(
      self,
      kv_cache: torch.Tensor,
      step_idx: torch.Tensor,
      num_kv_heads: int,
      head_dim: int,
  ):
    self._kv_cache = kv_cache
    self._step = step_idx  # 0-d int32 tensor
    self._num_kv_heads = num_kv_heads
    self._head_dim = head_dim
    self._max_cache_len = kv_cache.shape[2]
    self._kv_dim = kv_cache.shape[3]
    self._zeros = torch.zeros((), dtype=torch.int32, device=kv_cache.device)
    self._updated: list[torch.Tensor] = []

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      layer_idx: int,
      cache_kwargs: dict[str, Any] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Writes current K/V into slot `_step` and returns full cache tensors."""
    del cache_kwargs
    bsz = key_states.shape[0]
    k_layer = self._kv_cache[:, 2 * layer_idx : 2 * layer_idx + 1, :, :]
    v_layer = self._kv_cache[:, 2 * layer_idx + 1 : 2 * layer_idx + 2, :, :]
    k_slice = key_states.permute(0, 2, 1, 3).reshape(bsz, 1, 1, self._kv_dim)
    v_slice = value_states.permute(0, 2, 1, 3).reshape(bsz, 1, 1, self._kv_dim)
    indices = [self._zeros, self._zeros, self._step, self._zeros]
    k_new = tfl_dus.dynamic_update_slice(k_layer, k_slice, indices)
    v_new = tfl_dus.dynamic_update_slice(v_layer, v_slice, indices)
    self._updated.extend((k_new, v_new))
    k_full = k_new.reshape(
        bsz, self._max_cache_len, self._num_kv_heads, self._head_dim
    ).transpose(1, 2)
    v_full = v_new.reshape(
        bsz, self._max_cache_len, self._num_kv_heads, self._head_dim
    ).transpose(1, 2)
    return k_full, v_full

  def stacked(self) -> torch.Tensor:
    return torch.cat(self._updated, dim=1)


class QwenOmniTtsDecoderWrapper(nn.Module):
  """Single-step Talker codec decoder wrapping HF `Qwen2_5OmniTalker`.

  Upstream HF `Qwen2_5OmniTalkerForConditionalGeneration.forward()` cannot be
  exported directly to TFLite because it relies on dynamic `DynamicCache`,
  stateful `self.rope_deltas` / `get_rope_index`, in-place `inputs_embeds`
  mutation for prefill, and re-applies `thinker_to_talker_proj` (with bias).
  This wrapper reuses upstream Talker submodules (`talker.model`,
  `talker.codec_head`, `talker.thinker_to_talker_proj.weight`) directly to
  match the static single-step C++ `QwenOmniTtsStage` tensor contract.

  Alphabetical `a_`..`d_` input prefixes fix TFLite signature buffer order for
  C++ `QwenOmniTtsStage`.
  """

  _cache_positions: torch.Tensor

  def __init__(
      self,
      talker: hf_qwen25.Qwen2_5OmniTalkerForConditionalGeneration,
      max_cache_len: int,
  ):
    super().__init__()
    self.talker = talker
    llm.ensure_mrope_section(talker.config)
    talker.model.set_attn_implementation("eager")
    self.max_cache_len = max_cache_len
    self.num_kv_heads = int(talker.config.num_key_value_heads)
    self.head_dim = int(
        getattr(talker.config, "head_dim", None)
        or (talker.config.hidden_size // talker.config.num_attention_heads)
    )
    self._layer_types = tuple(
        set(getattr(talker.config, "layer_types", None) or ["full_attention"])
    )
    self.register_buffer(
        "_cache_positions",
        torch.arange(max_cache_len, dtype=torch.int32).view(
            1, 1, 1, max_cache_len
        ),
        persistent=False,
    )

  def forward(
      self,
      a_cond_emb: torch.Tensor,
      b_prev_audio_code: torch.Tensor,
      c_step_idx: torch.Tensor,
      d_kv_cache: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs one cached Talker decode step via upstream `talker.model`.

    Custom `forward()` is required to:
    1. Handle the `b_prev_audio_code < 0` sentinel (omits codec embedding on
       prompt prefill steps without breaking `nn.Embedding`).
    2. Project `codec_emb` with `thinker_to_talker_proj.weight` only (no bias),
       since `thinker_to_talker_proj.bias` was already added in `tts_projector`.
    3. Build a static `[1, 1, 1, max_cache_len]` additive causal mask and pass
       it as a `{layer_type: mask}` dict so `talker.model` skips HF's dynamic
       `create_causal_mask`.
    4. Wrap `d_kv_cache` in `_TalkerFlatKVCache` for `dynamic_update_slice`.

    Args:
      a_cond_emb: Projected conditioning, `float32[1, 1, talker_hidden_size]`.
      b_prev_audio_code: Previous codec ID, `int32[1, 1]` (`< 0` omits embed).
      c_step_idx: 0-based KV-cache / RoPE position, `int32[1]`.
      d_kv_cache: Flat KV cache, `float32[1, 2*L, max_cache_len, kv_dim]`.

    Returns:
      `(code_logits, new_kv_cache)` with shapes `[1, 1, tts_vocab_size]` and
      `[1, 2*L, max_cache_len, kv_dim]`.
    """
    safe_code = torch.clamp(b_prev_audio_code, min=0).to(torch.long)
    has_code = (b_prev_audio_code >= 0).unsqueeze(-1).to(a_cond_emb.dtype)
    codec_emb = self.talker.model.embed_tokens(safe_code)
    # Bias was already added once inside `tts_projector`.
    codec_proj = F.linear(codec_emb, self.talker.thinker_to_talker_proj.weight)
    hidden = a_cond_emb + codec_proj * has_code

    step = torch.clamp(
        c_step_idx.reshape(()).to(torch.int32), 0, self.max_cache_len - 1
    )
    valid = self._cache_positions <= step
    causal_mask = torch.where(
        valid,
        torch.zeros((), dtype=a_cond_emb.dtype, device=a_cond_emb.device),
        torch.full((), -1e9, dtype=a_cond_emb.dtype, device=a_cond_emb.device),
    )
    mask_dict = {lt: causal_mask for lt in self._layer_types}
    kv_cache: Any = _TalkerFlatKVCache(
        d_kv_cache, step, self.num_kv_heads, self.head_dim
    )
    outputs = self.talker.model(
        inputs_embeds=hidden,
        attention_mask=mask_dict,
        position_ids=step.to(torch.long).view(1, 1),
        past_key_values=kv_cache,
        use_cache=True,
    )
    logits = self.talker.codec_head(outputs.last_hidden_state)
    return logits, kv_cache.stacked()
