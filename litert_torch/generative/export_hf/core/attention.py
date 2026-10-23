# Copyright 2025 The LiteRT Torch Authors.
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
"""Optimized Attention layer for HuggingFace integration."""

import math
from typing import Any, Optional
import jaxtyping as jt
from litert_torch.generative.custom_ops import bmm_4d as bmm_lib
from litert_torch.generative.export_hf.experimental.composites import sdpa as gpu_sdpa
import torch
import torch.nn.functional as F
import transformers
from litert_torch.backend import composite


def _append_sink_logits(
    padded_logits: torch.Tensor, s_aux: torch.Tensor, g: int
) -> torch.Tensor:
  """Appends per-head attention sink logits along the key sequence dimension."""
  b_dim, kv_groups_dim, gt, _ = padded_logits.shape
  num_query_groups = s_aux.numel() // g
  b_actual = (b_dim * kv_groups_dim) // num_query_groups
  t = gt // g
  sink_logits = s_aux.to(padded_logits.dtype).reshape(1, num_query_groups, g, 1)
  if b_actual != 1 or t != 1:
    sink_logits = sink_logits.expand(b_actual, num_query_groups, g, t)
  sink_logits = sink_logits.reshape(b_dim, kv_groups_dim, gt, 1)
  return torch.cat([padded_logits, sink_logits], dim=-1)


def scaled_dot_product_attention_transposed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_size: int,
    k_ts_idx: int,
    v_ts_idx: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
    alibi_bias: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
):
  """Scaled dot product attention with transposed key and value.

  Args:
    query: Query tensor, with shape [B, T, N, H].
    key: Key tensor, with shape [B, T, KV_LEN, H].
    value: Value tensor, with shape [B, T, H, KV_LEN].
    head_size (int): head dimension.
    k_ts_idx (int): the index of time step dimension in the key tensor.
    v_ts_idx (int): the index of time step dimension in the value tensor.
    mask (torch.Tensor): the optional mask tensor.
    scale (float): the optional scale factor.
    softcap (float): the optional softcap for the logits.
    alibi_bias (torch.Tensor): optional alibi bias tensor.
    s_aux (torch.Tensor): optional attention sink logit offset tensor.

  Returns:
    The output tensor of scaled_dot_product_attention_transposed.
  """
  if scale is None:
    scale = 1.0 / math.sqrt(head_size)

  if alibi_bias is not None:
    alibi_bias = alibi_bias * scale
    if mask is None:
      mask = alibi_bias
    else:
      mask = mask + alibi_bias

  query = query * scale

  assert mask is not None, "Mask should not be None!"
  t = mask.shape[2]
  if k_ts_idx == 2:
    bmm_fn = bmm_lib.bmm_4d
  else:
    assert k_ts_idx == 3, "k_ts_idx must be 2 or 3."
    bmm_fn = lambda x, y: torch.einsum("abth,abhs->abts", x, y)
  if isinstance(key, tuple):
    key_past, key_slice = key
    logits0 = bmm_fn(query, key_past)
    logits1 = bmm_fn(query, key_slice)
    logits = torch.cat([logits0, logits1], dim=-1)
  else:
    logits = bmm_fn(query, key)

  _, _, gt, _ = logits.shape
  g = gt // t
  if softcap is not None:
    logits = torch.tanh(logits / softcap)
    logits = logits * softcap

  # broadcasting mask
  if g != 1:
    mask_to_bc = []
    for _ in range(g):
      mask_to_bc.append(mask)
    mask = torch.cat(mask_to_bc, dim=-2)  # 1, 1, gt, s

  padded_logits = logits + mask
  if s_aux is not None:
    padded_logits = _append_sink_logits(padded_logits, s_aux, g)
  # TODO(weiyiw): Re-enable this with when compiler is updated
  if padded_logits.dtype != torch.float32:
    attrs = {"axis": -1}
    builder = composite.StableHLOCompositeBuilder(
        name="odml.softmax", attr=attrs
    )
    padded_logits = builder.mark_inputs(padded_logits)
  else:
    builder = None
  probs = F.softmax(padded_logits, dim=-1)
  if builder is not None:
    probs = builder.mark_outputs(probs)
  if s_aux is not None:
    probs = probs[..., :-1]

  probs = probs.type_as(key[0] if isinstance(key, tuple) else key)
  if v_ts_idx == 3:
    bmm_fn = bmm_lib.bmm_4d
  else:
    assert v_ts_idx == 2, "v_ts_idx must be 2 or 3."
    bmm_fn = lambda x, y: torch.einsum("abts,absh->abth", x, y)
  if isinstance(value, tuple):
    probs0, probs1 = probs[..., :-t], probs[..., -t:]
    value_past, value_slice = value
    encoded0 = bmm_fn(probs0, value_past)
    encoded1 = bmm_fn(probs1, value_slice)
    encoded = encoded0 + encoded1
  else:
    encoded = bmm_fn(probs, value)

  return encoded  # 1, bk, gt, h


def transposed_attention(
    module: torch.nn.Module,
    query: jt.Float[torch.Tensor, "b n t h"],
    key: Any,
    value: Any,
    attention_mask: jt.Shaped[torch.Tensor, "1 1 t s"] | None,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
  """Transpose k/v to specific layout for LiteRT Optimized implementation.

  Args:
    module: The attention layer module.
    query: The query tensor.
    key: The key cache tensor. Note that the key cache tensor is pre-processed.
    value: The value tensor. Note that the key cache tensor is pre-processed.
    attention_mask: The attention mask tensor.
    scaling: The scaling factor.
    softcap: The softcap factor.
    **kwargs: Other keyword arguments.

  Returns:
    The attention output tensor.
  """

  b, n, seq_len, h = query.shape
  g = getattr(module, "num_key_value_groups", 1)
  num_query_groups = n // g
  key_ts_idx: int | None = kwargs.get("k_ts_idx", None)
  value_ts_idx: int | None = kwargs.get("v_ts_idx", None)
  if key_ts_idx is None or value_ts_idx is None:
    raise ValueError(
        "Timestamp indices not passed to attention module. The model is not"
        " passing the kwargs correctly."
    )

  s_aux = kwargs.get("s_aux", getattr(module, "sinks", None))
  apply_gpu_composites = bool(kwargs.get("apply_gpu_composites", False))
  use_sdpa_composite = bool(
      kwargs.get("use_sdpa_composite", False)
      or kwargs.get("sdpa_use_composite", False)
  )
  if s_aux is not None:
    use_sdpa_composite = False

  if apply_gpu_composites or use_sdpa_composite:
    is_global = kwargs.get("is_global", None)
    if is_global is None:
      is_sliding = (
          getattr(module, "is_sliding", False)
          or getattr(module, "sliding_window", None) is not None
          or kwargs.get("sliding_window", None) is not None
      )
      if hasattr(module, "layer_type"):
        is_sliding = getattr(module, "layer_type", "") == "sliding_attention"
      is_global = not is_sliding
    is_global = bool(is_global)
    # Only mark attention as purely causal when the export guarantees it
    # (text-only models). The GPU delegate then drops the global mask and
    # enforces causality itself, so a false positive would break e.g.
    # bidirectional attention over interleaved image tokens.
    is_causal = (
        is_global
        and bool(kwargs.get("sdpa_is_causal", False))
        and bool(getattr(module, "is_causal", True))
    )
    past_key_value = kwargs.get("past_key_value", None)
    enable_ring_buffer = bool(kwargs.get("enable_ring_buffer", False))
    layer_idx = getattr(module, "layer_idx", None)
    use_fused_sdpa_cache_update = (
        bool(kwargs.get("use_fused_sdpa_cache_update", False)) and s_aux is None
    )
    # Whether this layer leaves the KV cache unwritten. By default KV-shared
    # layers (e.g. Gemma 4) skip, since they reuse a preceding donor layer's
    # cache. With the fused op a model can pass `skip_cache_update` explicitly
    # to choose which layer writes a shared cache instead.
    if use_fused_sdpa_cache_update and "skip_cache_update" in kwargs:
      skip_cache_update = bool(kwargs["skip_cache_update"])
    else:
      skip_cache_update = bool(
          getattr(module, "is_kv_shared_layer", False)
          or kwargs.get("skip_cache_update", False)
          or kwargs.get("is_kv_shared_layer", False)
      )
    if use_fused_sdpa_cache_update:
      # Without prefill logits the last layer's attention output has no
      # consumer. A fused composite whose output 0 is dead crashes the
      # converter's `BuildStableHLOCompositePass`, so that layer stays
      # unfused.
      num_layers = getattr(
          getattr(module, "config", None), "num_hidden_layers", 0
      )
      use_fused_sdpa_cache_update = not (
          not kwargs.get("prefill_logits", True)
          and layer_idx == num_layers - 1
      )
    sdpa_out = gpu_sdpa.scaled_dot_product_attention_transposed(
        query=query,
        key=key,
        value=value,
        head_size=h,
        k_ts_idx=key_ts_idx,
        v_ts_idx=value_ts_idx,
        mask=attention_mask,
        scale=scaling,
        softcap=softcap,
        param_tensor=kwargs.get("param_tensor", None),
        is_global=is_global,
        is_causal=is_causal,
        use_sdpa_composite=use_sdpa_composite,
        enable_ring_buffer=enable_ring_buffer,
        past_key_value=past_key_value,
        layer_idx=layer_idx,
        skip_cache_update=skip_cache_update,
        use_fused_sdpa_cache_update=use_fused_sdpa_cache_update,
        runtime_param_tensor=kwargs.get("runtime_param_tensor", None),
        s_aux=s_aux,
    )
    return sdpa_out, None

  # bnth -> b(kg)th -> 1(bk)(gt)h
  query = query.reshape(1, b * num_query_groups, g * seq_len, h)
  # 1, bk, gt, h
  sdpa_out = scaled_dot_product_attention_transposed(
      query=query,
      key=key,
      value=value,
      head_size=h,
      k_ts_idx=key_ts_idx,
      v_ts_idx=value_ts_idx,
      mask=attention_mask,
      scale=scaling,
      softcap=softcap,
      s_aux=s_aux,
  )
  # b, kg, t, h
  sdpa_out = sdpa_out.reshape(b, -1, seq_len, h).permute(0, 2, 1, 3)
  return sdpa_out, None


transformers.AttentionInterface.register(
    "lrt_transposed_attention", transposed_attention
)
