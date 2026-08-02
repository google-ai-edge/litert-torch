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
from typing import Optional
import jaxtyping as jt
from litert_torch.generative.custom_ops import bmm_4d as bmm_lib
from litert_torch.generative.export_hf.experimental.composites import sdpa as gpu_sdpa
import torch
import torch.nn.functional as F
import transformers
from litert_torch.backend import composite


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
  attrs = {"axis": -1}
  builder = composite.StableHLOCompositeBuilder(
      name="odml.softmax", attr=attrs
  )
  padded_logits = builder.mark_inputs(padded_logits)
  probs = F.softmax(padded_logits, dim=-1)
  probs = builder.mark_outputs(probs)
  probs = probs.type_as(key)
  if v_ts_idx == 3:
    bmm_fn = bmm_lib.bmm_4d
  else:
    assert v_ts_idx == 2, "v_ts_idx must be 2 or 3."
    bmm_fn = lambda x, y: torch.einsum("abts,absh->abth", x, y)
  encoded = bmm_fn(probs, value)

  return encoded  # 1, bk, gt, h


def transposed_attention(
    module: torch.nn.Module,
    query: jt.Float[torch.Tensor, "b n t h"],
    key: jt.Float[torch.Tensor, "1 c s h"],
    value: jt.Float[torch.Tensor, "1 c h s"],
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
  # bnth -> b(kg)th -> 1(bk)(gt)h
  query = query.reshape(1, b * num_query_groups, g * seq_len, h)
  key_ts_idx: int | None = kwargs.get("k_ts_idx", None)
  value_ts_idx: int | None = kwargs.get("v_ts_idx", None)
  if key_ts_idx is None or value_ts_idx is None:
    raise ValueError(
        "Timestamp indices not passed to attention module. The model is not"
        " passing the kwargs correctly."
    )

  apply_gpu_composites = kwargs.get("apply_gpu_composites", False)
  use_custom_op = kwargs.get("sdpa_use_custom_op", True)
  if apply_gpu_composites:
    is_global = kwargs.get("is_global", None)
    if is_global is None:
      is_sliding = getattr(module, "is_sliding", False)
      is_global = not is_sliding
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
        use_custom_op=use_custom_op,
    )
    # b, kg, t, h
    sdpa_out = sdpa_out.reshape(b, -1, seq_len, h).permute(0, 2, 1, 3)
    return sdpa_out, None

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
  )
  # b, kg, t, h
  sdpa_out = sdpa_out.reshape(b, -1, seq_len, h).permute(0, 2, 1, 3)
  return sdpa_out, None


transformers.AttentionInterface.register(
    "lrt_transposed_attention", transposed_attention
)
