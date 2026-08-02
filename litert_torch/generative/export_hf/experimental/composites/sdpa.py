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
"""Optimized GPU cache update operation for attention."""

import math
from typing import Optional
from litert_torch.backend import composite
from litert_torch.generative.custom_ops import bmm_4d as bmm_lib
from litert_torch.generative.export_hf.experimental.composites import runtime_batched_matmul
import torch
import torch.nn.functional as F

runtime_bmm = runtime_batched_matmul.runtime_bmm

# Fill value for attention mask. 1e-30 vs 1e-4?
MASK_FILL_VALUE = -1e30


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
    param_tensor: Optional[torch.Tensor] = None,
    is_global: bool = False,
    use_custom_op: bool = True,
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
    param_tensor (torch.Tensor): optional param tensor for runtime bmm.
    is_global (bool): whether the attention is global.

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
  gt = query.shape[2]
  g = gt // t

  # broadcasting mask
  if param_tensor is not None:
    if mask.dtype != torch.bool:
      mask: torch.Tensor = mask == 0
    if g != 1:
      mask = mask.to(torch.float32)  # pyrefly: ignore[missing-attribute]
      mask = torch.cat([mask] * g, dim=1)
      mask: torch.Tensor = mask != 0
      mask = mask.reshape(1, 1, gt, -1)  # pyrefly: ignore[missing-attribute]
  else:
    if g != 1:
      mask_to_bc = []
      for _ in range(g):
        mask_to_bc.append(mask)
      mask = torch.cat(mask_to_bc, dim=-2)  # 1, 1, gt, s

  attrs = {}
  attrs.update({
      "k_ts_idx": k_ts_idx,
      "v_ts_idx": v_ts_idx,
  })
  if softcap is not None:
    attrs["softcap"] = softcap
  if use_custom_op:
    sdpa_builder = composite.StableHLOCompositeBuilder(
        name="litert_custom_op.sdpa_transposed", attr=attrs
    )
    query, key, value, mask, param_tensor = sdpa_builder.mark_inputs(
        query, key, value, mask, param_tensor
    )
  else:
    sdpa_builder = None

  if param_tensor is not None:
    bmm_fn = lambda x, y: runtime_batched_matmul.runtime_bmm(
        x, y, param_tensor, is_global=is_global, is_src=False
    )
  elif k_ts_idx == 2:
    bmm_fn = bmm_lib.bmm_4d
  else:
    assert k_ts_idx == 3, "k_ts_idx must be 2 or 3."
    bmm_fn = lambda x, y: torch.einsum("abth,abhs->abts", x, y)
  logits = bmm_fn(query, key)

  if softcap is not None:
    logits = torch.tanh(logits / softcap)
    logits = logits * softcap

  if mask.dtype == torch.bool:
    padded_logits = torch.where(
        mask, logits, torch.tensor(MASK_FILL_VALUE, dtype=logits.dtype)
    )
  else:
    padded_logits = logits + mask

  attrs = {"axis": -1}
  builder = composite.StableHLOCompositeBuilder(name="odml.softmax", attr=attrs)
  padded_logits = builder.mark_inputs(padded_logits)
  probs = F.softmax(padded_logits, dim=-1)
  probs = builder.mark_outputs(probs)
  probs = probs.type_as(key)
  if param_tensor is not None:
    bmm_fn = lambda x, y: runtime_batched_matmul.runtime_bmm(
        x, y, param_tensor, is_global=is_global, is_src=True
    )
  elif v_ts_idx == 3:
    bmm_fn = bmm_lib.bmm_4d
  else:
    assert v_ts_idx == 2, "v_ts_idx must be 2 or 3."
    bmm_fn = lambda x, y: torch.einsum("abts,absh->abth", x, y)
  encoded = bmm_fn(probs, value)
  if sdpa_builder is not None:
    encoded = sdpa_builder.mark_outputs(encoded)

  return encoded  # 1, bk, gt, h
