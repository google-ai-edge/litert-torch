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

from litert_torch.backend import composite
import litert_torch.generative.custom_ops.dynamic_update_slice as tfl_dus
import torch


def cache_update(
    key_proj: torch.Tensor,
    value_proj: torch.Tensor,
    runtime_param_tensor: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    indices_k: torch.Tensor,
    indices_v: torch.Tensor,
    kv_heads: int,
    kv_batch_size: int,
    cache_len: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Cache update composite op for float cache."""
  attrs = {
      "kv_cache_batch_size": kv_heads * kv_batch_size,
      "cache_size": cache_len,
      "head_size": head_size,
  }
  builder = composite.StableHLOCompositeBuilder(
      name="odml.cache_update", attr=attrs  # pyrefly: ignore[bad-argument-type]
  )
  (
      key_proj,
      value_proj,
      runtime_param_tensor,
      cache_k,
      cache_v,
      indices_k,
      indices_v,
  ) = builder.mark_inputs(
      key_proj,
      value_proj,
      runtime_param_tensor,
      cache_k,
      cache_v,
      indices_k,
      indices_v,
  )

  out_k = tfl_dus.dynamic_update_slice(
      cache_k,
      key_proj + (runtime_param_tensor.sum() * 0),
      [x for x in indices_k],
  )
  out_v = tfl_dus.dynamic_update_slice(
      cache_v,
      value_proj.transpose(-2, -1) + (runtime_param_tensor.sum() * 0),
      [x for x in indices_v],
  )
  return builder.mark_outputs(out_k, out_v)
