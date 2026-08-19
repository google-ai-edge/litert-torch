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
"""Optimized GPU batched matmul operation for attention."""

from litert_torch.backend import composite
import torch


def runtime_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    param_tensor: torch.Tensor,
    is_global: bool = False,
    is_src: bool = False,
    use_composite: bool = True,
) -> torch.Tensor:
  """Runtime BMM composite op."""
  if use_composite:
    attrs = {
        "is_global": is_global,
        "is_src": is_src,
        "rhs_cache_update": True,  # If input is coming from cache_update custom
    }
    builder = composite.StableHLOCompositeBuilder(
        name="odml.runtime_bmm", attr=attrs
    )
    a, b, param_tensor = builder.mark_inputs(a, b, param_tensor)
  else:
    builder = None

  out = torch.einsum("abcd,abed->abce", a, b) + (param_tensor.sum() * 0)
  if builder is not None:
    out = builder.mark_outputs(out)
  return out
