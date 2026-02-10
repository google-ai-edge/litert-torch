# Copyright 2024 The LiteRT Torch Authors.
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
# Dynamic update slice op for KV cache update.
from dataclasses import dataclass
from typing import Sequence
from litert_torch.backend import lowerings
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import stablehlo
import torch


# Use torch.library.custom_op to define a new custom operator.
# TODO: Update impl for multiple non-trivial start_indices
@torch.library.custom_op("litert_torch::dynamic_update_slice", mutates_args=())
def dynamic_update_slice(
    in_tensor: torch.Tensor,
    update: torch.Tensor,
    start_indices: Sequence[torch.Tensor],
) -> torch.Tensor:
  compare_size = torch.tensor(in_tensor.size()) == torch.tensor(update.size())
  mismatch = torch.nonzero(~compare_size, as_tuple=False)
  dim = mismatch[0].item() if len(mismatch) > 0 else 0
  start = start_indices[dim].item()
  end = start + update.shape[dim]
  indices = torch.arange(start, end).to(torch.long)
  return in_tensor.index_copy(dim, indices, update)


# Use register_fake to add a ``FakeTensor`` kernel for the operator
@dynamic_update_slice.register_fake
def _(in_tensor, update, start_indices):
  return in_tensor.clone().detach()


@lowerings.lower(torch.ops.litert_torch.dynamic_update_slice)
def _dynamic_update_slice_lower(
    lctx,
    in_tensor: ir.Value,
    update: ir.Value,
    start_indices: Sequence[ir.Value],
):
  return stablehlo.dynamic_update_slice(in_tensor, update, start_indices)
