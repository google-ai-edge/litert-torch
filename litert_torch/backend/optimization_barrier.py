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
"""Optimization barrier op definition and lowering."""

from litert_torch.backend import _torch_library
from litert_torch.backend.lowerings import registry
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import stablehlo
import torch
import torch.utils._pytree as pytree

_torch_library.LITERT_TORCH_LIB.define(
    "optimization_barrier(Tensor[] inputs) -> Tensor[]"
)

optimization_barrier_op = torch.ops.litert_torch.optimization_barrier.default


def optimization_barrier(*inputs: pytree.PyTree):
  """Apply optimization barrier to the tensors nested within arbitrary pytrees.

  Args:
    *inputs: A list of tensors or tensor pytrees.

  Returns:
    The tensors after optimization barrier in the same pytrees structures.
  """
  if len(inputs) == 1:
    inputs = inputs[0]
  tensors, spec = pytree.tree_flatten(inputs)
  tensors = optimization_barrier_op(tuple(tensors))
  outputs = pytree.tree_unflatten(tensors, spec)
  return outputs


@torch.library.impl(
    _torch_library.LITERT_TORCH_LIB,
    "optimization_barrier",
    "CompositeExplicitAutograd",
)
def _optimization_barrier_impl(inputs: tuple[torch.Tensor, ...]):
  return tuple(inputs)


@torch.library.impl(
    _torch_library.LITERT_TORCH_LIB,
    "optimization_barrier",
    "Meta",
)
def _optimization_barrier_fake(inputs: tuple[torch.Tensor, ...]):
  return tuple([torch.empty_like(x) for x in inputs])


@registry.lower(torch.ops.litert_torch.optimization_barrier.default)
def _optimization_barrier_lowering(
    lctx, inputs: tuple[ir.Value, ...]
) -> ir.Value:
  del lctx
  return stablehlo.optimization_barrier(inputs)
