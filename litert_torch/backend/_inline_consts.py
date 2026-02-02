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
"""The pre-lower pass to make exported program's constant inputs a MLIR ElementsAttr."""

from typing import Any
from litert_torch.backend import lowerings
from litert_torch.backend.lowerings import utils as lowering_utils
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import stablehlo
import torch


def _tensor_fingerprint(tensor: torch.Tensor) -> int:
  return (
      str(tensor.device),
      tensor.shape,
      tensor.stride(),
      tensor.untyped_storage().data_ptr(),
  )


def get_tensor_lowering_placeholder(
    const_lowering_cache: dict[Any, Any],
):
  """Returns a placeholder that will be lowered to elements attr."""

  def placeholder(x: torch.Tensor):
    return x

  @lowerings.lower(placeholder)
  def tensor_lowering(lctx, x: torch.Tensor):
    del lctx
    x_fingerprint = _tensor_fingerprint(x)
    elty = lowering_utils.torch_dtype_to_ir_element_type(x.dtype)
    tensor_type = ir.RankedTensorType.get(x.shape, elty)

    if x_fingerprint in const_lowering_cache:
      return const_lowering_cache[x_fingerprint]

    is_splat = torch.all(x == x.flatten()[0]).item()
    if is_splat:
      return lowering_utils.splat(
          x.flatten()[0].item(),
          tensor_type.element_type,
          tensor_type.shape,
      )
    else:
      attr = ir.DenseElementsAttr.get(x.detach().numpy())

    ir_value = stablehlo.ConstantOp(attr).results[0]
    const_lowering_cache[x_fingerprint] = ir_value
    return ir_value

  return placeholder


def inline_consts(
    exported_program: torch.export.ExportedProgram,
    const_lowering_cache: dict[Any, Any] | None = None,
):
  """Inline exported program's constant inputs by replacing with lazy_tensor_placeholder."""
  flat_user_inputs, _ = exported_program._get_flat_args_with_check(
      *exported_program.example_inputs
  )
  flat_inputs = exported_program._graph_module_flat_inputs(
      *exported_program.example_inputs
  )
  if flat_user_inputs:
    flat_consts = flat_inputs[: -len(flat_user_inputs)]
  else:
    flat_consts = flat_inputs

  if const_lowering_cache is None:
    const_lowering_cache = {}

  lowering_placeholder = get_tensor_lowering_placeholder(const_lowering_cache)

  for i, (tensor, node) in enumerate(
      zip(flat_consts, exported_program.graph.nodes)
  ):
    if node.op != "placeholder":
      raise ValueError(
          "Expect FX graph nodes to start with placeholder nodes, but got"
          f"{i}-th node: {node}"
      )
    node.op = "call_function"
    node.target = lowering_placeholder
    node.args = (tensor,)

  exported_program.graph_signature.input_specs = (
      exported_program.graph_signature.input_specs[len(flat_consts) :]
  )
