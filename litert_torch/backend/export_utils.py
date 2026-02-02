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
"""Utilities for ODML Torch export."""

import re
from typing import Any, Sequence, cast
from litert_torch.backend.lowerings import utils as lowering_utils
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import func
import torch
import torch.utils._pytree as pytree
from ai_edge_litert.mlir._mlir_libs import converter_api_ext

# std::numeric_limits<int64_t>::min()
IR_DYNAMIC = -9223372036854775808


def flat_dict_names(
    tree_spec: pytree.TreeSpec, context: pytree.Context
) -> list[str]:
  """Given a TreeSpec, this produces a list of names for the leaves.

  The list of names embeddeds the structure of the tree_spec. A nesting level is
  indicated by an `_` and elements in a list are indicated by `_<index>`.

  TODO b/361601485: The flattening of names is not collision-free and needs to
  be revised.

  Args:
    tree_spec: The TreeSpec to extract the names from.
    context: The context used to check if the provided spec belongs to a
      dictionary or a list.

  Returns:
    A list of flattened names.
  """

  def _flatten_list(l: list[Any]) -> list[Any]:
    flattened = []
    for item in l:
      if isinstance(item, list):
        flattened.extend(_flatten_list(item))
      else:
        flattened.append(item)
    return flattened

  flat_names = []
  if context is None:
    for i, spec in enumerate(tree_spec):
      if spec.children_specs:
        flat_names.extend([
            f"{i}_{name}"
            for name in flat_dict_names(spec.children_specs, spec.context)
        ])
      else:
        flat_names.append(f"{i}")
  else:
    flat_ctx = _flatten_list(context)
    for prefix, spec in zip(flat_ctx, tree_spec):
      leaf_flat_names = flat_dict_names(spec.children_specs, spec.context)
      if leaf_flat_names:
        flat_names.extend([f"{prefix}_{name}" for name in leaf_flat_names])
      else:
        flat_names.append(prefix)

  return flat_names


def is_ir_dynamic(v):
  return v == IR_DYNAMIC


def is_torch_dynamic(v):
  return isinstance(v, torch.SymInt)


def is_iterable(v):
  try:
    iter(v)
  except TypeError:
    return False
  return True


def create_ir_context():
  context = ir.Context()
  converter_api_ext.prepare_mlir_context(context)
  return context


def inline(
    symbol_table: ir.SymbolTable,
    block: ir.Block,
):
  """Recursively inlines all func.call ops in the block.

  The symbol_table must include all func.func called by func.call ops.
  This inliner in Python is implemented because MLIR inline pass from JAX's
  MLIR pybinding build in OSS cannot properly inline func.call ops.
  """
  while True:
    is_changed = False
    for op in block.operations:
      if (
          not hasattr(op, "OPERATION_NAME")
          or op.OPERATION_NAME != func.CallOp.OPERATION_NAME
      ):
        continue

      call_op = cast(func.CallOp, op)
      func_op = cast(func.FuncOp, symbol_table[call_op.callee.value])
      with ir.InsertionPoint(op):
        new_results = clone_func_body_ops(func_op, call_op.operands)

      for old_result, new_result in zip(call_op.results, new_results):
        old_result = cast(ir.Value, old_result)
        old_result.replace_all_uses_with(new_result)
      call_op.erase()
      is_changed = True

    if not is_changed:
      break

  for op in block.operations:
    for region in op.regions:
      for block in region.blocks:
        inline(symbol_table, block)


def clone_func_body_ops(func_op: func.FuncOp, ir_inputs: Sequence[ir.Value]):
  """Clone operations in the func_op's body by one into the current context."""
  func_args = list(func_op.arguments)
  ir_inputs = list(ir_inputs)
  assert len(func_args) == len(ir_inputs)

  value_mapping = {arg: ir_input for arg, ir_input in zip(func_args, ir_inputs)}

  for op in list(func_op.entry_block.operations):
    cloned_operands = [value_mapping[val] for val in op.operands]
    if (
        hasattr(op, "OPERATION_NAME")
        and op.OPERATION_NAME == func.ReturnOp.OPERATION_NAME
    ):
      return cloned_operands

    cloned = cast(ir.Operation, op.operation.clone())

    for i in range(len(op.operands)):
      cloned.operands[i] = cloned_operands[i]

    for i in range(len(op.results)):
      value_mapping[op.results[i]] = cloned.results[i]

  return []


def sanitize_aten_op_name(op, chars=":."):
  return re.sub("[{}]".format(chars), "_", str(op))


def build_ir_attr(val):
  if val is None:
    return ir.StringAttr.get("py_None")
  if isinstance(val, bool):
    return ir.BoolAttr.get(val)
  if isinstance(val, int):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), val)
  if isinstance(val, float):
    return ir.BoolAttr.get(val)
  if isinstance(val, str):
    return ir.StringAttr.get(val)
  if isinstance(val, dict):
    return ir.DictAttr.get({k: build_ir_attr(v) for k, v in val.items()})
  if isinstance(val, (list, tuple)):
    return ir.ArrayAttr.get([build_ir_attr(v) for v in val])

  # Stringify the value to a StringAttr by default
  return ir.StringAttr.get(str(val))


torch_dtype_to_ir_element_type = lowering_utils.torch_dtype_to_ir_element_type


def ir_element_type_to_torch_dtype(ty):
  if isinstance(ty, ir.F32Type):
    return torch.float32
  if isinstance(ty, ir.F64Type):
    return torch.float64
  if isinstance(ty, ir.F16Type):
    return torch.half
  if isinstance(ty, ir.IntegerType):
    if ty.is_signless:
      if ty.width == 64:
        return torch.long
      if ty.width == 32:
        return torch.int32
      if ty.width == 16:
        return torch.int16
      if ty.width == 1:
        return torch.bool
  raise RuntimeError(f"Unsupported ir element type: {ty}")
