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
"""Mixed precision optimization passes."""

from collections.abc import Callable
import pathlib

from ai_edge_litert.mlir import ir
from xdsl import irdl

from ai_edge_litert.tools import model_utils as mu
from ai_edge_litert.tools.model_utils.dialect import func
from ai_edge_litert.tools.model_utils.dialect import mlir
from ai_edge_litert.tools.model_utils.dialect import stablehlo
from ai_edge_litert.tools.model_utils.dialect import tfl


def fp32_predicate(op):
  """Returns true if the op should be kept in fp32."""
  if isinstance(op, stablehlo.CompositeOp):
    if "odml.rms_norm" == op.composite_name:
      return True
  # if isinstance(op, tfl.SelectV2Op):
  #   return True
  # if isinstance(op, tfl.EmbeddingLookupOp):
  #   return True
  # if isinstance(op, tfl.FullyConnectedOp):
  #   return True
  # if isinstance(op, tfl.BatchMatMulOp):
  #   return True
  # if isinstance(op, tfl.SoftmaxOp):
  #   return True
  # if isinstance(op, tfl.DivOp):
  #   return True
  # if isinstance(op, tfl.GeluOp):
  #   return True
  if isinstance(op, tfl.AddOp):
    return True
  # if isinstance(op, tfl.CosOp):
  #   return True
  # if isinstance(op, tfl.SinOp):
  #   return True
  return False


def convert_model_to_fp16(
    path: str | pathlib.Path,
    fp32_op_predicate: Callable[[irdl.Operation], bool] | None = None,
) -> bytes:
  if isinstance(path, str):
    path = pathlib.Path(path)

  module, ctx = mu.read_flatbuffer(path)
  convert_to_fp16(module, ctx, fp32_op_predicate)
  with ctx:
    return mu.write_flatbuffer(module)


def convert_to_fp16(
    module: mlir.ModuleOp,
    ctx: ir.Context,
    fp32_op_predicate: Callable[[irdl.Operation], bool] | None = None,
) -> None:
  """Converts the model to fp16."""
  with ctx:
    args_to_cast = []
    args_to_update = []
    ops_to_cast = []
    ops_to_update = []
    funcs_to_update = set()
    fp32_ops = set()
    visited = set()

    def _walk(original_op):

      for op in original_op.walk():
        if op not in visited:
          visited.add(op)
        else:
          continue

        if (
            op.parent
            and isinstance(op.parent, irdl.Block)
            and op.parent.parent
            and isinstance(op.parent.parent, irdl.Region)
            and op.parent.parent.parent
            and isinstance(op.parent.parent.parent, func.FuncOp)
            and op.parent.parent.parent in fp32_ops
        ):
          continue

        if op == original_op:
          continue

        if isinstance(op, func.ReturnOp):
          continue

        if fp32_op_predicate and fp32_op_predicate(op):
          fp32_ops.add(op)
          if isinstance(op, stablehlo.CompositeOp):
            fp32_ops.add(op.decomposition_func)
          elif isinstance(op, tfl.SelectV2Op):
            if isinstance(op.operands[2].op, tfl.ConstOp):
              fp32_ops.add(op.operands[2].op)
          continue

        if op in fp32_ops:
          continue

        if isinstance(op, func.FuncOp):
          funcs_to_update.add(op)

          for arg in op.body.block.args:
            if not isinstance(arg.type, mlir.RankedTensorType):
              continue

            if arg.type.elty != "f32":
              continue

            args_to_cast.append(arg)

          _walk(op)

        elif isinstance(op, tfl.ConstOp):
          should_add = False
          for result in op.results:
            if (
                isinstance(result.type, mlir.RankedTensorType)
                and result.type.elty == "f32"
            ):
              should_add = True
              break
          if should_add:
            ops_to_cast.append(op)

        elif isinstance(op, stablehlo.CompositeOp):
          funcs_to_update.add(op.decomposition_func)

          for arg in op.decomposition_func.body.block.args:
            if not isinstance(arg.type, mlir.RankedTensorType):
              continue

            if arg.type.elty != "f32":
              continue

            args_to_update.append(arg)

          _walk(op.decomposition_func)
          ops_to_update.append(op)

        else:
          ops_to_update.append(op)

    _walk(module)

    for arg in args_to_cast:
      arg.type = mlir.RankedTensorType(arg.type.shape, "f16")
      for use in arg.uses.copy():
        if use.operation in fp32_ops:
          with mu.OpBuildingContext(use.operation, insert_before=True):
            cast = tfl.cast(arg, "f32")
            use.operation.operands[use.index] = cast

    for arg in args_to_update:
      arg.type = mlir.RankedTensorType(arg.type.shape, "f16")

    for op in ops_to_cast:
      for result in op.results:
        for use in result.uses.copy():
          # Skip if the use is in a fp32 op. Used for constant tensors.
          if use.operation in fp32_ops:
            continue
          with mu.OpBuildingContext(use.operation, insert_before=True):
            cast = tfl.cast(result, "f16")
            use.operation.operands[use.index] = cast

    for op in ops_to_update:
      for result in op.results:
        if not isinstance(result.type, mlir.RankedTensorType):
          continue
        if result.type.elty != "f32":
          continue

        result.type = mlir.RankedTensorType(result.type.shape, "f16")

    for func_op in funcs_to_update:
      func_op.update_function_type()

    for op in fp32_ops:
      for i, operand in enumerate(op.operands):
        if (
            isinstance(operand.type, mlir.RankedTensorType)
            and operand.type.elty == "f16"
        ):
          with mu.OpBuildingContext(op, insert_before=True):
            cast = tfl.cast(operand, "f32")
            op.operands[i] = cast

      for result in op.results:
        if (
            not isinstance(result.type, mlir.RankedTensorType)
            or result.type.elty != "f32"
        ):
          continue

        for use in result.uses.copy():
          if use.operation not in fp32_ops:
            with mu.OpBuildingContext(use.operation, insert_before=True):
              cast = tfl.cast(result, "f16")
              use.operation.operands[use.index] = cast

      if isinstance(op, func.FuncOp):
        op.update_function_type()

    module.cleanup()
