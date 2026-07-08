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

from xdsl import irdl

from litert_converter.tools import model_utils as mu
from litert_converter.tools.model_utils.dialect import func
from litert_converter.tools.model_utils.dialect import mlir
from litert_converter.tools.model_utils.dialect import stablehlo
from litert_converter.tools.model_utils.dialect import tfl


def default_fp32_predicate(op):
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


def is_float(value: irdl.SSAValue):
  if not isinstance(value.type, mlir.RankedTensorType):
    return False
  return value.type.elty in ("f16", "f32")


def convert_model_to_fp16(
    path: str | pathlib.Path,
    fp32_op_predicate: (
        Callable[[irdl.Operation], bool] | None
    ) = default_fp32_predicate,
) -> bytes:
  if isinstance(path, str):
    path = pathlib.Path(path)

  module, ctx = mu.read_flatbuffer(path)
  with ctx:
    convert_to_fp16(module, fp32_op_predicate)
    return mu.write_flatbuffer(module)


def convert_to_fp16(
    module: mlir.ModuleOp,
    fp32_op_predicate: (
        Callable[[irdl.Operation], bool] | None
    ) = default_fp32_predicate,
) -> None:
  """Converts the model to fp16."""
  fp32_ops = set()

  def is_nested_fp32_op(op):
    while op:
      if isinstance(op, irdl.Operation) and op in fp32_ops:
        return True
      op = op.parent
    return False

  def collect_fp32_ops(original_op):
    orig_fp32_ops_len = len(fp32_ops)

    for op in original_op.walk():
      if is_nested_fp32_op(op):
        fp32_ops.add(op)
      elif fp32_op_predicate and fp32_op_predicate(op):
        fp32_ops.add(op)

        if isinstance(op, stablehlo.CompositeOp):
          fp32_ops.add(op.decomposition_func)
        elif isinstance(op, tfl.SelectV2Op):
          if isinstance(op.operands[2].op, tfl.ConstOp):  # pyrefly: ignore[missing-attribute]
            fp32_ops.add(op.operands[2].op)

    return orig_fp32_ops_len != len(fp32_ops)

  # Recursively collect fp32 ops until convergence.
  while collect_fp32_ops(module):
    continue

  for op in module.walk():
    # Do not change cast ops.
    if isinstance(op, tfl.CastOp):
      continue

    # fp32 op
    if op in fp32_ops:
      for i, x in enumerate(op.operands):
        if is_float(x):
          with mu.OpBuildingContext(op, insert_before=True):  # pyrefly: ignore[bad-argument-type]
            op.operands[i] = tfl.cast(x, "f32")
      continue

    # fp16 op
    if isinstance(op, func.FuncOp):
      for arg in op.body.block.args:
        if is_float(arg):
          arg.type = mlir.RankedTensorType(arg.type.shape, "f16")  # pyrefly: ignore[missing-attribute]
    else:
      has_float_operand = False
      for i, x in enumerate(op.operands):
        if is_float(x):
          has_float_operand = True
          with mu.OpBuildingContext(op, insert_before=True):  # pyrefly: ignore[bad-argument-type]
            op.operands[i] = tfl.cast(x, "f16")

      for x in op.results:
        if is_float(x):
          if not has_float_operand:
            # Assumption: if the op has no float operands but has a float
            # result, the result type is determined by the op semantic or
            # attributes. In such case, we need to insert a cast op to convert
            # the result to fp16 and rely on cleanups to propagate the type
            # change.
            with mu.OpBuildingContext(op, insert_after=True):  # pyrefly: ignore[bad-argument-type]
              cast = tfl.cast(x, "f16")
              x.replace_by(cast)
              cast.owner.operands[0] = x
          else:
            # Otherwise, the result type is determined by the input operand
            # types. We can directly update the result type to fp16.
            x.type = mlir.RankedTensorType(x.type.shape, "f16")  # pyrefly: ignore[missing-attribute]

  # Update function types with new argument/result types.
  for op in module.walk():
    if isinstance(op, func.FuncOp):
      op.update_function_type()

  # Canonicalize, CSE, constant folding, etc.
  module.cleanup()
