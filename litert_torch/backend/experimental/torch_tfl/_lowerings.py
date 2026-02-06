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
"""Torch-TFL op to MLIR lowerings."""

from typing import Sequence

from litert_torch.backend.experimental.torch_tfl import _ops
from litert_torch.backend.lowerings import context
from litert_torch.backend.lowerings import registry
from litert_torch.backend.lowerings import utils as lowering_utils
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import stablehlo
import numpy as np
import torch


lower = registry.lower
LoweringContext = context.LoweringContext


def _ir_operation(
    name: str,
    results: Sequence[ir.Type],
    operands: Sequence[ir.Value] | None = None,
    attributes: dict[str, ir.Attribute] | None = None,
):
  """Helper function to create an IR operation in StableHLO CustomCall carrier."""
  if not operands:
    operands = []
  attributes = ir.DictAttr.get(attributes if attributes else {})
  return stablehlo.custom_call(
      result=results,
      inputs=operands,
      call_target_name=ir.StringAttr.get(name),
      has_side_effect=ir.BoolAttr.get(False),
      backend_config=ir.StringAttr.get(str(attributes)),
  )


@lower(torch.ops.tfl.batch_matmul.default)
def _tfl_batch_matmul_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    y: ir.Value,
    adj_x: bool = False,
    adj_y: bool = False,
) -> ir.Value:
  return _ir_operation(
      "tfl.batch_matmul",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, y],
      attributes={
          "adj_x": ir.BoolAttr.get(adj_x),
          "adj_y": ir.BoolAttr.get(adj_y),
          "asymmetric_quantize_inputs": ir.BoolAttr.get(False),
      },
  )


@lower(torch.ops.tfl.add.default)
def _tfl_add_lowering(
    lctx: LoweringContext,
    lhs: ir.Value | int | float,
    rhs: ir.Value | int | float,
    fused_activation_function: str = "NONE",
) -> ir.Value:
  lhs = lowering_utils.convert_to_ir_value(lhs)
  rhs = lowering_utils.convert_to_ir_value(rhs)
  return _ir_operation(
      "tfl.add",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
      attributes={
          "fused_activation_function": ir.StringAttr.get(
              fused_activation_function
          ),
      },
  )


@lower(torch.ops.tfl.sub.default)
def _tfl_sub_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value | int | float,
    fused_activation_function: str = "NONE",
) -> ir.Value:
  rhs = lowering_utils.convert_to_ir_value(rhs)
  return _ir_operation(
      "tfl.sub",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
      attributes={
          "fused_activation_function": ir.StringAttr.get(
              fused_activation_function
          ),
      },
  )


@lower(torch.ops.tfl.mul.default)
def _tfl_mul_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value | int | float,
    fused_activation_function: str = "NONE",
) -> ir.Value:
  rhs = lowering_utils.convert_to_ir_value(rhs)
  return _ir_operation(
      "tfl.mul",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
      attributes={
          "fused_activation_function": ir.StringAttr.get(
              fused_activation_function
          ),
      },
  )


@lower(torch.ops.tfl.div.default)
def _tfl_div_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value | int | float,
    fused_activation_function: str = "NONE",
) -> ir.Value:
  rhs = lowering_utils.convert_to_ir_value(rhs)
  return _ir_operation(
      "tfl.div",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
      attributes={
          "fused_activation_function": ir.StringAttr.get(
              fused_activation_function
          ),
      },
  )


@lower(torch.ops.tfl.pow.default)
def _tfl_pow_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value | int | float,
) -> ir.Value:
  lhs = lowering_utils.convert_to_ir_value(lhs)
  rhs = lowering_utils.convert_to_ir_value(rhs)
  return _ir_operation(
      "tfl.pow",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
  )


@lower(torch.ops.tfl.logical_and.default)
def _tfl_logical_and_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.logical_and",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
  )


@lower(torch.ops.tfl.mean.default)
def _tfl_mean_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    dims: int | ir.Value | Sequence[int | ir.Value],
    keepdim: bool = False,
) -> ir.Value:
  if isinstance(dims, int) or isinstance(dims, ir.Value):
    dims_ir_value = lowering_utils.convert_to_ir_value(dims)
  else:
    dims_ir_value = lowering_utils.convert_shape_to_ir_value(dims)
  return _ir_operation(
      "tfl.mean",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, dims_ir_value],
      attributes={
          "keep_dims": ir.BoolAttr.get(keepdim),
      },
  )


@lower(torch.ops.tfl.greater.default)
def _tfl_greater_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.greater",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
  )


@lower(torch.ops.tfl.less.default)
def _tfl_less_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.less",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
  )


@lower(torch.ops.tfl.maximum.default)
def _tfl_maximum_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.maximum",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
  )


@lower(torch.ops.tfl.minimum.default)
def _tfl_minimum_lowering(
    lctx: LoweringContext,
    lhs: ir.Value,
    rhs: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.minimum",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[lhs, rhs],
  )


@lower(torch.ops.tfl.sin.default)
def _tfl_sin_lowering(
    lctx: LoweringContext,
    x: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.sin",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
  )


@lower(torch.ops.tfl.cos.default)
def _tfl_cos_lowering(
    lctx: LoweringContext,
    x: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.cos",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
  )


@lower(torch.ops.tfl.rsqrt.default)
def _tfl_rsqrt_lowering(
    lctx: LoweringContext,
    x: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.rsqrt",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
  )


@lower(torch.ops.tfl.neg.default)
def _tfl_neg_lowering(
    lctx: LoweringContext,
    x: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.neg",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
  )


@lower(torch.ops.tfl.gelu.default)
def _tfl_gelu_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    approximate: bool = False,
) -> ir.Value:
  return _ir_operation(
      "tfl.gelu",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
      attributes={
          "approximate": ir.BoolAttr.get(approximate),
      },
  )


@lower(torch.ops.tfl.transpose.default)
def _tfl_transpose_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    perm: Sequence[int],
) -> ir.Value:
  constant_perm = lowering_utils.numpy_array_constant(
      np.array(perm, dtype=np.int32)
  )
  return _ir_operation(
      "tfl.transpose",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, constant_perm],
  )


@lower(torch.ops.tfl.concatenation.default)
def _tfl_concatenation_lowering(
    lctx: LoweringContext,
    tensors: Sequence[ir.Value],
    axis: int,
    fused_activation_function: str = "NONE",
) -> ir.Value:
  return _ir_operation(
      "tfl.concatenation",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=tensors,
      attributes={
          "axis": ir.IntegerAttr.get(ir.IntegerType.get_signless(32), axis),
          "fused_activation_function": ir.StringAttr.get(
              fused_activation_function
          ),
      },
  )


@lower(torch.ops.tfl.fill.default)
def _tfl_fill_lowering(
    lctx: LoweringContext,
    dims: Sequence[int | ir.Value],
    fill_value: int | float | ir.Value,
) -> ir.Value:
  dims_ir_value = lowering_utils.convert_shape_to_ir_value(dims)
  fill_value_ir_value = lowering_utils.convert_to_ir_value(fill_value)

  # Ensure fill_value_ir_value is a scalar (0-D tensor) for TFLite Fill op.
  # The TFLite Fill kernel expects the value to be a 0-D tensor.
  if isinstance(fill_value_ir_value.type, ir.RankedTensorType):
    tensor_type = fill_value_ir_value.type
    # If it's a 1-D tensor with a single element, reshape to 0-D.
    if list(tensor_type.shape) == [1]:
      scalar_type = ir.RankedTensorType.get([], tensor_type.element_type)
      fill_value_ir_value = stablehlo.reshape(scalar_type, fill_value_ir_value)

  # Determine the target element type from the node's output definition.
  result_types = lowering_utils.node_meta_to_ir_types(lctx.node)
  if not result_types or not isinstance(result_types[0], ir.RankedTensorType):
    raise ValueError(
        "tfl.fill: Unable to determine result tensor type or result is not a"
        " ranked tensor."
    )
  target_element_type = result_types[0].element_type

  # Ensure fill_value_ir_value is a RankedTensorType to access its properties.
  if not isinstance(fill_value_ir_value.type, ir.RankedTensorType):
    raise TypeError(
        "tfl.fill: fill_value_ir_value expected to be RankedTensorType, got"
        f" {fill_value_ir_value.type}"
    )

  current_fill_tensor_type = fill_value_ir_value.type
  current_element_type = current_fill_tensor_type.element_type

  # If the element type of the (scalar) fill_value doesn't match the target
  # output element type, cast fill_value_ir_value to the target_element_type
  # while maintaining its current shape (which should be scalar).
  if current_element_type != target_element_type:
    cast_to_type = ir.RankedTensorType.get(
        current_fill_tensor_type.shape, target_element_type
    )
    fill_value_ir_value = stablehlo.convert(cast_to_type, fill_value_ir_value)

  return _ir_operation(
      "tfl.fill",
      results=result_types,
      operands=[dims_ir_value, fill_value_ir_value],
  )


@lower(torch.ops.tfl.reshape.default)
def _tfl_reshape_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    shape: Sequence[int | ir.Value],
) -> ir.Value:
  return _ir_operation(
      "tfl.reshape",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, lowering_utils.convert_shape_to_ir_value(shape)],
  )


@lower(torch.ops.tfl.range.default)
def _tfl_range_lowering(
    lctx: LoweringContext,
    start: int | float | ir.Value,
    limit: int | float | ir.Value,
    delta: int | float | ir.Value = 1,
) -> ir.Value:
  tensor_meta = lctx.node.meta.get("tensor_meta") or lctx.node.meta.get("val")
  output_torch_dtype = tensor_meta.dtype

  original_mlir_output_types = lowering_utils.node_meta_to_ir_types(lctx.node)
  if not original_mlir_output_types or not isinstance(
      original_mlir_output_types[0], ir.RankedTensorType
  ):
    raise ValueError(
        "tfl.range output type is not a RankedTensorType as expected."
    )

  original_mlir_output_type = original_mlir_output_types[0]
  original_output_shape = original_mlir_output_type.shape
  original_output_element_type = original_mlir_output_type.element_type
  tflite_op_internal_element_type = (
      lowering_utils.torch_dtype_to_ir_element_type(output_torch_dtype)
  )

  # All operands and the output of tfl.range must have the same element type.
  # We cast all operands to the expected output element type.
  operands = []
  for operand in [start, limit, delta]:
    if not isinstance(operand, ir.Value):
      # Convert python scalars to ir.Value.
      numpy_scalar_0d = (
          torch.tensor(operand, dtype=output_torch_dtype).detach().numpy()
      )
      operand = lowering_utils.numpy_array_constant(numpy_scalar_0d)

    # `operand` is now an ir.Value.
    # Cast its element type to the output element type if they don't match.
    operand_type = operand.type
    if not isinstance(operand_type, ir.RankedTensorType):
      raise TypeError(
          "tfl.range operand expected to be RankedTensorType, got"
          f" {operand_type}"
      )

    if operand_type.element_type != tflite_op_internal_element_type:
      cast_to_type = ir.RankedTensorType.get(
          operand_type.shape, tflite_op_internal_element_type
      )
      operand = stablehlo.convert(cast_to_type, operand)
    operands.append(operand)

  # Define the result type that the tfl.range *kernel* (the custom op) will
  # produce.
  tfl_op_kernel_output_type = ir.RankedTensorType.get(
      original_output_shape, tflite_op_internal_element_type
  )

  tfl_range_op_val = _ir_operation(
      "tfl.range",
      results=[tfl_op_kernel_output_type],
      operands=operands,
  )

  # The _tfl_range_lowering function must return a value of the
  # original_mlir_output_type.
  # If the tfl.range op's internal element type is different from the
  # original_output_element_type, we need to convert.
  if tflite_op_internal_element_type != original_output_element_type:
    # Convert the tfl.range output to the original expected type.
    final_output_val = stablehlo.convert(
        original_mlir_output_type, tfl_range_op_val
    )
  else:
    final_output_val = tfl_range_op_val

  return final_output_val


@lower(torch.ops.tfl.split_v.default)
def _tfl_split_v_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    size_splits: Sequence[int | ir.Value],
    dim: int | ir.Value,
) -> ir.Value:
  size_splits_ir_value = lowering_utils.convert_shape_to_ir_value(size_splits)
  dim_ir_value = lowering_utils.numpy_array_constant(
      np.array(dim, dtype=np.int32)
  )
  return _ir_operation(
      "tfl.split_v",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, size_splits_ir_value, dim_ir_value],
      attributes={
          "num_splits": ir.IntegerAttr.get(
              ir.IntegerType.get_signless(32), len(size_splits)
          ),
      },
  )


@lower(torch.ops.tfl.slice.default)
def _tfl_slice_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    begin: Sequence[int | ir.Value],
    size: Sequence[int | ir.Value],
) -> ir.Value:
  begin_ir_value = lowering_utils.convert_shape_to_ir_value(begin)
  size_ir_value = lowering_utils.convert_shape_to_ir_value(size)
  return _ir_operation(
      "tfl.slice",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, begin_ir_value, size_ir_value],
  )


@lower(torch.ops.tfl.expand_dims.default)
def _tfl_expand_dims_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    dim: int | ir.Value,
) -> ir.Value:
  dim_ir_value = lowering_utils.numpy_array_constant(
      np.array(dim, dtype=np.int32)
  )
  return _ir_operation(
      "tfl.expand_dims",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, dim_ir_value],
  )


@lower(torch.ops.tfl.broadcast_to.default)
def _tfl_broadcast_to_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    shape: Sequence[int | ir.Value],
) -> ir.Value:
  return _ir_operation(
      "tfl.broadcast_to",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, lowering_utils.convert_shape_to_ir_value(shape)],
  )


@lower(torch.ops.tfl.squeeze.default)
def _tfl_squeeze_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    squeeze_dims: Sequence[int | ir.Value],
) -> ir.Value:
  return _ir_operation(
      "tfl.squeeze",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
      attributes={
          "squeeze_dims": ir.ArrayAttr.get([
              ir.IntegerAttr.get(ir.IntegerType.get_signless(64), int(d))
              for d in squeeze_dims
          ]),
      },
  )


@lower(torch.ops.tfl.strided_slice.default)
def _tfl_strided_slice_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    begin: Sequence[int | ir.Value],
    end: Sequence[int | ir.Value],
    strides: Sequence[int | ir.Value],
    begin_mask: int = 0,
    end_mask: int = 0,
    ellipsis_mask: int = 0,
    new_axis_mask: int = 0,
    shrink_axis_mask: int = 0,
    offset: bool = False,
) -> ir.Value:
  begin_ir_value = lowering_utils.convert_shape_to_ir_value(begin)
  end_ir_value = lowering_utils.convert_shape_to_ir_value(end)
  strides_ir_value = lowering_utils.convert_shape_to_ir_value(strides)
  return _ir_operation(
      "tfl.strided_slice",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, begin_ir_value, end_ir_value, strides_ir_value],
      attributes={
          "begin_mask": ir.IntegerAttr.get(
              ir.IntegerType.get_signless(32), begin_mask
          ),
          "end_mask": ir.IntegerAttr.get(
              ir.IntegerType.get_signless(32), end_mask
          ),
          "ellipsis_mask": ir.IntegerAttr.get(
              ir.IntegerType.get_signless(32), ellipsis_mask
          ),
          "new_axis_mask": ir.IntegerAttr.get(
              ir.IntegerType.get_signless(32), new_axis_mask
          ),
          "shrink_axis_mask": ir.IntegerAttr.get(
              ir.IntegerType.get_signless(32), shrink_axis_mask
          ),
          "offset": ir.BoolAttr.get(offset),
      },
  )


@lower(torch.ops.tfl.select_v2.default)
def _tfl_select_v2_lowering(
    lctx: LoweringContext,
    condition: ir.Value,
    x: ir.Value,
    y: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.select_v2",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[condition, x, y],
  )


@lower(torch.ops.tfl.embedding_lookup.default)
def _tfl_embedding_lookup_lowering(
    lctx: LoweringContext,
    indices: ir.Value,
    weight: ir.Value,
) -> ir.Value:
  return _ir_operation(
      "tfl.embedding_lookup",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[indices, weight],
  )


@lower(torch.ops.tfl.gather.default)
def _tfl_gather_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    indices: ir.Value,
    axis: int,
    batch_dims: int = 0,
) -> ir.Value:
  return _ir_operation(
      "tfl.gather",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x, indices],
      attributes={
          "axis": ir.IntegerAttr.get(ir.IntegerType.get_signless(32), axis),
          "batch_dims": ir.IntegerAttr.get(
              ir.IntegerType.get_signless(32), batch_dims
          ),
      },
  )


@lower(torch.ops.tfl.softmax.default)
def _tfl_softmax_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    beta: float = 1.0,
) -> ir.Value:
  return _ir_operation(
      "tfl.softmax",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[x],
      attributes={
          "beta": ir.FloatAttr.get(ir.F32Type.get(), beta),
      },
  )


@lower(torch.ops.tfl.topk_v2.default)
def _tfl_topk_v2_lowering(
    lctx: LoweringContext,
    x: ir.Value,
    k: int,
) -> tuple[ir.Value, ir.Value]:
  return _ir_operation(
      "tfl.topk_v2",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[
          x,
          lowering_utils.numpy_array_constant(np.array(k, dtype=np.int32)),
      ],
      attributes={},
  )


@lower(torch.ops.tfl.multinomial.default)
def _tfl_multinomial_lowering(
    lctx: LoweringContext,
    logits: ir.Value,
    num_samples: int,
    replacement: bool = False,
) -> ir.Value:
  if replacement:
    raise ValueError("tfl.multinomial does not support with_replacement=True.")
  return _ir_operation(
      "tfl.multinomial",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[
          logits,
          lowering_utils.numpy_array_constant(
              np.array(num_samples, dtype=np.int32)
          ),
      ],
      attributes={},
  )
