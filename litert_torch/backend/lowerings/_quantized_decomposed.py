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
"""Lowerings for PT2E torch.ops.quantized_decomposed ops."""

from typing import Optional, Union, cast

from litert_torch.backend.lowerings import context
from litert_torch.backend.lowerings import registry
from litert_torch.backend.lowerings import utils
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import stablehlo
import torch
import torch.ao.quantization.fx._decomposed
import torch.utils._pytree as pytree

lower = registry.lower
LoweringContext = context.LoweringContext


def _uniform_quantized_type(
    stored_type: Union[str, ir.Type],
    expressed_type: Union[str, ir.Type],
    *,
    scale=Union[float, list[float], tuple[float]],
    zero_point=Union[float, list[float], tuple[float]],
    storage_type_min: Optional[int] = None,
    storage_type_max: Optional[int] = None,
    channel_axis: Optional[int] = None,
    channel_axis_size: Optional[int] = None,
):
  """Polyfill for quant.UniformQuantizedType."""
  if storage_type_min and storage_type_max:
    storage_min_max = f"<{storage_type_min}:{storage_type_max}>"
  else:
    storage_min_max = ""

  if channel_axis is not None:
    # Per-channel quantization
    # https://mlir.llvm.org/docs/Dialects/QuantDialect/#per-channel-quantization
    assert isinstance(scale, (list, tuple))
    assert isinstance(zero_point, (list, tuple))

    scale = list(scale)
    zero_point = list(zero_point)

    if len(scale) == 1:
      scale = scale * channel_axis_size
    if len(zero_point) == 1:
      zero_point = zero_point * channel_axis_size

    assert len(scale) == len(zero_point) == channel_axis_size
    scale_zp_strs = []
    for s, zp in zip(scale, zero_point):
      scale_zp_strs.append(f"{s}:{zp}")
    scale_zp = "{" + ",".join(scale_zp_strs) + "}"
    return ir.Type.parse(
        f"!quant.uniform<{stored_type}{storage_min_max}:{expressed_type}:{channel_axis},{scale_zp}>"
    )
  else:
    # Per-layer quantization
    # https://mlir.llvm.org/docs/Dialects/QuantDialect/#per-layer-quantization
    scale = pytree.tree_flatten([scale])[0][-1]
    zero_point = pytree.tree_flatten([zero_point])[0][-1]
    scale_zp = f"{scale}:{zero_point}"
    return ir.Type.parse(
        f"!quant.uniform<{stored_type}{storage_min_max}:{expressed_type},{scale_zp}>"
    )


# Quant dialect is not registered in the Python MLIR pybinding used by
# odml-torch. Therefore, stablehlo.uniform_quantize/uniform_dequantize ops and
# quant types are represented in stablehlo.custom_call to pass MLIR verification
# and VHLO serialization before converter.
# TODO(b/362798610) Build MLIR pybinding in litert-torch release.


# Schema:
#   - quantized_decomposed::quantize_per_tensor(Tensor input, float scale,
#       int zero_point, int quant_min, int quant_max,
#       ScalarType dtype) -> Tensor
#   - quantized_decomposed::quantize_per_tensor.tensor(Tensor input,
#       Tensor scale, Tensor zero_point, int quant_min, int quant_max,
#       ScalarType dtype) -> Tensor
#
# Scale and zero_point in tensors are automatically converted to list before
# lowering.
@lower(torch.ops.quantized_decomposed.quantize_per_tensor)
def _quantize_per_tensor(
    lctx: LoweringContext,
    input: ir.Value,
    scale: Union[float, list[float]],
    zero_point: Union[float, list[float]],
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
  input_ty = cast(ir.RankedTensorType, input.type)
  qty = _uniform_quantized_type(
      utils.torch_dtype_to_ir_element_type(dtype),
      input_ty.element_type,
      scale=scale,
      zero_point=zero_point,
      storage_type_min=quant_min,
      storage_type_max=quant_max,
  )
  return stablehlo.custom_call(
      call_target_name="odml_torch.uniform_quantize",
      inputs=[input],
      result=[input_ty],
      backend_config=ir.StringAttr.get(
          str(ir.RankedTensorType.get(input_ty.shape, qty))
      ),
  )


# Schema:
#   - quantized_decomposed::quantize_per_channel(Tensor input, Tensor scales,
#       Tensor zero_points, int axis, int quant_min, int quant_max,
#       ScalarType dtype) -> Tensor
#
# Scale and zero_point in tensors are automatically converted to list before
# lowering.
@lower(torch.ops.quantized_decomposed.quantize_per_channel)
def _quantize_per_channel(
    lctx: LoweringContext,
    input: ir.Value,
    scale: list[float],
    zero_point: list[float],
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
  input_ty = cast(ir.RankedTensorType, input.type)
  qty = _uniform_quantized_type(
      utils.torch_dtype_to_ir_element_type(dtype),
      input_ty.element_type,
      scale=scale,
      zero_point=zero_point,
      channel_axis=axis,
      channel_axis_size=input_ty.shape[axis],
      storage_type_min=quant_min,
      storage_type_max=quant_max,
  )
  return stablehlo.custom_call(
      call_target_name="odml_torch.uniform_quantize",
      inputs=[input],
      result=[input_ty],
      backend_config=ir.StringAttr.get(
          str(ir.RankedTensorType.get(input_ty.shape, qty))
      ),
  )


@lower(torch.ops.quantized_decomposed.dequantize_per_tensor)
@lower(torch.ops.quantized_decomposed.dequantize_per_channel)
def _dequantize(lctx: LoweringContext, input: ir.Value, *args, **kwargs):
  result_meta = lctx.node.meta.get("tensor_meta")
  result_elty = utils.torch_dtype_to_ir_element_type(result_meta.dtype)

  return stablehlo.custom_call(
      call_target_name="odml_torch.uniform_dequantize",
      inputs=[input],
      result=[ir.RankedTensorType.get(result_meta.shape, result_elty)],
  )
