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
"""Provides lowering for coreaten to stablehlo for Convolution."""

from typing import Optional

from litert_torch.backend.lowerings import registry
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import stablehlo
import torch


def make_padding(padding):
  """Change the padding from pytorch to stablehlo style.

  Stablehlo allows start and end padding for each dimension while aten only
  allows symmetric padding and so only has one number per dimension.

  Args:
    padding: The padding of the convolution

  Returns:
    The padding in stablehlo style
  """
  return tuple((p, p) for p in padding)


def create_conv_dimension_numbers(lhs, transposed: bool = False):
  """Create the dimension numbers for the convolution.

  Args:
    lhs: The input tensor
    transposed: Whether the convolution is transposed

  Returns:
    The dimension numbers for the convolution
  """
  num_spatial_dims = len(lhs.type.shape) - 2
  spatial_dimensions = []
  for i in range(0, num_spatial_dims):
    spatial_dimensions.append(i + 2)

  # Regular kernels are OIHW
  # TransposedConv kernels are IOHW
  dimension_numbers = stablehlo.ConvDimensionNumbers.get(
      input_batch_dimension=0,
      input_feature_dimension=1,
      input_spatial_dimensions=spatial_dimensions,
      kernel_input_feature_dimension=0 if transposed else 1,
      kernel_output_feature_dimension=1 if transposed else 0,
      kernel_spatial_dimensions=spatial_dimensions,
      output_batch_dimension=0,
      output_feature_dimension=1,
      output_spatial_dimensions=spatial_dimensions,
  )
  return dimension_numbers


def build_transpose_conv(
    lctx,
    output_type: ir.RankedTensorType,
    lhs: ir.Value,
    rhs: ir.Value,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    output_padding: list[int],
    groups: int,
):
  del lctx, output_padding
  lhs_type: ir.RankedTensorType = lhs.type
  num_spatial_dims = len(lhs_type.shape) - 2
  rhs = stablehlo.reverse(rhs, list(range(2, 2 + num_spatial_dims)))

  kernel_size = rhs.type.shape
  # We need to additional padding on the input to get the right output size.
  adjusted_padding = [
      dilation[dim] * (kernel_size[dim + 2] - 1) - padding[dim]
      for dim in range(num_spatial_dims)
  ]
  return stablehlo.convolution(
      result=output_type,
      lhs=lhs,
      rhs=rhs,
      dimension_numbers=create_conv_dimension_numbers(lhs, True),
      feature_group_count=groups,
      batch_group_count=1,
      padding=make_padding(adjusted_padding),
      lhs_dilation=stride,
      rhs_dilation=dilation,
  )


# convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride,
#   SymInt[] padding, SymInt[] dilation, bool transposed,
#   SymInt[] output_padding, SymInt groups) -> Tensor
@registry.lower(torch.ops.aten.convolution)
def _aten_convolution(
    lctx,
    lhs: ir.Value,
    rhs: ir.Value,
    bias: Optional[ir.Value],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
):

  # TODO(b/365559296) Add support for output_padding
  if any(output_padding):
    raise NotImplementedError(
        "Output padding on convolution is not implemented."
    )

  out_aval = lctx.node.meta.get("tensor_meta") or lctx.node.meta.get("val")
  lhs_type: ir.RankedTensorType = lhs.type
  output_type = ir.RankedTensorType.get(out_aval.shape, lhs_type.element_type)

  if transposed:
    res = build_transpose_conv(
        lctx,
        output_type,
        lhs,
        rhs,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
    )
  else:
    res = stablehlo.convolution(
        result=output_type,
        lhs=lhs,
        rhs=rhs,
        dimension_numbers=create_conv_dimension_numbers(lhs),
        feature_group_count=groups,
        batch_group_count=1,
        window_strides=stride,
        padding=make_padding(padding),
        rhs_dilation=dilation,
    )

  if bias is not None:
    # broadcast [C] to [NCHW]
    broadcasted_bias = stablehlo.broadcast_in_dim(
        output_type, bias, ir.DenseI64ArrayAttr.get([1])
    )
    res = stablehlo.add(
        lhs=res,
        rhs=broadcasted_bias,
    )

  return res
