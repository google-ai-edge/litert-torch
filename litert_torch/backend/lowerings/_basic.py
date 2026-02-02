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
import logging
import math
import operator
from typing import Optional, Union

from litert_torch.backend import export_utils
from litert_torch.backend.lowerings import context
from litert_torch.backend.lowerings import registry
from litert_torch.backend.lowerings import utils
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import stablehlo
import numpy as np
import torch


LoweringContext = context.LoweringContext
lower = registry.lower


# add(Tensor self, Tensor other) -> Tensor
# @lower(torch.ops.aten.add)
def _aten_add(lctx, x: ir.Value, y: ir.Value, alpha=1):
  x, y = utils.upcast_to_same_type(x, y)
  x, y = utils.broadcast_args_if_needed(x, y)
  if alpha == 1:
    return stablehlo.add(x, y)

  alpha_splat = utils.splat(alpha, y.type.element_type, y.type.shape)
  return stablehlo.add(x, stablehlo.multiply(y, alpha_splat))


# mul.Tensor(Tensor self, Tensor other) -> Tensor
# @lower(torch.ops.aten.mul.Tensor)
def _aten_mul_tensor(lctx, self: ir.Value, other: ir.Value):
  self, other = utils.upcast_to_same_type(self, other)
  self, other = utils.broadcast_args_if_needed(self, other)

  return stablehlo.multiply(self, other)


def _hann_window_impl(
    lctx: LoweringContext,
    size: int,
    periodic: bool,
    dtype: Optional[torch.dtype],
) -> ir.Value:
  if dtype is None:
    ir_dtype = ir.F32Type.get()
  else:
    ir_dtype = utils.torch_dtype_to_ir_element_type(dtype)

  if not isinstance(ir_dtype, ir.FloatType):
    raise ValueError("hann_window only supports float dtypes.")

  if size == 0:
    return stablehlo.ConstantOp(
        ir.RankedTensorType.get((0,), ir_dtype),
        ir.DenseElementsAttr.get_empty(ir.RankedTensorType.get((0,), ir_dtype)),
    ).result
  if size == 1:
    return utils.splat(1.0, ir_dtype, [1])

  denom = size if periodic else size - 1

  i64 = ir.IntegerType.get_signless(64)
  iota_type = ir.RankedTensorType.get((size,), i64)
  n_i64 = stablehlo.IotaOp(
      iota_type, iota_dimension=ir.IntegerAttr.get(i64, 0)
  ).result

  n_type = ir.RankedTensorType.get((size,), ir_dtype)
  n = stablehlo.convert(n_type, n_i64)

  pi_val = math.pi
  scale = 2.0 * pi_val / denom

  scale_splat = utils.splat(scale, ir_dtype, [size])
  arg_cos = stablehlo.multiply(n, scale_splat)
  cos_val = stablehlo.cosine(arg_cos)

  half_splat = utils.splat(0.5, ir_dtype, [size])
  scaled_cos = stablehlo.multiply(half_splat, cos_val)
  return stablehlo.subtract(half_splat, scaled_cos)


# hann_window(int size, *, ScalarType? dtype=None) -> Tensor
@lower(torch.ops.aten.hann_window.default)
def _aten_hann_window_default(
    lctx: LoweringContext,
    size: int,
    **kwargs,
) -> ir.Value:
  dtype = kwargs.pop("dtype", None)
  layout = kwargs.pop("layout", torch.strided)
  if layout != torch.strided:
    logging.warning("hann_window only supports torch.strided layout.")
  return _hann_window_impl(lctx, size, True, dtype)


# hann_window.periodic(int size, bool periodic, *, ScalarType? dtype=None) ->
# Tensor
@lower(torch.ops.aten.hann_window.periodic)
def _aten_hann_window_periodic(
    lctx: LoweringContext,
    size: int,
    periodic: bool,
    **kwargs,
) -> ir.Value:
  dtype = kwargs.pop("dtype", None)
  layout = kwargs.pop("layout", torch.strided)
  if layout != torch.strided:
    logging.warning("hann_window only supports torch.strided layout.")
  return _hann_window_impl(lctx, size, periodic, dtype)


# cat(Tensor[] tensors, int dim=0) -> Tensor
# @lower(torch.ops.aten.cat)
def _aten_cat(lctx, tensors: list[ir.Value], dim: int = 1):
  return stablehlo.ConcatenateOp(tensors, dim).result


# view(Tensor(a) self, SymInt[] size) -> Tensor(a)
# @lower(torch.ops.aten.view)
def _aten_view(lctx, self: ir.Value, size: list[int]):
  return stablehlo.ReshapeOp(
      ir.RankedTensorType.get(size, self.type.element_type), self
  ).result


# hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
@lower(torch.ops.aten.hardtanh)
def _aten_hardtanh(
    lctx,
    self: ir.Value,
    min_val: Union[int, float] = -1.0,
    max_val: Union[int, float] = 1.0,
):
  elty = self.type.element_type
  min_val = utils.splat(min_val, elty)
  max_val = utils.splat(max_val, elty)

  return stablehlo.clamp(min_val, self, max_val)


# mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
# mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *,
#   ScalarType? dtype=None) -> Tensor
@lower(torch.ops.aten.mean)
@lower(torch.ops.aten.mean.dim)
def _aten_mean_dim(
    lctx,
    self: ir.Value,
    dim: Optional[list[int]] = None,
    keepdim: bool = False,
    *,
    dtype=None,
):
  self_shape = self.type.shape
  self_elty = self.type.element_type
  if dim is None:
    dim = list(range(len(self_shape)))
  dim = [len(self_shape) + d if d < 0 else d for d in dim]
  dim_ = ir.DenseI64ArrayAttr.get(np.asarray(dim, np.int64))
  dim_to_keep = [d for d in range(len(self_shape)) if d not in dim]
  dim_to_keep_ = ir.DenseI64ArrayAttr.get(np.asarray(dim_to_keep, np.int64))

  zero_ = utils.splat(0.0, self_elty)

  reduce_result_shape = [
      s for d, s in enumerate(self_shape) if d in dim_to_keep
  ]
  reduce_result_ty = ir.RankedTensorType.get(reduce_result_shape, self_elty)
  reduce_op = stablehlo.ReduceOp([reduce_result_ty], [self], [zero_], dim_)

  reducer_arg_ty = ir.RankedTensorType.get(tuple(), self_elty)
  reducer = reduce_op.regions[0].blocks.append(reducer_arg_ty, reducer_arg_ty)
  with ir.InsertionPoint(reducer):
    stablehlo.return_(
        [stablehlo.add(reducer.arguments[0], reducer.arguments[1])]
    )

  sum_ = reduce_op.result
  if keepdim:
    sum_ = stablehlo.broadcast_in_dim(
        ir.RankedTensorType.get(
            [s if d in dim_to_keep else 1 for d, s in enumerate(self_shape)],
            self_elty,
        ),
        sum_,
        dim_to_keep_,
    )

  dim_els = math.prod([s for d, s in enumerate(self_shape) if d in dim])
  dim_els_ = utils.splat(dim_els, self_elty)
  div_ = stablehlo.broadcast_in_dim(
      sum_.type, dim_els_, ir.DenseI64ArrayAttr.get([])
  )
  mean_ = stablehlo.divide(sum_, div_)

  return mean_


# https://pytorch.org/docs/stable/generated/torch.clone.html
# https://github.com/pytorch/pytorch/blob/a95ceb51a23ae33c00b3a99224143c609b1b3eb3/aten/src/ATen/native/TensorFactories.cpp#L1730
@lower(torch.ops.aten.clone)
def _aten_clone(lctx, x: ir.Value, *, memory_format=None):
  return x


# https://pytorch.org/docs/stable/generated/torch.permute.html
# https://github.com/pytorch/pytorch/blob/519151a062a9bd4f0d32a9c7c7eae47d7ed847b2/aten/src/ATen/native/TensorShape.cpp#L1448
# https://github.com/openxla/stablehlo/blob/main/docs/spec.md#transpose
@lower(torch.ops.aten.permute)
def _aten_permute(lctx, x: ir.Value, dims: list[int]):
  dim = len(x.type.shape)
  return stablehlo.transpose(x, ir.DenseI64ArrayAttr.get(dims))


# https://pytorch.org/docs/stable/generated/torch.mm.html
# https://github.com/pytorch/pytorch/blob/ffabb25c489df1dc631a577c12a0c843c8b202f3/aten/src/ATen/native/LinearAlgebra.cpp#L193
# https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general
@lower(torch.ops.aten.mm)
def _aten_mm(mod, mat1: ir.Value, mat2: ir.Value) -> ir.Value:
  mat1_shape = mat1.type.shape
  mat2_shape = mat2.type.shape
  mat1_dims = len(mat1_shape)
  mat2_dims = len(mat2_shape)

  if mat1_dims != 2 or mat1_dims != 2:
    raise ValueError(
        "Both arguments must be 2D matrices, received dimensions %d and %d"
        % (mat1_dims, mat2_dims)
    )

  if mat1_shape[1] != mat2_shape[0]:
    raise ValueError(
        "mat1 and mat2 shapes cannot be multiplied, received shapes %s and %s"
        % (mat1_shape, mat2_shape)
    )

  dot_dnums = stablehlo.DotDimensionNumbers.get(
      lhs_batching_dimensions=[],
      rhs_batching_dimensions=[],
      lhs_contracting_dimensions=(1,),
      rhs_contracting_dimensions=(0,),
  )
  return stablehlo.dot_general(
      ir.RankedTensorType.get(
          (mat1.type.shape[0], mat2.type.shape[1]), mat1.type.element_type
      ),
      mat1,
      mat2,
      dot_dnums,
  )


# https://pytorch.org/docs/stable/generated/torch.div.html
# https://openxla.org/stablehlo/spec#divide
# TODO: support rounding mode and type promotion (see torch.div spec).
# @lower(torch.ops.aten.div)
def _aten_div(mod, x, y, *, rounding_mode=None, out=None) -> ir.Value:
  # By default, PyTorch performs a "true" division like Python 3. This requires
  # casting integer input types to float to achieve the same semantics using
  # stablehlo.divide.
  if isinstance(x.type.element_type, ir.IntegerType):
    x = utils.convert_int_to_float(x)
  if isinstance(y.type.element_type, ir.IntegerType):
    y = utils.convert_int_to_float(y)

  x, y = utils.broadcast_args_if_needed(x, y)

  return stablehlo.divide(x, y)


# https://pytorch.org/docs/stable/generated/torch.floor.html
# https://openxla.org/stablehlo/spec#floor
@lower(torch.ops.aten.floor)
def _aten_floor(lctx, x: ir.Value, *, out=None) -> ir.Value:
  return stablehlo.floor(x)


# Schema:
#   - aten::abs(Tensor input) -> Tensor
# Torch Reference:
#   - https://pytorch.org/docs/main/generated/torch.abs.html
@lower(torch.ops.aten.abs.default)
def _aten_abs(lctx, input: ir.Value, *, out=None) -> ir.Value:
  return stablehlo.abs(input)


# Schema:
#   - aten::cat(Tensor[] tensors, int dim=0) -> Tensor
# Torch Reference:
#   - https://pytorch.org/docs/main/generated/torch.cat.html
@lower(torch.ops.aten.cat.default)
def _aten_cat(lctx: LoweringContext, tensors, dim=0):
  assert tensors
  non_empty_tensors = [t for t in tensors if np.prod(t.type.shape) != 0]
  out_aval = lctx.node.meta.get("tensor_meta") or lctx.node.meta.get("val")
  if not non_empty_tensors:
    return utils.splat(
        0,
        export_utils.torch_dtype_to_ir_element_type(out_aval.dtype),
        out_aval.shape,
    )

  if dim < 0:
    dim = dim + len(out_aval.shape)
  dim = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), dim)

  return stablehlo.concatenate(non_empty_tensors, dim)


# Schema:
#   - aten::unfold(Tensor self, int dim, int size, int step) -> Tensor
# Torch Reference:
#   - https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
@lower(torch.ops.aten.unfold.default)
def _aten_unfold(lctx, x: ir.Value, dim: int, size: int, step: int):
  x_shape = x.type.shape
  rank = len(x_shape)
  if dim < 0:
    dim += rank

  num_windows = (x_shape[dim] - size) // step + 1
  batch_shape = list(x_shape[:dim]) + [num_windows] + list(x_shape[dim + 1 :])

  # Create start_indices for gather.
  # The shape of start_indices will be batch_shape + [rank].
  # start_indices[b_0,...,b_{rank-1}] will be [p_0,...,p_{rank-1}] where
  # p_j = b_j for j != dim and p_dim = b_dim * step.
  indices_parts = []
  i64 = ir.IntegerType.get_signless(64)
  for i in range(rank):
    bshape = [1] * rank
    bshape[i] = batch_shape[i]
    dim_len = batch_shape[i]

    iota = stablehlo.IotaOp(
        ir.RankedTensorType.get([dim_len], i64),
        iota_dimension=ir.IntegerAttr.get(i64, 0),
    ).result
    if i == dim:
      iota = stablehlo.multiply(iota, utils.splat(step, i64, [dim_len]))

    iota_reshaped = stablehlo.reshape(
        ir.RankedTensorType.get(bshape, i64), iota
    )
    indices_parts.append(
        stablehlo.broadcast_in_dim(
            ir.RankedTensorType.get(batch_shape, i64),
            iota_reshaped,
            ir.DenseI64ArrayAttr.get(list(range(rank))),
        )
    )

  # For each dimension i, indices_parts[i] contains the i-th coordinate
  # of start_indices. We unsqueeze each part to shape batch_shape + [1]
  # and concatenate along the new dimension to produce start_indices of
  # shape batch_shape + [rank].
  unsqueezed_parts = [
      stablehlo.reshape(ir.RankedTensorType.get(batch_shape + [1], i64), part)
      for part in indices_parts
  ]
  start_indices = stablehlo.concatenate(
      unsqueezed_parts, ir.IntegerAttr.get(i64, rank)
  )

  slice_sizes_list = [1] * rank
  slice_sizes_list[dim] = size
  slice_sizes = ir.DenseI64ArrayAttr.get(slice_sizes_list)

  collapsed_slice_dims_list = [i for i in range(rank) if i != dim]

  dnums = stablehlo.GatherDimensionNumbers.get(
      offset_dims=[rank],
      collapsed_slice_dims=collapsed_slice_dims_list,
      operand_batching_dims=[],
      start_indices_batching_dims=[],
      start_index_map=list(range(rank)),
      index_vector_dim=rank,
  )

  return stablehlo.gather(
      x,
      start_indices,
      dnums,
      slice_sizes,
      indices_are_sorted=ir.BoolAttr.get(False),
  )


# Schema:
#   - aten::slice_scatter(Tensor self, Tensor src, int dim=0, SymInt?
#       start=None, SymInt? end=None, SymInt step=1) -> Tensor
# Torch Reference:
#   - https://pytorch.org/docs/stable/generated/torch.slice_scatter.html
#   - https://github.com/pytorch/pytorch/blob/18f9331e5deb4c02ae5c206e133a9b4add49bd97/aten/src/ATen/native/TensorShape.cpp#L4002
@lower(torch.ops.aten.slice_scatter)
def _aten_slice_scatter(lctx, self, src, dim=0, start=None, end=None, step=1):
  start = start if start is not None else 0
  end = end if end is not None else self.type.shape[dim]

  start, end = np.clip(
      [start, end], -self.type.shape[dim], self.type.shape[dim]
  )

  if start < 0:
    start = self.type.shape[dim] + start
  if end < 0:
    end = self.type.shape[dim] + end

  if end <= start or np.prod(src.type.shape) == 0:
    return self

  end = start + step * math.ceil((end - start) / step) - (step - 1)
  padding_low = start
  padding_high = self.type.shape[dim] - end
  interior_padding = step - 1

  rank = len(self.type.shape)
  src = stablehlo.pad(
      src,
      utils.splat(0, src.type.element_type, []),
      edge_padding_low=[padding_low if i == dim else 0 for i in range(rank)],
      edge_padding_high=[padding_high if i == dim else 0 for i in range(rank)],
      interior_padding=[
          interior_padding if i == dim else 0 for i in range(rank)
      ],
  )

  slices = [
      slice(start, end, step) if i == dim else slice(None, None, None)
      for i in range(rank)
  ]
  pred = np.ones(self.type.shape, dtype=np.bool_)
  pred[np.index_exp[tuple(slices)]] = False
  pred = stablehlo.constant(
      ir.DenseElementsAttr.get(
          np.packbits(pred, bitorder="little"),
          type=ir.IntegerType.get_signless(1),
          shape=pred.shape,
      )
  )
  out = stablehlo.select(pred, self, src)
  return out


# Schema:
#   - aten::_to_copy(Tensor self, *, ScalarType? dtype=None,
#       Layout? layout=None, Device? device=None, bool? pin_memory=None,
#       bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
@lower(torch.ops.aten._to_copy.default)
def _aten_to_copy(
    lctx, x: ir.Value, dtype: torch.dtype | None = None, **kwargs
):
  if not dtype:
    return x

  return stablehlo.convert(
      ir.RankedTensorType.get(
          x.type.shape, utils.torch_dtype_to_ir_element_type(dtype)
      ),
      x,
  )


# Schema:
#   - aten::sym_size.int(Tensor self, int dim) -> SymInt
@lower(torch.ops.aten.sym_size.int)
def _aten_sym_size_int(lctx, x: ir.Value, dim: int):
  return stablehlo.get_dimension_size(x, dim)


# Lowering for the addition operator (`+`).
# Handles cases where one operand is an integer (scalar) and the other is a
# tensor, broadcasting the scalar to the tensor's shape before addition.
@lower(operator.add)
def _operator_add(lctx, self: int | ir.Value, other: int | ir.Value):
  if isinstance(self, int) and isinstance(other, ir.Value):
    self = utils.splat(self, other.type.element_type, other.type.shape)
  if isinstance(other, int) and isinstance(self, ir.Value):
    other = utils.splat(other, self.type.element_type, self.type.shape)
  return stablehlo.add(self, other)


# Lowering for the subtraction operator (`-`).
# Handles cases where one operand is an integer (scalar) and the other is a
# tensor, broadcasting the scalar to the tensor's shape before subtraction.
@lower(operator.sub)
def _operator_sub(lctx, self: int | ir.Value, other: int | ir.Value):
  if isinstance(self, int) and isinstance(other, ir.Value):
    self = utils.splat(self, other.type.element_type, other.type.shape)
  if isinstance(other, int) and isinstance(self, ir.Value):
    other = utils.splat(other, self.type.element_type, self.type.shape)
  return stablehlo.subtract(self, other)


# Lowering for the multiplication operator (`*`).
# Handles cases where one operand is an integer (scalar) and the other is a
# tensor, broadcasting the scalar to the tensor's shape before multiplication.
@lower(operator.mul)
def _operator_mul(lctx, self: int | ir.Value, other: int | ir.Value):
  if isinstance(self, int) and isinstance(other, ir.Value):
    self = utils.splat(self, other.type.element_type, other.type.shape)
  if isinstance(other, int) and isinstance(self, ir.Value):
    other = utils.splat(other, self.type.element_type, self.type.shape)
  return stablehlo.multiply(self, other)
