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
"""Torch ops to Torch-TFL decompositions."""
from typing import Sequence
from litert_torch.backend.experimental.torch_tfl import _ops
import torch

decomps = {}


def register_decomp(op):
  global decomps
  ops = [op]
  if isinstance(op, torch._ops.OpOverloadPacket):
    ops = [getattr(op, overload) for overload in op.overloads()]

  def register(decomp_fn):
    for op in ops:
      decomps[op] = decomp_fn
    return decomp_fn

  return register


@register_decomp(torch.ops.aten.mm.default)
def _aten_mm_decomp(x, y):
  return torch.ops.tfl.batch_matmul(x, y)


@register_decomp(torch.ops.aten.bmm.default)
def _aten_bmm_decomp(x, y):
  return torch.ops.tfl.batch_matmul(x, y)


def _promote_types_for_binary_op(x, y):
  """Promotes operand types for a binary op."""
  # TFLite's binary ops require operands to have the same element type.
  # We promote the types before calling the op.
  # Handle scalar operand by converting scalar to a tensor.
  if not isinstance(x, torch.Tensor):
    x = torch.scalar_tensor(x)
  elif not isinstance(y, torch.Tensor):
    y = torch.scalar_tensor(y)

  target_dtype = torch.promote_types(x.dtype, y.dtype)
  if x.dtype != target_dtype:
    x = x.to(target_dtype)
  if y.dtype != target_dtype:
    y = y.to(target_dtype)
  return x, y


@register_decomp(torch.ops.aten.add.Tensor)
def _aten_add_tensor_decomp(x, y, alpha=1):
  if alpha == 1:
    x, y = _promote_types_for_binary_op(x, y)
    return torch.ops.tfl.add(x, y)

  # The op is add(x, mul(y, alpha))
  y, alpha = _promote_types_for_binary_op(y, alpha)
  mul_out = torch.ops.tfl.mul(y, alpha)
  x, mul_out = _promote_types_for_binary_op(x, mul_out)
  return torch.ops.tfl.add(x, mul_out)


@register_decomp(torch.ops.aten.sub.Tensor)
def _aten_sub_tensor_decomp(x, y, alpha=1):
  if alpha == 1:
    x, y = _promote_types_for_binary_op(x, y)
    return torch.ops.tfl.sub(x, y)

  # The op is sub(x, mul(y, alpha))
  y, alpha = _promote_types_for_binary_op(y, alpha)
  mul_out = torch.ops.tfl.mul(y, alpha)
  x, mul_out = _promote_types_for_binary_op(x, mul_out)
  return torch.ops.tfl.sub(x, mul_out)


@register_decomp(torch.ops.aten.mul.Tensor)
def _aten_mul_tensor_decomp(x, y):
  x, y = _promote_types_for_binary_op(x, y)
  return torch.ops.tfl.mul(x, y)


@register_decomp(torch.ops.aten.mul.Scalar)
def _aten_mul_scalar_decomp(x, y):
  x, y = _promote_types_for_binary_op(x, y)
  return torch.ops.tfl.mul(x, y)


@register_decomp(torch.ops.aten.div.Tensor)
def _aten_div_tensor_decomp(x, y):
  x, y = _promote_types_for_binary_op(x, y)
  return torch.ops.tfl.div(x, y)


@register_decomp(torch.ops.aten.pow.Scalar)
def _aten_pow_scalar_decomp(x, y):
  x, y = _promote_types_for_binary_op(x, y)
  return torch.ops.tfl.pow(x, y)


@register_decomp(torch.ops.aten.pow.Tensor_Scalar)
def _aten_pow_tensor_scalar_decomp(x, y):
  x, y = _promote_types_for_binary_op(x, y)
  return torch.ops.tfl.pow(x, y)


@register_decomp(torch.ops.aten.pow.Tensor_Tensor)
def _aten_pow_tensor_tensor_decomp(x, y):
  x, y = _promote_types_for_binary_op(x, y)
  return torch.ops.tfl.pow(x, y)


@register_decomp(torch.ops.aten.bitwise_and.Tensor)
def _aten_bitwise_and_tensor_decomp(x, y):
  if not (
      isinstance(x, torch.Tensor)
      and x.dtype == torch.bool
      and isinstance(y, torch.Tensor)
      and y.dtype == torch.bool
  ):
    raise TypeError(
        "Input tensors for aten.bitwise_and only supports bool for now."
    )
  return torch.ops.tfl.logical_and(x, y)


@register_decomp(torch.ops.aten.mean.dim)
def _aten_mean_dim_decomp(x, dim, keepdim=False):
  return torch.ops.tfl.mean(x, dim, keepdim)


@register_decomp(torch.ops.aten.gt.Tensor)
def _aten_gt_tensor_decomp(x, y):
  x, y = _promote_types_for_binary_op(x, y)
  return torch.ops.tfl.greater(x, y)


@register_decomp(torch.ops.aten.lt.Tensor)
def _aten_lt_tensor_decomp(x, y):
  x, y = _promote_types_for_binary_op(x, y)
  return torch.ops.tfl.less(x, y)


@register_decomp(torch.ops.aten.maximum.default)
def _aten_maximum_tensor_decomp(x, y):
  return torch.ops.tfl.maximum(x, y)


@register_decomp(torch.ops.aten.minimum.default)
def _aten_minimum_tensor_decomp(x, y):
  return torch.ops.tfl.minimum(x, y)


@register_decomp(torch.ops.aten.sin.default)
def _aten_sin_decomp(x):
  return torch.ops.tfl.sin(x)


@register_decomp(torch.ops.aten.cos.default)
def _aten_cos_decomp(x):
  return torch.ops.tfl.cos(x)


@register_decomp(torch.ops.aten.rsqrt.default)
def _aten_rsqrt_decomp(x):
  return torch.ops.tfl.rsqrt(x)


@register_decomp(torch.ops.aten.neg.default)
def _aten_neg_decomp(x):
  return torch.ops.tfl.neg(x)


@register_decomp(torch.ops.aten.gelu.default)
def _aten_gelu_decomp(x, approximate="none"):
  return torch.ops.tfl.gelu(x, approximate != "none")


@register_decomp(torch.ops.aten.permute.default)
def _aten_permute_decomp(x, dims: Sequence[int]):
  return torch.ops.tfl.transpose(x, dims)


def _prepare_tensors_for_concatenation(
    tensors: Sequence[torch.Tensor], axis: int
) -> Sequence[torch.Tensor]:
  """Prepares PyTorch tensors for concatenation by reshaping 1D (0,) tensors if needed."""
  max_rank = 0
  # First pass: determine max_rank among all input tensors
  for t_val_rank_check in tensors:
    max_rank = max(max_rank, t_val_rank_check.dim())

  ref_tensor_for_shape_inference = None
  # If max_rank > 1, we might need to reshape. Find a reference tensor.
  if max_rank > 1:
    for t_val_ref_check in tensors:
      if t_val_ref_check.dim() == max_rank:
        ref_tensor_for_shape_inference = t_val_ref_check
        break

  processed_operands = []
  # Perform reshaping of 1D (0,) tensors only if concatenating multi-dimensional
  # tensors and a valid reference tensor was found.
  perform_reshaping = (
      max_rank > 1 and ref_tensor_for_shape_inference is not None
  )

  if perform_reshaping:
    ref_shape = list(ref_tensor_for_shape_inference.shape)
    for t_val in tensors:
      current_val = t_val

      # Check if this tensor is 1D, shape (0,), and we are in a context
      # where reshaping to max_rank is needed.
      if torch.numel(t_val) == 0:
        new_shape = list(ref_shape)
        new_shape[axis] = 0
        current_val = torch.ops.tfl.reshape(t_val, new_shape)
      processed_operands.append(current_val)
  else:
    # No reshaping needed, use tensors as they are.
    processed_operands = list(tensors)
  return processed_operands


@register_decomp(torch.ops.aten.cat.default)
def _aten_cat_decomp(tensors, dim=0):
  processed_tensors = _prepare_tensors_for_concatenation(tensors, dim)
  return torch.ops.tfl.concatenation(processed_tensors, dim)


@register_decomp(torch.ops.aten.full.default)
def _aten_full_decomp(
    size,
    fill_value,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
  return torch.ops.tfl.fill(tuple(size), fill_value)


@register_decomp(torch.ops.aten.full_like.default)
def _aten_full_like_decomp(
    x,
    fill_value,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
  return torch.ops.tfl.fill(tuple(x.shape), fill_value)


@register_decomp(torch.ops.aten.view.default)
def _aten_view_decomp(x, shape: Sequence[int]):
  return torch.ops.tfl.reshape(x, shape)


@register_decomp(torch.ops.aten.arange.start_step)
def _aten_arange_start_step_decomp(
    start, end, step=1, dtype=None, layout=None, device=None, pin_memory=None
):
  return torch.ops.tfl.range(start, end, step)


@register_decomp(torch.ops.aten.split_with_sizes.default)
def _aten_split_with_sizes_decomp(x, split_sizes, dim=0):
  outputs = []
  offset = 0
  for size in split_sizes:
    begin = [0] * x.dim()
    begin[dim] = offset
    output_size = list(x.shape)
    output_size[dim] = size
    outputs.append(torch.ops.tfl.slice(x, begin, output_size))
    offset += size
  return tuple(outputs)


@register_decomp(torch.ops.aten.unsqueeze.default)
def _aten_unsqueeze_decomp(x, dim):
  return torch.ops.tfl.expand_dims(x, dim)


@register_decomp(torch.ops.aten.expand.default)
def _aten_expand_decomp(x, shape: Sequence[int]):
  return torch.ops.tfl.broadcast_to(x, shape)


@register_decomp(torch.ops.aten.squeeze.dims)
def _aten_squeeze_dims_decomp(x, squeeze_dims: Sequence[int]):
  if len(squeeze_dims) > 8:
    raise ValueError(
        "torch.ops.tfl.squeeze supports squeezing at most 8 dimensions, but got"
        f" {len(squeeze_dims)} dimensions."
    )
  return torch.ops.tfl.squeeze(x, squeeze_dims)


@register_decomp(torch.ops.aten.select.int)
def _aten_select_int_decomp(x, dim, index):
  rank = len(x.shape)

  # Initialize begin, end, strides
  begin = [0] * rank
  end = list(x.shape)
  strides = [1] * rank

  # Select the index on the given dim
  begin[dim] = index
  end[dim] = index + 1

  # Perform the strided slice
  sliced = torch.ops.tfl.strided_slice(x, begin, end, strides)

  # Remove the selected dimension
  return torch.ops.tfl.squeeze(sliced, [dim])


@register_decomp(torch.ops.aten.slice.Tensor)
def _aten_slice_tensor_decomp(x, dim=0, start=None, end=None, step=1):
  rank = x.dim()
  dim_size = x.shape[dim]

  # Initialize begin, end, strides for tfl.strided_slice
  begin = [0] * rank
  end_vec = list(x.shape)
  strides = [1] * rank

  # The logic below is to match PyTorch's `slice` behavior.
  # `start` and `end` can be negative, which means they count from the end.
  # `start=None` defaults to 0.
  # `end=None` or a large number defaults to `dim_size` after clamping.

  start_val = 0 if start is None else start
  if start_val < 0:
    start_val += dim_size

  end_val = dim_size if end is None else end
  if end_val < 0:
    end_val += dim_size

  # Clamp start and end to be within the dimension size, following PyTorch's
  # logic.
  start_val = max(0, min(start_val, dim_size))
  end_val = max(start_val, min(end_val, dim_size))

  begin[dim], end_vec[dim], strides[dim] = start_val, end_val, step
  return torch.ops.tfl.strided_slice(x, begin, end_vec, strides)


@register_decomp(torch.ops.aten.where.self)
def _aten_where_self_decomp(condition, x, y):
  x, y = _promote_types_for_binary_op(x, y)
  return torch.ops.tfl.select_v2(condition, x, y)


@register_decomp(torch.ops.aten.embedding.default)
def _aten_embedding_decomp(weight, indices, padding_idx=-1):
  # The `tfl.gather` op only supports 1D indices, so we need to flatten the
  # indices and then reshape the output to the correct shape.
  original_indices_shape = list(indices.shape)
  flat_indices = torch.ops.tfl.reshape(indices, [-1])
  # Need to convert indices to int32 for tfl.embedding_lookup.
  flat_indices = flat_indices.to(torch.int32)
  output = torch.ops.tfl.embedding_lookup(flat_indices, weight)
  output_shape = original_indices_shape + [weight.shape[-1]]
  return torch.ops.tfl.reshape(output, output_shape)


@register_decomp(torch.ops.aten._softmax.default)
def _aten__softmax_decomp(
    x, dim: int, half_to_float: bool  # pylint: disable=unused-argument
):
  if dim == -1 or dim == x.dim() - 1:
    return torch.ops.tfl.softmax(x)
  else:
    dims = list(range(x.dim()))
    # Transpose the input by swapping the dim with the last dimension.
    dims[dim], dims[-1] = dims[-1], dims[dim]
    x_permuted = torch.ops.tfl.transpose(x, dims)
    # Compute the softmax on the last dimension.
    softmax_result = torch.ops.tfl.softmax(x_permuted)
    # Transpose the result back to the original dimensions.
    return torch.ops.tfl.transpose(softmax_result, dims)


@register_decomp(torch.ops.aten.multinomial.default)
def _aten_multinomial_decomp(x, num_samples, replacement=False, generator=None):
  is_1d = x.dim() == 1
  if is_1d:
    x = torch.ops.aten.unsqueeze.default(x, 0)
  logits = torch.log(x)
  indices = torch.ops.tfl.multinomial(logits, num_samples, replacement)
  if is_1d:
    indices = torch.ops.aten.squeeze.dims(indices, [0])
  return indices.to(torch.int64)


@register_decomp(torch.ops.aten.topk.default)
def _aten_topk_decomp(self, k, dim=-1, largest=True, sorted=True):
  if not largest:
    raise ValueError("Only largest=True is supported for torch.topk.")

  if dim < 0:
    dim = self.dim() + dim

  if dim != self.dim() - 1:
    self = torch.transpose(self, dim, -1)

  # Ignores sorted value: tfl.topk_v2 only supports sorted=True, but it doesn't
  # affect the correctness of the output.
  out, indices = torch.ops.tfl.topk_v2(self, k)

  if dim != self.dim() - 1:
    out = torch.transpose(out, dim, -1)
    indices = torch.transpose(indices, dim, -1)

  # torch.topk returns int64 indices, but tfl.topk_v2 returns indices in int32.
  indices = indices.to(torch.int64)
  return out, indices
