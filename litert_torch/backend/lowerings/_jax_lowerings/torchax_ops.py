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
"""Torchax ATen ops implemented in JAX.

Branched from https://github.com/google/torchax/blob/main/torchax/ops/jaten.py.
"""

import math
import sys
from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
from litert_torch.backend.lowerings._jax_lowerings import interop
import numpy as np
import torch

__all__ = ['ALL_OPS']

ALL_OPS = {}

maybe_convert_constant_dtype = interop.maybe_convert_constant_dtype
convert_dtype = interop.convert_dtype
promote_int_input = interop.promote_int_input
t2j_dtype = interop.t2j_dtype


def op(*aten_ops, **kwargs):
  del kwargs

  def inner(func):
    resolved_ops = []
    for aten_op in aten_ops:
      resolved_ops.append(aten_op)
      if isinstance(aten_op, torch._ops.OpOverloadPacket):
        resolved_ops.extend(
            [getattr(aten_op, overload) for overload in aten_op.overloads()]
        )

    for resolved_op in resolved_ops:
      ALL_OPS[resolved_op] = func

    return func

  return inner


@op(
    torch.ops.aten.view_copy,
    torch.ops.aten.view,
    torch.ops.aten._unsafe_view,
    torch.ops.aten.reshape,
)
def _aten_unsafe_view(x, shape):
  return jnp.reshape(x, shape)


@op(torch.ops.aten.clone)
def _aten_clone(x, memory_format=None):
  return x


@op(torch.ops.aten.trunc)
def _aten_trunc(x):
  res = jnp.trunc(x)
  return res.astype(x.dtype)


@op(torch.ops.aten.index_copy)
def _aten_index_copy(x, dim, indexes, source):
  if x.ndim == 0:
    return source
  if x.ndim == 1:
    source = jnp.squeeze(source)
  if dim < 0:
    dim = dim + x.ndim
  dims = []
  for i in range(len(x.shape)):
    if i == dim:
      dims.append(indexes)
    else:
      dims.append(slice(None, None, None))
  return x.at[tuple(dims)].set(source)


@op(torch.ops.aten.select)
def _aten_select(x, dim, indexes):
  return jax.lax.index_in_dim(x, index=indexes, axis=dim, keepdims=False)


@op(torch.ops.aten.index_select)
@op(torch.ops.aten.select_copy)
def _aten_index_select(x, dim, index):
  if x.shape == ():
    return x
  # aten.index_select requires in-range indices, so clamping is a valid
  # realization. mode='clip' is StableHLO gather's native semantics; the default
  # mode='fill' emits NumPy-compat out-of-bounds machinery that becomes ~9
  # extra TFLite ops around the gather.
  return jnp.take(x, index, dim, mode='clip')


@op(torch.ops.aten.mean)
def _aten_mean(x, dim=None, keepdim=False):
  if x.shape == () and dim is not None:
    dim = None
  return jnp.mean(x, dim, keepdims=keepdim)


def _torch_binary_scalar_type(scalar, tensor):
  if 'float' in str(tensor.dtype) or 'complex' in str(tensor.dtype):
    return tensor.dtype
  if isinstance(scalar, int):
    if 'int' in str(tensor.dtype):
      return tensor.dtype
  return jnp.float32


@op(torch.ops.aten.mm)
def _aten_mm(x, y):
  res = x @ y
  return res


@op(torch.ops.aten.silu)
@op(torch.ops.aten.silu.default)
def _aten_silu(x):
  return jax.nn.silu(x)


@op(torch.ops.aten.t)
def _aten_t(x):
  return jnp.transpose(x)


@op(torch.ops.aten.transpose)
@op(torch.ops.aten.transpose_copy)
def _aten_transpose(x, dim0, dim1):
  if x.ndim == 0:
    return x
  dim0 = dim0 if dim0 >= 0 else dim0 + x.ndim
  dim1 = dim1 if dim1 >= 0 else dim1 + x.ndim
  return jnp.swapaxes(x, dim0, dim1)


@op(torch.ops.aten.triu)
def _aten_triu(m, k=0):
  return jnp.triu(m, k)


@op(torch.ops.aten.slice)
@op(torch.ops.aten.slice_copy)
def _aten_slice(self, dim=0, start=None, end=None, step=1):
  if dim < 0:
    dim += self.ndim
  if end == sys.maxsize:
    end = self.shape[dim]
  sl = slice(start, end, step)
  dims = []
  for i in range(len(self.shape)):
    if i == dim:
      dims.append(sl)
    else:
      dims.append(slice(None, None, None))
  return self[tuple(dims)]


@op(torch.ops.aten.positive)
@op(torch.ops.aten.detach)
def _aten_detach(self):
  return self


@op(torch.ops.aten.view_as_real)
def _aten_view_as_real(x):
  real = jnp.real(x)
  im = jnp.imag(x)
  res = jnp.stack([real, im], -1)
  return res


@op(torch.ops.aten.stack)
def _aten_stack(tensors, dim=0):
  return jnp.stack(tensors, dim)


@op(torch.ops.aten._softmax)
@op(torch.ops.aten.softmax)
@op(torch.ops.aten.softmax.int)
def _aten_softmax(x, dim, halftofloat=False):
  if x.shape == ():
    return jax.nn.softmax(x.reshape([1]), axis=0).reshape([])
  return jax.nn.softmax(x, dim)


def _is_int(x):
  if isinstance(x, int):
    return True
  if isinstance(x, jax.Array) and (
      x.dtype.name.startswith('int') or x.dtype.name.startswith('uint')
  ):
    return True
  return False


def highest_precision_int_dtype(tensor1, tensor2):
  if isinstance(tensor1, int):
    return tensor2.dtype
  if isinstance(tensor2, int):
    return tensor1.dtype
  dtype_hierarchy = {
      'uint8': 8,
      'int8': 8,
      'uint16': 16,
      'int16': 16,
      'uint32': 32,
      'int32': 32,
      'uint64': 64,
      'int64': 64,
  }
  return max(
      tensor1.dtype,
      tensor2.dtype,
      key=lambda dtype: dtype_hierarchy[str(dtype)],
  )


@op(torch.ops.aten.pow)
def _aten_pow(x, y):
  y_orig = y
  if isinstance(y, int):
    y = float(y)
  if _is_int(x) and _is_int(y_orig):
    res = jnp.power(jnp.astype(x, jnp.dtype('float')), y)
    return res.astype(highest_precision_int_dtype(x, y_orig))
  res = jnp.power(x, y)
  if isinstance(x, float):
    return res.astype(_torch_binary_scalar_type(x, y_orig))
  if isinstance(y_orig, float):
    return res.astype(_torch_binary_scalar_type(y_orig, x))
  return res


@op(torch.ops.aten.view_as_complex)
def _aten_view_as_complex(input):
  if input.dtype == jnp.bfloat16:
    input = input.astype(jnp.float32)
  x, y = (input[..., 0], input[..., 1])
  return jax.lax.complex(x, y)


@op(torch.ops.aten.div)
def _aten_div(x, y, rounding_mode=''):
  res_dtype = None
  if _is_int(x) and _is_int(y):
    res_dtype = jnp.dtype('float32')
  if isinstance(x, float) or isinstance(y, float):
    res_dtype = new_dtype = t2j_dtype(torch.get_default_dtype())
  if rounding_mode == 'floor':
    res = jnp.floor_divide(x, y)
    if _is_int(x) and _is_int(y):
      res_dtype = jnp.dtype('int64')
  else:
    res = x / y
  if rounding_mode == 'trunc':
    res = jnp.trunc(res)
    if _is_int(x) and _is_int(y):
      res_dtype = jnp.dtype('int64')
  if res_dtype:
    res = res.astype(res_dtype)
  return res


@op(torch.ops.aten.true_divide)
def _aten_true_divide(x, y):
  return x / y


@op(torch.ops.aten.bmm)
def _aten_bmm(x, y):
  res = x @ y
  return res


@op(torch.ops.aten.embedding)
def _aten_embedding(
    a, w, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
  # aten.embedding requires in-range indices (see _aten_index_select for why
  # mode='clip' is used).
  if padding_idx == -1:
    return jnp.take(a, w, axis=0, mode='clip')

  if padding_idx is not None and padding_idx != -1:
    a = a.at[padding_idx].set(0.0)

  return jnp.take(a, w, axis=0, mode='clip')


@op(torch.ops.aten.rsqrt)
@promote_int_input
def _aten_rsqrt(x):
  return jax.lax.rsqrt(x)


@op(torch.ops.aten.expand)
@op(torch.ops.aten.expand_copy)
def _aten_expand(x, dims):

  def fix_dims(d, xs):
    if d == -1:
      return xs
    return d

  shape = list(x.shape)
  if len(shape) < len(dims):
    shape = [1] * (len(dims) - len(shape)) + shape
  dims = [fix_dims(p, s) for p, s in zip(dims, shape)]
  return jnp.broadcast_to(x, dims)


@op(torch.ops.aten.dot)
def _aten_dot(x, y):
  return jnp.dot(x, y)


@op(torch.ops.aten.empty)
@convert_dtype(use_default_dtype=False)
def _aten_empty(size: Sequence[int], *, dtype=None, **kwargs):
  return jnp.empty(size, dtype=dtype)


@op(torch.ops.aten.full)
@convert_dtype()
def _full(size: Sequence[int], fill_value, *, dtype=None, **kwargs):
  return jnp.full(size, fill_value, dtype=dtype)


@op(torch.ops.aten.index_put)
def _aten_index_put(self, indexes, values, accumulate=False):
  indexes = [slice(None, None, None) if i is None else i for i in indexes]
  indexes = tuple(indexes)
  if accumulate:
    return self.at[indexes].add(values)
  else:
    return self.at[indexes].set(values)


@op(torch.ops.aten.index)
@op(torch.ops.aten._unsafe_index)
@op(torch.ops.aten.index.Tensor)
def _aten_index(self, indexes):
  indexes = [slice(None, None, None) if i is None else i for i in indexes]
  indexes = tuple(indexes)
  return self[indexes]


@op(torch.ops.aten.split)
@op(torch.ops.aten.split_copy)
@op(torch.ops.aten.split_with_sizes)
def split_with_sizes(x, sizes, dim=0):
  """Splits an array `x` into sub-arrays based on static sizes `sizes`.

  Args:
    x: The input array to split.
    sizes: A 1D array of integer sizes for each sub-array.

  Returns:
    A list of sub-arrays.
  """
  if isinstance(sizes, int):
    new_sizes = [sizes] * -(-x.shape[dim] // sizes)
    sizes = new_sizes
  rank = x.ndim
  splits = np.cumsum(sizes)

  def make_range(rank, dim, start, end):
    res = [slice(None, None, None)] * rank
    res[dim] = slice(start, end)
    return tuple(res)

  return [
      x[make_range(rank, dim, start, end)]
      for start, end in zip([0] + list(splits[:-1]), splits)
  ]


@op(torch.ops.aten.permute)
@op(torch.ops.aten.permute_copy)
def permute(t, dims):
  return jnp.transpose(t, dims)


@op(torch.ops.aten.unsqueeze)
@op(torch.ops.aten.unsqueeze_copy)
def _aten_unsqueeze(self, dim):
  if dim < 0:
    dim += self.ndim + 1
  return jnp.expand_dims(self, dim)


@op(torch.ops.aten.ne)
def _aten_ne(x, y):
  return jnp.not_equal(x, y)


@op(torch.ops.aten.cumsum)
def _aten_cumsum(x, y, dtype=None):
  if dtype:
    dtype = t2j_dtype(dtype)
  if not x.shape:
    return x
  res = jnp.cumsum(x, y, dtype)
  return res


@op(torch.ops.aten.addmm)
@op(torch.ops.aten.addmv)
def _aten_addmm(self, mat1, mat2, *, beta=1.0, alpha=1.0):
  alpha = jnp.array(alpha).astype(mat1.dtype)
  beta = jnp.array(beta).astype(mat1.dtype)
  self *= beta
  self += alpha * jnp.matmul(mat1, mat2)
  return self


@op(torch.ops.aten.addbmm.default)
def _aten_addbmm(input, batch1, batch2, *, beta=1, alpha=1):
  alpha = jnp.array(alpha).astype(batch1.dtype)
  beta = jnp.array(beta).astype(batch1.dtype)
  mm = jnp.einsum('bxy, byz -> xz', batch1, batch2)
  return jax.lax.cond(
      beta == 0, lambda: alpha * mm, lambda: beta * input + alpha * mm
  )


@op(torch.ops.aten.gelu)
def _aten_gelu(self, *, approximate='none'):
  approx = approximate == 'tanh'
  return jax.nn.gelu(self, approx)


@op(torch.ops.aten.squeeze)
@op(torch.ops.aten.squeeze_copy)
def _aten_squeeze_dim(self, dim=None):
  if self.ndim == 0:
    return self
  if dim is not None:
    if isinstance(dim, int):
      if self.shape[dim] != 1:
        return self
      if dim < 0:
        dim += self.ndim
    else:
      dim = [i if i >= 0 else i + self.ndim for i in dim if self.shape[i] == 1]
  return jnp.squeeze(self, dim)


@op(torch.ops.aten._native_batch_norm_legit.default)
def _aten__native_batch_norm_legit(
    input, weight, bias, running_mean, running_var, training, momentum, eps
):
  """JAX implementation of batch normalization with optional parameters.

  Refers to
  https://github.com/pytorch/pytorch/blob/cd3a71f754a2248bcfe500de7c9860bd7d2002bf/torch/_decomp/decompositions.py#L1713.

  Args:
    input (DeviceArray): Input data (N, C, H, W).
    running_mean ([DeviceArray]): Running mean of input (C,).
    running_var ([DeviceArray]): Running variance of input (C,).
    weight (Optional[DeviceArray]): Scaling factor (gamma) (C,). Can be None.
    bias (Optional[DeviceArray]): Shift factor (beta) (C,). Can be None.
    training (bool): If True, use batch statistics for normalization. If False,
      use running statistics.
    momentum (float): Momentum factor for updating running statistics.
    eps (float): Small constant for numerical stability.

  Returns:
    DeviceArray: Normalized output
    DeviceArray: Batch mean (C,) or empty if training is False
    DeviceArray: Reversed batch variance (C,) or empty if training is False
  """
  reshape_dims = [1, -1] + [1] * (input.ndim - 2)
  if training:
    raise ValueError('training is not supported')

  rstd = jax.lax.rsqrt(running_var.reshape(reshape_dims) + eps)
  saved_mean = jnp.array([], dtype=input.dtype)
  saved_rstd = jnp.array([], dtype=input.dtype)
  x_hat = (input - running_mean.reshape(reshape_dims)) * rstd

  if weight is not None:
    x_hat *= weight.reshape(reshape_dims)
  if bias is not None:
    x_hat += bias.reshape(reshape_dims)
  return (x_hat, saved_mean, saved_rstd)


@op(torch.ops.aten._native_batch_norm_legit_no_training)
def _aten__native_batch_norm_legit_no_training(
    input, weight, bias, running_mean, running_var, momentum, eps
):
  return _aten__native_batch_norm_legit(
      input, weight, bias, running_mean, running_var, False, momentum, eps
  )


@op(torch.ops.aten.relu)
def _aten_relu(self):
  return jax.nn.relu(self)


def _ceil_mode_padding(
    padding: list[int],
    input_shape: list[int],
    kernel_size: list[int],
    stride: list[int],
    dilation: list[int],
    ceil_mode: bool,
):
  """Creates low and high padding specification for the given padding (which is symmetric) and ceil mode.

  Additional high padding could be required when ceil mode is set.
  """
  ceil_mode_padding = []
  for i in range(len(padding)):
    left_padding = padding[i]
    right_padding = left_padding
    input_size = input_shape[2 + i]
    output_size_rem = (
        input_size + 2 * left_padding - (kernel_size[i] - 1) * dilation[i] - 1
    ) % stride[i]
    if ceil_mode and output_size_rem != 0:
      extra_padding = stride[i] - output_size_rem
      new_output_size = (
          input_size
          + left_padding
          + right_padding
          + extra_padding
          - (kernel_size[i] - 1) * dilation[i]
          - 1
          + stride[i]
          - 1
      ) // stride[i] + 1
      size_to_compare = input_size + left_padding
      if (new_output_size - 1) * stride[i] < size_to_compare:
        right_padding += extra_padding
    ceil_mode_padding.append((left_padding, right_padding))
  return ceil_mode_padding


@op(torch.ops.aten.min)
def _aten_min(x, dim=None, keepdim=False):
  if dim is not None:
    return (
        _with_reduction_scalar(jnp.min, x, dim, keepdim),
        _with_reduction_scalar(jnp.argmin, x, dim, keepdim).astype(jnp.int64),
    )
  else:
    return _with_reduction_scalar(jnp.min, x, dim, keepdim)


@op(torch.ops.aten.amin)
def _aten_amin(x, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.amin, x, dim, keepdim)


@op(torch.ops.aten.argmin)
def _aten_argmin(self, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.argmin, self, dim, keepdim)


@op(torch.ops.aten.sin)
@promote_int_input
def _aten_sin(x):
  return jnp.sin(x)


@op(torch.ops.aten.var.correction)
@op(torch.ops.prims.var)
def _aten_var(x, dim=None, *, correction=1, keepdim=False, out=None):
  return jnp.var(x, axis=dim, ddof=correction, keepdims=keepdim)


@op(torch.ops.prims.broadcast_in_dim)
def _prims_broadcast_in_dim(t, shape, broadcast_dimensions):
  return jax.lax.broadcast_in_dim(
      t, shape, broadcast_dimensions=broadcast_dimensions
  )


@op(torch.ops.aten.linalg_vector_norm)
def _aten_linalg_vector_norm(self, ord=2, dim=None, keepdim=False, dtype=None):
  """Calculates the vector norm along specified dimensions.

  Args:
      self: The input tensor.
      ord: The order of the norm. Can be a float or 'inf', '-inf', 'fro'.
        Default is 2 (Euclidean norm).
      dim: Dimensions along which to calculate the norm. If None, the norm is
        calculated over all dimensions.
      keepdim: Whether to keep the reduced dimensions.
      dtype: Optional data type for the output.

  Returns:
      The tensor containing the calculated vector norms.
  """
  if ord not in {2, float('inf'), float('-inf'), 'fro'} and (
      not isinstance(ord, (int, float))
  ):
    raise ValueError(
        f'Unsupported ord value: {ord}. Supported values are 2, inf, -inf, and'
        " 'fro'."
    )
  if ord == 0:
    if self.shape == ():
      result = jnp.astype(jnp.array(float(self != 0)), self.dtype)
    else:
      result = _with_reduction_scalar(
          jnp.sum, jnp.where(self != 0, 1, 0), dim, keepdim
      )
  elif ord == 2:
    result = jnp.sqrt(
        _with_reduction_scalar(jnp.sum, jnp.abs(self) ** 2, dim, keepdim)
    )
  elif ord == float('inf'):
    result = _with_reduction_scalar(jnp.max, jnp.abs(self), dim, keepdim)
  elif ord == float('-inf'):
    result = _with_reduction_scalar(jnp.min, jnp.abs(self), dim, keepdim)
  elif ord == 'fro':
    result = jnp.sqrt(
        _with_reduction_scalar(jnp.sum, jnp.abs(self) ** 2, dim, keepdim)
    )
  else:
    result = _with_reduction_scalar(
        jnp.sum, jnp.abs(self) ** ord, dim, keepdim
    ) ** (1.0 / ord)
  if dtype is not None:
    result = jnp.astype(result, self.dtype)
  new_dtype = t2j_dtype(torch.get_default_dtype())
  if result.dtype == jax.numpy.int64:
    result = result.astype(new_dtype)
  return result


@op(torch.ops.aten.reflection_pad1d)
def _aten_reflection_pad1d(input, padding):
  rank = len(input.shape)
  pad_size = [(0, 0)] * rank
  pad_size[-1] = padding
  return jnp.pad(input, pad_size, mode='reflect')


@op(torch.ops.aten.alias)
def _aten_alias(self, *args):
  return self


@op(torch.ops.aten.sinh)
@promote_int_input
def _aten_sinh(self):
  return jnp.sinh(self)


@op(torch.ops.aten.native_layer_norm_backward)
def _aten_native_layer_norm_backward(
    grad_out, input, normalized_shape, weight, bias, eps=1e-05
):
  """Implements the backward pass of layer normalization in Jax as defined by `aten::native_layer_norm_backward`.

  Args:
    grad_out: The gradient of the output tensor.
    input: The input tensor.
    normalized_shape: A list of integer dimensions to be normalized over.
    weight: Optional weight tensor for the affine transformation.
    bias: Optional bias tensor for the affine transformation.
    eps: A small epsilon value for numerical stability.

  Returns:
    A tuple of (grad_input, grad_weight, grad_bias).
  """
  return jax.lax.native_layer_norm_backward(
      grad_out, input, normalized_shape, weight, bias, eps
  )


@op(torch.ops.aten.atanh)
@promote_int_input
def _aten_atanh(self):
  res = jnp.arctanh(self)
  return res


@op(torch.ops.aten.bitwise_not)
def _aten_bitwise_not(self):
  return ~self


@op(torch.ops.aten.sum)
def _aten_sum(self, dim=None, keepdim=False, dtype=None):
  if not dim:
    dim = None
  return _with_reduction_scalar(jnp.sum, self, dim, keepdim)


@op(torch.ops.aten.sqrt)
@promote_int_input
def _aten_sqrt(self):
  return jnp.sqrt(self)


@op(torch.ops.aten.tan)
@promote_int_input
def _aten_tanh(self):
  res = jnp.tan(self)
  return res


@op(torch.ops.aten.tanh)
@promote_int_input
def _aten_tanh(self):
  res = jnp.tanh(self)
  return res


@op(torch.ops.aten.ceil)
def _aten_ceil(self):
  return jnp.ceil(self).astype(self.dtype)


@op(torch.ops.aten.asin)
@promote_int_input
def _aten_asin(self):
  res = jnp.arcsin(self)
  return res


@op(torch.ops.aten.minimum)
def _aten_minimum(self, other):
  return jnp.minimum(self, other)


def _scatter_index(dim, index):
  """Returns a tuple of indexes;

  The first is to select in input (to modify),
  the second is to select from the values.
  """
  index_shape = list(index.shape)
  input_indexes = []
  source_indexes = []
  if dim < 0:
    dim += len(index_shape)
  for i in range(len(index_shape)):
    source_indexes.append(slice(0, index_shape[i]))
    if i == dim:
      input_indexes.append(index)
    else:
      target_shape = [1] * len(index_shape)
      target_shape[i] = index_shape[i]
      input_indexes.append(
          jnp.broadcast_to(
              jnp.arange(index_shape[i]).reshape(target_shape), index_shape
          )
      )
  return (tuple(input_indexes), tuple(source_indexes))


@op(torch.ops.aten.scatter_add)
def _aten_scatter_add(input, dim, index, src):
  """JAX implementation of scatter, mimicking torch.scatter behavior"""
  input_indexes, source_indexes = _scatter_index(dim, index)
  return input.at[input_indexes].add(src[source_indexes])


@op(torch.ops.aten.sign)
def _aten_sign(x):
  return jnp.sign(x)


@op(torch.ops.aten.sigmoid)
@promote_int_input
def _aten_sigmoid(x):
  return jax.nn.sigmoid(x)


@op(torch.ops.aten.asinh)
@promote_int_input
def _aten_asinh(self):
  res = jnp.arcsinh(self)
  return res


@op(torch.ops.aten.atan)
@promote_int_input
def _aten_atan(self):
  res = jnp.arctan(self)
  return res


@op(torch.ops.aten.scatter_reduce)
@op(torch.ops.aten.scatter)
def _aten_scatter_reduce(
    input, dim, index, src, reduce=None, *, include_self=True
):
  if not isinstance(src, jnp.ndarray):
    src = jnp.array(src, dtype=input.dtype)
  input_indexes, source_indexes = _scatter_index(dim, index)
  if not include_self:
    if reduce in ['sum', 'mean']:
      base_input = jnp.zeros_like(src)
    elif reduce == 'prod':
      base_input = jnp.ones_like(src)
    elif reduce == 'amax':
      base_input = jnp.full_like(src, -jnp.inf)
    else:
      base_input = jnp.full_like(src, jnp.inf)
    input = input.at[input_indexes].set(base_input[source_indexes])
  if reduce == 'sum' or reduce == 'add':
    return input.at[input_indexes].add(src[source_indexes])
  elif reduce == 'prod' or reduce == 'multiply':
    return input.at[input_indexes].multiply(src[source_indexes])
  elif reduce == 'mean':
    if include_self:
      count = jnp.ones_like(input)
    else:
      count = jnp.zeros_like(input)
    count = count.at[input_indexes].add(jnp.ones_like(src)[source_indexes])
    count = jnp.clip(count, min=1)
    mean = input.at[input_indexes].add(src[source_indexes])
    if _is_int(input):
      return mean // count
    return mean / count
  elif reduce == 'amax':
    return input.at[input_indexes].max(src[source_indexes])
  elif reduce == 'amin':
    return input.at[input_indexes].min(src[source_indexes])
  else:
    return input.at[input_indexes].set(src[source_indexes])


@op(torch.ops.aten.acos)
@promote_int_input
def _aten_acos(self):
  return jnp.arccos(self)


@op(torch.ops.aten.gt)
def _aten_gt(self, other):
  return self > other


def pool(inputs, init, reduce_fn, window_shape, strides, padding):
  """Helper function to define pooling functions.

  Pooling functions are implemented using the ReduceWindow XLA op.
  NOTE: Be aware that pooling is not generally differentiable.
  That means providing a reduce_fn that is differentiable does not imply that
  pool is differentiable.

  Args:
    inputs: input data with dimensions (batch, window dims..., features).
    init: the initial value for the reduction
    reduce_fn: a reduce function of the form ``(T, T) -> T``.
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of ``n`` integers, representing the inter-window strides
      (default: ``(1, ..., 1)``).
    padding: either the string ``'SAME'``, the string ``'VALID'``, or a sequence
      of ``n`` ``(low, high)`` integer pairs that give the padding to apply
      before and after each spatial dimension.

  Returns:
    The output of the reduction for each window slice.
  """
  num_batch_dims = inputs.ndim - (len(window_shape) + 1)
  strides = strides or (1,) * len(window_shape)
  assert len(window_shape) == len(
      strides
  ), f'len({window_shape}) must equal len({strides})'
  strides = (1,) * (1 + num_batch_dims) + strides
  dims = (1,) * (1 + num_batch_dims) + window_shape
  is_single_input = False
  if num_batch_dims == 0:
    inputs = inputs[None]
    strides = (1,) + strides
    dims = (1,) + dims
    is_single_input = True
  assert inputs.ndim == len(dims), f'len({inputs.shape}) != len({dims})'
  if not isinstance(padding, str):
    padding = tuple(map(tuple, padding))
    assert len(padding) == len(window_shape), (
        f'padding {padding} must specify pads for same number of dims as'
        f' window_shape {window_shape}'
    )
    assert all(
        [len(x) == 2 for x in padding]
    ), f'each entry in padding {padding} must be length 2'
    padding = ((0, 0), (0, 0)) + padding
  y = jax.lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
  if is_single_input:
    y = jnp.squeeze(y, axis=0)
  return y


@op(torch.ops.aten._adaptive_avg_pool2d)
@op(torch.ops.aten._adaptive_avg_pool3d)
def adaptive_avg_pool2or3d(
    input: jnp.ndarray, output_size: Tuple[int, int]
) -> jnp.ndarray:
  """Applies a 2/3D adaptive average pooling over an input signal composed of several input planes.

  See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

  Args:
      input: input tensor
      output_size: the target output size (single integer or double-integer
        tuple)

  Context:
    https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L2401
  """
  shape = input.shape
  ndim = len(shape)
  out_dim = len(output_size)
  num_spatial_dim = ndim - out_dim
  assert ndim in (out_dim + 1, out_dim + 2), (
      f'adaptive_avg_pool{num_spatial_dim}d(): Expected {num_spatial_dim + 1}D'
      f' or {num_spatial_dim + 2}D tensor, but got {ndim}'
  )
  for d in input.shape[-2:]:
    assert d != 0, (
        'adaptive_avg_pool{num_spactial_dim}d(): Expected input to have'
        ' non-zero size for non-batch dimensions, but input has shape'
        f' {tuple(shape)}.'
    )
  if all((s % o == 0 for o, s in zip(output_size, shape[-out_dim:]))):
    stride = tuple((i // o for i, o in zip(shape[-out_dim:], output_size)))
    kernel = tuple((
        i - (o - 1) * s
        for i, o, s in zip(shape[-out_dim:], output_size, stride)
    ))
    return _aten_avg_pool(input, kernel, strides=stride)

  def start_index(a, b, c):
    return a * c // b

  def end_index(a, b, c):
    return ((a + 1) * c + b - 1) // b

  def compute_idx(in_size, out_size):
    orange = jnp.arange(out_size, dtype=jnp.int64)
    i0 = start_index(orange, out_size, in_size)
    maxlength = in_size // out_size + 1
    in_size_mod = in_size % out_size
    adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
    if adaptive:
      maxlength += 1
    elif in_size_mod == 0:
      maxlength -= 1
    range_max = jnp.arange(maxlength, dtype=jnp.int64)
    idx = i0[:, None] + range_max
    if adaptive:
      idx = jnp.minimum(idx, in_size - 1)
      i1 = end_index(orange, out_size, in_size)
      length = i1 - i0
    else:
      length = maxlength
    return (idx, length, range_max, adaptive)

  idx, length, range_max, adaptive = [[None] * out_dim for _ in range(4)]
  for i, (s, o) in enumerate(zip(shape[-out_dim:], output_size)):
    idx[i], length[i], range_max[i], adaptive[i] = compute_idx(s, o)

  def _unsqueeze_to_dim(x, dim):
    ndim = len(x.shape)
    return jax.lax.expand_dims(x, tuple(range(ndim, dim)))

  if out_dim == 2:
    vals = input[..., _unsqueeze_to_dim(idx[0], 4), idx[1]]
    reduce_axis = (-3, -1)
  else:
    assert out_dim == 3
    vals = input[
        ..., _unsqueeze_to_dim(idx[0], 6), _unsqueeze_to_dim(idx[1], 4), idx[2]
    ]
    reduce_axis = (-5, -3, -1)
  if not any(adaptive):
    return jnp.mean(vals, axis=reduce_axis)

  def maybe_mask(vals, length, range_max, adaptive, dim):
    if isinstance(length, int):
      return (vals, length)
    else:
      assert dim < 0
      mask = range_max >= length[:, None]
      if dim == -2:
        mask = _unsqueeze_to_dim(mask, 4)
      elif dim == -3:
        mask = _unsqueeze_to_dim(mask, 6)
      vals = jnp.where(mask, 0.0, vals)
      length = _unsqueeze_to_dim(length, -dim)
      return (vals, length)

  for i in range(len(length)):
    vals, length[i] = maybe_mask(
        vals, length[i], range_max[i], adaptive=adaptive[i], dim=i - out_dim
    )
  ret = jnp.sum(vals, axis=reduce_axis)
  return ret / math.prod(length)


@op(torch.ops.aten.avg_pool1d)
@op(torch.ops.aten.avg_pool2d)
@op(torch.ops.aten.avg_pool3d)
def _aten_avg_pool(
    inputs,
    kernel_size,
    strides=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
  num_batch_dims = len(inputs.shape) - len(kernel_size) - 1
  kernel_size = tuple(kernel_size)
  strides = tuple(strides) if strides else kernel_size
  if isinstance(padding, list) and len(padding) == 1:
    padding = padding[0]
  if isinstance(padding, int):
    padding = [padding for _ in range(len(kernel_size))]
  input_shape = inputs.shape
  if num_batch_dims == 0:
    input_shape = [1, *input_shape]
  padding = _ceil_mode_padding(
      padding,
      input_shape,
      kernel_size,
      strides,
      [1] * len(kernel_size),
      ceil_mode,
  )
  y = pool(inputs, 0.0, jax.lax.add, kernel_size, strides, padding)
  if divisor_override is not None:
    y = y / jnp.array(divisor_override, y.dtype)
  elif count_include_pad:
    div_shape = list(y.shape)
    div_by = jnp.ones(div_shape, y.dtype) * np.prod(kernel_size)
    unequal_paddings = map(lambda pad: pad[0] != pad[1], padding)
    unequal_padding_indices = np.where(list(unequal_paddings))[0]
    if len(unequal_padding_indices) > 0:
      offset = len(div_shape) - len(padding)
      skip_indices = list(map(lambda x: x + offset, unequal_padding_indices))
      indices = _generate_indices(div_shape, skip_dim_indices=skip_indices)
      new_kernel_size = list(kernel_size)
      for j in unequal_padding_indices:
        new_kernel_size[j] = kernel_size[j] - padding[j][1] + padding[j][0]
      for idx in indices:
        for j in unequal_padding_indices:
          idx[j + offset] = -1
        div_by = div_by.at[tuple(idx)].set(np.prod(new_kernel_size))
    y = y / div_by
  else:
    div_shape = list(inputs.shape)
    div_shape[num_batch_dims] = 1
    div_shape = tuple(div_shape)
    if len(div_shape) - 2 == len(kernel_size):
      div_shape = (1,) + div_shape[1:]
    y = y / pool(
        jnp.ones(div_shape, y.dtype),
        jnp.array(0.0, y.dtype),
        jax.lax.add,
        kernel_size,
        strides,
        padding,
    )
  return y.astype(inputs.dtype)


def _generate_indices(dims, skip_dim_indices=[]):
  res = []

  def _helper(curr_dim_idx, sofar):
    if curr_dim_idx in skip_dim_indices:
      _helper(curr_dim_idx + 1, sofar[:])
      return
    if curr_dim_idx >= len(dims):
      res.append(sofar)
      return
    for i in range(dims[curr_dim_idx]):
      sofar[curr_dim_idx] = i
      _helper(curr_dim_idx + 1, sofar[:])

  _helper(0, [0 for _ in dims])
  return res


@op(torch.ops.aten.reciprocal)
def _aten_reciprocal(a):
  if _is_int(a):
    return (1 / a).astype(jnp.dtype('float32'))
  return 1 / a


@op(torch.ops.aten.select_scatter)
def _aten_select_scatter(input, src, dim, index):
  input_indexes = []
  if dim < 0:
    dim += len(input.shape)
  for x in range(len(input.shape)):
    if x == dim:
      input_indexes.append(index)
    else:
      input_indexes.append(slice(None, None, None))
  return input.at[tuple(input_indexes)].set(src)


@op(torch.ops.aten.scatter.src)
def _aten_scatter_src(input, dim, index, src, reduce=None):
  input_index, source_indexes = _scatter_index(dim, index)
  return input.at[input_index].set(src[source_indexes])


@op(torch.ops.aten.scatter.value)
def _aten_scatter(input, dim, index, src, reduce=None):
  input_index, source_indexes = _scatter_index(dim, index)
  return input.at[input_index].set(src)


@op(torch.ops.aten.acosh)
@promote_int_input
def _aten_acosh(self):
  return jnp.arccosh(self)


@op(torch.ops.aten.round)
def _aten_round(input, decimals=0):
  return jnp.round(input, decimals)


@op(torch.ops.aten.max)
def _aten_max(self, dim=None, keepdim=False):
  if dim is not None:
    return (
        _with_reduction_scalar(jnp.max, self, dim, keepdim),
        _with_reduction_scalar(jnp.argmax, self, dim, keepdim).astype(
            jnp.int64
        ),
    )
  else:
    return _with_reduction_scalar(jnp.max, self, dim, keepdim)


@op(torch.ops.aten.maximum)
def _aten_maximum(self, other):
  return jnp.maximum(self, other)


@op(torch.ops.aten.amax)
def _aten_amax(self, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.amax, self, dim, keepdim)


def _with_reduction_scalar(jax_func, self, dim, keepdim):
  expanded = False
  if self.ndim == 0:
    expanded = True
    self = jnp.expand_dims(self, 0)
  res = jax_func(self, axis=dim, keepdims=keepdim)
  if expanded:
    res = res.squeeze()
  return res


@op(torch.ops.aten.any)
def _aten_any(self, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.any, self, dim, keepdim)


@op(torch.ops.aten.arange.start_step)
@op(torch.ops.aten.arange.start)
@op(torch.ops.aten.arange.default)
@convert_dtype(use_default_dtype=False)
def _aten_arange(
    start,
    end=None,
    step=None,
    *,
    dtype=None,
    layout=None,
    requires_grad=False,
    device=None,
    pin_memory=False,
):
  return jnp.arange(
      maybe_convert_constant_dtype(start, dtype),
      maybe_convert_constant_dtype(end, dtype),
      maybe_convert_constant_dtype(step, dtype),
      dtype=dtype,
  )


@op(torch.ops.aten.argmax)
def _aten_argmax(self, dim=None, keepdim=False):
  return _with_reduction_scalar(jnp.argmax, self, dim, keepdim)


def _strided_index(sizes, strides, storage_offset=None):
  ind = jnp.zeros(sizes, dtype=jnp.int32)
  for i, (size, stride) in enumerate(zip(sizes, strides)):
    result_shape = (1,) * i + (size,) + (1,) * (len(sizes) - i - 1)
    indexes = (jnp.arange(size) * stride).reshape(result_shape)
    ind += indexes
  if storage_offset is not None:
    ind += storage_offset
  return ind


def _is_contiguous_view(sizes, strides) -> bool:
  """Whether (sizes, strides) walks row-major storage with no gaps or repeats.

  Strides of size-1 dims are ignored: they never contribute to an offset, and
  eager torch leaves them arbitrary (e.g. `mean` of a channels-last tensor
  yields sizes [1, C, 1, 1] with strides [C, 1, C, C]).
  """
  expected = 1
  for size, stride in reversed(list(zip(sizes, strides))):
    if size != 1 and stride != expected:
      return False
    expected *= size
  return True


@op(torch.ops.aten.as_strided)
@op(torch.ops.aten.as_strided_copy)
def _aten_as_strided(x, sizes, strides, storage_offset=None):
  offset = storage_offset or 0
  static = all(isinstance(v, int) for v in (*sizes, *strides, offset))
  if static and _is_contiguous_view(sizes, strides):
    # A contiguous window of storage is a slice + reshape; avoid the generic
    # index gather, which GPU delegates generally cannot run.
    numel = math.prod(sizes)
    if offset + numel <= x.size:
      flattened = jnp.ravel(x)
      if offset != 0 or numel != x.size:
        flattened = jax.lax.slice(flattened, (offset,), (offset + numel,))
      return jnp.reshape(flattened, sizes)

  ind = _strided_index(sizes, strides, storage_offset)
  flattened = jnp.ravel(x)
  return flattened[ind]


@op(torch.ops.aten.atan2)
@promote_int_input
def _aten_atan2(input, other):
  return jnp.arctan2(input, other)


@op(torch.ops.aten.bitwise_and)
@op(torch.ops.aten.__and__)
def _aten_bitwise_and(self, other):
  return self & other


@op(torch.ops.aten.bitwise_or)
def _aten_bitwise_or(self, other):
  return self | other


@op(torch.ops.aten.bitwise_xor)
def _aten_bitwise_xor(self, other):
  return self ^ other


@op(torch.ops.aten.clamp.default)
@op(torch.ops.aten.clamp.Tensor)
def _aten_clamp(self, min=None, max=None):
  return jnp.clip(self, min, max)


@op(torch.ops.aten.constant_pad_nd)
def _aten_constant_pad_nd(input, padding, value=0):
  m = len(padding)
  rev_padding = [(padding[i - 1], padding[i], 0) for i in range(m - 1, 0, -2)]
  pad_dim = tuple([(0, 0, 0)] * (len(input.shape) - m // 2) + rev_padding)
  value_casted = jax.numpy.array(value, dtype=input.dtype)
  return jax.lax.pad(input, padding_value=value_casted, padding_config=pad_dim)


@op(torch.ops.aten.lift_fresh_copy)
def _aten_lift_fresh_copy(x):
  return jnp.copy(x)


@op(torch.ops.aten._cdist_forward)
def _aten_cdist_forward(x1, x2, p, compute_mode=''):
  x1 = jnp.expand_dims(x1, len(x1.shape) - 1)
  x2 = jnp.expand_dims(x2, len(x2.shape) - 2)
  return jnp.linalg.norm(x1 - x2, ord=p, axis=-1)


@op(torch.ops.aten._pdist_forward)
def _aten__pdist_forward(x, p=2):
  pairwise_dists = _aten_cdist_forward(x, x, p)
  condensed_dists = pairwise_dists[
      jnp.triu_indices(pairwise_dists.shape[0], k=1)
  ]
  return condensed_dists


@op(torch.ops.aten.cos)
@promote_int_input
def _aten_cos(input):
  return jnp.cos(input)


@op(torch.ops.aten.cosh)
@promote_int_input
def _aten_cosh(input):
  return jnp.cosh(input)


@op(torch.ops.aten.diagonal)
@op(torch.ops.aten.diagonal_copy)
def _aten_diagonal(input, offset=0, dim1=0, dim2=1):
  return jnp.diagonal(input, offset, dim1, dim2)


@op(torch.ops.aten.eq)
def _aten_eq(input1, input2):
  return input1 == input2


@op(torch.ops.aten.erf)
@promote_int_input
def _aten_erf(x):
  return jax.lax.erf(x)


@op(torch.ops.aten.exp)
def _aten_exp(input):
  res = jnp.exp(input)
  new_dtype = t2j_dtype(torch.get_default_dtype())
  if input.dtype == jax.numpy.int64:
    res = res.astype(new_dtype)
  return res


@op(torch.ops.aten.expm1)
def _aten_expm1(input):
  res = jnp.expm1(input)
  new_dtype = t2j_dtype(torch.get_default_dtype())
  if input.dtype == jax.numpy.int64:
    res = res.astype(new_dtype)
  return res


@op(torch.ops.aten.fill)
@op(torch.ops.aten.full_like)
def _aten_fill(
    x, value, dtype=None, pin_memory=None, memory_format=None, device=None
):
  if dtype is None:
    dtype = x.dtype
  else:
    dtype = t2j_dtype(dtype)
  return jnp.full(x.shape, value, dtype)


@op(torch.ops.aten.flip)
def _aten_flip(input, dims):
  if dims is not None:
    return jnp.flip(input, tuple(dims))
  else:
    return jnp.flip(input)


@op(torch.ops.aten.fmod)
def _aten_fmod(input, other):
  return input - other * _aten_div(input, other, 'trunc')


@op(torch.ops.aten.gather)
def _aten_gather(input, dim, index):
  if input.ndim == 0:
    return jnp.broadcast_to(input, index.shape)
  if not all(index.shape):
    return jnp.zeros(index.shape, dtype=input.dtype)
  if dim < 0:
    dim += input.ndim
  input_indexes, source_indexes = _scatter_index(dim, index)
  return input[input_indexes]


@op(torch.ops.aten.ge)
def _aten_ge(self, other):
  return self >= other


@op(torch.ops.aten.glu)
def _aten_glu(x, dim=-1):
  return jax.nn.glu(x, dim)


@op(torch.ops.aten.hardtanh)
def _aten_hardtanh(input, min_val=-1, max_val=1, inplace=False):
  if (
      input.dtype == np.int64
      and isinstance(max_val, float)
      and isinstance(min_val, float)
  ):
    min_val = int(min_val)
    max_val = int(max_val)
  return jnp.clip(input, min_val, max_val)


@op(torch.ops.aten.isinf)
def _aten_isinf(input):
  return jnp.isinf(input)


@op(torch.ops.aten.isnan)
def _aten_isnan(input):
  return jnp.isnan(input)


@op(torch.ops.aten.le)
def _aten_le(self, other):
  return self <= other


@op(torch.ops.aten.leaky_relu)
def _aten_leaky_relu(x, negative_slope=0.01):
  return jax.nn.leaky_relu(x, negative_slope)


@op(torch.ops.aten.log)
@promote_int_input
def _aten_log(x):
  return jnp.log(x)


@op(torch.ops.aten.log10)
@promote_int_input
def _aten_log10(x):
  return jnp.log10(x)


@op(torch.ops.aten.log1p)
@promote_int_input
def _aten_log1p(x):
  return jnp.log1p(x)


@op(torch.ops.aten.log2)
@promote_int_input
def _aten_log2(x):
  return jnp.log2(x)


@op(torch.ops.aten.logical_and)
@op(torch.ops.aten.__and__)
def _aten_logical_and(self, other):
  return jnp.logical_and(self, other)


@op(torch.ops.aten.logical_or)
@op(torch.ops.aten.__or__)
def _aten_logical_or(self, other):
  return jnp.logical_or(self, other)


@op(torch.ops.aten.logical_not)
def _aten_logical_not(self):
  return jnp.logical_not(self)


@op(torch.ops.aten._log_softmax)
def _aten_log_softmax(self, axis=-1, half_to_float=False):
  if self.shape == ():
    return jnp.astype(0.0, self.dtype)
  return jax.nn.log_softmax(self, axis)


@op(torch.ops.aten.logical_xor)
@op(torch.ops.aten.__xor__)
def _aten_logical_xor(self, other):
  return jnp.logical_xor(self, other)


@op(torch.ops.aten.neg)
def _aten_neg(x):
  return -1 * x


@op(torch.ops.aten.nonzero)
def _aten_nonzero(x, as_tuple=False):
  if jnp.ndim(x) == 0 and (as_tuple or x.item() == 0):
    return torch.empty(0, 0, dtype=torch.int64)
  if jnp.ndim(x) == 0:
    res = torch.empty(1, 0, dtype=torch.int64)
    return jnp.array(res.numpy())
  index_tuple = jnp.nonzero(x)
  index_tuple = [jnp.expand_dims(p, -1) for p in index_tuple]
  return jnp.concatenate(index_tuple, axis=-1)


@op(torch.ops.aten.prod)
def _aten_prod(input, dim=None, keepdim=False, *, dtype=None):
  if dtype:
    input = input.astype(t2j_dtype(dtype))
  return _with_reduction_scalar(jnp.prod, input, dim, keepdim)


@op(torch.ops.aten.remainder)
def _aten_remainder(inputs, other):
  return inputs % other


@op(torch.ops.aten.repeat)
def _aten_repeat(x, reps):
  return jnp.tile(x, reps)


@op(torch.ops.aten.roll)
def _aten_roll(input, shifts, dims=None):
  return jnp.roll(input, shifts, dims)


@op(torch.ops.aten.sort)
def _aten_sort(a, dim=-1, descending=False, stable=False):
  if a.shape == ():
    return (a, jnp.astype(0, 'int64'))
  return (
      jnp.sort(a, axis=dim, stable=stable, descending=descending),
      jnp.argsort(a, axis=dim, stable=stable, descending=descending),
  )


@op(torch.ops.aten.unbind_copy)
def _aten_unbind(a, dim=0):
  return [
      jax.lax.index_in_dim(a, i, dim, keepdims=False)
      for i in range(a.shape[dim])
  ]


@op(torch.ops.aten.where)
@op(torch.ops.aten.where.self)
@op(torch.ops.aten.where.ScalarSelf)
@op(torch.ops.aten.where.ScalarOther)
@op(torch.ops.aten.where.Scalar)
def _aten_where(condition, x=None, y=None):
  return jnp.where(condition, x, y)


@op(torch.ops.aten.to.dtype)
def _aten_to_dtype(
    a, dtype, non_blocking=False, copy=False, memory_format=None
):
  if not dtype:
    raise ValueError('dtype should be specified.')
  jaxdtype = t2j_dtype(dtype)
  return a.astype(jaxdtype)


@op(torch.ops.aten.copy)
def _aten_copy(self, src):
  return jnp.broadcast_to(src, self.shape).astype(self.dtype)


@op(torch.ops.aten.var_mean.correction)
def _aten_var_mean_correction(tensor, dim=None, correction=1, keepdim=False):
  if correction is None:
    correction = 1
  mean = jnp.mean(tensor, axis=dim, keepdims=keepdim)
  var = jnp.var(tensor, axis=dim, ddof=correction, keepdims=keepdim)
  return (var, mean)


@op(torch.ops.aten.scalar_tensor)
@convert_dtype()
def _aten_scalar_tensor(
    s, dtype=None, layout=None, device=None, pin_memory=None
):
  return jnp.array(s, dtype=dtype)


@op(torch.ops.aten.to.device)
def _aten_to_device(x, device, dtype):
  return x


@op(torch.ops.aten.max_pool2d_with_indices_backward)
def max_pool2d_with_indices_backward_custom(
    grad_output,
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    indices,
):
  """Approximates the gradient calculation of PyTorch's max_pool2d_with_indices_backward.

  Args:
      grad_output: The gradient tensor from the preceding layer.
      self: The input tensor on which the original max pooling was performed.
      kernel_size: The size of the pooling window.
      stride: The stride of the pooling window.
      padding: The padding applied during max pooling.
      dilation: The dilation factor for the pooling operation.
      ceil_mode: Whether to use ceil or floor when calculating output shapes.
      indices: The indices of the maximum values, as produced by
        max_pool2d_with_indices.

  Returns:
      The calculated gradient with respect to the input (grad_input).
  """
  kH, kW = kernel_size
  dH, dW = stride
  padH, padW = padding
  dilH, dilW = dilation
  out_shape = jnp.array(self.shape)
  grad_input = jnp.zeros_like(self)
  for i, idx in enumerate(indices.flatten()):
    out_y, out_x = (i // grad_output.shape[3], i % grad_output.shape[3])
    in_y = out_y * dH - padH + out_y * (dilH - 1)
    in_x = out_x * dW - padW + out_x * (dilW - 1)
    for y in range(in_y, in_y + kH):
      for x in range(in_x, in_x + kW):
        if 0 <= y < grad_input.shape[2] and 0 <= x < grad_input.shape[3]:
          grad_input = grad_input.at[y, x].add(grad_output.flatten()[i])
  return grad_input


@op(torch.ops.aten._local_scalar_dense)
def _aten_local_scalar_dense(x):
  return x.item()


@op(torch.ops.aten.tensor_split.sections)
def _aten_tensor_split(ary, indices_or_sections, axis=0):
  return jnp.array_split(ary, indices_or_sections, axis)


@op(torch.ops.aten.outer)
def _aten_outer(a, b):
  return jnp.outer(a, b)


@op(torch.ops.aten.allclose)
def _aten_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
  return jnp.allclose(input, other, rtol, atol, equal_nan)


@op(torch.ops.aten.native_batch_norm)
def _aten_native_batch_norm(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
  if running_mean is None:
    running_mean = jnp.zeros(input.shape[1], dtype=input.dtype)
  if running_var is None:
    running_var = jnp.ones(input.shape[1], dtype=input.dtype)
  if training:
    return _aten__native_batch_norm_legit(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
  else:
    return _aten__native_batch_norm_legit_no_training(
        input, weight, bias, running_mean, running_var, momentum, eps
    )
