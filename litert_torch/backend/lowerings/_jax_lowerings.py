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
import functools
import logging

import jax
import jax.numpy as jnp
from litert_torch.backend import jax_bridge
from litert_torch.backend.lowerings import context
from litert_torch.backend.lowerings import registry
from ai_edge_litert.mlir import ir
import numpy as np
import torch
import torch_xla2.ops.jaten  # Import to load torch_xla2 ops
import torch_xla2.ops.ops_registry  # Import to load torch_xla2 ops

LoweringContext = context.LoweringContext


@functools.cache
def _log_usage(op):
  logging.info("Use JAX lowering: %s", str(op))


def lower_by_jax(op, ir_input_names=None):
  def inner(lowering):
    bridged = jax_bridge.wrap(lowering, ir_input_names)

    @registry.lower(op)
    def _jax_lowering(lctx, *args, **kwargs):
      _log_usage(op)
      return bridged(lctx, *args, **kwargs)

    return lowering

  return inner


_TORCH_XLA2_IMPLS = {}

for op, torch_xla2_op in torch_xla2.ops.ops_registry.all_aten_ops.items():
  if not torch_xla2_op.is_jax_function:
    continue
  if isinstance(op, torch._ops.OpOverloadPacket):
    ops = [getattr(op, overload) for overload in op.overloads()] + [op]
  else:
    ops = [op]

  for op in ops:
    _TORCH_XLA2_IMPLS[op] = torch_xla2_op.func


def lower_by_torch_xla2(op):
  return lower_by_jax(op)(_TORCH_XLA2_IMPLS[op])


lower_by_torch_xla2(torch.ops.aten._adaptive_avg_pool2d)
lower_by_torch_xla2(torch.ops.aten._adaptive_avg_pool3d)
lower_by_torch_xla2(torch.ops.aten._cdist_forward)
lower_by_torch_xla2(torch.ops.aten._local_scalar_dense)
lower_by_torch_xla2(torch.ops.aten._local_scalar_dense)
lower_by_torch_xla2(torch.ops.aten._log_softmax)
lower_by_torch_xla2(torch.ops.aten._pdist_forward)
lower_by_torch_xla2(torch.ops.aten._softmax)
lower_by_torch_xla2(torch.ops.aten._unsafe_index)
lower_by_torch_xla2(torch.ops.aten._unsafe_view)
lower_by_torch_xla2(torch.ops.aten.acos)
lower_by_torch_xla2(torch.ops.aten.acosh)
lower_by_torch_xla2(torch.ops.aten.addbmm.default)
lower_by_torch_xla2(torch.ops.aten.addmm)
lower_by_torch_xla2(torch.ops.aten.addmv)
lower_by_torch_xla2(torch.ops.aten.alias)
lower_by_torch_xla2(torch.ops.aten.allclose)
lower_by_torch_xla2(torch.ops.aten.amax)
lower_by_torch_xla2(torch.ops.aten.amin)
lower_by_torch_xla2(torch.ops.aten.any)
lower_by_torch_xla2(torch.ops.aten.arange.default)
lower_by_torch_xla2(torch.ops.aten.arange.start)
lower_by_torch_xla2(torch.ops.aten.arange.start_step)
lower_by_torch_xla2(torch.ops.aten.argmax)
lower_by_torch_xla2(torch.ops.aten.argmin)
lower_by_torch_xla2(torch.ops.aten.as_strided)
lower_by_torch_xla2(torch.ops.aten.as_strided_copy)
lower_by_torch_xla2(torch.ops.aten.asin)
lower_by_torch_xla2(torch.ops.aten.asinh)
lower_by_torch_xla2(torch.ops.aten.atan)
lower_by_torch_xla2(torch.ops.aten.atan2)
lower_by_torch_xla2(torch.ops.aten.atanh)
lower_by_torch_xla2(torch.ops.aten.avg_pool2d)
lower_by_torch_xla2(torch.ops.aten.avg_pool3d)
lower_by_torch_xla2(torch.ops.aten.bitwise_and)
lower_by_torch_xla2(torch.ops.aten.bitwise_not)
lower_by_torch_xla2(torch.ops.aten.bitwise_or)
lower_by_torch_xla2(torch.ops.aten.bitwise_xor)
lower_by_torch_xla2(torch.ops.aten.bmm)
lower_by_torch_xla2(torch.ops.aten.ceil)
lower_by_torch_xla2(torch.ops.aten.clamp.Tensor)
lower_by_torch_xla2(torch.ops.aten.clamp.default)
lower_by_torch_xla2(torch.ops.aten.clone)
lower_by_torch_xla2(torch.ops.aten.clone.default)
lower_by_torch_xla2(torch.ops.aten.constant_pad_nd)
lower_by_torch_xla2(torch.ops.aten.cos)
lower_by_torch_xla2(torch.ops.aten.cosh)
lower_by_torch_xla2(torch.ops.aten.cumsum)
lower_by_torch_xla2(torch.ops.aten.detach)
lower_by_torch_xla2(torch.ops.aten.diagonal)
lower_by_torch_xla2(torch.ops.aten.dot)
lower_by_torch_xla2(torch.ops.aten.embedding)
lower_by_torch_xla2(torch.ops.aten.empty)
lower_by_torch_xla2(torch.ops.aten.eq)
lower_by_torch_xla2(torch.ops.aten.erf)
lower_by_torch_xla2(torch.ops.aten.exp)
lower_by_torch_xla2(torch.ops.aten.expand)
lower_by_torch_xla2(torch.ops.aten.expand_copy)
lower_by_torch_xla2(torch.ops.aten.expm1)
lower_by_torch_xla2(torch.ops.aten.fill)
lower_by_torch_xla2(torch.ops.aten.flip)
lower_by_torch_xla2(torch.ops.aten.fmod)
lower_by_torch_xla2(torch.ops.aten.full)
lower_by_torch_xla2(torch.ops.aten.full_like)
lower_by_torch_xla2(torch.ops.aten.gather)
lower_by_torch_xla2(torch.ops.aten.ge)
lower_by_torch_xla2(torch.ops.aten.gelu)
lower_by_torch_xla2(torch.ops.aten.glu)
lower_by_torch_xla2(torch.ops.aten.glu.default)
lower_by_torch_xla2(torch.ops.aten.gt)
lower_by_torch_xla2(torch.ops.aten.hardtanh)
lower_by_torch_xla2(torch.ops.aten.index)
lower_by_torch_xla2(torch.ops.aten.index.Tensor)
lower_by_torch_xla2(torch.ops.aten.index_copy)
lower_by_torch_xla2(torch.ops.aten.index_put)
lower_by_torch_xla2(torch.ops.aten.index_select)
lower_by_torch_xla2(torch.ops.aten.isinf)
lower_by_torch_xla2(torch.ops.aten.isnan)
lower_by_torch_xla2(torch.ops.aten.le)
lower_by_torch_xla2(torch.ops.aten.leaky_relu)
lower_by_torch_xla2(torch.ops.aten.lift_fresh_copy)
lower_by_torch_xla2(torch.ops.aten.linalg_vector_norm)
lower_by_torch_xla2(torch.ops.aten.log)
lower_by_torch_xla2(torch.ops.aten.log10)
lower_by_torch_xla2(torch.ops.aten.log1p)
lower_by_torch_xla2(torch.ops.aten.log2)
lower_by_torch_xla2(torch.ops.aten.logical_and)
lower_by_torch_xla2(torch.ops.aten.logical_not)
lower_by_torch_xla2(torch.ops.aten.logical_or)
lower_by_torch_xla2(torch.ops.aten.logical_xor)
lower_by_torch_xla2(torch.ops.aten.max)
lower_by_torch_xla2(torch.ops.aten.max_pool2d_with_indices_backward)
lower_by_torch_xla2(torch.ops.aten.max_pool2d_with_indices_backward)
lower_by_torch_xla2(torch.ops.aten.maximum)
lower_by_torch_xla2(torch.ops.aten.mean)
lower_by_torch_xla2(torch.ops.aten.min)
lower_by_torch_xla2(torch.ops.aten.minimum)
lower_by_torch_xla2(torch.ops.aten.mm)
lower_by_torch_xla2(torch.ops.aten.native_batch_norm)
lower_by_torch_xla2(torch.ops.aten.native_layer_norm_backward)
lower_by_torch_xla2(torch.ops.aten.ne)
lower_by_torch_xla2(torch.ops.aten.neg)
lower_by_torch_xla2(torch.ops.aten.nonzero)
lower_by_torch_xla2(torch.ops.aten.outer)
lower_by_torch_xla2(torch.ops.aten.permute)
lower_by_torch_xla2(torch.ops.aten.permute_copy)
lower_by_torch_xla2(torch.ops.aten.pow)
lower_by_torch_xla2(torch.ops.aten.prod)
lower_by_torch_xla2(torch.ops.aten.reciprocal)
lower_by_torch_xla2(torch.ops.aten.reflection_pad1d)
lower_by_torch_xla2(torch.ops.aten.relu)
lower_by_torch_xla2(torch.ops.aten.remainder)
lower_by_torch_xla2(torch.ops.aten.repeat)
lower_by_torch_xla2(torch.ops.aten.reshape)
lower_by_torch_xla2(torch.ops.aten.roll)
lower_by_torch_xla2(torch.ops.aten.round)
lower_by_torch_xla2(torch.ops.aten.rsqrt)
lower_by_torch_xla2(torch.ops.aten.scalar_tensor)
lower_by_torch_xla2(torch.ops.aten.scatter.src)
lower_by_torch_xla2(torch.ops.aten.scatter.value)
lower_by_torch_xla2(torch.ops.aten.scatter_add)
lower_by_torch_xla2(torch.ops.aten.scatter_reduce)
lower_by_torch_xla2(torch.ops.aten.select)
lower_by_torch_xla2(torch.ops.aten.select_copy)
lower_by_torch_xla2(torch.ops.aten.select_scatter)
lower_by_torch_xla2(torch.ops.aten.sigmoid)
lower_by_torch_xla2(torch.ops.aten.sign)
lower_by_torch_xla2(torch.ops.aten.silu)
lower_by_torch_xla2(torch.ops.aten.sin)
lower_by_torch_xla2(torch.ops.aten.sinh)
lower_by_torch_xla2(torch.ops.aten.slice)
lower_by_torch_xla2(torch.ops.aten.slice_copy)
lower_by_torch_xla2(torch.ops.aten.sort)
lower_by_torch_xla2(torch.ops.aten.split)
lower_by_torch_xla2(torch.ops.aten.split_copy)
lower_by_torch_xla2(torch.ops.aten.split_with_sizes)
lower_by_torch_xla2(torch.ops.aten.sqrt)
lower_by_torch_xla2(torch.ops.aten.squeeze)
lower_by_torch_xla2(torch.ops.aten.squeeze_copy)
lower_by_torch_xla2(torch.ops.aten.stack)
lower_by_torch_xla2(torch.ops.aten.sum)
lower_by_torch_xla2(torch.ops.aten.t)
lower_by_torch_xla2(torch.ops.aten.tan)
lower_by_torch_xla2(torch.ops.aten.tanh)
lower_by_torch_xla2(torch.ops.aten.tensor_split.sections)
lower_by_torch_xla2(torch.ops.aten.tensor_split.sections)
lower_by_torch_xla2(torch.ops.aten.to.device)
lower_by_torch_xla2(torch.ops.aten.to.device)
lower_by_torch_xla2(torch.ops.aten.to.dtype)
lower_by_torch_xla2(torch.ops.aten.transpose)
lower_by_torch_xla2(torch.ops.aten.transpose_copy)
lower_by_torch_xla2(torch.ops.aten.triu)
lower_by_torch_xla2(torch.ops.aten.true_divide)
lower_by_torch_xla2(torch.ops.aten.trunc)
lower_by_torch_xla2(torch.ops.aten.unbind_copy)
lower_by_torch_xla2(torch.ops.aten.unsqueeze)
lower_by_torch_xla2(torch.ops.aten.unsqueeze.default)
lower_by_torch_xla2(torch.ops.aten.unsqueeze_copy)
lower_by_torch_xla2(torch.ops.aten.var.correction)
lower_by_torch_xla2(torch.ops.aten.var_mean.correction)
lower_by_torch_xla2(torch.ops.aten.view)
lower_by_torch_xla2(torch.ops.aten.view_as_complex)
lower_by_torch_xla2(torch.ops.aten.view_as_real)
lower_by_torch_xla2(torch.ops.aten.view_copy)
lower_by_torch_xla2(torch.ops.aten.where.ScalarOther)
lower_by_torch_xla2(torch.ops.aten.where.ScalarSelf)
lower_by_torch_xla2(torch.ops.prims.broadcast_in_dim)
lower_by_torch_xla2(torch.ops.prims.var)


def _ceil_mode_padding(
    padding: list[int],
    input_shape: list[int],
    kernel_size: list[int],
    stride: list[int],
    dilation: list[int],
    ceil_mode: bool,
):
  """Creates low and high padding specification for ceil mode.

  This is for the given padding (which is symmetric). Additional high padding
  could be required when ceil mode is set.
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
      # Ensure that the last pooling starts inside the image.
      size_to_compare = input_size + left_padding

      if (new_output_size - 1) * stride[i] < size_to_compare:
        right_padding += extra_padding

    ceil_mode_padding.append((left_padding, right_padding))
  return ceil_mode_padding


def max_pool(
    inputs,
    kernel_size,
    strides=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    with_index=False,
):
  num_spatial_dims = len(kernel_size)
  num_batch_dims = inputs.ndim - num_spatial_dims - 1
  kernel_size_tup = tuple(kernel_size)
  # Default stride is kernel_size
  strides_tup = tuple(strides) if strides else kernel_size_tup
  if isinstance(padding, int):
    padding_list = [padding for _ in range(num_spatial_dims)]
  elif not padding:  # padding can be [], meaning all zeros.
    padding_list = [0 for _ in range(num_spatial_dims)]
  else:
    padding_list = padding

  if isinstance(dilation, int):
    dilation_tup = tuple(dilation for _ in range(num_spatial_dims))
  elif not dilation:
    dilation_tup = tuple(1 for _ in range(num_spatial_dims))
  elif isinstance(dilation, list):
    dilation_tup = tuple(dilation)
  else:
    dilation_tup = dilation

  input_shape_for_ceil = inputs.shape
  if num_batch_dims == 0:
    input_shape_for_ceil = [1, *input_shape_for_ceil]
  padding_pairs = _ceil_mode_padding(
      padding_list,
      input_shape_for_ceil,
      kernel_size_tup,
      strides_tup,
      dilation_tup,
      ceil_mode,
  )

  assert len(kernel_size_tup) == len(
      strides_tup
  ), f"len({kernel_size_tup=}) must equal len({strides_tup=})"
  assert len(kernel_size_tup) == len(
      dilation_tup
  ), f"len({kernel_size_tup=}) must equal len({dilation_tup=})"

  is_single_input = False
  if num_batch_dims == 0:
    inputs = inputs[None]
    is_single_input = True

  reduce_window_strides = (1,) * (inputs.ndim - num_spatial_dims) + strides_tup
  reduce_window_dims = (1,) * (inputs.ndim - num_spatial_dims) + kernel_size_tup
  reduce_window_dilation = (1,) * (
      inputs.ndim - num_spatial_dims
  ) + dilation_tup

  assert inputs.ndim == len(
      reduce_window_dims
  ), f"len({inputs.shape}) != len({reduce_window_dims})"
  if not isinstance(padding_pairs, str):
    padding_pairs_tup = tuple(padding_pairs)
    assert all(
        [len(x) == 2 for x in padding_pairs_tup]
    ), f"each entry in padding {padding_pairs_tup} must be length 2"
    padding_lax = ((0, 0),) * (
        inputs.ndim - len(padding_pairs_tup)
    ) + padding_pairs_tup
  else:
    padding_lax = padding_pairs

  indices = jnp.arange(
      np.prod(inputs.shape[-num_spatial_dims:]), dtype=jnp.int64
  )
  indices = indices.reshape(inputs.shape[-num_spatial_dims:])
  indices_shape = (1,) * (inputs.ndim - indices.ndim) + indices.shape
  indices = jnp.broadcast_to(indices.reshape(indices_shape), inputs.shape)

  return_dtype = inputs.dtype
  if jnp.issubdtype(inputs.dtype, jnp.integer):
    init_val = jnp.int32(jnp.iinfo(jnp.int32).min)
    inputs = inputs.astype(jnp.int32)
  else:
    init_val = jnp.float32(-jnp.inf)
    inputs = inputs.astype(jnp.float32)

  if not with_index:
    y = jax.lax.reduce_window(
        inputs,
        init_val,
        jax.lax.max,
        reduce_window_dims,
        reduce_window_strides,
        padding_lax,
        window_dilation=reduce_window_dilation,
    )
    if is_single_input:
      y = jnp.squeeze(y, axis=0)
    return y.astype(return_dtype)
  else:

    def reduce_fn(a, b):
      ai, av = a
      bi, bv = b
      which = av >= bv
      return jnp.where(which, ai, bi), jnp.where(which, av, bv)

    indices, y = jax.lax.reduce_window(
        (indices, inputs),
        (jnp.int64(0), init_val),
        reduce_fn,
        reduce_window_dims,
        reduce_window_strides,
        padding_lax,
        window_dilation=reduce_window_dilation,
    )
    if is_single_input:
      indices = jnp.squeeze(indices, axis=0)
      y = jnp.squeeze(y, axis=0)
    y = y.astype(return_dtype)
    return y, indices


@lower_by_jax(torch.ops.aten.max_pool2d_with_indices)
def _aten_max_pool2d_with_indices(
    self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
  stride = stride if stride is not None else []
  y = max_pool(
      self,
      kernel_size,
      strides=stride,
      padding=padding,
      dilation=dilation,
      ceil_mode=ceil_mode,
      with_index=False,
  )
  # TFLite's reduce_window kernel doesn't support multiple inputs/outputs,
  # so we emit reduce_window with a single output and return dummy indices.
  return y, jnp.zeros_like(y, dtype=jnp.int64)


@lower_by_jax(torch.ops.aten.max_pool3d_with_indices.default)
def _aten_max_pool3d_with_indices(
    self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
):
  stride = stride if stride is not None else []
  y = max_pool(
      self,
      kernel_size,
      strides=stride,
      padding=padding,
      dilation=dilation,
      ceil_mode=ceil_mode,
      with_index=False,
  )
  # TFLite's reduce_window kernel doesn't support multiple inputs/outputs,
  # so we emit reduce_window with a single output and return dummy indices.
  return y, jnp.zeros_like(y, dtype=jnp.int64)


@lower_by_jax(torch.ops.aten.pixel_shuffle)
def _aten_pixel_shuffle(x, upscale_factor):
  """PixelShuffle implementation in JAX lowering.

  Args:
    x: Input tensor. Typically a feature map.
    upscale_factor: Integer by which to upscale the spatial dimensions.

  Returns:
    Tensor after PixelShuffle operation.
  """

  batch_size, channels, height, width = x.shape

  if channels % (upscale_factor**2) != 0:
    raise ValueError(
        "Number of channels must be divisible by the square of the upscale"
        " factor."
    )

  new_channels = channels // (upscale_factor**2)
  new_height = height * upscale_factor
  new_width = width * upscale_factor

  x = x.reshape(
      batch_size, new_channels, upscale_factor, upscale_factor, height, width
  )
  x = jnp.transpose(
      x, (0, 1, 4, 2, 5, 3)
  )  # Move channels to spatial dimensions
  x = x.reshape(batch_size, new_channels, new_height, new_width)

  return x


@lower_by_jax(torch.ops.aten.unbind)
def _aten_copy(self, *args, **kwargs):
  return _TORCH_XLA2_IMPLS[torch.ops.aten.unbind_copy](self, *args, **kwargs)


@lower_by_jax(torch.ops.aten.copy, ir_input_names=["src"])
def _aten_copy(self, src, **kwargs):
  return _TORCH_XLA2_IMPLS[torch.ops.aten.copy](self, src)


@lower_by_jax(torch.ops.aten.elu.default)
def _aten_elu(self, alpha=1.0, scale=1.0, input_scale=1.0):
  pos_coef = scale
  neg_coef = alpha * scale
  neg_input_coef = input_scale

  pos_branch = self * pos_coef
  neg_branch = jnp.expm1(self * neg_input_coef) * neg_coef
  return jnp.where(self >= 0, pos_branch, neg_branch)


@registry.lower(torch.ops.aten.add.Scalar)
def _aten_add_scalar(lctx: LoweringContext, self, other):
  _log_usage(torch.ops.aten.add.Scalar)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    other_dtype = jnp.result_type(other)
    promoted_type = jnp.promote_types(self.dtype, other_dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    return jnp.add(
        self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
    )

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.add.Tensor)
def _aten_add_tensor(lctx: LoweringContext, self, other):
  _log_usage(torch.ops.aten.add.Tensor)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    promoted_type = jnp.promote_types(self.dtype, other.dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    return jnp.add(self.astype(promoted_type), other.astype(promoted_type))

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.sub.Scalar)
def _aten_sub_scalar(lctx: LoweringContext, self, other):
  _log_usage(torch.ops.aten.sub.Scalar)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    other_dtype = jnp.result_type(other)
    promoted_type = jnp.promote_types(self.dtype, other_dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    return jnp.subtract(
        self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
    )

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.sub.Tensor)
def _aten_sub_tensor(lctx: LoweringContext, self, other):
  _log_usage(torch.ops.aten.sub.Tensor)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    promoted_type = jnp.promote_types(self.dtype, other.dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    return jnp.subtract(self.astype(promoted_type), other.astype(promoted_type))

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.lt.Scalar)
def _aten_lt_scalar(lctx: LoweringContext, self, other):
  _log_usage(torch.ops.aten.lt.Scalar)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    other_dtype = jnp.result_type(other)
    promoted_type = jnp.promote_types(self.dtype, other_dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    return jnp.less(
        self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
    )

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.lt.Tensor)
def _aten_lt_tensor(lctx: LoweringContext, self, other):
  _log_usage(torch.ops.aten.lt.Tensor)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    promoted_type = jnp.promote_types(self.dtype, other.dtype)
    return jnp.less(self.astype(promoted_type), other.astype(promoted_type))

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.mul.Scalar)
def _aten_mul_scalar(lctx: LoweringContext, self, other):
  _log_usage(torch.ops.aten.mul.Scalar)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    other_dtype = jnp.result_type(other)
    promoted_type = jnp.promote_types(self.dtype, other_dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    elif promoted_type == jnp.int64:
      promoted_type = jnp.int32
    return jnp.multiply(
        self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
    )

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.mul.Tensor)
def _aten_mul_tensor(lctx: LoweringContext, self, other):
  _log_usage(torch.ops.aten.mul.Tensor)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    other_dtype = jnp.result_type(other)
    promoted_type = jnp.promote_types(self.dtype, other_dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    elif promoted_type == jnp.int64:
      promoted_type = jnp.int32
    return jnp.multiply(
        self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
    )

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.div.Scalar)
def _aten_div_scalar(lctx: LoweringContext, self, other):
  _log_usage(torch.ops.aten.div.Scalar)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    other_dtype = jnp.result_type(other)
    promoted_type = jnp.promote_types(self.dtype, other_dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    elif promoted_type == jnp.int64:
      promoted_type = jnp.int32
    return jnp.divide(
        self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
    )

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.div.Scalar_mode)
def _aten_div_scalar_mode(lctx: LoweringContext, self, other, rounding_mode=""):
  _log_usage(torch.ops.aten.div.Scalar_mode)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    other_dtype = jnp.result_type(other)
    promoted_type = jnp.promote_types(self.dtype, other_dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    elif promoted_type == jnp.int64:
      promoted_type = jnp.int32
    if rounding_mode == "floor":
      return jnp.floor_divide(
          self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
      )
    result = jnp.divide(
        self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
    )
    if rounding_mode == "trunc":
      result = jnp.trunc(result)
    return result

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.div.Tensor)
def _aten_div_tensor(lctx: LoweringContext, self, other):
  _log_usage(torch.ops.aten.div.Tensor)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    other_dtype = jnp.result_type(other)
    promoted_type = jnp.promote_types(self.dtype, other_dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    elif promoted_type == jnp.int64:
      promoted_type = jnp.int32
    return jnp.divide(
        self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
    )

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.div.Tensor_mode)
def _aten_div_tensor_mode(lctx: LoweringContext, self, other, rounding_mode=""):
  _log_usage(torch.ops.aten.div.Tensor_mode)

  @jax_bridge.wrap
  def jax_lowering(self, other):
    other_dtype = jnp.result_type(other)
    promoted_type = jnp.promote_types(self.dtype, other_dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    elif promoted_type == jnp.int64:
      promoted_type = jnp.int32
    if rounding_mode == "floor":
      return jnp.floor_divide(
          self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
      )
    result = jnp.divide(
        self.astype(promoted_type), jnp.array(other, dtype=promoted_type)
    )
    if rounding_mode == "trunc":
      result = jnp.trunc(result)
    return result

  return jax_lowering(lctx, self, other)


@registry.lower(torch.ops.aten.where.self)
def _aten_where_self(lctx: LoweringContext, condition, self, other):
  _log_usage(torch.ops.aten.where.self)

  @jax_bridge.wrap
  def jax_lowering(condition, self, other):
    promoted_type = jnp.promote_types(self.dtype, other.dtype)
    if promoted_type == jnp.float64:
      promoted_type = jnp.float32
    return jnp.where(
        condition,
        self.astype(promoted_type),
        other.astype(promoted_type),
    )

  return jax_lowering(lctx, condition, self, other)


# Schema:
#   - aten::einsum(str equation, Tensor[] tensors, *, int[]? path=None)
#       -> Tensor
# Torch Reference:
#   - https://pytorch.org/docs/stable/generated/torch.einsum.html
#   - https://github.com/pytorch/pytorch/blob/1b3f8b75896720e88362cbec7db32abc52afa83e/aten/src/ATen/native/Linear.cpp#L255
@registry.lower(torch.ops.aten.einsum.default)
def _aten_einsum_default(
    lctx: LoweringContext,
    equation: str,
    tensors: list[ir.Value],
    path=None,
):
  _log_usage(torch.ops.aten.einsum.default)

  @jax_bridge.wrap
  def jax_lowering(operands):
    # Ignore the input path and let JAX determine the path.
    return jnp.einsum(equation, *operands, optimize="optimal")

  return jax_lowering(lctx, tuple(tensors))


@registry.lower(torch.ops.aten.topk)
def _aten_topk(
    lctx: LoweringContext, self, k, dim=-1, largest=True, sorted=True
):
  _log_usage(torch.ops.aten.topk)

  if not sorted:
    logging.warning(
        "aten.topk lowering ignores `sorted=False` and always returns sorted"
        " results."
    )

  @jax_bridge.wrap
  def jax_lowering(self, k):
    if not largest:
      self = -self
    # jax.lax.top_k always sorts and operates on the last dimension.
    move_dim_to_last = dim != -1 and dim != self.ndim - 1
    if move_dim_to_last:
      input_tensor = jnp.moveaxis(self, dim, -1)
    else:
      input_tensor = self
    values, indices = jax.lax.top_k(input_tensor, k)
    if move_dim_to_last:
      values = jnp.moveaxis(values, -1, dim)
      indices = jnp.moveaxis(indices, -1, dim)
    if not largest:
      values = -values
    return values, indices

  return jax_lowering(lctx, self, k)
