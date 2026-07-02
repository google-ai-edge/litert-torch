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
"""Torch export decompositions to run before lowering."""

import functools
from litert_torch import fx_infra
import torch


# Fork from pytorch/torch/_decomp/decompositions.py
def upsample_compute_output_size(input_size, output_size, scale_factors):
  spatial_dimensions = len(input_size) - 2
  if output_size is not None:
    torch._check(
        scale_factors is None,
        lambda: "Must specify exactly one of output_size and scale_factors",
    )
    torch._check(len(output_size) == spatial_dimensions, lambda: "")
    return output_size
  if scale_factors is not None:
    # NB: this isn't necessary lol
    torch._check(
        output_size is None,
        lambda: "Must specify exactly one of output_size and scale_factors",
    )
    torch._check(len(scale_factors) == spatial_dimensions, lambda: "")
    output_size = []
    for i, s in enumerate(scale_factors):
      if int(s) == s:
        output_size.append(input_size[i + 2] * int(s))
      else:
        output_size.append(torch.sym_int(input_size[i + 2] * s))
    return output_size
  torch._check(
      False, lambda: "Must specify exactly one of output_size and scale_factors"
  )


# Fork from pytorch/torch/_decomp/decompositions.py
def _compute_upsample_nearest_indices(input, output_size, scales, exact=False):
  indices = []
  num_spatial_dims = len(output_size)
  offset = 0.5 if exact else 0.0

  for d in range(num_spatial_dims):
    osize = output_size[d]
    isize = input.shape[-num_spatial_dims + d]
    scale = (
        isize / (isize * scales[d]) if scales[d] is not None else isize / osize
    )

    output_indices = torch.arange(
        osize, dtype=torch.float32, device=input.device
    )
    input_indices = ((output_indices + offset) * scale).to(torch.int64)
    for _ in range(num_spatial_dims - 1 - d):
      input_indices = input_indices.unsqueeze(-1)
    indices.append(input_indices)
  return tuple(indices)


# Fork from pytorch/torch/_decomp/decompositions.py
def _upsample_nearest2d_common(input, h_indices, w_indices):
  result = torch.ops.aten.index(input, (None, None, h_indices, w_indices))
  result = result.contiguous()
  return result


fx_infra.decomp.update_pre_lower_decomp(
    torch._decomp.get_decompositions([
        torch.ops.aten._native_batch_norm_legit.no_stats,
        torch.ops.aten._native_batch_norm_legit_functional,
        torch.ops.aten._adaptive_avg_pool2d,
        torch.ops.aten._adaptive_avg_pool3d,
        torch.ops.aten.grid_sampler_2d,
        torch.ops.aten.native_group_norm,
        torch.ops.aten.native_dropout,
        torch.ops.aten.reflection_pad1d,
        torch.ops.aten.reflection_pad3d,
        torch.ops.aten.replication_pad1d,
        torch.ops.aten.replication_pad3d,
        torch.ops.aten.upsample_bilinear2d.vec,
        torch.ops.aten.addmm,
        torch.ops.aten.upsample_nearest2d.vec,
    ])
)


@functools.partial(
    fx_infra.decomp.add_pre_lower_decomp,
    torch.ops.aten.upsample_nearest2d.default,
)
@fx_infra.annotate_force_decomp
def upsample_nearest2d(input, output_size, scales_h=None, scales_w=None):
  h_indices, w_indices = _compute_upsample_nearest_indices(
      input, output_size, (scales_h, scales_w)
  )
  return _upsample_nearest2d_common(input, h_indices, w_indices)


def get_scale_value(scales, idx):
  if scales is None:
    return None
  return scales[idx]





fx_infra.decomp.remove_pre_lower_decomp(torch.ops.aten.roll)

# Torch's default einsum impl/decompositions is less efficient and
# optimized through converter than JAX's impl. Disable einsum
# decomposition to use JAX bridge for a more efficient lowering.
fx_infra.decomp.remove_pre_lower_decomp(torch.ops.aten.einsum.default)


# Override noop aten op decompositions for faster run_decompositions.
fx_infra.decomp.add_pre_convert_decomp(
    torch.ops.aten.alias.default, lambda x: x
)
fx_infra.decomp.add_pre_convert_decomp(
    torch.ops.aten.detach.default, lambda x: x
)

# Override _safe_softmax decompositions with regular softmax.
# _safe_softmax introduces additional check-select ops to guard extreme
# input values to softmax, which could make the converted model inefficient
# on-device.
if hasattr(torch.ops.aten, "_safe_softmax"):
  fx_infra.decomp.add_pre_convert_decomp(
      torch.ops.aten._safe_softmax.default,
      torch.softmax,
  )


# Promote 1D convolution to 2D in FX layer to prevent rank 3D-4D-3D round trips
def _conv1d_decomp(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
  """Decomposes 1D convolution to 2D convolution by expanding spatial dims."""
  if x.dim() == 3:
    if isinstance(stride, (int, torch.SymInt)):
      stride = [stride]
    if isinstance(padding, (int, torch.SymInt)):
      padding = [padding]
    if isinstance(dilation, (int, torch.SymInt)):
      dilation = [dilation]
    x4 = x.unsqueeze(-2)
    w4 = weight.unsqueeze(-2)
    out = torch.ops.aten.convolution.default(
        x4,
        w4,
        bias,
        [1, *stride],
        [0, *padding],
        [1, *dilation],
        False,
        [0],
        groups,
    )
    return out.squeeze(-2)
  return NotImplemented


def _convolution_decomp(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    output_padding=0,
    groups=1,
):
  """Decomposes rank-3 convolution to 2D convolution by expanding spatial dims."""
  if x.dim() == 3:
    if isinstance(stride, (int, torch.SymInt)):
      stride = [stride]
    if isinstance(padding, (int, torch.SymInt)):
      padding = [padding]
    if isinstance(dilation, (int, torch.SymInt)):
      dilation = [dilation]

    if isinstance(output_padding, (int, torch.SymInt)):
      output_padding = [output_padding]
    x4 = x.unsqueeze(-2)
    w4 = weight.unsqueeze(-2)
    out = torch.ops.aten.convolution.default(
        x4,
        w4,
        bias,
        [1, *stride],
        [0, *padding],
        [1, *dilation],
        transposed,
        [0, *output_padding],
        groups,
    )
    return out.squeeze(-2)
  return NotImplemented


fx_infra.decomp.add_pre_convert_decomp(
    torch.ops.aten.convolution.default,
    _convolution_decomp,
)
fx_infra.decomp.add_pre_convert_decomp(
    torch.ops.aten.conv1d.default,
    _conv1d_decomp,
)
fx_infra.decomp.add_pre_convert_decomp(
    torch.ops.aten.conv1d,
    _conv1d_decomp,
)


# Override the _prelu_kernel decomposition, where(x > 0, x, w * x), which
# legalizes to GREATER + SELECT, both rejected by the TFLite GPU delegate.
# The relu form below is numerically identical (including NaN propagation)
# and legalizes to RELU/MUL/SUB, which are GPU-clean.
def _prelu_kernel_relu_form(self, weight):
  return torch.relu(self) - weight * torch.relu(torch.neg(self))


fx_infra.decomp.add_pre_convert_decomp(
    torch.ops.aten._prelu_kernel.default, _prelu_kernel_relu_form
)


# Override the pixel_shuffle/pixel_unshuffle decompositions, which go through
# a rank-6 reshape + permute while GPU delegates cap tensors at rank 4. The
# forms below interleave one spatial axis at a time (folding batch and
# channel into one dimension) so every intermediate tensor stays rank 4.
def _pixel_shuffle_rank4(self, upscale_factor):
  *batch, c, h, w = self.shape
  r = upscale_factor
  oc = c // (r * r)
  x = self.reshape(-1, r, r, h * w)
  x = x.transpose(2, 3)
  x = x.reshape(-1, r, h, w * r)
  x = x.transpose(1, 2)
  return x.reshape(*batch, oc, h * r, w * r)


def _pixel_unshuffle_rank4(self, downscale_factor):
  *batch, c, height, width = self.shape
  r = downscale_factor
  h = height // r
  w = width // r
  x = self.reshape(-1, h, r, width)
  x = x.transpose(1, 2)
  x = x.reshape(-1, r, h * w, r)
  x = x.transpose(2, 3)
  return x.reshape(*batch, c * r * r, h, w)


fx_infra.decomp.add_pre_convert_decomp(
    torch.ops.aten.pixel_shuffle.default, _pixel_shuffle_rank4
)
fx_infra.decomp.add_pre_convert_decomp(
    torch.ops.aten.pixel_unshuffle.default, _pixel_unshuffle_rank4
)
