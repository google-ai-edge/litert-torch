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
        torch.ops.aten.reflection_pad2d,
        torch.ops.aten.reflection_pad3d,
        torch.ops.aten.replication_pad1d,
        torch.ops.aten.replication_pad2d,
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
