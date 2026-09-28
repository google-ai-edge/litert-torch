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
"""Optimized GPU short convolution decode step composite operation compatible with MLDrift."""

from typing import Optional
from litert_torch.backend import composite
import torch


def apply_short_conv_step(
    in_proj_out: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    conv_L_cache: int = 3,
    is_gated: bool = True,
    use_silu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Computes a single decode step of 1D depthwise short convolution.

  Args:
    in_proj_out: Input projection tensor [batch, 1, 3 * hidden_size] if
      is_gated=True, or [batch, 1, hidden_size] if is_gated=False.
    conv_state: Convolution state tensor [batch, hidden_size, conv_L_cache - 1].
    conv_weight: Depthwise convolution weight [hidden_size, 1, conv_L_cache] or
      [hidden_size, conv_L_cache].
    conv_bias: Optional bias tensor [hidden_size].
    conv_L_cache: Kernel length / cache size (default 3).
    is_gated: If True, splits in_proj_out into (b, c, x_proj) and computes
      y = c * conv(b * x_proj). If False, applies depthwise conv directly to
      in_proj_out.
    use_silu: If True, applies SiLU activation to the convolution output.

  Returns:
    y: Output activation tensor [batch, 1, hidden_size] ready for out_proj.
    next_state: Updated state tensor [batch, hidden_size, conv_L_cache - 1].
  """
  attrs: dict[str, int | bool] = {
      "conv_L_cache": int(conv_L_cache),
  }
  if not is_gated:
    attrs["is_gated"] = False
  if use_silu:
    attrs["use_silu"] = True
  builder = composite.StableHLOCompositeBuilder(
      name="odml.short_conv_step", attr=attrs
  )
  if conv_bias is not None:
    in_proj_out, conv_state, conv_weight, conv_bias = builder.mark_inputs(
        in_proj_out, conv_state, conv_weight, conv_bias
    )
  else:
    in_proj_out, conv_state, conv_weight = builder.mark_inputs(
        in_proj_out, conv_state, conv_weight
    )

  # Fallback PyTorch execution during export tracing:
  w = (
      conv_weight.squeeze(1).unsqueeze(0)
      if conv_weight.dim() == 3
      else conv_weight.unsqueeze(0)
  )
  if is_gated:
    b, c, x_proj = in_proj_out.chunk(3, dim=-1)
    conv_input = b * x_proj
    conv_input_s = conv_input.squeeze(1).unsqueeze(-1)
    padded_input = torch.cat([conv_state, conv_input_s], dim=-1)
    next_state = padded_input[:, :, -(conv_L_cache - 1) :]
    conv_out = (padded_input * w).sum(dim=-1)
    if conv_bias is not None:
      conv_out = conv_out + conv_bias.unsqueeze(0)
    if use_silu:
      conv_out = torch.nn.functional.silu(conv_out)
    y = c * conv_out.unsqueeze(1)
  else:
    conv_input_s = in_proj_out.transpose(1, 2)
    padded_input = torch.cat([conv_state, conv_input_s], dim=-1)
    next_state = padded_input[:, :, -(conv_L_cache - 1) :]
    conv_out = (padded_input * w).sum(dim=-1)
    if conv_bias is not None:
      conv_out = conv_out + conv_bias.unsqueeze(0)
    if use_silu:
      conv_out = torch.nn.functional.silu(conv_out)
    y = conv_out.unsqueeze(1)

  y, next_state = builder.mark_outputs(y, next_state)
  return y, next_state
