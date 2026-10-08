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
"""Optimized GPU short convolution composites compatible with MLDrift."""

from typing import Optional
from litert_torch.backend import composite
import torch


def apply_short_conv_step(
    in_proj_out: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
    conv_L_cache: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Computes 1D depthwise short convolution for decode (S=1) or prefill (S>=1).

  When `num_valid_tokens` is provided, the valid tokens must form a prefix of
  the sequence and the remaining positions are padding: `y` is computed for
  every position, while `next_state` covers inputs up to and including the last
  valid token.

  Args:
    in_proj_out: Fused input projection tensor [batch, seq_len, 3 *
      hidden_size].
    conv_state: Convolution state tensor [batch, hidden_size, conv_L_cache - 1].
    conv_weight: Depthwise convolution weight [hidden_size, 1, conv_L_cache] or
      [hidden_size, conv_L_cache].
    conv_bias: Optional bias tensor [hidden_size].
    num_valid_tokens: Optional int32 tensor [1] with the number of valid
      (non-padding) tokens at the start of the sequence, in [0, seq_len]. When
      omitted, all `seq_len` tokens are treated as valid.
    conv_L_cache: Kernel length / cache size (default 3).

  Returns:
    y: Output activation tensor [batch, seq_len, hidden_size] ready for
      out_proj.
    next_state: Updated state tensor [batch, hidden_size, conv_L_cache - 1].
  """
  attrs = {
      "conv_L_cache": int(conv_L_cache),
  }
  builder = composite.StableHLOCompositeBuilder(
      name="odml.short_conv_step", attr=attrs
  )
  if conv_bias is not None and num_valid_tokens is not None:
    in_proj_out, conv_state, conv_weight, conv_bias, num_valid_tokens = (
        builder.mark_inputs(
            in_proj_out, conv_state, conv_weight, conv_bias, num_valid_tokens
        )
    )
  elif conv_bias is not None:
    in_proj_out, conv_state, conv_weight, conv_bias = builder.mark_inputs(
        in_proj_out, conv_state, conv_weight, conv_bias
    )
  elif num_valid_tokens is not None:
    in_proj_out, conv_state, conv_weight, num_valid_tokens = (
        builder.mark_inputs(
            in_proj_out, conv_state, conv_weight, num_valid_tokens
        )
    )
  else:
    in_proj_out, conv_state, conv_weight = builder.mark_inputs(
        in_proj_out, conv_state, conv_weight
    )

  # Fallback PyTorch execution during export tracing:
  b, c, x_proj = in_proj_out.chunk(3, dim=-1)
  conv_input = b * x_proj
  if in_proj_out.shape[1] == 1 and num_valid_tokens is None:
    conv_input_s = conv_input.squeeze(1).unsqueeze(-1)
    padded_input = torch.cat([conv_state, conv_input_s], dim=-1)
    next_state = padded_input[:, :, -(conv_L_cache - 1) :]
    w = (
        conv_weight.squeeze(1).unsqueeze(0)
        if conv_weight.dim() == 3
        else conv_weight.unsqueeze(0)
    )
    conv_out = (padded_input * w).sum(dim=-1)
    if conv_bias is not None:
      conv_out = conv_out + conv_bias.unsqueeze(0)
    y = c * conv_out.unsqueeze(1)
  else:
    conv_input_t = conv_input.transpose(1, 2)
    padded_input = torch.cat([conv_state, conv_input_t], dim=-1)
    if num_valid_tokens is not None:
      # Select padded_input[:, :, n : n + conv_L_cache - 1] with a one-hot
      # matmul so the dynamic offset lowers to static-shape ops.
      rows = torch.arange(
          padded_input.shape[-1], device=padded_input.device, dtype=torch.int32
      ).unsqueeze(1)
      cols = (
          torch.arange(
              conv_L_cache - 1, device=padded_input.device, dtype=torch.int32
          )
          + num_valid_tokens
      ).unsqueeze(0)
      selection = (rows == cols).to(padded_input.dtype)
      next_state = torch.matmul(padded_input, selection)
    else:
      next_state = padded_input[:, :, -(conv_L_cache - 1) :]
    w = conv_weight if conv_weight.dim() == 3 else conv_weight.unsqueeze(1)
    conv_out = torch.nn.functional.conv1d(
        padded_input, w, conv_bias, groups=w.shape[0]
    )
    y = c * conv_out.transpose(1, 2)

  y, next_state = builder.mark_outputs(y, next_state)
  return y, next_state
