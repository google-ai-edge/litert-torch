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
"""Short convolution: causal depthwise conv1d with a rolling state.

Shared by the LFM2 short convolution blocks and the Qwen3.5 Gated DeltaNet
layers. `apply_short_conv_step` wraps the LFM2 gated short convolution in the
`odml.short_conv_step` composite, which MLDrift runs as one GPU kernel in both
prefill and decode.
"""

from typing import Optional
from litert_torch.backend import composite
from litert_torch.backend.torch_tfl import _ops as _  # Registers torch.ops.tfl.*
import torch
import torch.nn.functional as F


def short_conv(
    in_proj_out: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
    is_gated: bool = True,
    use_silu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Computes gated (`y = C * conv(B * x)`) or ungated short convolution.

  This is the decomposition of `odml.short_conv_step`, and what runs when the
  composite is disabled. Arguments and results are the same as
  `apply_short_conv_step`.

  Args:
    in_proj_out: Input projection tensor [batch, seq_len, 3 * hidden_size] if
      is_gated=True, or [batch, seq_len, hidden_size] if is_gated=False.
    conv_state: Convolution state tensor [batch, L - 1, hidden_size].
    conv_weight: Depthwise convolution weight [hidden_size, 1, L] or
      [hidden_size, L].
    conv_bias: Optional bias tensor [hidden_size].
    num_valid_tokens: Optional int32 tensor [1] with the number of valid
      (non-padding) tokens at the start of the sequence, in [0, seq_len]. When
      omitted, all `seq_len` tokens are treated as valid.
    is_gated: If True, splits in_proj_out into (b, c, x_proj) and computes y = c
      * conv(b * x_proj). If False, applies depthwise conv directly to
      in_proj_out.
    use_silu: If True, applies SiLU activation to the convolution output.

  Returns:
    y: Output activation tensor [batch, seq_len, hidden_size] ready for
      out_proj.
    next_state: Updated state tensor [batch, L - 1, hidden_size].
  """
  if is_gated:
    b, c, x_proj = in_proj_out.chunk(3, dim=-1)
    conv_input = b * x_proj
  else:
    c = None
    conv_input = in_proj_out

  state_len = conv_state.shape[1]
  w = conv_weight.squeeze(1) if conv_weight.dim() == 3 else conv_weight
  # Transpose weight [channels, L] -> [L, channels] if needed.
  if w.dim() == 2 and w.shape[0] == conv_input.shape[-1]:
    w = w.transpose(0, 1)
  if w.dim() == 2:
    w = w.unsqueeze(0)  # [1, L, channels]

  if in_proj_out.shape[1] == 1 and num_valid_tokens is None:
    # Single-token decode with NWC channels-last state [batch, state_len, channels].
    # Compute taps directly with binary operations to avoid the need for slicing
    # and concatenation.
    conv_out = conv_state[:, 0, :] * w[:, 0, :]
    for k in range(1, state_len):
      conv_out = conv_out + conv_state[:, k, :] * w[:, k, :]
    conv_out = conv_out + conv_input.squeeze(1) * w[:, state_len, :]
    if conv_bias is not None:
      conv_out = conv_out + conv_bias.unsqueeze(0)
    if use_silu:
      conv_out = F.silu(conv_out)
    y = c * conv_out.unsqueeze(1) if c is not None else conv_out.unsqueeze(1)
    if state_len > 1:
      next_state = torch.cat([conv_state[:, 1:, :], conv_input], dim=1)
    else:
      next_state = conv_input
    return y, next_state

  # Prefill with NWC channels-last state [batch, state_len, channels].
  padded = torch.cat([conv_state, conv_input], dim=1)
  seq_len = in_proj_out.shape[1]
  num_taps = w.shape[1]
  conv_out = padded[:, 0 : seq_len, :] * w[:, 0 : 1, :]
  for k in range(1, num_taps):
    conv_out = conv_out + padded[:, k : k + seq_len, :] * w[:, k : k + 1, :]
  if conv_bias is not None:
    conv_out = conv_out + conv_bias.unsqueeze(0).unsqueeze(0)
  if num_valid_tokens is None:
    next_state = padded[:, seq_len : seq_len + state_len, :]
  else:
    indices = num_valid_tokens + torch.arange(
        state_len, device=padded.device, dtype=torch.int32
    )
    next_state = torch.ops.tfl.gather(padded, indices, 1)
  if use_silu:
    conv_out = F.silu(conv_out)
  y = c * conv_out if c is not None else conv_out
  return y, next_state


def apply_short_conv_step(
    in_proj_out: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
    conv_L_cache: int = 3,
    is_gated: bool = True,
    use_silu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Emits the short convolution as `odml.short_conv_step`.

  Covers decode (seq_len = 1) and prefill (seq_len >= 1). When
  `num_valid_tokens` is provided, the valid tokens must form a prefix of the
  sequence and the remaining positions are padding: `y` is computed for every
  position, while `next_state` covers inputs up to and including the last
  valid token. The composite decomposes to `short_conv`.

  Args:
    in_proj_out: Input projection tensor [batch, seq_len, 3 * hidden_size] if
      is_gated=True, or [batch, seq_len, hidden_size] if is_gated=False.
    conv_state: Convolution state tensor [batch, conv_L_cache - 1, hidden_size].
    conv_weight: Depthwise convolution weight [hidden_size, 1, conv_L_cache] or
      [hidden_size, conv_L_cache].
    conv_bias: Optional bias tensor [hidden_size].
    num_valid_tokens: Optional int32 tensor [1] with the number of valid
      (non-padding) tokens at the start of the sequence, in [0, seq_len]. When
      omitted, all `seq_len` tokens are treated as valid.
    conv_L_cache: Kernel length / cache size (default 3).
    is_gated: If True, splits in_proj_out into (b, c, x_proj) and computes y = c
      * conv(b * x_proj). If False, applies depthwise conv directly to
      in_proj_out.
    use_silu: If True, applies SiLU activation to the convolution output.

  Returns:
    y: Output activation tensor [batch, seq_len, hidden_size] ready for
      out_proj.
    next_state: Updated state tensor [batch, conv_L_cache - 1, hidden_size].
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

  y, next_state = short_conv(
      in_proj_out,
      conv_state,
      conv_weight,
      conv_bias,
      num_valid_tokens,
      is_gated=is_gated,
      use_silu=use_silu,
  )
  y, next_state = builder.mark_outputs(y, next_state)
  return y, next_state
