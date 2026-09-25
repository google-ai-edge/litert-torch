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
from litert_torch.backend.experimental.torch_tfl import _ops as _  # Registers torch.ops.tfl.*
import torch
import torch.nn.functional as F


def causal_conv1d_with_state(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Computes a causal depthwise conv1d over `concat(conv_state, x)`.

  Args:
    x: Conv input [batch, channels, seq_len]. When `num_valid_tokens` is set,
      the valid tokens must form a prefix of the sequence.
    conv_state: [batch, channels, L - 1], the last `L - 1` inputs of the
      previous call, where `L` is the kernel size.
    conv_weight: Depthwise weight [channels, 1, L] or [channels, L].
    conv_bias: Optional bias [channels].
    num_valid_tokens: Optional int32 tensor with one element (shape [] or [1]):
      the number of valid tokens at the start of `x`, in [0, seq_len]. When
      omitted, all tokens are valid.

  Returns:
    conv_out: [batch, channels, seq_len].
    next_state: The last `L - 1` inputs up to and including the last valid
      token, [batch, channels, L - 1].
  """
  state_len = conv_state.shape[-1]
  padded = torch.cat([conv_state, x], dim=-1)
  weight = conv_weight if conv_weight.dim() == 3 else conv_weight.unsqueeze(1)
  conv_out = F.conv1d(padded, weight, conv_bias, groups=weight.shape[0])
  if num_valid_tokens is None:
    next_state = padded[:, :, -state_len:]
  else:
    # padded[:, :, n : n + L - 1] for a runtime n. tfl.gather keeps the
    # dynamic offset lowerable to the GPU delegates.
    indices = num_valid_tokens + torch.arange(
        state_len, device=padded.device, dtype=torch.int32
    )
    next_state = torch.ops.tfl.gather(padded, indices, 2)
  return conv_out, next_state


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
    conv_state: Convolution state tensor [batch, hidden_size, L - 1].
    conv_weight: Depthwise convolution weight [hidden_size, 1, L] or
      [hidden_size, L].
    conv_bias: Optional bias tensor [hidden_size].
    num_valid_tokens: Optional int32 tensor [1] with the number of valid
      (non-padding) tokens at the start of the sequence, in [0, seq_len]. When
      omitted, all `seq_len` tokens are treated as valid.
    is_gated: If True, splits in_proj_out into (b, c, x_proj) and computes
      y = c * conv(b * x_proj). If False, applies depthwise conv directly to
      in_proj_out.
    use_silu: If True, applies SiLU activation to the convolution output.

  Returns:
    y: Output activation tensor [batch, seq_len, hidden_size] ready for
      out_proj.
    next_state: Updated state tensor [batch, hidden_size, L - 1].
  """
  if is_gated:
    b, c, x_proj = in_proj_out.chunk(3, dim=-1)
    conv_input = b * x_proj
  else:
    c = None
    conv_input = in_proj_out

  if in_proj_out.shape[1] == 1 and num_valid_tokens is None:
    # Single-token decode: the depthwise conv is a multiply-add over the L
    # taps. Reshapes instead of transposes keep it a direct dot product.
    padded_input = torch.cat(
        [conv_state, conv_input.squeeze(1).unsqueeze(-1)], dim=-1
    )
    next_state = padded_input[:, :, -conv_state.shape[-1] :]
    w = conv_weight.squeeze(1) if conv_weight.dim() == 3 else conv_weight
    conv_out = (padded_input * w.unsqueeze(0)).sum(dim=-1)
    if conv_bias is not None:
      conv_out = conv_out + conv_bias.unsqueeze(0)
    if use_silu:
      conv_out = F.silu(conv_out)
    y = c * conv_out.unsqueeze(1) if c is not None else conv_out.unsqueeze(1)
    return y, next_state

  conv_out, next_state = causal_conv1d_with_state(
      conv_input.transpose(1, 2),
      conv_state,
      conv_weight,
      conv_bias,
      num_valid_tokens,
  )
  if use_silu:
    conv_out = F.silu(conv_out)
  conv_out_t = conv_out.transpose(1, 2)
  return (c * conv_out_t if c is not None else conv_out_t), next_state


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
    conv_state: Convolution state tensor [batch, hidden_size, conv_L_cache - 1].
    conv_weight: Depthwise convolution weight [hidden_size, 1, conv_L_cache] or
      [hidden_size, conv_L_cache].
    conv_bias: Optional bias tensor [hidden_size].
    num_valid_tokens: Optional int32 tensor [1] with the number of valid
      (non-padding) tokens at the start of the sequence, in [0, seq_len]. When
      omitted, all `seq_len` tokens are treated as valid.
    conv_L_cache: Kernel length / cache size (default 3).
    is_gated: If True, splits in_proj_out into (b, c, x_proj) and computes
      y = c * conv(b * x_proj). If False, applies depthwise conv directly to
      in_proj_out.
    use_silu: If True, applies SiLU activation to the convolution output.

  Returns:
    y: Output activation tensor [batch, seq_len, hidden_size] ready for
      out_proj.
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
