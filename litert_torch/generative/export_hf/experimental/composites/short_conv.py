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

The conv state is always channels-last, `[batch, L - 1, channels]`, where `L`
is the kernel size, so every tap reads one contiguous row of channels.
"""

from typing import Optional
from litert_torch.backend import composite
from litert_torch.backend.torch_tfl import _ops as _  # Registers torch.ops.tfl.*
import torch
import torch.nn.functional as F


def _to_channels_last_weight(
    conv_weight: torch.Tensor, hidden_size: int
) -> torch.Tensor:
  """Normalizes depthwise conv weight to channels-last `[1, L, hidden_size]`."""
  if conv_weight.dim() == 2:
    return conv_weight.transpose(0, 1).unsqueeze(0).contiguous()
  if (
      conv_weight.shape[0] == 1
      and conv_weight.shape[2] == hidden_size
      and (hidden_size != 1 or conv_weight.shape[1] != 1)
  ):
    return conv_weight
  return conv_weight.squeeze(1).transpose(0, 1).unsqueeze(0).contiguous()


def _check_conv_state(conv_state: torch.Tensor, weight: torch.Tensor) -> None:
  """Raises if `conv_state` is not channels-last `[batch, L - 1, channels]`.

  Args:
    conv_state: The conv state.
    weight: Channels-last depthwise weight `[1, L, channels]`.
  """
  num_taps, channels = weight.shape[1], weight.shape[2]
  if (
      conv_state.dim() != 3
      or conv_state.shape[1] != num_taps - 1
      or conv_state.shape[2] != channels
  ):
    raise ValueError(
        "conv_state must be channels-last [batch, L - 1, channels] ="
        f" [batch, {num_taps - 1}, {channels}], got"
        f" {tuple(conv_state.shape)}"
    )


def causal_conv1d_with_state(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Computes a causal depthwise conv1d over `concat(conv_state, x)`.

  Args:
    x: Conv input [batch, seq_len, channels]. When `num_valid_tokens` is set,
      the valid tokens must form a prefix of the sequence.
    conv_state: [batch, L - 1, channels], the last `L - 1` inputs of the
      previous call, where `L` is the kernel size.
    conv_weight: Depthwise weight [channels, 1, L], [channels, L] or
      channels-last [1, L, channels].
    conv_bias: Optional bias [channels].
    num_valid_tokens: Optional int32 tensor with one element (shape [] or [1]):
      the number of valid tokens at the start of `x`, in [0, seq_len]. When
      omitted, all tokens are valid.

  Returns:
    conv_out: [batch, seq_len, channels].
    next_state: The last `L - 1` inputs up to and including the last valid
      token, [batch, L - 1, channels].
  """
  w = _to_channels_last_weight(conv_weight, conv_state.shape[-1])
  _check_conv_state(conv_state, w)
  state_len = conv_state.shape[1]
  seq_len = x.shape[1]
  padded = torch.cat([conv_state, x], dim=1)
  # Tap-wise multiply-add instead of `F.conv1d`: inside the
  # `odml.short_conv_step` decomposition `conv_weight` is a subgraph argument
  # rather than a constant, so a lowered `tfl.depthwise_conv_2d` would get a
  # non-static filter, which the quantizer skips and XNNPACK rejects, leaving
  # the slow builtin float kernel. Slices, muls and adds stay on XNNPACK.
  num_taps = w.shape[1]
  conv_out = padded[:, 0:seq_len, :] * w[:, 0:1, :]
  for k in range(1, num_taps):
    conv_out = conv_out + padded[:, k : k + seq_len, :] * w[:, k : k + 1, :]
  if conv_bias is not None:
    conv_out = conv_out + conv_bias.view(1, 1, -1)
  if num_valid_tokens is None:
    next_state = padded[:, -state_len:, :]
  else:
    # padded[:, n : n + L - 1, :] for a runtime n. tfl.gather keeps the
    # dynamic offset lowerable to the GPU delegates.
    indices = num_valid_tokens + torch.arange(
        state_len, device=padded.device, dtype=torch.int32
    )
    next_state = torch.ops.tfl.gather(padded, indices, 1)
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
    conv_state: Channels-last convolution state [batch, L - 1, hidden_size].
    conv_weight: Depthwise convolution weight [hidden_size, 1, L], [hidden_size,
      L], or channels-last [1, L, hidden_size].
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
    next_state: Updated channels-last state [batch, L - 1, hidden_size].

  Raises:
    ValueError: If `conv_state` is not channels-last.
  """
  if is_gated:
    b, c, x_proj = in_proj_out.chunk(3, dim=-1)
    conv_input = b * x_proj
  else:
    c = None
    conv_input = in_proj_out

  w = _to_channels_last_weight(conv_weight, conv_state.shape[-1])
  _check_conv_state(conv_state, w)
  state_len = conv_state.shape[1]
  if in_proj_out.shape[1] == 1 and num_valid_tokens is None:
    conv_out = conv_input * w[:, state_len : state_len + 1, :]
    for k in range(state_len):
      conv_out = conv_out + conv_state[:, k : k + 1, :] * w[:, k : k + 1, :]
    if conv_bias is not None:
      conv_out = conv_out + conv_bias.view(1, 1, -1)
    next_state = (
        torch.cat([conv_state[:, 1:, :], conv_input], dim=1)
        if state_len > 1
        else conv_input
    )
  else:
    conv_out, next_state = causal_conv1d_with_state(
        conv_input, conv_state, w, conv_bias, num_valid_tokens
    )
  if use_silu:
    conv_out = F.silu(conv_out)
  return (c * conv_out if c is not None else conv_out), next_state


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

  `conv_weight` is folded to channels-last `[1, conv_L_cache, hidden_size]`
  before it is marked as a composite input, so the transpose is constant-folded
  at export time. Channels-last is the only supported layout; the runtime
  identifies it from the tensor shapes.

  Args:
    in_proj_out: Input projection tensor [batch, seq_len, 3 * hidden_size] if
      is_gated=True, or [batch, seq_len, hidden_size] if is_gated=False.
    conv_state: Channels-last convolution state [batch, conv_L_cache - 1,
      hidden_size].
    conv_weight: Depthwise convolution weight [hidden_size, 1, conv_L_cache],
      [hidden_size, conv_L_cache], or [1, conv_L_cache, hidden_size].
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
    next_state: Updated channels-last state [batch, conv_L_cache - 1,
      hidden_size].

  Raises:
    ValueError: If `conv_state` is not channels-last.
  """
  conv_weight = _to_channels_last_weight(conv_weight, conv_state.shape[-1])
  _check_conv_state(conv_state, conv_weight)
  if conv_weight.shape[1] != conv_L_cache:
    raise ValueError(
        f"conv_weight has {conv_weight.shape[1]} taps, expected conv_L_cache ="
        f" {conv_L_cache}"
    )
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
