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
"""Short convolutions for LFM2."""

from typing import Optional
import torch
from transformers.models.lfm2 import modeling_lfm2


class Lfm2ShortConv(modeling_lfm2.Lfm2ShortConv):
  """Short convolutions for LFM2, suitable for LiteRT inference."""

  def __init__(
      self,
      config: modeling_lfm2.Lfm2Config,
      layer_idx: int,
  ):
    super().__init__(config, layer_idx)
    self.conv = torch.nn.Conv1d(
        in_channels=config.hidden_size,
        out_channels=config.hidden_size,
        kernel_size=self.L_cache,
        groups=config.hidden_size,
        bias=self.bias,
        padding=0,  # Padding is done in forward as part of state management.
    )

  def forward(  # pyrefly: ignore[bad-override]
      self,
      hidden_states: torch.Tensor,
      past_key_values=None,
      cache_position: Optional[torch.LongTensor] = None,
      attention_mask: Optional[torch.Tensor] = None,
      valid_mask: Optional[torch.Tensor] = None,
  ):
    seq_len = hidden_states.shape[1]

    # Compute valid_mask
    if seq_len == 1:  # Decode
      valid_mask = torch.ones(
          (1,), dtype=torch.bool, device=hidden_states.device
      )
    elif valid_mask is None:
      # Fallback to cache_position
      if cache_position is not None:
        valid_mask = torch.ones_like(cache_position, dtype=torch.bool)
        valid_mask[1:] = cache_position[1:] > cache_position[:-1]
      else:
        valid_mask = None
    else:
      # Prefill with valid_mask passed
      if valid_mask.dim() > 1:
        valid_mask = valid_mask.squeeze(0)

    # Apply mask to hidden_states
    if valid_mask is not None:
      hidden_states = hidden_states * valid_mask.unsqueeze(0).unsqueeze(-1)

    b, c, x_proj = self.in_proj(hidden_states).chunk(3, dim=-1)
    conv_input = b * x_proj
    conv_input_t = conv_input.transpose(1, 2)
    state = past_key_values.layers[self.layer_idx].conv_states  # pyrefly: ignore[missing-attribute]
    padded_input = torch.cat([state, conv_input_t], dim=-1)

    if seq_len > 1:  # Prefill
      if valid_mask is not None:
        L_state = self.L_cache - 1
        B, C, S = padded_input.shape
        start = valid_mask.to(torch.float32).sum().to(torch.int32)

        cols = (
            torch.arange(
                L_state, device=hidden_states.device, dtype=torch.int32
            )
            + start
        ).unsqueeze(0)

        rows = torch.arange(
            S, device=hidden_states.device, dtype=torch.int32
        ).unsqueeze(1)

        mask = (rows == cols).to(padded_input.dtype)

        next_state = torch.matmul(padded_input, mask)
      else:
        next_state = padded_input[:, :, -(self.L_cache - 1) :]
    else:  # Decode
      next_state = padded_input[:, :, -(self.L_cache - 1) :]

    conv_out = self.conv(padded_input)
    conv_out = conv_out.transpose(1, 2)
    y = c * conv_out
    y = self.out_proj(y)
    past_key_values.layers[self.layer_idx].conv_states = next_state  # pyrefly: ignore[missing-attribute]
    return y
