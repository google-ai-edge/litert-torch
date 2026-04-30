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
import transformers
import torch

modeling_lfm2 = transformers.models.lfm2.modeling_lfm2


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

  def forward(
      self,
      hidden_states: torch.Tensor,
      past_key_values=None,
      cache_position: Optional[torch.LongTensor] = None,
      attention_mask: Optional[torch.Tensor] = None,
  ):
    x = modeling_lfm2.apply_mask_to_padding_states(
        hidden_states, attention_mask
    )
    b, c, x_proj = self.in_proj(x).chunk(3, dim=-1)
    conv_input = b * x_proj
    conv_input_t = conv_input.transpose(1, 2)
    state = past_key_values.layers[self.layer_idx].conv_state
    padded_input = torch.cat([state, conv_input_t], dim=-1)
    next_state = padded_input[:, :, -(self.L_cache - 1) :]
    conv_out = self.conv(padded_input)
    conv_out = conv_out.transpose(1, 2)
    y = c * conv_out
    y = self.out_proj(y)
    past_key_values.layers[self.layer_idx].conv_state = next_state
    return y
