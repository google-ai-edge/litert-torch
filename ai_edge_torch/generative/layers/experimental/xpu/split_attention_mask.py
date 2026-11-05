# Copyright 2025 The AI Edge Torch Authors.
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
#

"""Split attention mask."""


import torch
from torch import nn


class SplitAttentionMask(nn.Module):
  """Split attention mask."""

  def __init__(self, context_size):
    super().__init__()
    self.context_size = context_size

  def build_mask(self, num_input_tokens):
    row_index, col_index = torch.meshgrid(
        torch.arange(self.context_size),
        torch.arange(self.context_size),
        indexing='ij',
    )
    mask = torch.logical_or(
        torch.logical_or(
            torch.less(row_index, col_index),
            torch.greater_equal(col_index, num_input_tokens),
        ),
        torch.greater_equal(row_index, num_input_tokens),
    )
    mask = torch.logical_not(mask)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

  def forward(self, input_tokens, time_step):
    # input_tokens: [1, T]
    # time_step: []
    paddings = torch.sum(input_tokens == 0, dim=-1)
    paddings = paddings[0]
    _, T = input_tokens.shape  # pylint: disable=invalid-name

    num_paddings = torch.sum(paddings)
    num_input_tokens = T - num_paddings

    orig_mask = self.build_mask(num_input_tokens + time_step)
    orig_mask = orig_mask[:, :, time_step : time_step + T, :]

    new_mask = orig_mask[:, :, :, time_step : time_step + T].clone()
    zeroed_mask = torch.zeros_like(new_mask)

    orig_mask[:, :, :, time_step : time_step + T] = zeroed_mask

    mask = torch.cat([orig_mask, new_mask], dim=-1)

    return {'mask': mask}
