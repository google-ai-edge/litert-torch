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
"""Sliding Window attention mask."""

import torch


def generate_causal_right_with_valid_mask(
    valid_mask, W: int | None  # pylint: disable=invalid-name
) -> torch.Tensor:
  """Generates causal mask for right."""

  L = valid_mask.shape[1]  # pylint: disable=invalid-name
  row_indices = torch.arange(L, dtype=torch.int32).unsqueeze(
      1
  )  # [L, 1] (query indices)
  col_indices = torch.arange(L, dtype=torch.int32).unsqueeze(
      0
  )  # [1, L] (key indices)

  # Query at 'i' only attends to keys at 'j' that are at or before 'i'.
  causal_mask = col_indices <= row_indices  # [L, L]

  pads = valid_mask.squeeze(0)
  mask_rows = pads.unsqueeze(1)
  mask_cols = pads.unsqueeze(0)
  padding_mask = mask_rows & mask_cols
  global_mask = causal_mask & padding_mask

  if W is not None:
    # Key at 'j' is within the past window to 'i'.
    window_lower_bound_mask = col_indices >= (row_indices - W + 1)  # [L, L]

    local_mask = global_mask & window_lower_bound_mask
    return local_mask

  return global_mask


def generate_causal_left_with_ring_buffer(
    valid_mask, W: int | None, S: int, time_step: torch.Tensor  # pylint: disable=invalid-name
) -> torch.Tensor:
  """Generates causal mask for left."""
  time_step = time_step.clone().unsqueeze(0)
  L = valid_mask.shape[1]  # pylint: disable=invalid-name
  row_indices = (
      torch.arange(L, dtype=torch.int32).unsqueeze(1) + time_step
  )  # [L, 1] (query indices)
  col_indices = (
      time_step
      - torch.tensor([1], dtype=torch.int32)
      - (
          time_step
          - torch.tensor([1], dtype=torch.int32)
          - torch.arange(S, dtype=torch.int32)
      )
      % S
  ).unsqueeze(
      0
  )  # [1, S] (key indices)

  # Query at 'i' only attends to keys at 'j' that are at or before 'i'.
  causal_mask = col_indices <= row_indices  # [L, S]
  mask_cols = (col_indices >= 0) & (col_indices < time_step)

  causal_mask &= mask_cols

  if W is not None:
    # Key at 'j' is within the past window to 'i'.
    window_lower_bound_mask = col_indices >= (row_indices - W + 1)  # [L, S]

    final_mask = causal_mask & window_lower_bound_mask
    return final_mask

  return causal_mask


def build_full_mask_with_valid_mask(
    valid_mask: torch.Tensor,
    W: int | None,  # pylint: disable=invalid-name
    S: int,  # pylint: disable=invalid-name
    time_step: torch.Tensor,
    use_bool_mask: bool = False,
):
  """Builds full attention mask."""
  left_mask = generate_causal_left_with_ring_buffer(valid_mask, W, S, time_step)
  right_mask = generate_causal_right_with_valid_mask(valid_mask, W)
  mask = torch.cat([left_mask, right_mask], dim=-1)
  mask = torch.logical_not(mask.unsqueeze(0).unsqueeze(0))
  if use_bool_mask:
    return mask
  else:
    return mask * -1e4
