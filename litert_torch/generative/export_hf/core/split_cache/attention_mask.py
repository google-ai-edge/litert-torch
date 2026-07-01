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
"""Split attention mask."""

import dataclasses
from typing import Any, List, Tuple
import torch
from torch import nn
import torch.utils._pytree as pytree


# TODO(weiyiw) - Move this class to a common place.
@dataclasses.dataclass
class Mask:
  """Attention mask."""

  mask: torch.Tensor | None
  local_masks: dict[int, torch.Tensor] | torch.Tensor | None

  @classmethod
  def _create_mask(cls, mask_len, kv_cache_max_len):
    mask = torch.full(
        (mask_len, kv_cache_max_len),
        float('-inf'),
        dtype=torch.float32,
    )
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    return mask


@dataclasses.dataclass
class SplitMask(Mask):
  """Split attention mask."""

  @classmethod
  def _create_mask(cls, mask_len, kv_cache_max_len):
    mask = torch.full(
        (mask_len, kv_cache_max_len + mask_len),
        float('-inf'),
        dtype=torch.float32,
    )
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    return mask


_MaskPyTreeContext = list[str]


def _flatten_mask(mask: Mask) -> Tuple[List[torch.Tensor], _MaskPyTreeContext]:
  """Flattens a mask into a list of tensors and a context."""
  if mask.mask is None:
    flattened = []
    flat_names = []
  else:
    flattened = [mask.mask]
    flat_names = ['global']
  if mask.local_masks is not None:
    if isinstance(mask.local_masks, dict):
      for window_size, local_mask in mask.local_masks.items():
        flattened.append(local_mask)
        flat_names.append(f'local_{window_size}')
    else:
      flattened.append(mask.local_masks)
      flat_names.append('local')
  return flattened, flat_names


def _unflatten_mask(
    values: List[torch.Tensor],
    context: _MaskPyTreeContext,
) -> Mask:
  """Unflattens a list of tensors into a mask."""
  flat_names = context
  global_mask = None
  local_masks = {}
  for flat_name in flat_names:
    if flat_name == str('global'):
      global_mask = values[flat_names.index(flat_name)]
      continue
    if flat_name == 'local':
      local_masks = values[flat_names.index(flat_name)]
      continue
    assert isinstance(
        local_masks, dict
    ), 'local_masks should be a dict when there are multiple local masks.'
    assert flat_name.startswith('local'), f'Invalid flat_name: {flat_name}'
    window_size = int(flat_name.split('_')[-1])
    local_masks[window_size] = values[flat_names.index(flat_name)]
  if not len(local_masks):
    local_masks = None
  return Mask(mask=global_mask, local_masks=local_masks)


def _flatten_mask_with_keys(
    mask: Mask,
) -> tuple[list[tuple[pytree.KeyEntry, Any]], Any]:
  flattened, (flat_names, _) = _flatten_mask(mask)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


pytree.register_pytree_node(
    Mask,
    _flatten_mask,
    _unflatten_mask,
    flatten_with_keys_fn=_flatten_mask_with_keys,
    serialized_type_name='',
)

pytree.register_pytree_node(
    SplitMask,
    _flatten_mask,
    _unflatten_mask,
    flatten_with_keys_fn=_flatten_mask_with_keys,
    serialized_type_name='',
)


def generate_causal_right(
    input_tokens, W: int | None, valid_mask: torch.Tensor  # pylint: disable=invalid-name
) -> torch.Tensor:
  """Generates causal mask for right."""

  L = input_tokens.shape[1]  # pylint: disable=invalid-name
  row_indices = torch.arange(L, dtype=torch.int32).unsqueeze(
      1
  )  # [L, 1] (query indices)
  col_indices = torch.arange(L, dtype=torch.int32).unsqueeze(
      0
  )  # [1, L] (key indices)

  # Query at 'i' only attends to keys at 'j' that are at or before 'i'.
  causal_mask = col_indices <= row_indices  # [L, L]

  pads = valid_mask[0]
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


def generate_causal_left(
    input_tokens, W: int | None, S: int, time_step: torch.Tensor  # pylint: disable=invalid-name
) -> torch.Tensor:
  """Generates causal mask for left."""
  L = input_tokens.shape[1]  # pylint: disable=invalid-name
  row_indices = (
      torch.arange(L, dtype=torch.int32).unsqueeze(1) + time_step
  )  # [L, 1] (query indices)
  col_indices = torch.arange(S, dtype=torch.int32).unsqueeze(
      0
  )  # [1, S] (key indices)

  # Query at 'i' only attends to keys at 'j' that are at or before 'i'.
  causal_mask = col_indices <= row_indices  # [L, L]
  mask_cols = col_indices < time_step

  causal_mask &= mask_cols

  if W is not None:
    # Key at 'j' is within the past window to 'i'.
    window_lower_bound_mask = col_indices >= (row_indices - W + 1)  # [L, L]

    final_mask = causal_mask & window_lower_bound_mask
    return final_mask

  return causal_mask


def build_full_mask(
    input_tokens: torch.Tensor,
    W: int | None,  # pylint: disable=invalid-name
    S: int,  # pylint: disable=invalid-name
    time_step: torch.Tensor,
    valid_mask: torch.Tensor,
):
  """Builds full attention mask."""
  left_mask = generate_causal_left(input_tokens, W, S, time_step)
  right_mask = generate_causal_right(input_tokens, W, valid_mask)

  mask = torch.cat([left_mask, right_mask], dim=-1)
  mask = torch.logical_not(mask.unsqueeze(0).unsqueeze(0)) * -100.0
  return mask


class SplitAttentionMask(nn.Module):
  """Split attention mask."""

  def __init__(
      self,
      context_size: int,
      sliding_window_size: int | None = None,
  ):
    super().__init__()
    self.context_size = context_size
    self.sliding_window_size = sliding_window_size

  def forward(self, input_tokens, time_step, valid_mask):
    # input_tokens: [1, T]
    # valid_mask: [1, T]
    # time_step: []
    return build_full_mask(
        input_tokens,
        self.sliding_window_size,
        self.context_size,
        time_step,
        valid_mask,
    )
