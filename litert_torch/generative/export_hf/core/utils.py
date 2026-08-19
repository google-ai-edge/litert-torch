# Copyright 2025 The LiteRT Torch Authors.
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
"""Utility functions."""

import os
import re
import torch


# TODO(weiyiw): Remove this when we support external sliding mask.
def create_sliding_mask(
    segment_pos: torch.Tensor,  # [B, L]
    cache_len: int,
    sliding_window_size: int,
    mask_value: float = -1e4,
    use_bool_mask: bool = False,
) -> torch.Tensor:
  """Creates mask for sliding window attention (PyTorch)."""
  # Use torch.arange to create a tensor with a range of integers in a
  # Dynamo-friendly way.
  cache_positions = torch.arange(cache_len, dtype=torch.int32)
  cache_positions = cache_positions.view(1, 1, -1)  # [1, 1, cache_len]
  segment_pos_expanded = segment_pos.clone().unsqueeze(-1)  # [B, seq_len, 1]

  # Create boolean masks for window boundaries.
  if use_bool_mask:
    left_boundary = segment_pos_expanded >= cache_positions
    right_boundary = (
        segment_pos_expanded < cache_positions + sliding_window_size
    )
  else:
    left_boundary = cache_positions > segment_pos_expanded - sliding_window_size
    right_boundary = (
        cache_positions < segment_pos_expanded + sliding_window_size
    )

  # Combine boolean masks (AND).
  sliding_mask_bool = left_boundary & right_boundary

  if use_bool_mask:
    return sliding_mask_bool.unsqueeze(0)

  # Convert boolean mask to float mask with 0 and -inf.
  sliding_mask = torch.where(
      sliding_mask_bool,
      torch.zeros_like(sliding_mask_bool, dtype=torch.float),
      torch.full_like(sliding_mask_bool, mask_value, dtype=torch.float),
  )

  return sliding_mask.unsqueeze(0)


WARNING_MESSAGE = r"""
                   .
                  / \
                 / ! \
                /  .  \
               /_______\
              / WARNING \
             /___________\
            /_____________\

"""


ERROR_MESSAGE = r"""
                   .
                  / \
                 / ! \
                /  .  \
               /_______\
              /  ERROR  \
             /___________\
            /_____________\
           /_______________\

"""


def has_local_rope(model):
  config = model.config
  if hasattr(config, 'text_config'):
    config = config.text_config
  return 'sliding_attention' in config.rope_parameters


def has_sliding_attention(model):
  config = model.config
  if hasattr(config, 'text_config'):
    config = config.text_config
  sliding_window = getattr(config, 'sliding_window', None)
  if not sliding_window:
    return False
  layer_types = getattr(config, 'layer_types', None)
  return layer_types is not None and 'sliding_attention' in layer_types


def get_model_path_type(path_str: str) -> str:
  """Determines if a string is a local path or a Hugging Face Repo ID.

  Args:
      path_str: The string to check.

  Returns:
      "local": If the path exists on disk.
      "repo_id": If it looks like a Hub ID (e.g., 'meta-llama/Llama-2-7b').
      "local_not_found": If it looks like a file path but doesn't exist.
      "unknown": If it matches neither pattern clearly.
  """
  # 1. Absolute Truth: Does it exist on the disk?
  if os.path.exists(path_str):
    return 'local'

  # 2. Heuristic: Does it have explicit path markers?
  # Starts with "./", "/", "~", or contains Windows backslashes
  if path_str.startswith(('.', '/', '~')) or '\\' in path_str:
    return 'local_not_found'

  # 3. Heuristic: Does it look like a Repo ID?
  # Pattern: username/repo_name (e.g. "mistralai/Mistral-7B")
  # or just repo_name for official models (e.g. "gpt2", "bert-base-uncased")
  # Allowed chars: Alphanumeric, underscores, hyphens, periods.
  repo_id_pattern = r'^(?:[\w\-\.]+\/)?[\w\-\.]+$'
  if re.match(repo_id_pattern, path_str):
    return 'repo_id'

  return 'unknown'


def is_magic_number_(num: int) -> bool:
  """Returns true if the number is a magic number, i.e. prime number > 10."""
  if num < 10:
    return False
  if num % 2 == 0:
    return False
  for i in range(3, int(num / 2), 2):
    if num % i == 0:
      return False
  return True


def get_magic_number_for(org_number: int) -> int:
  """Returns the magic number for the given original number."""
  while not is_magic_number_(org_number):
    org_number += 1
  return org_number

