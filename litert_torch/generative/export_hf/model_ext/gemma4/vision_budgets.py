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
"""Gemma 4 vision soft-token budget flag parsing.

Gemma 4 supports variable image resolution through a per-image "soft token
budget" (see
https://ai.google.dev/gemma/docs/capabilities/vision#variable-resolution). Each
exported budget `N` becomes a `vision_{N}` signature in `vision_encoder.tflite`
(and a `vision_adapter_{N}` signature in `vision_adapter.tflite`); the LiteRT-LM
runtime selects the smallest signature that fits the preprocessed image.
"""

from collections.abc import Mapping, Sequence
from typing import Any

# Soft-token budgets supported by the Gemma 4 image processor.
SUPPORTED_VISION_TOKEN_BUDGETS: tuple[int, ...] = (70, 140, 280, 560, 1120)

DEFAULT_VISION_TOKEN_BUDGETS: tuple[int, ...] = (140,)

# CLI / `export(...)` flag selecting the budgets to export, e.g.
# `--gemma4_vision_token_budgets=70,140,280,560,1120`.
VISION_TOKEN_BUDGETS_FLAG = 'gemma4_vision_token_budgets'
# Legacy single-budget flag, still honored when `VISION_TOKEN_BUDGETS_FLAG` is
# not set.
LEGACY_VISION_MAX_SOFT_TOKENS_FLAG = 'gemma4_vision_max_soft_tokens'

EXTRA_KWARGS_KEYS: tuple[str, ...] = (
    VISION_TOKEN_BUDGETS_FLAG,
    LEGACY_VISION_MAX_SOFT_TOKENS_FLAG,
)


def _parse_budgets(value: Any, flag_name: str) -> tuple[int, ...]:
  """Parses an int, comma-separated string, or sequence into budgets."""
  if isinstance(value, bool):
    raise ValueError(f'`{flag_name}` must be an int or a list of ints.')
  if isinstance(value, int):
    raw = [value]
  elif isinstance(value, str):
    raw = [
        v for v in value.strip().strip('[]()').replace(' ', '').split(',') if v
    ]
  elif isinstance(value, Sequence):
    raw = list(value)
  else:
    raise ValueError(
        f'`{flag_name}` must be an int or a list of ints, got {value!r}.'
    )
  try:
    budgets = [int(v) for v in raw]
  except (TypeError, ValueError) as e:
    raise ValueError(
        f'`{flag_name}` must be an int or a list of ints, got {value!r}.'
    ) from e
  if not budgets:
    raise ValueError(f'`{flag_name}` must specify at least one budget.')
  unsupported = sorted(
      set(b for b in budgets if b not in SUPPORTED_VISION_TOKEN_BUDGETS)
  )
  if unsupported:
    raise ValueError(
        f'Unsupported Gemma 4 vision token budget(s) {unsupported} in'
        f' `{flag_name}`; supported budgets are'
        f' {list(SUPPORTED_VISION_TOKEN_BUDGETS)}.'
    )
  return tuple(sorted(set(budgets)))


def get_vision_token_budgets(kwargs: Mapping[str, Any]) -> tuple[int, ...]:
  """Returns the sorted, deduplicated vision soft-token budgets to export.

  Reads `gemma4_vision_token_budgets` (preferred) or the legacy
  `gemma4_vision_max_soft_tokens` from `kwargs` (typically
  `ExportableModuleConfig.extra_kwargs`). Each accepts an int (`70`), a
  comma-separated string (`'70,140'`), or a sequence of ints (`[70, 140]`).
  Defaults to `DEFAULT_VISION_TOKEN_BUDGETS` when neither flag is set.

  Args:
    kwargs: Mapping holding the export flags.

  Returns:
    Tuple of budgets in ascending order.

  Raises:
    ValueError: If a budget is malformed or unsupported, or both flags are set.
  """
  budgets = kwargs.get(VISION_TOKEN_BUDGETS_FLAG)
  legacy = kwargs.get(LEGACY_VISION_MAX_SOFT_TOKENS_FLAG)
  if budgets is not None and legacy is not None:
    raise ValueError(
        f'Specify only one of `{VISION_TOKEN_BUDGETS_FLAG}` and'
        f' `{LEGACY_VISION_MAX_SOFT_TOKENS_FLAG}`.'
    )
  if budgets is not None:
    return _parse_budgets(budgets, VISION_TOKEN_BUDGETS_FLAG)
  if legacy is not None:
    return _parse_budgets(legacy, LEGACY_VISION_MAX_SOFT_TOKENS_FLAG)
  return DEFAULT_VISION_TOKEN_BUDGETS
