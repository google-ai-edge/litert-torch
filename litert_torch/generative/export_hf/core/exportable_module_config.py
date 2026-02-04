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
"""Exportable modules."""

import dataclasses
import torch


@dataclasses.dataclass
class ExportableModuleConfig:
  """Config for exportable modules."""

  batch_size: int = 1
  cache_length: int = 1280
  prefill_lengths: list[int] = dataclasses.field(default_factory=lambda: [128])
  # For dynamic shape
  cache_length_dim: torch.export.Dim | None = None
  prefill_length_dim: torch.export.Dim | None = None

  # Export configs
  externalize_embedder: bool = False
  single_token_embedder: bool = False
  externalize_rope: bool = False

  split_cache: bool = False
  cache_implementation: str = "LiteRTLMCache"
  k_ts_idx: int = 2
  v_ts_idx: int = 3
