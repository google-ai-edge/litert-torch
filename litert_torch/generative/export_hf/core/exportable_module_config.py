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
import enum
import pprint
from typing import Any

import torch


class ExportTask(str, enum.Enum):
  TEXT_GENERATION = "text_generation"
  IMAGE_TEXT_TO_TEXT = "image_text_to_text"


@dataclasses.dataclass
class ExportableModuleConfig:
  """Config for exportable modules."""

  model: str
  output_dir: str | None = None
  task: ExportTask | str = ExportTask.TEXT_GENERATION
  keep_temporary_files: bool = False
  trust_remote_code: bool = False
  prefill_lengths: list[int] = dataclasses.field(default_factory=lambda: [128])
  cache_length: int = 4096
  # For quantization
  quantization_recipe: str | None = "dynamic_wi8_afp32"
  # For dynamic shape
  enable_dynamic_shape: bool = False
  # Export configs
  externalize_embedder: bool = False
  single_token_embedder: bool = False
  k_ts_idx: int = 2
  v_ts_idx: int = 3
  split_cache: bool = False
  cache_implementation: str | None = None
  auto_model_override: str | None = None
  use_jinja_template: bool = True
  bundle_litert_lm: bool = True
  # Experimental configs
  experimental_use_mixed_precision: bool = False
  export_vision_encoder: bool = True
  # TODO(weiyiw): Update when b/481323182 is fixed.
  # For now, for vision encoder, if there's conv op, set weight_only_wi8_afp32
  # if you intend to run on CPU, and set dynamic_wi8_afp32 if you intend to run
  # on GPU.
  vision_encoder_quantization_recipe: str | None = "dynamic_wi8_afp32"
  litert_lm_model_type_override: str | None = None
  litert_lm_llm_metadata_override: str | None = None
  sampler_top_p: float | None = None
  sampler_temperature: float | None = None
  sampler_top_k: int | None = None
  jinja_chat_template_override: str | None = None

  experimental_lightweight_conversion: bool = False
  experimental_transpile_chat_template_for_minijinja: bool = False

  # AOT Compilation.
  aot_backend: str | None = None
  aot_soc_model: str | None = None
  aot_compilation_config_dict: dict[str, Any] | None = None

  extra_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Internal configs
  work_dir: str | None = None
  batch_size: int = 1
  cache_length_dim: torch.export.Dim | None = None
  prefill_length_dim: torch.export.Dim | None = None
  externalize_rope: bool = False

  def __post_init__(self):
    """Refines configuration based on task-specific rules."""
    # pylint: disable=g-bool-id-comparison
    if self.split_cache:
      self.externalize_embedder = True
      self.externalize_rope = True
      if self.cache_implementation is None:
        self.cache_implementation = "LiteRTLMSplitCache"

    match self.task:
      case ExportTask.IMAGE_TEXT_TO_TEXT:
        if self.export_vision_encoder:
          self.externalize_embedder = True
          self.single_token_embedder = True
      case _:
        self.export_vision_encoder = False

    if self.enable_dynamic_shape:
      self.prefill_length_dim = torch.export.Dim(
          "prefill_length", min=1, max=self.cache_length
      )
      self.cache_length_dim = torch.export.Dim("cache_length")

    if self.cache_implementation is None:
      self.cache_implementation = "LiteRTLMCache"
    # pylint: enable=g-bool-id-comparison

  def __repr__(self):
    """Returns a pretty-printed string representation of the config."""

    data = dataclasses.asdict(self)
    lines = [f"{' Export Configuration ':=^50}"]
    for key, value in sorted(data.items()):
      val_str = pprint.pformat(value, width=60, compact=True)
      if "\n" in val_str:
        val_str = val_str.replace("\n", "\n" + " " * 25)
      lines.append(f"{key:<22} : {val_str}")
    lines.append("=" * 50)

    return "\n".join(lines)

  def print_summary(self):
    """Directly prints the formatted configuration."""
    print(self.__repr__())
