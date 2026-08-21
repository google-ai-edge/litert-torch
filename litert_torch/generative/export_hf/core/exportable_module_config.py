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

from litert_torch.generative.export_hf.core import utils
from litert_torch.generative.export_hf.experimental.npu_export.configs import vendor_configs
import torch


class ExportTask(str, enum.Enum):
  TEXT_GENERATION = "text_generation"
  IMAGE_TEXT_TO_TEXT = "image_text_to_text"
  AUTOMATIC_SPEECH_RECOGNITION = "automatic_speech_recognition"
  TEXT_TO_SPEECH = "text_to_speech"


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
  sliding_window_ring_buffer_size: int | None = None
  # For quantization
  quantization_recipe: str | None = "dynamic_wi8_afp32"
  # For dynamic shape
  enable_dynamic_shape: bool = False
  # If True, prefill lengths are adjusted to magic numbers for GPU execution.
  enable_gpu_dynamic_prefill: bool = False
  # If True, cache length is adjusted to a magic number for GPU execution.
  enable_gpu_dynamic_cache: bool = False
  # Export configs
  externalize_embedder: bool = False
  single_token_embedder: bool = False
  k_ts_idx: int | None = None
  v_ts_idx: int | None = None
  split_cache: bool = False
  cache_implementation: str | None = None
  auto_model_override: str | None = None
  use_jinja_template: bool = True
  bundle_litert_lm: bool = True
  # Experimental configs
  experimental_use_mixed_precision: bool = False
  experimental_use_fp16: bool = False
  export_vision_encoder: bool = True
  export_audio_encoder: bool = True
  fuse_gate_up: bool = False
  fuse_qkv: bool = False
  use_rope_composite: bool = False
  use_qkv_norm_rope_composite: bool = False
  use_sdpa_composite: bool = False
  input_sec: float = 1.0
  # If >= 0, the model runs in stateful mode after this many tokens.
  stateful_after: int = -1
  # TODO(weiyiw): Update when b/481323182 is fixed.
  # For now, for vision encoder, if there's conv op, set weight_only_wi8_afp32
  # if you intend to run on CPU, and set dynamic_wi8_afp32 if you intend to run
  # on GPU.
  vision_encoder_quantization_recipe: str | None = "dynamic_wi8_afp32"
  litert_lm_model_type_override: str | None = None
  litert_lm_llm_metadata_override: str | None = None
  tokenizer_path_override: str | None = None
  llm_metadata_max_num_tokens_override: int | None = None
  sampler_top_p: float | None = None
  sampler_temperature: float | None = None
  sampler_top_k: int | None = None
  jinja_chat_template_override: str | None = None
  use_random_weights: bool = False

  experimental_lightweight_conversion: bool = False
  experimental_transpile_chat_template_for_minijinja: bool = False

  assistant_model: str | None = None
  mtp_verifier_step: int = 5

  moe_exports_implementation: str = "litert_moe"

  # AOT Compilation.
  aot_backend: str | None = None
  aot_soc_model: str | None = None
  aot_compilation_config_dict: dict[str, Any] | None = None

  # New Compiler & Pipeline settings
  use_litert_lm_compiler: bool = False
  compile_configs: str | None = None

  # Calibration & Static Quantization settings
  calibration_dataset_dir: str | None = None
  calibration_dataset_format: str = "jsonl"
  calibration_eval_task_names: str | list[str] = "ALL"
  max_calibration_decode_steps: int = 32
  calibration_range_scale: float = 1.0
  use_float_input_output_normalizer: bool = False
  skip_mlir_passes: bool = True
  use_profiler_based_calibration: bool = True
  enable_min_max_calibration_update: bool = True
  ema_smoothing_factor: float = 0.1
  static_quantization_recipe: str | None = None

  extra_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Internal configs
  work_dir: str | None = None
  batch_size: int = 1
  cache_length_dim: torch.export.Dim | None = None
  prefill_length_dim: torch.export.Dim | None = None
  externalize_rope: bool = False

  def __post_init__(self):
    """Refines configuration based on task-specific rules."""
    if self.aot_backend:
      backend_clean = self.aot_backend.lower()
      if backend_clean in vendor_configs.VENDOR_CONFIGS:
        defaults = vendor_configs.VENDOR_CONFIGS[backend_clean]
        if self.k_ts_idx is None:
          self.k_ts_idx = defaults.get("k_ts_idx")
        if self.v_ts_idx is None:
          self.v_ts_idx = defaults.get("v_ts_idx")
        self.split_cache = True
        self.externalize_embedder = True

    if self.k_ts_idx is None:
      self.k_ts_idx = 2
    if self.v_ts_idx is None:
      self.v_ts_idx = 3

    if isinstance(self.prefill_lengths, int):
      self.prefill_lengths = [self.prefill_lengths]
    elif isinstance(self.prefill_lengths, str):
      self.prefill_lengths = [
          int(x) for x in self.prefill_lengths.split(",") if x
      ]

    # pylint: disable=g-bool-id-comparison
    match self.task:
      case ExportTask.IMAGE_TEXT_TO_TEXT:
        if self.export_vision_encoder:
          self.externalize_embedder = True
          self.single_token_embedder = True
        self.export_audio_encoder = False
      case ExportTask.AUTOMATIC_SPEECH_RECOGNITION:
        self.export_vision_encoder = False
        self.split_cache = False
        self.externalize_embedder = False
        self.externalize_rope = False
        self.bundle_litert_lm = False
      case ExportTask.TEXT_TO_SPEECH:
        self.export_vision_encoder = False
        self.export_audio_encoder = False
        self.split_cache = False
        self.externalize_embedder = True
        self.single_token_embedder = True
        self.externalize_rope = False
        self.bundle_litert_lm = False
      case _:
        self.export_vision_encoder = False
        self.export_audio_encoder = False

    if self.split_cache:
      self.externalize_embedder = True
      self.externalize_rope = True
      if self.cache_implementation is None:
        self.cache_implementation = "LiteRTLMSplitCache"
      self.moe_exports_implementation = "litert_moe_sequential"

    if self.enable_gpu_dynamic_prefill or self.enable_gpu_dynamic_cache:
      if self.enable_dynamic_shape:
        raise ValueError(
            "enable_dynamic_shape and enable_gpu_dynamic_prefill/cache"
            " cannot be both True."
        )
    if self.enable_gpu_dynamic_prefill:
      self.prefill_lengths = [
          utils.get_magic_number_for(l) for l in self.prefill_lengths
      ]
    if self.enable_gpu_dynamic_cache:
      self.cache_length = utils.get_magic_number_for(self.cache_length)

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
