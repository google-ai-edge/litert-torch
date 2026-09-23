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
  MULTIMODAL_LM = "multimodal_lm"
  AUTOMATIC_SPEECH_RECOGNITION = "automatic_speech_recognition"
  TEXT_TO_SPEECH = "text_to_speech"


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _parse_dtype(dtype: str) -> torch.dtype:
  """Parses a dtype string (e.g. "bfloat16") into a torch.dtype."""
  parsed = _DTYPE_ALIASES.get(str(dtype).lower().strip())
  if parsed is None:
    raise ValueError(
        f"Unsupported dtype: {dtype!r}. Supported: {sorted(_DTYPE_ALIASES)}."
    )
  return parsed


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
  cache_lengths: list[int] | None = None
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
  # Model weight/activation dtype ("float32", "float16" or "bfloat16"). If
  # None, the model is loaded in float32.
  dtype: str | None = None
  # Experimental configs
  experimental_use_mixed_precision: bool = False
  experimental_use_fp16: bool = False
  export_vision_encoder: bool = True
  export_audio_encoder: bool = True
  fuse_gate_up: bool = False
  fuse_qkv: bool = False
  use_rope_composite: bool = False
  use_swiglu_composite: bool = False
  use_qkv_norm_rope_composite: bool = False
  use_short_conv_composite: bool = False
  use_sdpa_composite: bool = False
  # Whether the prefill signature returns logits for the final position.
  # The LiteRT-LM runtime samples from the decode signature and never reads
  # the prefill logits, so emitting them adds a vocabulary-sized `lm_head`
  # matmul to every prefill call -- cheap on GPU, very visible on CPU/YNNPACK.
  #
  # Multi-output composites whose `pos=0` activation output is not part of
  # `past_key_values` (`odml.qkv_norm_rope` and `odml.short_conv`) require a
  # live consumer in the final layer so FX/MLIR DCE does not delete `pos=0`
  # while keeping `pos>=1`, which crashes `BuildStableHLOCompositePass`.
  # Note that `apply_gpu_composites` and `use_sdpa_composite` are also used by
  # YNNPACK and only emit single-output or full-cache-output composites, so
  # `prefill_logits` stays `False` for them by default.
  prefill_logits: bool | None = None
  input_sec: float = 1.0
  # If >= 0, the model runs in stateful mode after this many tokens.
  stateful_after: int = -1
  # TODO(weiyiw): Update when b/481323182 is fixed.
  # For now, for vision encoder, if there's conv op, set weight_only_wi8_afp32
  # if you intend to run on CPU, and set dynamic_wi8_afp32 if you intend to run
  # on GPU.
  vision_encoder_quantization_recipe: str | None = "dynamic_wi8_afp32"
  audio_encoder_quantization_recipe: str | None = "dynamic_wi8_afp32"
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

  # If set, the model will be exported with the specified MoE implementation.
  # However it requires transformers>=5.7.0.
  moe_exports_implementation: str | None = None

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

  # LiteRT V2 export settings
  use_v2: bool = False
  delete_in_memory_params: bool = False

  extra_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  # Internal configs
  work_dir: str | None = None
  batch_size: int = 1
  cache_length_dim: torch.export.Dim | None = None
  prefill_length_dim: torch.export.Dim | None = None
  externalize_rope: bool = False

  def __post_init__(self):
    """Refines configuration based on task-specific rules."""
    if self.prefill_logits is None:
      self.prefill_logits = bool(
          self.use_qkv_norm_rope_composite or self.use_short_conv_composite
      )
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

    if isinstance(self.cache_lengths, int):
      self.cache_lengths = [self.cache_lengths]
    elif isinstance(self.cache_lengths, str):
      self.cache_lengths = [int(x) for x in self.cache_lengths.split(",") if x]

    if not self.cache_lengths:
      self.cache_lengths = [self.cache_length]

    if self.enable_dynamic_shape and len(self.cache_lengths) > 1:
      raise ValueError(
          "Dynamic shape is not supported with multiple cache lengths."
      )
    # pylint: disable=g-bool-id-comparison
    match self.task:
      case ExportTask.IMAGE_TEXT_TO_TEXT:
        if self.export_vision_encoder:
          self.externalize_embedder = True
          self.single_token_embedder = True
        self.export_audio_encoder = False
      case ExportTask.MULTIMODAL_LM:
        if self.export_vision_encoder:
          self.externalize_embedder = True
          self.single_token_embedder = True
        if self.export_audio_encoder:
          self.externalize_embedder = True
          self.single_token_embedder = True
      case ExportTask.AUTOMATIC_SPEECH_RECOGNITION:
        self.export_vision_encoder = False
        self.split_cache = False
        self.externalize_rope = False
        self.externalize_embedder = self.bundle_litert_lm
        self.export_audio_encoder = self.bundle_litert_lm
        self.single_token_embedder = self.bundle_litert_lm
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

    if self.sliding_window_ring_buffer_size is not None:
      # Round up to the nearest 32.
      self.sliding_window_ring_buffer_size = (
          (self.sliding_window_ring_buffer_size + 31) // 32
      ) * 32

    self._validate_dtype_and_v2_options()

  def _validate_dtype_and_v2_options(self) -> None:
    """Rejects unsupported or conflicting dtype / Converter V2 options."""
    if self.dtype is not None:
      _parse_dtype(self.dtype)  # Raises on unsupported values.
      if self.experimental_use_fp16:
        raise ValueError(
            "`dtype` and `experimental_use_fp16` cannot be set together."
        )
    if self.delete_in_memory_params and not self.use_v2:
      raise ValueError("`delete_in_memory_params` requires `use_v2=True`.")
    if not self.use_v2:
      return
    unsupported = []
    if self.task != ExportTask.TEXT_GENERATION:
      unsupported.append(f"task={self.task}")
    if self.split_cache:
      unsupported.append("split_cache")
    if self.moe_exports_implementation:
      unsupported.append("moe_exports_implementation")
    if self.experimental_use_mixed_precision:
      unsupported.append("experimental_use_mixed_precision")
    if unsupported:
      raise ValueError(
          "`use_v2=True` does not support: " + ", ".join(unsupported)
      )

  def get_torch_dtype(self) -> torch.dtype:
    """Returns the dtype the source model is loaded and exported in."""
    if self.dtype is None:
      return torch.float32
    return _parse_dtype(self.dtype)

  def get_cache_dtype(self) -> torch.dtype:
    """Returns the KV cache dtype."""
    if self.experimental_use_fp16:
      return torch.float16
    return self.get_torch_dtype()

  @property
  def is_16bit(self) -> bool:
    """Returns whether the exported model uses 16-bit activations."""
    return self.get_cache_dtype() in (torch.float16, torch.bfloat16)

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
