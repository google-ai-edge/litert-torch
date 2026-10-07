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
"""Extension for HF integration."""

from typing import Any
from litert_torch.generative.export_hf.core import exportable_module_config

_QWEN3_GPU_PRESET: dict[str, bool] = {
    "fuse_qkv": True,
    "fuse_gate_up": True,
    "use_qkv_norm_rope_composite": True,
    "use_swiglu_composite": True,
    "use_sdpa_composite": True,
    "use_bool_mask": True,
    "enable_gpu_dynamic_cache": True,
}

_QWEN3_5_GPU_PRESET: dict[str, bool] = {
    "fuse_qkv": True,
    "fuse_gate_up": True,
    "use_rope_composite": True,
    "use_swiglu_composite": True,
    "use_sdpa_composite": True,
    "use_bool_mask": True,
    "enable_gpu_dynamic_cache": True,
}

_LFM2_GPU_PRESET: dict[str, bool] = {
    "fuse_qkv": True,
    "fuse_gate_up": True,
    "use_qkv_norm_rope_composite": True,
    "use_swiglu_composite": True,
    "use_sdpa_composite": True,
    "use_short_conv_composite": True,
    "use_bool_mask": True,
    "enable_gpu_dynamic_cache": True,
}

_GEMMA4_GPU_PRESET: dict[str, bool] = {
    "fuse_qkv": True,
    "fuse_gate_up": True,
    "use_qkv_norm_rope_composite": True,
    "use_swiglu_composite": True,
    "use_sdpa_composite": True,
    "use_bool_mask": True,
    "enable_gpu_dynamic_cache": True,
}

_GEMMA3_GPU_PRESET: dict[str, bool] = {
    "fuse_qkv": True,
    "fuse_gate_up": True,
    "use_rope_composite": True,
    "use_sdpa_composite": True,
    "use_bool_mask": True,
    "enable_gpu_dynamic_cache": True,
}

_MODEL_GPU_COMPOSITE_PRESETS: dict[str, dict[str, bool]] = {
    "qwen3": _QWEN3_GPU_PRESET,
    "qwen3_text": _QWEN3_GPU_PRESET,
    "qwen3_5": _QWEN3_5_GPU_PRESET,
    "qwen3_5_text": _QWEN3_5_GPU_PRESET,
    "lfm2": _LFM2_GPU_PRESET,
    "lfm2_vl": _LFM2_GPU_PRESET,
    "gemma4": _GEMMA4_GPU_PRESET,
    "gemma4_text": _GEMMA4_GPU_PRESET,
    "gemma3": _GEMMA3_GPU_PRESET,
    "gemma3_text": _GEMMA3_GPU_PRESET,
}


def _resolve_model_type(
    model_config: Any,
) -> str | None:
  """Extracts the primary or text-backbone model_type from a PretrainedConfig."""
  if model_config is None:
    return None
  model_type = getattr(model_config, "model_type", None)
  if model_type in _MODEL_GPU_COMPOSITE_PRESETS:
    return model_type
  text_config = getattr(model_config, "text_config", None)
  text_model_type = getattr(text_config, "model_type", None)
  if text_model_type in _MODEL_GPU_COMPOSITE_PRESETS:
    return text_model_type
  return model_type or text_model_type


def update_export_config(
    export_config: exportable_module_config.ExportableModuleConfig,
    model_config: Any,
) -> exportable_module_config.ExportableModuleConfig:
  """Updates export config with model-family GPU optimization presets."""
  explicit_fields = getattr(export_config, "_explicit_fields", None)
  apply_preset = bool(
      export_config.apply_gpu_composites
      and (explicit_fields is None or "apply_gpu_composites" in explicit_fields)
  )
  if not apply_preset:
    return export_config

  model_type = _resolve_model_type(model_config)
  preset = _MODEL_GPU_COMPOSITE_PRESETS.get(model_type or "")
  if preset is None:
    return export_config

  explicit = explicit_fields or set()
  for flag_name, default_val in preset.items():
    if flag_name in explicit:
      continue
    if flag_name == "use_qkv_norm_rope_composite":
      if export_config.use_rope_composite:
        continue
      if "fuse_qkv" in explicit and not export_config.fuse_qkv:
        continue
    if flag_name == "use_swiglu_composite":
      if "fuse_gate_up" in explicit and not export_config.fuse_gate_up:
        continue
    if flag_name == "enable_gpu_dynamic_cache":
      if export_config.enable_dynamic_shape:
        continue
    setattr(export_config, flag_name, default_val)

  export_config.resolve_gpu_composite_dependencies()
  return export_config
