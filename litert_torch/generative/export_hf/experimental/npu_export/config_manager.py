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
"""Hierarchical Pipeline Configuration Manager for LiteRT NPU Export."""

import dataclasses
import json
import os
from typing import Any

from litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler import soc_utils
from litert_torch.generative.export_hf.experimental.npu_export.configs import vendor_configs

gfile = None


def _open(path: str, mode: str = "r"):
  if gfile:
    return gfile.Open(path, mode)
  return open(path, mode)


@dataclasses.dataclass
class NpuPipelineConfig:
  """Unified end-to-end pipeline configuration for LiteRT NPU export."""

  model_id: str
  backend: str
  soc_model: str
  overwrite: bool = True
  weight_quantization_recipe: str = "dynamic_wi8_afp32"
  use_16bits_activations: bool = True
  k_ts_idx: int = 3
  v_ts_idx: int = 2
  allow_float_operations: bool = False
  skip_mlir_passes: bool = True
  prefill_lengths: list[int] = dataclasses.field(default_factory=lambda: [128])
  cache_length: int = 1280
  max_decode_steps: int = 32

  # Stage 2 Calibration Configuration Defaults
  calib_dataset_format: str = "jsonl"
  calib_eval_task_names: str | list[str] = "ALL"
  use_profiler_based_calibration: bool = True
  enable_min_max_calibration_update: bool = True
  ema_smoothing_factor: float = 0.1

  # Stage 3 Static Range Quantization Defaults
  align_kv_cache: bool = True
  calibration_range_scale: float = 1.0

  @property
  def quantization_recipe(self) -> str:
    """Backward-compatible property alias pointing to weight_quantization_recipe."""
    return self.weight_quantization_recipe

  @quantization_recipe.setter
  def quantization_recipe(self, value: str) -> None:
    self.weight_quantization_recipe = value

  @property
  def a16w8(self) -> bool:
    """Backward-compatible property alias pointing to use_16bits_activations."""
    return self.use_16bits_activations

  @a16w8.setter
  def a16w8(self, value: bool) -> None:
    self.use_16bits_activations = value

  def print_summary(self) -> None:
    """Prints a formatted summary table of all resolved configuration fields."""
    print("=== Resolved NPU Pipeline Configuration ===")
    print(f"  Model ID              : {self.model_id}")
    print(
        f"  Target Hardware       : {self.backend.upper()} ({self.soc_model})"
    )
    printed = {
        "model_id",
        "backend",
        "soc_model",
        "quantization_recipe",
        "a16w8",
    }
    for field in dataclasses.fields(self):
      if field.name in printed:
        continue
      val = getattr(self, field.name)
      label = field.name.replace("_", " ").title()
      print(f"  {label:<22}: {val}")
      printed.add(field.name)
    for k, val in self.__dict__.items():
      if k in printed:
        continue
      label = k.replace("_", " ").title()
      print(f"  {label:<22}: {val}")
    print("===========================================")

  def to_dict(self) -> dict[str, Any]:
    """Returns a dictionary containing all dataclass fields plus any extra instance attributes."""
    d = dataclasses.asdict(self)
    for k, v in self.__dict__.items():
      if k not in d and not k.startswith("_"):
        d[k] = v
    return d

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "NpuPipelineConfig":
    """Reconstructs NpuPipelineConfig from a dictionary, attaching extra attributes dynamically."""
    data = dict(data)
    if (
        "quantization_recipe" in data
        and "weight_quantization_recipe" not in data
    ):
      data["weight_quantization_recipe"] = data.pop("quantization_recipe")
    if "a16w8" in data and "use_16bits_activations" not in data:
      data["use_16bits_activations"] = data.pop("a16w8")
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    clean_kwargs = {k: v for k, v in data.items() if k in valid_fields}
    cfg = cls(**clean_kwargs)
    for k, v in data.items():
      if k not in valid_fields and not k.startswith("_") and k != "a16w8":
        setattr(cfg, k, v)
    return cfg

  def to_json(self, indent: int = 2) -> str:
    """Serializes the configuration to a JSON string."""
    return json.dumps(self.to_dict(), indent=indent)

  @classmethod
  def from_json(cls, json_str: str) -> "NpuPipelineConfig":
    """Deserializes NpuPipelineConfig from a JSON string."""
    return cls.from_dict(json.loads(json_str))

  def save_json(self, filepath: str) -> None:
    """Saves the serialized JSON configuration snapshot directly to disk at filepath."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with _open(filepath, "w") as f:
      f.write(self.to_json())

  @classmethod
  def load_json(cls, filepath: str) -> "NpuPipelineConfig":
    """Loads a serialized JSON configuration snapshot directly from filepath."""
    with _open(filepath, "r") as f:
      return cls.from_json(f.read())


def build_pipeline_config(
    model_id: str,
    target_vendor: str,
    target_soc: str,
    prefill_lengths: list[int] | None = None,
    cache_length: int | None = None,
    max_decode_steps: int | None = None,
    **user_overrides: Any,
) -> NpuPipelineConfig:
  """Factory function that builds a validated and resolved NpuPipelineConfig.

  Args:
    model_id: HuggingFace model identifier (e.g. 'google/gemma-3-270m-it').
    target_vendor: Key in VENDOR_CONFIGS (e.g. 'qualcomm').
    target_soc: Target SoC ID (e.g. 'sm8850'). Validated against supported SoCs.
    prefill_lengths: User-specified prefill sequence buckets.
    cache_length: User-specified KV cache capacity.
    max_decode_steps: User-specified calibration decode sample steps.
    **user_overrides: Additional ad-hoc property overrides for
      NpuPipelineConfig.

  Returns:
    Resolved NpuPipelineConfig.
  """
  clean_vendor = target_vendor.strip().lower()
  if clean_vendor not in vendor_configs.VENDOR_CONFIGS:
    raise ValueError(
        f"Unknown target vendor '{target_vendor}'. Supported vendors:"
        f" {list(vendor_configs.VENDOR_CONFIGS.keys())}"
    )

  clean_soc = soc_utils.validate_soc(clean_vendor, target_soc)

  merged: dict[str, Any] = {
      "model_id": model_id,
  }

  # 1. Vendor Level
  vendor_def = vendor_configs.VENDOR_CONFIGS.get(clean_vendor, {})
  merged.update(vendor_def)

  # 2. Per-SoC Overrides (if present in sparse SOC_CONFIGS map)
  soc_def = vendor_configs.SOC_CONFIGS.get(clean_soc, {})
  merged.update(soc_def)

  # Explicitly bind validated vendor and SoC
  merged["backend"] = clean_vendor
  merged["soc_model"] = clean_soc

  # 4. User Overrides
  if prefill_lengths is not None:
    merged["prefill_lengths"] = prefill_lengths
  if cache_length is not None:
    merged["cache_length"] = cache_length
  if max_decode_steps is not None:
    merged["max_decode_steps"] = max_decode_steps
  for k, v in user_overrides.items():
    if v is not None:
      merged[k] = v

  if (
      "quantization_recipe" in merged
      and "weight_quantization_recipe" not in merged
  ):
    merged["weight_quantization_recipe"] = merged.pop("quantization_recipe")
  if "a16w8" in merged and "use_16bits_activations" not in merged:
    merged["use_16bits_activations"] = merged.pop("a16w8")

  valid_fields = {f.name for f in dataclasses.fields(NpuPipelineConfig)}
  clean_kwargs = {k: v for k, v in merged.items() if k in valid_fields}

  cfg = NpuPipelineConfig(**clean_kwargs)
  for k, v in merged.items():
    if k not in valid_fields and v is not None and k != "a16w8":
      setattr(cfg, k, v)
  return cfg


def get_available_models() -> list[str]:
  """Returns the list of available model keys registered in MODEL_CONFIGS."""
  return sorted(list(model_configs.MODEL_CONFIGS.keys()))


def get_available_socs(vendor: str) -> list[str]:
  """Returns the authoritative list of SoCs registered for a vendor in supported_soc.csv."""
  return soc_utils.get_supported_socs(vendor)


# Aliases for convenience in tests and tools
get_supported_socs_from_csv = get_available_socs
validate_soc_against_csv = soc_utils.validate_soc
