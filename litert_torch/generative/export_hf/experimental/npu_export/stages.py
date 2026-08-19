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
"""NPU-specific stage wrappers that dynamically bind NpuPipelineConfig to primitives."""

import inspect
import os
import pathlib
from typing import Any, Callable

from absl import flags
from litert_torch.generative.export_hf.experimental.calib.calibrate import calibrate as calib_func
from litert_torch.generative.export_hf.experimental.calib.quantize import quantize as quant_func
from litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler.litert_lm_npu_compiler import compile_litertlm as compile_func
from litert_torch.generative.export_hf.experimental.npu_export.config_manager import NpuPipelineConfig
from litert_torch.generative.export_hf.export import export as export_func

# Mark absl flags as parsed to allow programmatic execution in environments
# like Jupyter/VSCode Notebooks where absl.app.run is not used.
try:
  flags.FLAGS.mark_as_parsed()
except Exception:  # pylint: disable=broad-except
  pass


def _bind_cfg(
    cfg: NpuPipelineConfig,
    func: Callable[..., Any],
    overrides: dict[str, Any],
    name_map: dict[str, str] | None = None,
) -> dict[str, Any]:
  """Automatically inspects target function arguments and binds matching fields from cfg."""
  sig = inspect.signature(func)
  name_map = name_map or {}

  bound = {}
  # 1. Inspect function signature and pull matching fields/properties from cfg
  for param_name in sig.parameters:
    cfg_key = name_map.get(param_name, param_name)
    if hasattr(cfg, cfg_key):
      val = getattr(cfg, cfg_key)
      if val is not None:
        bound[param_name] = val

  # 2. User-supplied explicit kwargs always take highest precedence
  bound.update(overrides)
  return bound


def npu_export(
    cfg: NpuPipelineConfig,
    output_dir: str,
    keep_temporary_files: bool = False,
    **kwargs: Any,
) -> str:
  """Runs Stage 1 Model Export bound to NpuPipelineConfig."""
  overrides = {
      "output_dir": output_dir,
      "split_cache": True,  # Mandatory for NPU backends
      "keep_temporary_files": keep_temporary_files,
      **kwargs,
  }
  bound_args = _bind_cfg(
      cfg,
      export_func,
      overrides,
      name_map={
          "model": "model_id",
          "quantization_recipe": "weight_quantization_recipe",
      },
  )
  export_func(**bound_args)
  return os.path.join(output_dir, "model.litertlm")


def npu_calibrate(
    cfg: NpuPipelineConfig,
    input_litertlm: str,
    calibration_dataset_dir: str,
    calibration_result_save_dir: str,
    **kwargs: Any,
) -> None:
  """Runs Stage 2 Profiler Calibration bound to NpuPipelineConfig."""
  overrides = {
      "input_litertlm": input_litertlm,
      "calibration_dataset_dir": calibration_dataset_dir,
      "calibration_result_save_dir": calibration_result_save_dir,
      **kwargs,
  }
  name_map = {
      "kv_cache_max_len": "cache_length",
  }
  bound_args = _bind_cfg(cfg, calib_func, overrides, name_map=name_map)
  calib_func(**bound_args)


def npu_quantize(
    cfg: NpuPipelineConfig,
    input_litertlm: str,
    output_litertlm: str,
    calibration_dir: str,
    **kwargs: Any,
) -> None:
  """Runs Stage 3 Static Range Quantization bound to NpuPipelineConfig."""
  overrides = {
      "input_litertlm": input_litertlm,
      "output_litertlm": output_litertlm,
      "calibration_dir": calibration_dir,
      **kwargs,
  }
  bound_args = _bind_cfg(
      cfg,
      quant_func,
      overrides,
  )
  if "quantization_recipe" not in overrides:
    bound_args.pop("quantization_recipe", None)
  quant_func(**bound_args)


def npu_compile(
    cfg: NpuPipelineConfig,
    input_litertlm: str | pathlib.Path,
    output_litertlm: str | pathlib.Path,
    keep_temporary_files: bool = False,
    intermediate_dir: str | pathlib.Path | None = None,
    **kwargs: Any,
) -> None:
  """Runs Stage 4 Backend Compilation bound to NpuPipelineConfig."""
  overrides = {
      "input_litertlm": input_litertlm,
      "output_litertlm": output_litertlm,
      "keep_temporary_files": keep_temporary_files,
      "intermediate_dir": intermediate_dir,
      **kwargs,
  }
  bound_args = _bind_cfg(cfg, compile_func, overrides)
  compile_func(**bound_args)
