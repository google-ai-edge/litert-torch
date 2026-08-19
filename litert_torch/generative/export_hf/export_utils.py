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
"""Post-Training Quantization (PTQ) and NPU compilation pipeline utilities."""

import os
import shutil
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.experimental.calib.calibrate import calibrate as calib_func
from litert_torch.generative.export_hf.experimental.calib.quantize import quantize as quant_func
from litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler.litert_lm_npu_compiler import compile_litertlm as compile_func
import transformers


def validate_pipeline_config(
    export_config: exportable_module.ExportableModuleConfig,
) -> None:
  """Validates pipeline configuration before running export tasks.

  Args:
    export_config: The export configuration to validate.

  Raises:
    ValueError: If task names are given without a calibration dataset directory,
        or if calibration dataset is provided with disabled quantization, or if
        AOT backend is provided without an SoC model.
  """
  if (
      export_config.calibration_eval_task_names != "ALL"
      and export_config.calibration_dataset_dir is None
  ):
    raise ValueError(
        "`calibration_eval_task_names` was specified, but"
        " `calibration_dataset_dir` is missing. Please specify"
        " `--calibration_dataset_dir` to run calibration."
    )

  if export_config.calibration_dataset_dir is not None:
    if (
        export_config.static_quantization_recipe
        and export_config.static_quantization_recipe.lower()
        in ("none", "null", "false")
    ):
      raise ValueError(
          "Calibration dataset was provided (--calibration_dataset_dir), but"
          " static quantization was disabled"
          " (--static_quantization_recipe=none). Calibration profiles are only"
          " used for static quantization, so running calibration is"
          " unnecessary when static quantization is disabled. Please remove"
          " --calibration_dataset_dir or enable a static quantization recipe."
      )

  if (
      export_config.aot_backend is not None
      and export_config.aot_soc_model is None
  ):
    raise ValueError("aot_soc_model must be specified when aot_backend is set.")


def run_npu_compilation_pipeline(
    export_config: exportable_module.ExportableModuleConfig,
    input_litertlm: str,
    work_dir: str,
    output_dir: str,
) -> None:
  """Runs the post-export NPU calibration, quantization, and compilation stages.

  This orchestrates the post-export stages of the NPU compilation flow:
    Stage 1 (Calibrate): [Optional] Runs interpreter-based profiler calibration
        over representative sample datasets to record activation ranges.
    Stage 2 (Quantize): [Optional] Static range-quantizes the intermediate
        graphs using the collected calibration profiles.
    Stage 3 (Compile): [Always] Invokes target SoC compilers (e.g. Qualcomm QNN,
        MediaTek NeuroPilot) to compile the model into target NPU instructions.
        If compilation is disabled, copies the model to the final output dir.

  Args:
    export_config: Unified configuration resolving target hardware, prefill/
      cache lengths, and quantization recipes.
    input_litertlm: Path to the raw unquantized exported LiteRT-LM model bundle.
    work_dir: Intermediate workspace directory for temporary calibration
      profiles and compiler artifacts.
    output_dir: Final target directory where the compiled `.litertlm` container
      will be written.
  """
  current_litertlm = input_litertlm

  if export_config.calibration_dataset_dir is not None:
    print("--- NPU Pipeline: Running Stage 2 Calibration ---")
    calib_dir = os.path.join(work_dir, "calibration_results")
    calib_func(
        input_litertlm=current_litertlm,
        calibration_dataset_dir=export_config.calibration_dataset_dir,
        calibration_result_save_dir=calib_dir,
        kv_cache_max_len=export_config.cache_length,
        calibration_dataset_format=export_config.calibration_dataset_format,
        calibration_eval_task_names=export_config.calibration_eval_task_names,
        max_calibration_decode_steps=export_config.max_calibration_decode_steps,
        use_profiler_based_calibration=export_config.use_profiler_based_calibration,
        enable_min_max_calibration_update=export_config.enable_min_max_calibration_update,
        ema_smoothing_factor=export_config.ema_smoothing_factor,
    )

    print("--- NPU Pipeline: Running Stage 3 Static Quantization ---")
    quantized_litertlm = os.path.join(work_dir, "quantized.litertlm")
    quant_func(
        input_litertlm=current_litertlm,
        output_litertlm=quantized_litertlm,
        calibration_dir=calib_dir,
        use_float_input_output_normalizer=export_config.use_float_input_output_normalizer,
        skip_mlir_passes=True,
        calibration_range_scale=export_config.calibration_range_scale,
        quantization_recipe=export_config.static_quantization_recipe,
    )
    current_litertlm = quantized_litertlm

  final_litertlm = os.path.join(output_dir, "model.litertlm")
  if export_config.aot_backend:
    if export_config.aot_soc_model is None:
      raise ValueError(
          "aot_soc_model must be specified when aot_backend is set."
      )
    print(
        "--- NPU Pipeline: Running Stage 4 NPU Compilation"
        f" ({export_config.aot_backend}) ---"
    )
    try:
      config = transformers.AutoConfig.from_pretrained(
          export_config.model,
          trust_remote_code=export_config.trust_remote_code,
      )
      model_name = config.model_type
    except Exception:  # pylint: disable=broad-except
      model_name = None

    compile_func(
        input_litertlm=current_litertlm,
        output_litertlm=final_litertlm,
        backend=export_config.aot_backend,
        soc_model=export_config.aot_soc_model,
        compile_configs=export_config.compile_configs,
        model_name=model_name,
        overwrite=True,
    )
  else:
    print("--- NPU Pipeline: Copying model to output ---")
    shutil.copy(current_litertlm, final_litertlm)

  print(f"NPU compilation complete. Model saved to: {final_litertlm}")
