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
"""Export functions for HuggingFace Transformers models."""

import dataclasses
import gc
import os
import shutil
import tempfile
from typing import Any
import warnings
from litert_torch import progress
from litert_torch.generative.export_hf import export_utils
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder

ExportTask = exportable_module_config.ExportTask


@progress.task('LiteRT GenAI Export')
def run_export_tasks(
    export_tasks,
    export_config: exportable_module.ExportableModuleConfig,
) -> export_lib.ExportedModelArtifacts:
  """Runs export tasks."""
  model_path = export_config.model
  trust_remote_code = export_config.trust_remote_code
  auto_model_override = export_config.auto_model_override
  task = export_config.task
  source_model_artifacts = export_lib.load_model(
      model_path,
      export_config,
      trust_remote_code=trust_remote_code,
      auto_model_override=auto_model_override,
      task=task,
  )
  export_config = export_lib.update_export_config(
      export_config, source_model_artifacts
  )

  exported_model_artifacts = export_lib.ExportedModelArtifacts()

  # Suppress deprecation warnings to be compatible with older PyTorch.
  with warnings.catch_warnings():
    warnings.filterwarnings(
        'ignore',
        category=FutureWarning,
        message=r'.*isinstance\(treespec, LeafSpec\)` is deprecated.*',
    )
    warnings.filterwarnings(
        'ignore',
        category=FutureWarning,
        message=r'.*treespec\.children_specs` is deprecated.*',
    )
    for export_task in export_tasks:
      exported_model_artifacts = export_task(
          source_model_artifacts,
          export_config,
          exported_model_artifacts,
      )
      gc.collect()
  return exported_model_artifacts


def export(
    model: str,
    output_dir: str,
    task: ExportTask | str = ExportTask.TEXT_GENERATION,
    keep_temporary_files: bool = False,
    # target_accelerator: str | None = None,
    # TODO(weiyiw): Remove the following flags.
    # pylint: disable=unused-argument
    trust_remote_code: bool = False,
    prefill_lengths: list[int] | None = None,
    cache_length: int | None = None,
    decode_cache_length: int | None = None,
    quantization_recipe: str | None = None,
    enable_dynamic_shape: bool | None = None,
    enable_gpu_dynamic_prefill: bool | None = None,
    enable_gpu_dynamic_cache: bool | None = None,
    use_rope_composite: bool | None = None,
    use_swiglu_composite: bool | None = None,
    use_qkv_norm_rope_composite: bool | None = None,
    use_sdpa_composite: bool | None = None,
    externalize_embedder: bool | None = None,
    single_token_embedder: bool | None = None,
    k_ts_idx: int | None = None,
    v_ts_idx: int | None = None,
    split_cache: bool | None = None,
    cache_implementation: str | None = None,
    auto_model_override: str | None = None,
    use_jinja_template: bool | None = None,
    bundle_litert_lm: bool | None = None,
    experimental_use_mixed_precision: bool | None = None,
    export_vision_encoder: bool | None = None,
    vision_encoder_quantization_recipe: str | None = None,
    export_audio_encoder: bool | None = None,
    input_sec: float | None = None,
    stateful_after: int | None = None,
    litert_lm_model_type_override: str | None = None,
    litert_lm_llm_metadata_override: str | None = None,
    sampler_top_p: float | None = None,
    sampler_temperature: float | None = None,
    sampler_top_k: int | None = None,
    aot_backend: str | None = None,
    aot_soc_model: str | None = None,
    aot_compilation_config_dict: dict[str, Any] | None = None,
    experimental_lightweight_conversion: bool = False,
    experimental_transpile_chat_template_for_minijinja: bool = False,
    sliding_window_ring_buffer_size: int | None = None,
    decode_sliding_window_ring_buffer_size: int | None = None,
    use_litert_lm_compiler: bool = False,
    compile_configs: str | None = None,
    calibration_dataset_dir: str | None = None,
    calibration_dataset_format: str = 'jsonl',
    calibration_eval_task_names: str | list[str] = 'ALL',
    max_calibration_decode_steps: int = 32,
    calibration_range_scale: float = 1.0,
    use_float_input_output_normalizer: bool = False,
    use_profiler_based_calibration: bool = True,
    enable_min_max_calibration_update: bool = True,
    ema_smoothing_factor: float = 0.1,
    static_quantization_recipe: str | None = None,
    # pylint: enable=unused-argument
    **kwargs,
):
  """Exports HuggingFace Transformers model to tflite.

  Args:
    model: The name of the HuggingFace Transformers model to export, or the path
      to the safetensors directory.
    output_dir: The directory to export the model to.
    task: The task to export the model for. Use 'text_generation' for text only
      LLMs, 'image_text_to_text' for Vision LLMs, 'automatic_speech_recognition'
      for ASR, and 'text_to_speech' for TTS.
    keep_temporary_files: Whether to keep the temporary files.
    trust_remote_code: Whether to trust remote code.
    prefill_lengths: The lengths of the prefill input, separated by comma.
    cache_length: The length of the cache.
    decode_cache_length: Optional distinct KV-cache length for the decode
      signature's full_attention (global) layers. When unset, decode reuses
      cache_length. When set, prefill uses cache_length and decode uses this
      value so both reach the same total context (e.g. context=16384: prefill
      16256 + chunk 128, decode 16383 + 1). Sliding layers are unaffected (they
      use sliding_window_ring_buffer_size for both). Export-only unless the
      runtime is changed to allow per-signature KV-cache sizes.
    quantization_recipe: The quantization recipes to use, separated by comma.
    enable_dynamic_shape: Whether to enable dynamic shape.
    enable_gpu_dynamic_prefill: Whether to enable GPU dynamic shapes (magic
      numbers) for prefill lengths.
    enable_gpu_dynamic_cache: Whether to enable GPU dynamic shapes (magic
      numbers) for cache length.
    use_rope_composite: Whether to enable the RoPE composite.
    use_qkv_norm_rope_composite: Whether to enable the QKV norm rope composite.
    use_sdpa_composite: Whether to enable the fused transposed SDPA composite.
    externalize_embedder: Whether to externalize the embedder.
    single_token_embedder: Whether to use a single token embedder.
    k_ts_idx: The index of time step dimension in the key tensor.
    v_ts_idx: The index of time step dimension in the value tensor.
    split_cache: Whether to use split cache attention.
    cache_implementation: The cache implementation to use.
    auto_model_override: Overriding the AutoModel class to use for export.
    use_jinja_template: Whether to use jinja template.
    bundle_litert_lm: Whether to bundle the model as a LiteRT LM file.
    experimental_use_mixed_precision: Whether to enable mixed precision.
    export_vision_encoder: Whether to export the vision encoder.
    vision_encoder_quantization_recipe: The quantization recipe to use for the
      vision encoder.
    export_audio_encoder: Whether to export the audio encoder.
    input_sec: Input audio length in seconds.
    stateful_after: If >= 0, the model runs in stateful mode after this many
      tokens.
    litert_lm_model_type_override: Overriding the LiteRT LM model type.
    litert_lm_llm_metadata_override: Overriding the LiteRT LM LLM metadata.
    aot_backend: The backend to use for AOT compilation.
    aot_soc_model: The SoC model to use for AOT compilation.
    aot_compilation_config_dict: The configuration dictionary for AOT
      compilation.
    experimental_lightweight_conversion: Whether to use lightweight conversion,
      which might speed up large model conversion, but might not work for all
      models.
    sliding_window_ring_buffer_size: The size of the sliding window ring buffer
      (the local past cache) used for both signatures by default.
    decode_sliding_window_ring_buffer_size: Optional distinct ring-buffer (local
      past) size for the decode signature's sliding_attention layers. When unset,
      decode reuses sliding_window_ring_buffer_size. When set, prefill uses
      sliding_window_ring_buffer_size and decode uses this value so both reach the
      same sliding window (e.g. window=1024: prefill 896 + chunk 128, decode
      1023 + 1). The window bound (= model sliding_window) is unchanged for both.
      Ring-buffer equivalent of decode_cache_length; export-only unless the
      runtime is changed to allow per-signature local cache sizes.
    sampler_top_p: The top_p sampling parameter.
    sampler_temperature: The temperature sampling parameter.
    sampler_top_k: The top_k sampling parameter.
    aot_backend: The NPU backend to AOT compile the model for.
    aot_soc_model: The NPU SoC model to AOT compile for.
    aot_compilation_config_dict: Additional configuration options for AOT
      compilation.
    experimental_lightweight_conversion: Whether to use lightweight conversion.
    experimental_transpile_chat_template_for_minijinja: Whether to transpile the
      chat template for MiniJinja.
    use_litert_lm_compiler: Whether to compile using the LiteRT LM NPU compiler.
    compile_configs: Path to compiler configuration files.
    calibration_dataset_dir: Directory containing the calibration dataset. If
      specified without `calibration_eval_task_names`, the pipeline expects a
      file named `ALL.{calibration_dataset_format}` (e.g., `ALL.jsonl`) inside
      this directory.
    calibration_dataset_format: Format of the calibration dataset (e.g.,
      'jsonl').
    calibration_eval_task_names: Calibration task/file name(s) (excluding format
      extension) to load from `calibration_dataset_dir`. Can be a single string
      or a list of strings. Maps to
      `{calibration_dataset_dir}/{task_name}.{calibration_dataset_format}`.
      Defaults to 'ALL' (loading `{calibration_dataset_dir}/ALL.jsonl`).
    max_calibration_decode_steps: Maximum decode steps for calibration.
    calibration_range_scale: Scale factor for calibration range.
    use_float_input_output_normalizer: Keep normalizer layers (e.g. RMS Norm,
      residual ADD) in float in the quantized model.
    use_profiler_based_calibration: Use profiler based calibration.
    enable_min_max_calibration_update: Enable min-max calibration range updates.
    ema_smoothing_factor: Exponential moving average smoothing factor.
    static_quantization_recipe: Quantization recipe for static quantization
      stage.
    **kwargs: Additional keyword arguments to pass to the exportable module
      config.

  Raises:
    RuntimeError: If bundle_litert_lm is False when calibration is triggered.
  """
  provided_args = {
      k: v
      for k, v in locals().items()
      if v is not None and k not in ['model', 'output_dir', 'task', 'kwargs']
  }
  provided_args.update(kwargs)

  os.makedirs(output_dir, exist_ok=True)
  if not keep_temporary_files:
    work_dir = tempfile.mkdtemp(dir=output_dir)
  else:
    work_dir = output_dir

  valid_fields = {
      f.name
      for f in dataclasses.fields(exportable_module.ExportableModuleConfig)
  }
  config_args = {}
  extra_args = {}
  for key, value in provided_args.items():
    if key in valid_fields:
      config_args[key] = value
    else:
      extra_args[key] = value

  export_config = exportable_module.ExportableModuleConfig(
      model=model,
      output_dir=output_dir,
      work_dir=work_dir,
      task=task,
      extra_kwargs=extra_args,
      **config_args,
  )

  export_config.print_summary()

  export_utils.validate_pipeline_config(export_config)

  # TODO(weiyiw): Move this to the exportable module config.
  export_tasks = []
  legacy_compile_triggered = False

  if task == ExportTask.AUTOMATIC_SPEECH_RECOGNITION:
    export_tasks.append(export_lib.export_asr_models)
    if export_config.aot_backend is not None:
      export_tasks.append(export_lib.aot_compile_model)
      legacy_compile_triggered = True
  elif task == ExportTask.TEXT_TO_SPEECH:
    export_tasks.append(export_lib.export_tts_models)
    if export_config.aot_backend is not None:
      export_tasks.append(export_lib.aot_compile_model)
      legacy_compile_triggered = True
  else:
    export_tasks.append(export_lib.export_text_prefill_decode_model)
    if (
        export_config.aot_backend is not None
        and not export_config.use_litert_lm_compiler
    ):
      export_tasks.append(export_lib.aot_compile_model)
      legacy_compile_triggered = True
    if export_config.externalize_embedder:
      export_tasks.append(export_lib.export_embedder_model)
    if export_config.split_cache:
      export_tasks.append(export_lib.export_auxiliary_model)
    export_tasks.append(export_lib.export_additional_models)
    if export_config.export_vision_encoder:
      export_tasks.append(export_lib.export_vision_encoder_models)
    export_tasks.append(export_lib.export_tokenizer)
    if export_config.bundle_litert_lm:
      export_tasks.append(litert_lm_builder.package_model)

  if legacy_compile_triggered:
    if task == ExportTask.AUTOMATIC_SPEECH_RECOGNITION:
      print(
          '\nWARNING: The legacy AOT compiler path (compiling individual models'
          ' before packaging) is deprecated and will be removed soon. ASR task'
          ' compilation will be migrated to the new package-based compiler'
          ' path in a future release.\n'
      )
    elif task == ExportTask.TEXT_TO_SPEECH:
      print(
          '\nWARNING: The legacy AOT compiler path (compiling individual models'
          ' before packaging) is deprecated and will be removed soon. TTS task'
          ' compilation will be migrated to the new package-based compiler'
          ' path in a future release.\n'
      )
    else:
      print(
          '\nWARNING: The legacy AOT compiler path (compiling individual models'
          ' before packaging) is deprecated and will be removed soon. Please'
          ' switch to the new LiteRT-LM compiler by passing'
          ' --use_litert_lm_compiler=True, or use the unified PTQ calibration'
          ' pipeline by passing --dataset_dir.\n'
      )

  run_compilation_pipeline = (
      export_config.aot_backend is not None
      and export_config.use_litert_lm_compiler
  ) or export_config.calibration_dataset_dir is not None

  orig_keep_temp = export_config.keep_temporary_files
  if run_compilation_pipeline:
    export_config.keep_temporary_files = True

  exported_model_artifacts = run_export_tasks(
      export_tasks,
      export_config,
  )

  if run_compilation_pipeline:
    temp_litertlm = exported_model_artifacts.litert_lm_model_path
    if not temp_litertlm:
      raise RuntimeError(
          'NPU compilation pipeline requires bundle_litert_lm = True.'
      )

    export_utils.run_npu_compilation_pipeline(
        export_config=export_config,
        input_litertlm=temp_litertlm,
        work_dir=work_dir,
        output_dir=output_dir,
    )
    export_config.keep_temporary_files = orig_keep_temp
    keep_temporary_files = orig_keep_temp

  if not export_config.bundle_litert_lm:
    keep_temporary_files = True
  if not keep_temporary_files:
    print('Cleaning up temporary files.')
    shutil.rmtree(work_dir)
  print(
      'Export complete. Model saved to:'
      f' {exported_model_artifacts.litert_lm_model_path or output_dir}'
  )
