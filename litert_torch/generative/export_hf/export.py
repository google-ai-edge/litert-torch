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
    quantization_recipe: str | None = None,
    enable_dynamic_shape: bool | None = None,
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
    # pylint: enable=unused-argument
    **kwargs,
):
  """Exports HuggingFace Transformers model to tflite.

  Args:
    model: The name of the HuggingFace Transformers model to export, or the path
      to the safetensors directory.
    output_dir: The directory to export the model to.
    task: The task to export the model for. Use 'text_generation' for text only
      LLMs, and 'image_text_to_text' for Vision LLMs.
    keep_temporary_files: Whether to keep the temporary files.
    trust_remote_code: Whether to trust remote code.
    prefill_lengths: The lengths of the prefill input, separated by comma.
    cache_length: The length of the cache.
    quantization_recipe: The quantization recipes to use, separated by comma.
    enable_dynamic_shape: Whether to enable dynamic shape.
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
    litert_lm_model_type_override: Overriding the LiteRT LM model type.
    litert_lm_llm_metadata_override: Overriding the LiteRT LM LLM metadata.
    **kwargs: Additional keyword arguments to pass to the exportable module
      config.
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

  # TODO(weiyiw): Move this to the exportable module config.
  export_tasks = []
  export_tasks.append(export_lib.export_text_prefill_decode_model)
  if export_config.aot_backend is not None:
    export_tasks.append(export_lib.aot_compile_model)
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

  exported_model_artifacts = run_export_tasks(
      export_tasks,
      export_config,
  )
  if not export_config.bundle_litert_lm:
    keep_temporary_files = True
  if not keep_temporary_files:
    print('Cleaning up temporary files.')
    shutil.rmtree(work_dir)
  print(
      'Export complete. Model saved to:'
      f' {exported_model_artifacts.litert_lm_model_path or output_dir}'
  )
