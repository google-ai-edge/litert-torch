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

from collections.abc import Mapping, Sequence
import dataclasses
import gc
import json
import os
import pathlib
import shutil
import tempfile
import types
from typing import Any
import warnings

import huggingface_hub
from litert_torch import progress
from litert_torch.generative.export_hf import export_utils
from litert_torch.generative.export_hf import model_ext as _
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.experimental.npu_export.configs import vendor_configs
from litert_torch.generative.export_hf.model_ext import builtin_extensions as _
from litert_torch.generative.export_hf.model_ext import extension as model_ext_extension

ExportTask = exportable_module_config.ExportTask

TASK_AUTO = 'auto'
TARGET_CPU_GPU = 'cpu_gpu'
TARGET_CPU = 'cpu'
TARGET_GPU = 'gpu'
TARGET_NPU = 'npu'
CPU_GPU_TARGETS = (TARGET_CPU_GPU, TARGET_CPU, TARGET_GPU)

# Where a resolved export argument (or the task) came from.
SOURCE_USER = 'user'
SOURCE_MODEL = 'model'
SOURCE_TARGET = 'target'
SOURCE_DEFAULT = 'default'

_LLM_TASKS = frozenset({
    ExportTask.TEXT_GENERATION.value,
    ExportTask.IMAGE_TEXT_TO_TEXT.value,
    ExportTask.MULTIMODAL_LM.value,
})

# Grand fallback settings for LLM/VLM export on CPU/GPU (`cpu_gpu`): GPU
# composites (CPU runs their reference decompositions) and externalized weights.
# A short 128-token prefill signature is added next to the 1024-token one
# because a short prompt padded to 1024 tokens prefills 4-7x slower on CPU. The
# graph KV cache is exported at 32771 while `LlmMetadata.max_num_tokens` is
# capped at 4096 by default so CPU/GPU runtimes allocate a 4K context window
# unless overridden at runtime.
_CPU_GPU_LLM_KWARGS = types.MappingProxyType({
    'apply_gpu_composites': True,
    'externalize_weights': True,
    'prefill_lengths': (128, 1024),
    'cache_length': 32771,
    'llm_metadata_max_num_tokens_override': 4096,
    'use_jinja_template': False,
})

# Generic NPU split-cache export settings for LLMs/VLMs (`npu`, `npu:<vendor>`).
# TODO(weiyiw): Verify and support context bucketing (multiple cache_lengths)
# for NPU targets.
_NPU_LLM_KWARGS = types.MappingProxyType({
    'split_cache': True,
    'apply_gpu_composites': False,
    'externalize_embedder': True,
    'prefill_lengths': (128,),
    'cache_length': 1024,
    'quantization_recipe': 'dynamic_wi8_afp32',
    'use_jinja_template': False,
})

# Architecture suffixes that export_hf's generic text-generation path loads with
# `AutoModelForCausalLM`.
_CAUSAL_LM_ARCHITECTURE_SUFFIXES = ('ForCausalLM', 'LMHeadModel')

# Tasks tried, in order, to identify checkpoints without a `model_type`.
_INFERENCE_TASKS = (
    None,
    ExportTask.TEXT_TO_IMAGE.value,
    ExportTask.TEXT_TO_SPEECH.value,
    ExportTask.ANY_TO_ANY.value,
)


@dataclasses.dataclass(frozen=True)
class TargetSpec:
  """Parsed hardware target specification.

  Attributes:
    raw: Original target string (or `'cpu_gpu'` when omitted).
    family: Target family (`'cpu_gpu'` or `'npu'`).
    vendor: Canonical NPU vendor key in `vendor_configs.VENDOR_CONFIGS` (e.g.
      `'qualcomm'`, `'mediatek'`, `'intel'`, `'google_tensor'`, `'samsung'`), or
      `None`.
    soc: Optional NPU SoC identifier (e.g. `'sm8750'`), or `None`.
  """

  raw: str
  family: str
  vendor: str | None = None
  soc: str | None = None

  @property
  def is_npu(self) -> bool:
    """Returns True if this target is in the `'npu'` family."""
    return self.family == TARGET_NPU

  @property
  def canonical(self) -> str:
    """Returns the canonical normalized target string."""
    if not self.is_npu:
      return TARGET_CPU_GPU
    if self.vendor and self.soc:
      return f'{TARGET_NPU}:{self.vendor}:{self.soc}'
    if self.vendor:
      return f'{TARGET_NPU}:{self.vendor}'
    return TARGET_NPU


def _supported_npu_vendors_help() -> str:
  vendors = sorted(vendor_configs.VENDOR_CONFIGS.keys())
  return ', '.join(vendors)


def parse_target(target: str | None) -> TargetSpec:
  """Parses and validates a `target` specification string.

  Accepted forms:
  - `None`, `''`, `'cpu_gpu'`, `'cpu'`, `'gpu'`: Resolves to the `'cpu_gpu'`
    family.
  - `'npu'`: Generic uncompiled NPU split-cache export.
  - `'npu:<vendor>'`: Vendor-specific uncompiled NPU split-cache export (where
    `<vendor>` is a key or alias in `vendor_configs.VENDOR_CONFIGS`, e.g.
    `'npu:qualcomm'`, `'npu:qc'`, `'npu:mediatek'`, `'npu:mtk'`, `'npu:intel'`,
    `'npu:google_tensor'`, `'npu:samsung'`, `'npu:slsi'`).
  - `'npu:<vendor>:<soc>'`: Vendor + SoC NPU target (reserved for SoC-specific
    AOT compilation).

  Args:
    target: Raw target string, or `None` for the default `'cpu_gpu'` target.

  Returns:
    A validated `TargetSpec`.

  Raises:
    ValueError: If `target` is malformed or names an unknown target or vendor.
  """
  if target is None or not target.strip():
    return TargetSpec(raw=TARGET_CPU_GPU, family=TARGET_CPU_GPU)
  raw = target.strip()
  lowered = raw.lower()
  if lowered in CPU_GPU_TARGETS:
    return TargetSpec(raw=lowered, family=TARGET_CPU_GPU)
  if lowered == TARGET_NPU:
    return TargetSpec(raw=TARGET_NPU, family=TARGET_NPU)
  if lowered.startswith(f'{TARGET_NPU}:'):
    parts = [p.strip() for p in raw.split(':')]
    if len(parts) not in (2, 3) or any(not p for p in parts):
      raise ValueError(
          f"Invalid NPU target '{target}'. Expected 'npu', 'npu:<vendor>', or"
          " 'npu:<vendor>:<soc>' (e.g. 'npu:qualcomm' or"
          " 'npu:qualcomm:sm8750')."
      )
    vendor = vendor_configs.normalize_vendor(parts[1])
    if vendor not in vendor_configs.VENDOR_CONFIGS:
      raise ValueError(
          f"Unknown NPU vendor '{parts[1]}' in target '{target}'. Supported"
          f' vendors: {_supported_npu_vendors_help()}. If'
          f" '{parts[1]}' is a chip/SoC name, include the vendor as"
          f" 'npu:<vendor>:{parts[1]}' (e.g. 'npu:qualcomm:{parts[1]}')."
      )
    soc = parts[2] if len(parts) == 3 else None
    return TargetSpec(raw=raw, family=TARGET_NPU, vendor=vendor, soc=soc)

  vendor_candidate = vendor_configs.normalize_vendor(lowered)
  if vendor_candidate in vendor_configs.VENDOR_CONFIGS:
    raise ValueError(
        f"Invalid target '{target}'. Prefix NPU vendors with 'npu:', e.g."
        f" 'npu:{vendor_candidate}'."
    )
  raise ValueError(
      f"Unknown target '{target}'. Use one of: {', '.join(CPU_GPU_TARGETS)},"
      " 'npu', 'npu:<vendor>' (vendors:"
      f" {_supported_npu_vendors_help()}), or 'npu:<vendor>:<soc>' (e.g."
      " 'npu:qualcomm:sm8750')."
  )


def _normalize_kwarg_value(value: Any) -> Any:
  return list(value) if isinstance(value, tuple) else value


def _target_llm_kwargs(target_spec: TargetSpec) -> dict[str, Any]:
  """Returns the target-level LLM/VLM export kwargs for `target_spec`."""
  base = _NPU_LLM_KWARGS if target_spec.is_npu else _CPU_GPU_LLM_KWARGS
  kwargs = {k: _normalize_kwarg_value(v) for k, v in base.items()}
  if target_spec.is_npu and target_spec.vendor:
    vendor_cfg = vendor_configs.VENDOR_CONFIGS.get(target_spec.vendor, {})
    for key in ('k_ts_idx', 'v_ts_idx'):
      if key in vendor_cfg:
        kwargs[key] = vendor_cfg[key]
  return kwargs


@dataclasses.dataclass(frozen=True)
class ExportPlan:
  """Resolved export settings for a checkpoint.

  Attributes:
    model: Hugging Face repo id or local checkpoint directory.
    task: Resolved export task.
    task_source: Where `task` came from (`'user'`, `'model'` or `'default'`).
    model_type: Detected `model_type`, or `None` if unknown.
    export_kwargs: Keyword arguments resolved for `export` (besides `model`,
      `output_dir`, `task`, `target`, and `dry_run`).
    sources: Where each `export_kwargs` entry came from (`'user'`, `'model'` or
      `'target'`).
    target: Canonical hardware target string (e.g. `'cpu_gpu'`, `'npu'`,
      `'npu:qualcomm'`).
  """

  model: str
  task: str
  task_source: str
  model_type: str | None
  export_kwargs: Mapping[str, Any]
  sources: Mapping[str, str]
  target: str = TARGET_CPU_GPU

  def summary_rows(self) -> list[tuple[str, str, str]]:
    """Returns `(setting, value, source)` rows describing the plan."""
    rows = [
        ('model_type', str(self.model_type), 'detected'),
        ('task', self.task, self.task_source),
        ('target', self.target, SOURCE_TARGET),
    ]
    for key in sorted(self.export_kwargs):
      rows.append((key, repr(self.export_kwargs[key]), self.sources[key]))
    return rows

  def __str__(self) -> str:
    """Formats the resolved export plan as a human-readable summary table."""
    rows = self.summary_rows()
    key_width = max(len('setting'), *(len(r[0]) for r in rows))
    val_width = max(len('value'), *(len(r[1]) for r in rows))
    lines = [
        f'ExportPlan(model={self.model!r}):',
        f"  {'setting':<{key_width}}  {'value':<{val_width}}  source",
        f"  {'-' * key_width}  {'-' * val_width}  ------",
    ]
    for setting, value, source in rows:
      lines.append(f'  {setting:<{key_width}}  {value:<{val_width}}  {source}')
    return '\n'.join(lines)


AutoExportPlan = ExportPlan


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError):
    # Missing or malformed file: detection falls back to the other sources.
    return None


def _fetch_hub_json(repo_id: str, filename: str) -> dict[str, Any] | None:
  """Downloads a JSON file from a Hub repo, or returns None if unavailable."""
  try:
    path = huggingface_hub.hf_hub_download(repo_id, filename)
  except (
      huggingface_hub.utils.EntryNotFoundError,
      huggingface_hub.utils.HfHubHTTPError,
      OSError,
      ValueError,
  ):
    # Missing file, gated / missing repo, offline or invalid repo id: the
    # export itself reports any real error.
    return None
  return _read_json(pathlib.Path(path))


def load_checkpoint_config(
    model: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
  """Returns the raw `config.json` and `model_index.json` of a checkpoint.

  Args:
    model: Hugging Face repo id or local checkpoint directory.

  Returns:
    `(config, model_index)`: `config` is `{}` if the checkpoint has no
    `config.json`; `model_index` is `None` unless it is a Diffusers pipeline.
  """
  local = pathlib.Path(model)
  if local.is_dir():
    return (
        _read_json(local / 'config.json') or {},
        _read_json(local / 'model_index.json'),
    )
  config = _fetch_hub_json(model, 'config.json') or {}
  if config.get('model_type'):
    # A regular Transformers checkpoint, not a Diffusers pipeline.
    return config, None
  return config, _fetch_hub_json(model, 'model_index.json')


def _normalize_task(task: ExportTask | str | None) -> str | None:
  """Returns the `ExportTask` value for `task`, or None for automatic."""
  if task is None or task == TASK_AUTO:
    return None
  try:
    return ExportTask(task).value
  except ValueError:
    raise ValueError(
        f"Unknown task '{task}'. Use '{TASK_AUTO}' or one of:"
        f" {', '.join(t.value for t in ExportTask)}."
    ) from None


def _find_extensions(
    model: str,
    config: Mapping[str, Any],
    model_index: Mapping[str, Any] | None,
    task: str | None,
    model_extension: Any = None,
) -> tuple[str | None, list[model_ext_extension.ModelExtension]]:
  """Returns the detected model type and its extensions (most specific first)."""
  model_type, extensions = _find_registered_extensions(
      model, config, model_index, task
  )
  if model_extension is None:
    return model_type, extensions
  explicit = model_ext_extension.compose_extensions(*(
      model_extension
      if isinstance(model_extension, (list, tuple))
      else (model_extension,)
  ))
  if model_type is None and explicit.model_types:
    model_type = explicit.model_types[0]
  return model_type, [explicit, *extensions]


def _find_registered_extensions(
    model: str,
    config: Mapping[str, Any],
    model_index: Mapping[str, Any] | None,
    task: str | None,
) -> tuple[str | None, list[model_ext_extension.ModelExtension]]:
  """Returns the model type and its registered extensions."""
  candidates = [
      config.get('model_type'),
      model_ext_extension.text_config_dict(config).get('model_type'),
  ]
  model_type = None
  extensions = []
  for candidate in candidates:
    if not candidate:
      continue
    extension = model_ext_extension.get_model_extension(candidate)
    if extension is not None and extension not in extensions:
      model_type = model_type or candidate
      extensions.append(extension)
  if extensions:
    return model_type, extensions

  # Diffusers pipelines, standalone TTS checkpoints, and checkpoints that
  # reuse a generic `model_type` (e.g. MiniCPM5 with `model_type='llama'`) are
  # matched via `ModelExtension.match_checkpoint`.
  for inference_task in (task,) if task else _INFERENCE_TASKS:
    config_dict = dict(
        model_index
        if inference_task == ExportTask.TEXT_TO_IMAGE.value and model_index
        else config
    )
    inferred = model_ext_extension.infer_model_type_from_extensions(
        model, task=inference_task, config_dict=config_dict
    )
    if inferred:
      extension = model_ext_extension.get_model_extension(inferred)
      return (
          config.get('model_type') or inferred,
          [extension] if extension is not None else [],
      )
  if config.get('model_type'):
    # A regular checkpoint without a registered extension (e.g. a generic
    # causal LM): export_hf's generic path handles it.
    return config['model_type'], []
  return None, []


def _check_generic_text_generation(
    model: str,
    config: Mapping[str, Any],
    extensions: Sequence[model_ext_extension.ModelExtension],
    requested_task: str | None,
) -> None:
  """Rejects checkpoints the generic text-generation path cannot export."""
  architectures = list(config.get('architectures') or [])
  if extensions or requested_task or not architectures:
    return
  if any(
      arch.endswith(_CAUSAL_LM_ARCHITECTURE_SUFFIXES) for arch in architectures
  ):
    return
  raise ValueError(
      f"'{model}' has architectures {architectures} (model_type"
      f" {config.get('model_type')!r}), which no litert-torch ModelExtension"
      ' handles and which are not an AutoModelForCausalLM. Pass a task, or'
      ' load an extension for this architecture with `extension_modules`.'
  )


def _resolve_plan(
    model: str,
    *,
    task: ExportTask | str | None = TASK_AUTO,
    target: str | None = None,
    extension_modules: Sequence[str] | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExportPlan:
  """Resolves the task, target defaults, and extension defaults for `model`."""
  overrides = {
      k: v
      for k, v in (overrides or {}).items()
      if v is not None or k in _NULLABLE_QUANTIZATION_KEYS
  }
  if target is None:
    aot_backend = overrides.get('aot_backend')
    if aot_backend:
      norm_vendor = vendor_configs.normalize_vendor(str(aot_backend))
      if norm_vendor in vendor_configs.VENDOR_CONFIGS:
        target = f'{TARGET_NPU}:{norm_vendor}'
      else:
        target = TARGET_NPU
    elif overrides.get('split_cache'):
      target = TARGET_NPU
    else:
      target = TARGET_CPU_GPU

  target_spec = parse_target(target)
  if target_spec.soc is not None:
    raise NotImplementedError(
        f"SoC-targeted NPU AOT compilation ('{target_spec.canonical}') is not"
        f" yet supported in export(). Use target='npu:{target_spec.vendor}' to"
        ' export the uncompiled split-cache model.'
    )

  requested_task = _normalize_task(task)
  if extension_modules:
    model_ext_extension.load_extension_modules(extension_modules)
    overrides['extension_modules'] = extension_modules
  model_ext_extension.discover_entry_point_extensions()

  config, model_index = load_checkpoint_config(model)
  model_type, extensions = _find_extensions(
      model,
      config,
      model_index,
      requested_task,
      model_extension=overrides.get('model_extension'),
  )
  _check_generic_text_generation(model, config, extensions, requested_task)
  defaults = model_ext_extension.ExportDefaults()
  config_dict = config or dict(model_index or {})
  for extension in extensions:
    extension_defaults = extension.get_export_defaults(
        config_dict, requested_task, target_spec.canonical
    )
    if extension_defaults is not None:
      defaults = extension_defaults
      break

  if requested_task:
    resolved_task, task_source = requested_task, SOURCE_USER
  elif defaults.task:
    resolved_task, task_source = ExportTask(defaults.task).value, SOURCE_MODEL
  else:
    resolved_task = ExportTask.TEXT_GENERATION.value
    task_source = SOURCE_DEFAULT

  export_kwargs: dict[str, Any] = {}
  sources: dict[str, str] = {}
  if resolved_task in _LLM_TASKS:
    for key, value in _target_llm_kwargs(target_spec).items():
      export_kwargs[key] = value
      sources[key] = SOURCE_TARGET
  for key, value in defaults.export_kwargs.items():
    export_kwargs[key] = _normalize_kwarg_value(value)
    sources[key] = SOURCE_MODEL
  for key, value in overrides.items():
    if (
        key == 'litert_lm_llm_metadata_override'
        and (value is None or str(value).lower() == 'none')
    ):
      export_kwargs[key] = None
      sources[key] = SOURCE_USER
      if (
          'use_jinja_template' not in overrides
          and not export_kwargs.get('jinja_chat_template_override')
      ):
        export_kwargs['use_jinja_template'] = False
        sources['use_jinja_template'] = SOURCE_TARGET
      continue
    export_kwargs[key] = value
    sources[key] = SOURCE_USER

  return ExportPlan(
      model=model,
      task=resolved_task,
      task_source=task_source,
      model_type=model_type,
      export_kwargs=export_kwargs,
      sources=sources,
      target=target_spec.canonical,
  )


def _build_export_tasks(
    export_config: exportable_module.ExportableModuleConfig,
) -> tuple[list[Any], bool]:
  """Builds the ordered export pipeline stage list from `export_config`.

  Extracted as a pure helper so `run_export_tasks` can recompute the task list
  automatically when `ModelExtension.update_export_config` toggles flags such
  as `externalize_embedder`, `split_cache`, or `export_vision_encoder`.

  Args:
    export_config: Current `ExportableModuleConfig`.

  Returns:
    A 2-tuple `(export_tasks, legacy_compile_triggered)` containing the ordered
    stage callables and whether the deprecated pre-package AOT path was added.
  """
  task = export_config.task
  export_tasks = []
  legacy_compile_triggered = False

  if (
      task == ExportTask.AUTOMATIC_SPEECH_RECOGNITION
      and export_config.export_audio_encoder
  ):
    # Audio encoder + LLM (e.g. Qwen3-ASR), packaged with `LlmMetadata`.
    # `ModelExtension.update_export_config` enables `export_audio_encoder` for
    # LLM-based ASR models once the model is loaded.
    export_tasks.append(export_lib.export_text_prefill_decode_model)
    if export_config.externalize_embedder:
      export_tasks.append(export_lib.export_embedder_model)
    export_tasks.append(export_lib.export_audio_encoder_models)
    if export_config.bundle_litert_lm:
      export_tasks.append(export_lib.export_tokenizer)
      export_tasks.append(litert_lm_builder.package_model)
  elif task == ExportTask.AUTOMATIC_SPEECH_RECOGNITION:
    # Standalone (encoder-decoder or CTC) ASR model in one TFLite model;
    # `package_model` packages it with its `Speech2TextMetadata`.
    export_tasks.append(export_lib.export_asr_models)
    if export_config.bundle_litert_lm:
      export_tasks.append(export_lib.export_tokenizer)
      export_tasks.append(litert_lm_builder.package_model)
    elif export_config.aot_backend is not None:
      export_tasks.append(export_lib.aot_compile_model)
      legacy_compile_triggered = True
  elif task == ExportTask.TEXT_TO_SPEECH:
    export_tasks.append(export_lib.export_tts_models)
    if export_config.aot_backend is not None:
      export_tasks.append(export_lib.aot_compile_model)
      legacy_compile_triggered = True
  elif task == ExportTask.TEXT_TO_IMAGE:
    export_tasks.append(export_lib.export_image_gen_models)
    export_tasks.append(export_lib.export_tokenizer)
    if export_config.bundle_litert_lm:
      export_tasks.append(litert_lm_builder.package_model)
  elif task == ExportTask.ANY_TO_ANY:
    export_tasks.append(export_lib.export_any2any_models)
    export_tasks.append(export_lib.export_tokenizer)
    if export_config.bundle_litert_lm:
      export_tasks.append(litert_lm_builder.package_model)
  else:
    # Submodule exports must run before prefill/decode: with Converter V2 and
    # `delete_in_memory_params`, prefill/decode frees the model's parameters.
    if export_config.externalize_embedder:
      export_tasks.append(export_lib.export_embedder_model)
    export_tasks.append(export_lib.export_additional_models)
    if export_config.export_vision_encoder:
      export_tasks.append(export_lib.export_vision_encoder_models)
    if export_config.export_audio_encoder:
      export_tasks.append(export_lib.export_audio_encoder_models_for_llm)
    export_tasks.append(export_lib.export_text_prefill_decode_model)
    if (
        export_config.aot_backend is not None
        and not export_config.use_litert_lm_compiler
    ):
      export_tasks.append(export_lib.aot_compile_model)
      legacy_compile_triggered = True
    if export_config.split_cache:
      export_tasks.append(export_lib.export_auxiliary_model)
    export_tasks.append(export_lib.export_tokenizer)
    if export_config.bundle_litert_lm:
      export_tasks.append(litert_lm_builder.package_model)

  return export_tasks, legacy_compile_triggered


def _should_run_compilation_pipeline(
    export_config: exportable_module.ExportableModuleConfig,
) -> bool:
  """Returns whether the post-export NPU compilation/calibration pipeline runs."""
  return (
      export_config.aot_backend is not None
      and export_config.use_litert_lm_compiler
  ) or (export_config.calibration_dataset_dir is not None)


def _warn_if_legacy_compile(
    task: ExportTask | str, legacy_compile_triggered: bool
) -> None:
  """Prints a deprecation warning when the legacy AOT compile task is scheduled."""
  if not legacy_compile_triggered:
    return
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
  prev_default_tasks, prev_legacy_compile = _build_export_tasks(export_config)
  updated_config = export_lib.update_export_config(
      export_config, source_model_artifacts
  )
  if updated_config is not export_config:
    for field in dataclasses.fields(export_config):
      setattr(export_config, field.name, getattr(updated_config, field.name))
    export_utils.validate_pipeline_config(export_config)
    if _should_run_compilation_pipeline(export_config):
      export_config.keep_temporary_files = True
  if export_tasks == prev_default_tasks:
    export_tasks, new_legacy_compile = _build_export_tasks(export_config)
    if new_legacy_compile and not prev_legacy_compile:
      _warn_if_legacy_compile(export_config.task, new_legacy_compile)

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


_UNSET: Any = object()
_NULLABLE_QUANTIZATION_KEYS = frozenset({
    'quantization_recipe',
    'vision_encoder_quantization_recipe',
    'audio_encoder_quantization_recipe',
    'litert_lm_llm_metadata_override',
    'jinja_chat_template_override',
})


def export(
    model: str,
    output_dir: str | None = None,
    task: ExportTask | str | None = TASK_AUTO,
    target: str | None = None,
    keep_temporary_files: bool | None = None,
    dry_run: bool = False,
    # TODO(weiyiw): Remove the following flags.
    # pylint: disable=unused-argument
    trust_remote_code: bool | None = None,
    prefill_lengths: list[int] | None = None,
    cache_length: int | None = None,
    cache_lengths: list[int] | None = None,
    quantization_recipe: str | None = _UNSET,
    enable_dynamic_shape: bool | None = None,
    enable_gpu_dynamic_prefill: bool | None = None,
    enable_gpu_dynamic_cache: bool | None = None,
    use_rope_composite: bool | None = None,
    use_swiglu_composite: bool | None = None,
    use_qkv_norm_rope_composite: bool | None = None,
    use_short_conv_composite: bool | None = None,
    use_sdpa_composite: bool | None = None,
    use_sdpa_composite_for_vision: bool | None = None,
    apply_gpu_composites: bool | None = None,
    use_bool_mask: bool | None = None,
    use_fused_sdpa_cache_update: bool | None = None,
    prefill_logits: bool | None = None,
    externalize_embedder: bool | None = None,
    single_token_embedder: bool | None = None,
    k_ts_idx: int | None = None,
    v_ts_idx: int | None = None,
    split_cache: bool | None = None,
    cache_implementation: str | None = None,
    auto_model_override: str | None = None,
    use_jinja_template: bool | None = None,
    jinja_chat_template_override: str | None = _UNSET,
    bundle_litert_lm: bool | None = None,
    externalize_weights: bool | None = None,
    experimental_use_mixed_precision: bool | None = None,
    export_vision_encoder: bool | None = None,
    vision_encoder_quantization_recipe: str | None = _UNSET,
    export_audio_encoder: bool | None = None,
    audio_encoder_quantization_recipe: str | None = _UNSET,
    input_sec: float | None = None,
    t2i_output_image_size: int | None = None,
    stateful_after: int | None = None,
    litert_lm_model_type_override: str | None = None,
    litert_lm_llm_metadata_override: str | None = _UNSET,
    sampler_top_p: float | None = None,
    sampler_temperature: float | None = None,
    sampler_top_k: int | None = None,
    aot_backend: str | None = None,
    aot_soc_model: str | None = None,
    aot_compilation_config_dict: dict[str, Any] | None = None,
    experimental_lightweight_conversion: bool | None = None,
    experimental_transpile_chat_template_for_minijinja: bool | None = None,
    sliding_window_ring_buffer_size: int | None = None,
    use_litert_lm_compiler: bool | None = None,
    compile_configs: str | None = None,
    calibration_dataset_dir: str | None = None,
    calibration_dataset_format: str | None = None,
    calibration_eval_task_names: str | list[str] | None = None,
    max_calibration_decode_steps: int | None = None,
    calibration_range_scale: float | None = None,
    use_float_input_output_normalizer: bool | None = None,
    use_profiler_based_calibration: bool | None = None,
    enable_min_max_calibration_update: bool | None = None,
    ema_smoothing_factor: float | None = None,
    static_quantization_recipe: str | None = None,
    model_dtype: str | None = None,
    use_v2: bool | None = None,
    delete_in_memory_params: bool | None = None,
    model_extension: Any = None,
    extension_modules: Sequence[str] | str | None = None,
    # pylint: enable=unused-argument
    **kwargs,
) -> ExportPlan:
  """Exports a HuggingFace Transformers or Diffusers checkpoint to LiteRT.

  Resolution order for every export argument (later wins):
  1. `ExportableModuleConfig` field defaults.
  2. Hardware `target` grand fallback defaults for LLM/VLM tasks (`cpu_gpu` vs
     `npu` / `npu:<vendor>`).
  3. `ModelExtension.get_export_defaults` of the checkpoint's extension.
  4. Explicit caller arguments passed to `export(...)`.

  Note on built-in `LlmMetadata` and chat template defaults:
    Built-in `ModelExtension`s for several families (`qwen2`, `qwen3`,
    `qwen3_5`, `lfm2`, `lfm2_vl`, `minicpm5`, `gemma4`, `gemma4_unified`) set
    `litert_lm_llm_metadata_override` (fetched from the
    `google-ai-edge/LiteRT-LM` `models/` directory at `LITERT_LM_MODELS_REF`,
    defaulting to `main`) or `jinja_chat_template_override` tuned for official
    base/instruct checkpoints. These defaults may be wrong for custom
    fine-tunes, derivative checkpoints with modified chat templates or stop/BOS
    tokens, or non-standard vision patch counts. Pass
    `litert_lm_llm_metadata_override='none'` (and/or
    `jinja_chat_template_override=''`) to synthesize metadata from the
    checkpoint's own tokenizer, or pass a custom `.pbtext` / `.jinja` path.

  Args:
    model: The name of the HuggingFace Transformers model to export, or the path
      to the safetensors directory.
    output_dir: The directory to export the model to. Required unless
      `dry_run=True`.
    task: The task to export the model for (`'auto'` / `None` to detect from the
      checkpoint's `ModelExtension` or `config.json`, `'text_generation'`,
      `'image_text_to_text'`, `'multimodal_lm'`,
      `'automatic_speech_recognition'`, `'text_to_speech'`, `'text_to_image'`,
      or `'any_to_any'`).
    target: Hardware target preset: `'cpu_gpu'` (default; also `'cpu'` or
      `'gpu'`), `'npu'`, `'npu:<vendor>'` (e.g. `'npu:qualcomm'`,
      `'npu:mediatek'`, `'npu:intel'`, `'npu:google_tensor'`, `'npu:samsung'`),
      or `'npu:<vendor>:<soc>'`.
    keep_temporary_files: Whether to keep the temporary files.
    dry_run: When `True`, resolves and returns the `ExportPlan` (reading only
      `config.json` / `model_index.json`) without creating `output_dir` or
      loading/exporting the model.
    trust_remote_code: Whether to trust remote code.
    prefill_lengths: The lengths of the prefill input, separated by comma. If
      multiple prefill_lengths and multiple cache_lengths are provided, the
      export processes the outer product of prefill_lengths and cache_lengths,
      creating prefill/decode signatures for each combination (e.g.,
      `prefill_{prefill_length}_cache_{cache_length}`).
    cache_length: The length of the cache.
    cache_lengths: The lengths of the cache, separated by comma. If multiple
      cache_lengths and multiple prefill_lengths are provided, the export
      processes the outer product of prefill_lengths and cache_lengths, creating
      prefill/decode signatures for each combination (e.g.,
      `prefill_{prefill_length}_cache_{cache_length}`).
    quantization_recipe: The quantization recipes to use, separated by comma.
    enable_dynamic_shape: Whether to enable dynamic shape.
    enable_gpu_dynamic_prefill: Whether to enable GPU dynamic shapes (magic
      numbers) for prefill lengths.
    enable_gpu_dynamic_cache: Whether to enable GPU dynamic shapes (magic
      numbers) for cache length.
    use_rope_composite: Whether to enable the `odml.rope` composite. Defaults to
      `True` when RoPE is not externalized (`split_cache=False`) so GPU fp16
      inference computes rotary angles in fp32 past token position 2048, and
      `False` when `split_cache=True`.
    use_swiglu_composite: Whether to enable the SwiGLU composite.
    use_qkv_norm_rope_composite: Whether to enable the QKV norm rope composite.
    use_short_conv_composite: Whether to enable the short conv composite.
    use_sdpa_composite: Whether to enable the fused transposed SDPA composite in
      both prefill and decode signatures.
    use_sdpa_composite_for_vision: Whether the vision encoder (currently Gemma 4
      only) emits the `odml.scaled_dot_product_attention` composite for fused
      (flash) attention on GPU. Defaults to `apply_gpu_composites`.
    apply_gpu_composites: Master switch for GPU composite emission. Implied by
      use_sdpa_composite.
    use_bool_mask: Whether to use a boolean attention mask instead of
      materializing fp32 mask constants.
    use_fused_sdpa_cache_update: Whether sliding-window layers emit the
      `odml.fused_sdpa_cache_update` composite (attention plus ring buffer write
      in one op). Requires `apply_gpu_composites` and
      `sliding_window_ring_buffer_size`.
    prefill_logits: Whether the prefill signature returns logits for the final
      position. Defaults to on only when multi-output composites
      (`use_qkv_norm_rope_composite` or `use_short_conv_composite`) are enabled,
      which require a live consumer for output `pos=0` in the final layer.
      Exports using `apply_gpu_composites` or `use_sdpa_composite` (including
      YNNPACK/CPU exports) leave prefill logits off by default.
    externalize_embedder: Whether to externalize the embedder.
    single_token_embedder: Whether to use a single token embedder.
    k_ts_idx: The index of time step dimension in the key tensor.
    v_ts_idx: The index of time step dimension in the value tensor.
    split_cache: Whether to use split cache attention.
    cache_implementation: The cache implementation to use.
    auto_model_override: Overriding the AutoModel class to use for export.
    use_jinja_template: Whether to use jinja template.
    jinja_chat_template_override: Optional local file path or Hugging Face repo
      id containing `chat_template.jinja` to override the chat template. Note
      that some built-in extensions (e.g. `gemma4` E2B/E4B) set a default remote
      template for official checkpoints; pass `jinja_chat_template_override=''`
      when exporting custom fine-tunes whose own `tokenizer.chat_template`
      should be preserved.
    bundle_litert_lm: Whether to bundle the model as a LiteRT LM file.
    externalize_weights: Whether to move eligible TFLite weight tensors into a
      separate `tflite_weights` section when packaging the `.litertlm` file.
    experimental_use_mixed_precision: Whether to enable mixed precision.
    export_vision_encoder: Whether to export the vision encoder.
    vision_encoder_quantization_recipe: The quantization recipe to use for the
      vision encoder.
    export_audio_encoder: Whether to export the audio encoder.
    audio_encoder_quantization_recipe: The quantization recipe to use for the
      audio encoder.
    input_sec: Input audio length in seconds.
    t2i_output_image_size: Output image height/width for text_to_image task.
    stateful_after: If >= 0, the model runs in stateful mode after this many
      tokens.
    litert_lm_model_type_override: Overriding the LiteRT LM model type.
    litert_lm_llm_metadata_override: Overriding the LiteRT LM LLM metadata (a
      local `.pbtext`/`.pb` path, inline textproto, canonical family key in
      `litert_lm_builder.CANONICAL_LITERT_LM_FAMILIES`, or `'none'` to disable
      any built-in extension metadata preset and synthesize `LlmMetadata` from
      the checkpoint's own tokenizer/config, which is recommended for custom
      fine-tunes).
    sampler_top_p: The top_p sampling parameter.
    sampler_temperature: The temperature sampling parameter.
    sampler_top_k: The top_k sampling parameter.
    aot_backend: The NPU backend to AOT compile the model for.
    aot_soc_model: The NPU SoC model to AOT compile for.
    aot_compilation_config_dict: Additional configuration options for AOT
      compilation.
    experimental_lightweight_conversion: Whether to use lightweight conversion,
      which might speed up large model conversion, but might not work for all
      models.
    experimental_transpile_chat_template_for_minijinja: Whether to transpile the
      chat template for MiniJinja.
    sliding_window_ring_buffer_size: The size of the sliding window ring buffer.
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
    model_dtype: Dtype the model is loaded and traced in (weights and graph):
      'float32' (default) or 'bfloat16'. Not the deployment activation dtype;
      see `experimental_use_fp16`. Mutually exclusive with it.
    use_v2: Whether to use Converter V2. Supports `text_generation`,
      `image_text_to_text`, `multimodal_lm`, and `text_to_speech`.
    delete_in_memory_params: With `use_v2`, loads the model on the meta device
      and streams weights from the checkpoint during serialization instead of
      keeping them in memory.
    model_extension: Optional `model_ext.ModelExtension` subclass/instance (or
      sequence of extensions) customizing patching, exportables, and metadata
      for this export run (see `model_ext/extension.py`).
    extension_modules: Optional comma-separated string or list of dotted Python
      module paths or `.py` file paths to import before export so custom
      `ModelExtension` subclasses and their `extra_kwargs_keys` CLI flags are
      registered (e.g. `['my_pkg.my_ext']` or `'./my_arch_extension.py'`).
    **kwargs: Additional keyword arguments to pass to the exportable module
      config.

  Returns:
    The resolved `ExportPlan`.

  Raises:
    ValueError: If `output_dir` is omitted when `dry_run=False`, or if `task` /
      `target` is invalid.
    NotImplementedError: If `target` specifies a chip/SoC (`npu:<vendor>:<soc>`)
      not yet supported via `target`.
    RuntimeError: If bundle_litert_lm is False when calibration is triggered.
  """
  provided_args = {
      k: v
      for k, v in locals().items()
      if v is not _UNSET
      and (v is not None or k in _NULLABLE_QUANTIZATION_KEYS)
      and k
      not in (
          'model',
          'output_dir',
          'task',
          'target',
          'dry_run',
          'kwargs',
      )
  }
  for k, v in kwargs.items():
    if v is not _UNSET and (v is not None or k in _NULLABLE_QUANTIZATION_KEYS):
      provided_args[k] = v

  if dry_run:
    return _resolve_plan(
        model,
        task=task,
        target=target,
        extension_modules=extension_modules,
        overrides=provided_args,
    )

  if not output_dir:
    raise ValueError('output_dir is required when dry_run=False.')

  os.makedirs(output_dir, exist_ok=True)
  if not keep_temporary_files:
    work_dir = tempfile.mkdtemp(dir=output_dir)
  else:
    work_dir = output_dir

  plan = _resolve_plan(
      model,
      task=task,
      target=target,
      extension_modules=extension_modules,
      overrides=provided_args,
  )
  resolved_task = ExportTask(plan.task)

  valid_fields = {
      f.name
      for f in dataclasses.fields(exportable_module.ExportableModuleConfig)
  }
  config_args = {}
  extra_args = {}
  for key, value in plan.export_kwargs.items():
    if key == 'extra_kwargs':
      if isinstance(value, dict):
        extra_args.update(value)
    elif key in valid_fields:
      config_args[key] = value
    else:
      extra_args[key] = value

  export_config = exportable_module.ExportableModuleConfig(
      model=model,
      output_dir=output_dir,
      work_dir=work_dir,
      task=resolved_task,
      extra_kwargs=extra_args,
      **config_args,
  )

  export_config.print_summary()

  export_utils.validate_pipeline_config(export_config)

  # TODO(weiyiw): Move this to the exportable module config.
  export_tasks, legacy_compile_triggered = _build_export_tasks(export_config)
  _warn_if_legacy_compile(resolved_task, legacy_compile_triggered)

  run_compilation_pipeline = _should_run_compilation_pipeline(export_config)

  orig_keep_temp = export_config.keep_temporary_files
  if run_compilation_pipeline:
    export_config.keep_temporary_files = True

  exported_model_artifacts = run_export_tasks(
      export_tasks,
      export_config,
  )

  run_compilation_pipeline = _should_run_compilation_pipeline(export_config)

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

  keep_temp = export_config.keep_temporary_files
  if not export_config.bundle_litert_lm:
    keep_temp = True
  if not keep_temp:
    print('Cleaning up temporary files.')
    shutil.rmtree(work_dir)
  print(
      'Export complete. Model saved to:'
      f' {exported_model_artifacts.litert_lm_model_path or output_dir}'
  )
  return plan
