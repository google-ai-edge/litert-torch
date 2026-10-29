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
"""One-call `export_hf` export with task detection and recommended defaults.

`export.export` keeps conservative, backward-compatible defaults: the task is
`text_generation` unless set, GPU composites and externalized weights are off,
and per-architecture settings (Whisper's 30 s window, Gemma's external embedder,
Qwen3's quantization) must be spelled out by the caller. This module resolves
those settings from the checkpoint's `ModelExtension` so tools such as LiteRT
CLI can export any supported checkpoint without model-specific code:

    plan = auto_export.export('Qwen/Qwen3-0.6B', '/tmp/qwen3')

Resolution order for every export argument (later wins):
1. `export.export` / `ExportableModuleConfig` defaults.
2. The preset: with `preset='recommended'`, LLM and VLM tasks get the settings
   from the "Hugging Face model export to LiteRT-LM" guide (GPU composites,
   externalized weights, 128+1024 prefill signatures, 32K KV cache).
3. `ModelExtension.get_export_defaults` of the checkpoint (applied with the
   `recommended` preset).
4. Explicit caller arguments.

`resolve` computes the plan without loading the model (it only reads
`config.json` / `model_index.json`), so callers can print it first.
"""

import dataclasses
import json
import pathlib
import types
from typing import Any, Mapping, Sequence

import huggingface_hub
from litert_torch.generative.export_hf import export as export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import builtin_extensions as _  # Registers built-in extensions.
from litert_torch.generative.export_hf.model_ext import extension as extension_lib

ExportTask = exportable_module_config.ExportTask

TASK_AUTO = 'auto'
PRESET_RECOMMENDED = 'recommended'
PRESET_NONE = 'none'
PRESETS = (PRESET_RECOMMENDED, PRESET_NONE)

# Where a resolved export argument (or the task) came from.
SOURCE_USER = 'user'
SOURCE_MODEL = 'model'
SOURCE_PRESET = 'preset'
SOURCE_DEFAULT = 'default'

_LLM_TASKS = frozenset({
    ExportTask.TEXT_GENERATION.value,
    ExportTask.IMAGE_TEXT_TO_TEXT.value,
    ExportTask.MULTIMODAL_LM.value,
})

# "Hugging Face model export to LiteRT-LM" guide: GPU composites (CPU runs their
# reference decompositions), externalized weights and a 32K KV cache. A short
# 128-token prefill signature is added next to the 1024-token one because a
# short prompt padded to 1024 tokens prefills 4-7x slower on CPU.
_RECOMMENDED_LLM_KWARGS = types.MappingProxyType({
    'apply_gpu_composites': True,
    'externalize_weights': True,
    'prefill_lengths': (128, 1024),
    'cache_length': 32768,
})

# Tasks tried, in order, to identify checkpoints without a `model_type`.
_INFERENCE_TASKS = (
    None,
    ExportTask.TEXT_TO_IMAGE.value,
    ExportTask.TEXT_TO_SPEECH.value,
    ExportTask.ANY_TO_ANY.value,
)


@dataclasses.dataclass(frozen=True)
class AutoExportPlan:
  """Resolved export settings for a checkpoint.

  Attributes:
    model: Hugging Face repo id or local checkpoint directory.
    task: Resolved export task.
    task_source: Where `task` came from (`'user'`, `'model'` or `'default'`).
    model_type: Detected `model_type`, or `None` if unknown.
    export_kwargs: Keyword arguments passed to `export.export` (besides `model`,
      `output_dir` and `task`).
    sources: Where each `export_kwargs` entry came from (`'user'`, `'model'` or
      `'preset'`).
    litert_lm_model_family: LiteRT-LM `models/<family>` with the canonical
      `LlmMetadataProto.pbtext`, if the extension declares one.
    litert_lm_chat_template: Canonical chat template path relative to LiteRT-LM
      `models/`, if the extension declares one.
    runtime_model_name: LiteRT-LM `omni_runner --model_name`, if declared.
  """

  model: str
  task: str
  task_source: str
  model_type: str | None
  export_kwargs: Mapping[str, Any]
  sources: Mapping[str, str]
  litert_lm_model_family: str | None = None
  litert_lm_chat_template: str | None = None
  runtime_model_name: str | None = None

  def summary_rows(self) -> list[tuple[str, str, str]]:
    """Returns `(setting, value, source)` rows describing the plan."""
    rows = [
        ('model_type', str(self.model_type), 'detected'),
        ('task', self.task, self.task_source),
    ]
    for key in sorted(self.export_kwargs):
      rows.append((key, repr(self.export_kwargs[key]), self.sources[key]))
    return rows


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
  return (
      _fetch_hub_json(model, 'config.json') or {},
      _fetch_hub_json(model, 'model_index.json'),
  )


def _normalize_task(task: str | None) -> str | None:
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
) -> tuple[str | None, list[extension_lib.ModelExtension]]:
  """Returns the detected model type and its extensions (most specific first)."""
  candidates = [
      config.get('model_type'),
      extension_lib.text_config_dict(config).get('model_type'),
  ]
  model_type = None
  extensions = []
  for candidate in candidates:
    if not candidate:
      continue
    extension = extension_lib.get_model_extension(candidate)
    if extension is not None and extension not in extensions:
      model_type = model_type or candidate
      extensions.append(extension)
  if extensions:
    return model_type, extensions
  if config.get('model_type'):
    # A regular checkpoint without a registered extension (e.g. a generic
    # causal LM): export_hf's generic path handles it.
    return config['model_type'], []

  # Diffusers pipelines and standalone TTS checkpoints have no `model_type`.
  for inference_task in (task,) if task else _INFERENCE_TASKS:
    config_dict = dict(
        model_index
        if inference_task == ExportTask.TEXT_TO_IMAGE.value and model_index
        else config
    )
    inferred = extension_lib.infer_model_type_from_extensions(
        model, task=inference_task, config_dict=config_dict
    )
    if inferred:
      extension = extension_lib.get_model_extension(inferred)
      return inferred, [extension] if extension is not None else []
  return None, []


def resolve(
    model: str,
    *,
    task: str | None = TASK_AUTO,
    preset: str = PRESET_RECOMMENDED,
    extension_modules: Sequence[str] | str | None = None,
    **overrides: Any,
) -> AutoExportPlan:
  """Resolves the task and export arguments for a checkpoint.

  Args:
    model: Hugging Face repo id or local checkpoint directory.
    task: Export task, or `'auto'` / `None` to use the extension's default
      (`text_generation` when no extension declares one).
    preset: `'recommended'` applies the LiteRT-LM recommended settings and the
      extension's defaults; `'none'` keeps the `export.export` defaults.
    extension_modules: Out-of-tree `ModelExtension` modules or `.py` files to
      load before detection; also forwarded to `export.export`.
    **overrides: Explicit `export.export` keyword arguments. They win over the
      preset and the extension defaults; `None` values are ignored.

  Returns:
    The resolved `AutoExportPlan`.

  Raises:
    ValueError: If `task` or `preset` is unknown.
  """
  if preset not in PRESETS:
    raise ValueError(
        f"Unknown preset '{preset}'. Use one of: {', '.join(PRESETS)}."
    )
  requested_task = _normalize_task(task)
  overrides = {k: v for k, v in overrides.items() if v is not None}
  if extension_modules:
    extension_lib.load_extension_modules(extension_modules)
    overrides['extension_modules'] = extension_modules

  config, model_index = load_checkpoint_config(model)
  model_type, extensions = _find_extensions(
      model, config, model_index, requested_task
  )
  defaults = extension_lib.ExportDefaults()
  config_dict = config or dict(model_index or {})
  for extension in extensions:
    extension_defaults = extension.get_export_defaults(
        config_dict, requested_task
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
  if preset == PRESET_RECOMMENDED:
    if resolved_task in _LLM_TASKS:
      for key, value in _RECOMMENDED_LLM_KWARGS.items():
        export_kwargs[key] = list(value) if isinstance(value, tuple) else value
        sources[key] = SOURCE_PRESET
    for key, value in defaults.export_kwargs.items():
      export_kwargs[key] = value
      sources[key] = SOURCE_MODEL
  for key, value in overrides.items():
    export_kwargs[key] = value
    sources[key] = SOURCE_USER

  return AutoExportPlan(
      model=model,
      task=resolved_task,
      task_source=task_source,
      model_type=model_type,
      export_kwargs=export_kwargs,
      sources=sources,
      litert_lm_model_family=defaults.litert_lm_model_family,
      litert_lm_chat_template=defaults.litert_lm_chat_template,
      runtime_model_name=defaults.runtime_model_name,
  )


def export(
    model: str,
    output_dir: str,
    *,
    task: str | None = TASK_AUTO,
    preset: str = PRESET_RECOMMENDED,
    extension_modules: Sequence[str] | str | None = None,
    **overrides: Any,
) -> AutoExportPlan:
  """Resolves the plan for `model` (see `resolve`) and runs `export.export`.

  Args:
    model: Hugging Face repo id or local checkpoint directory.
    output_dir: Directory to write the exported artifacts to.
    task: Export task, or `'auto'` / `None` to detect it.
    preset: `'recommended'` or `'none'`; see `resolve`.
    extension_modules: Out-of-tree `ModelExtension` modules or `.py` files.
    **overrides: Explicit `export.export` keyword arguments.

  Returns:
    The `AutoExportPlan` that was exported.
  """
  plan = resolve(
      model,
      task=task,
      preset=preset,
      extension_modules=extension_modules,
      **overrides,
  )
  export_lib.export(
      model=model, output_dir=output_dir, task=plan.task, **plan.export_kwargs
  )
  return plan
