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
"""Pluggable model extension contract and registry for HF export."""

import contextlib
import importlib
from importlib import metadata as importlib_metadata
from typing import Any, Callable, ClassVar, NamedTuple, Sequence

from litert_torch.generative.export_hf.core import exportable_module_config
import transformers


ENTRY_POINT_GROUP = 'litert_torch.model_extensions'


class VisionExportablesSpec(NamedTuple):
  """Exportable module classes for vision encoder, adapter, and EOI."""

  encoder_cls: Any
  adapter_cls: Any = None
  eoi_cls: Any = None


class AudioExportablesSpec(NamedTuple):
  """Exportable module classes for audio encoder, adapter, and EOA."""

  encoder_cls: Any
  adapter_cls: Any = None
  eoa_cls: Any = None


class ModelExtension:
  """Pluggable model architecture extension for `export_hf`.

  Subclasses can override any combination of hooks to customize how a model
  architecture is patched, wrapped in exportable modules, and packaged with
  LiteRT-LM metadata. All hooks default to `None` or a no-op so extensions can
  be selectively implemented and composed via `CompositeModelExtension`.
  """

  model_types: ClassVar[Sequence[str]] = ()

  def update_export_config(
      self,
      export_config: exportable_module_config.ExportableModuleConfig,
      model_config: transformers.PretrainedConfig,
  ) -> exportable_module_config.ExportableModuleConfig:
    """Validates or updates `export_config` for this architecture."""
    del model_config
    return export_config

  @contextlib.contextmanager
  def patch_context(self):
    """Class-level monkey-patch context entered around model loading."""
    yield

  @contextlib.contextmanager
  def patch_model(
      self,
      model: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
  ):
    """Instance-level patch context entered around model tracing/export."""
    del model, export_config
    yield

  def get_prefill_decode_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> tuple[Any, Any] | None:
    """Returns `(prefill_cls, decode_cls)` or `None` to use generic ones."""
    del model_config, export_config
    return None

  def get_vision_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> VisionExportablesSpec | tuple[Any, Any, Any] | None:
    """Returns `(encoder_cls, adapter_cls, eoi_cls)` or `None`."""
    del model_config, export_config
    return None

  def get_audio_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> AudioExportablesSpec | tuple[Any, Any, Any] | None:
    """Returns `(encoder_cls, adapter_cls, eoa_cls)` or `None`."""
    del model_config, export_config
    return None

  def get_additional_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> dict[str, Any]:
    """Returns `{submodel_name: exportable_cls}` for auxiliary submodels."""
    del model_config, export_config
    return {}

  def get_speech_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> tuple[Any, ...] | None:
    """Returns speech (ASR) exportable module classes or `None`."""
    del model_config, export_config
    return None

  def get_speech_model_cls(self, model_type: str) -> Any | None:
    """Returns the ASR model wrapper class or `None`."""
    del model_type
    return None

  def get_tts_model_cls(self, model_type: str) -> Any | None:
    """Returns the TTS model wrapper class or `None`."""
    del model_type
    return None

  @classmethod
  def is_matching_tts_checkpoint(
      cls, config_dict: dict[str, Any], model_path: str
  ) -> bool:
    """Returns True if a typeless TTS checkpoint matches this extension."""
    del config_dict, model_path
    return False

  @classmethod
  def match_checkpoint(
      cls,
      model_path: str,
      task: Any = None,
      config_dict: dict[str, Any] | None = None,
  ) -> bool:
    """Returns True if a checkpoint directory/repo matches this extension."""
    if config_dict is not None and str(task) == 'text_to_speech':
      return cls.is_matching_tts_checkpoint(config_dict, model_path)
    return False

  def matches_checkpoint(
      self,
      model_path: str,
      task: Any = None,
      config_dict: dict[str, Any] | None = None,
  ) -> bool:
    """Instance wrapper around `match_checkpoint`."""
    return self.match_checkpoint(model_path, task=task, config_dict=config_dict)

  def get_image_gen_model_cls(self, model_type: str) -> Any | None:
    """Returns the ImageGen model wrapper class or `None`."""
    del model_type
    return None

  def get_any2any_model_cls(self, model_type: str) -> Any | None:
    """Returns the Any2Any multimodal model wrapper class or `None`."""
    del model_type
    return None

  def get_litert_lm_section_types(self) -> dict[str, Any]:
    """Returns `{submodel_name: section_type}` for `.litertlm` packaging."""
    return {}

  def get_llm_model_type_proto(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> Any | None:
    """Returns an `LlmModelType` proto message or `None` for default."""
    del model_config, export_config
    return None

  def build_llm_metadata(
      self,
      source_model_artifacts: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
      exported_model_artifacts: Any,
      llm_metadata: Any,
  ) -> Any:
    """Customizes `LlmMetadata` proto in-place or returns a new one."""
    del source_model_artifacts, export_config, exported_model_artifacts
    return llm_metadata

  def get_executor_metadata_builder(
      self,
      model_config: transformers.PretrainedConfig,
  ) -> Callable[..., Any] | None:
    """Returns a custom executor metadata builder callable or `None`."""
    del model_config
    return None


class CompositeModelExtension(ModelExtension):
  """Composes multiple `ModelExtension` instances in priority order.

  For context managers (`patch_context`, `patch_model`) and transforms
  (`update_export_config`, `build_llm_metadata`), extensions are applied in
  sequence. For selector hooks (`get_*`), the last extension in the sequence
  that returns a non-None value wins (allowing later extensions to override
  earlier ones), while `get_additional_exportables` merges dictionaries.
  """

  def __init__(self, extensions: Sequence[ModelExtension]):
    self._extensions = tuple(extensions)

  @property
  def extensions(self) -> tuple[ModelExtension, ...]:
    return self._extensions

  def update_export_config(
      self,
      export_config: exportable_module_config.ExportableModuleConfig,
      model_config: transformers.PretrainedConfig,
  ) -> exportable_module_config.ExportableModuleConfig:
    for ext in self._extensions:
      export_config = ext.update_export_config(export_config, model_config)
    return export_config

  @contextlib.contextmanager
  def patch_context(self):
    with contextlib.ExitStack() as stack:
      for ext in self._extensions:
        stack.enter_context(ext.patch_context())
      yield

  @contextlib.contextmanager
  def patch_model(
      self,
      model: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
  ):
    with contextlib.ExitStack() as stack:
      for ext in self._extensions:
        stack.enter_context(ext.patch_model(model, export_config))
      yield

  def get_prefill_decode_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> tuple[Any, Any] | None:
    for ext in reversed(self._extensions):
      res = ext.get_prefill_decode_exportables(model_config, export_config)
      if res is not None:
        return res
    return None

  def get_vision_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> VisionExportablesSpec | tuple[Any, Any, Any] | None:
    for ext in reversed(self._extensions):
      res = ext.get_vision_exportables(model_config, export_config)
      if res is not None:
        return res
    return None

  def get_audio_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> AudioExportablesSpec | tuple[Any, Any, Any] | None:
    for ext in reversed(self._extensions):
      res = ext.get_audio_exportables(model_config, export_config)
      if res is not None:
        return res
    return None

  def get_additional_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for ext in self._extensions:
      merged.update(ext.get_additional_exportables(model_config, export_config))
    return merged

  def get_speech_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> tuple[Any, ...] | None:
    for ext in reversed(self._extensions):
      res = ext.get_speech_exportables(model_config, export_config)
      if res is not None:
        return res
    return None

  def get_speech_model_cls(self, model_type: str) -> Any | None:
    for ext in reversed(self._extensions):
      res = ext.get_speech_model_cls(model_type)
      if res is not None:
        return res
    return None

  def get_tts_model_cls(self, model_type: str) -> Any | None:
    for ext in reversed(self._extensions):
      res = ext.get_tts_model_cls(model_type)
      if res is not None:
        return res
    return None

  def get_image_gen_model_cls(self, model_type: str) -> Any | None:
    for ext in reversed(self._extensions):
      res = ext.get_image_gen_model_cls(model_type)
      if res is not None:
        return res
    return None

  def get_any2any_model_cls(self, model_type: str) -> Any | None:
    for ext in reversed(self._extensions):
      res = ext.get_any2any_model_cls(model_type)
      if res is not None:
        return res
    return None

  def matches_checkpoint(
      self,
      model_path: str,
      task: Any = None,
      config_dict: dict[str, Any] | None = None,
  ) -> bool:
    return any(
        ext.matches_checkpoint(model_path, task=task, config_dict=config_dict)
        for ext in self._extensions
    )

  def get_litert_lm_section_types(self) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for ext in self._extensions:
      merged.update(ext.get_litert_lm_section_types())
    return merged

  def get_llm_model_type_proto(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> Any | None:
    for ext in reversed(self._extensions):
      res = ext.get_llm_model_type_proto(model_config, export_config)
      if res is not None:
        return res
    return None

  def build_llm_metadata(
      self,
      source_model_artifacts: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
      exported_model_artifacts: Any,
      llm_metadata: Any,
  ) -> Any:
    for ext in self._extensions:
      llm_metadata = ext.build_llm_metadata(
          source_model_artifacts,
          export_config,
          exported_model_artifacts,
          llm_metadata,
      )
    return llm_metadata

  def get_executor_metadata_builder(
      self,
      model_config: transformers.PretrainedConfig,
  ) -> Callable[..., Any] | None:
    for ext in reversed(self._extensions):
      res = ext.get_executor_metadata_builder(model_config)
      if res is not None:
        return res
    return None


def compose_extensions(
    *extensions: ModelExtension | type[ModelExtension] | None,
) -> ModelExtension:
  """Normalizes and composes one or more `ModelExtension` classes/instances."""
  normalized: list[ModelExtension] = []
  for ext in extensions:
    if ext is None:
      continue
    if isinstance(ext, type) and issubclass(ext, ModelExtension):
      ext = ext()
    if isinstance(ext, CompositeModelExtension):
      normalized.extend(ext.extensions)
    elif isinstance(ext, ModelExtension):
      normalized.append(ext)
    else:
      raise TypeError(
          'Expected ModelExtension subclass or instance, got'
          f' {type(ext).__name__}.'
      )
  if not normalized:
    return ModelExtension()
  if len(normalized) == 1:
    return normalized[0]
  return CompositeModelExtension(normalized)


# Map of model_type -> ModelExtension instance.
_EXTENSION_REGISTRY: dict[str, ModelExtension] = {}
_ENTRY_POINTS_LOADED = False


def _coerce_extension(
    ext: ModelExtension | type[ModelExtension],
) -> ModelExtension:
  if isinstance(ext, type) and issubclass(ext, ModelExtension):
    return ext()
  if isinstance(ext, ModelExtension):
    return ext
  raise TypeError(
      f'Expected ModelExtension subclass or instance, got {type(ext).__name__}.'
  )


def register_model_extension(
    model_types: str | Sequence[str] | None = None,
    *,
    override: bool = True,
):
  """Decorator or function to register a `ModelExtension` class or instance.

  Usage as a decorator:
    @register_model_extension(['my_arch', 'my_arch_text'])
    class MyArchExtension(ModelExtension):
      ...

  Or using `model_types` defined on the class:
    @register_model_extension()
    class MyArchExtension(ModelExtension):
      model_types = ('my_arch',)

  Args:
    model_types: Single model_type string or sequence of model_type strings. If
      omitted, uses `cls.model_types`.
    override: Whether to overwrite an existing registration for the same key.

  Returns:
    A decorator that registers the extension and returns the original class or
    instance.
  """

  def wrapper(ext_cls_or_instance: ModelExtension | type[ModelExtension]):
    instance = _coerce_extension(ext_cls_or_instance)
    types = model_types
    if types is None:
      types = getattr(ext_cls_or_instance, 'model_types', ())
    if isinstance(types, str):
      types = (types,)
    if not types:
      raise ValueError(
          'register_model_extension requires non-empty model_types either via'
          ' argument or class attribute `model_types`.'
      )
    for mt in types:
      if not override and mt in _EXTENSION_REGISTRY:
        continue
      _EXTENSION_REGISTRY[mt] = instance
    return ext_cls_or_instance

  return wrapper


def unregister_model_extension(model_type: str) -> ModelExtension | None:
  """Removes and returns a registered `ModelExtension` for `model_type`."""
  return _EXTENSION_REGISTRY.pop(model_type, None)


def registered_model_types() -> tuple[str, ...]:
  """Returns the tuple of currently registered model types."""
  return tuple(_EXTENSION_REGISTRY.keys())


def load_extension_modules(
    modules: Sequence[str] | str | None,
) -> None:
  """Imports dotted Python module paths so their `@register_*` decorators run.

  Args:
    modules: A comma-separated string or sequence of dotted module paths.
  """
  if not modules:
    return
  if isinstance(modules, str):
    module_list = [m.strip() for m in modules.split(',') if m.strip()]
  else:
    module_list = [m.strip() for m in modules if m and m.strip()]
  for module_path in module_list:
    importlib.import_module(module_path)


def _load_entry_point_extensions(reload_entry_points: bool = False) -> None:
  """Discovers and loads OSS wheel plugins from `importlib.metadata`."""
  global _ENTRY_POINTS_LOADED
  if _ENTRY_POINTS_LOADED and not reload_entry_points:
    return
  _ENTRY_POINTS_LOADED = True
  try:
    eps = importlib_metadata.entry_points(group=ENTRY_POINT_GROUP)
  except Exception:  # pylint: disable=broad-exception-caught
    return

  for ep in eps:
    try:
      loaded = ep.load()
      if isinstance(loaded, type) and issubclass(loaded, ModelExtension):
        if ep.name and ep.name not in _EXTENSION_REGISTRY:
          register_model_extension(ep.name, override=False)(loaded)
        elif getattr(loaded, 'model_types', ()):
          register_model_extension(override=False)(loaded)
      elif isinstance(loaded, ModelExtension):
        if ep.name and ep.name not in _EXTENSION_REGISTRY:
          register_model_extension(ep.name, override=False)(loaded)
        elif getattr(loaded, 'model_types', ()):
          register_model_extension(override=False)(loaded)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Warning: failed to load model extension entry point {ep}: {e}')


def _try_lazy_import_builtin(model_type: str) -> None:
  """Attempts to lazily import built-in `model_ext.<model_type>` modules."""
  if not model_type or not model_type.isidentifier():
    return
  base_pkg = f'litert_torch.generative.export_hf.model_ext.{model_type}'
  for suffix in ('extension', 'patch', ''):
    mod_name = f'{base_pkg}.{suffix}' if suffix else base_pkg
    try:
      importlib.import_module(mod_name)
    except ImportError:
      continue


def get_model_extension(model_type: str | None) -> ModelExtension | None:
  """Looks up a registered `ModelExtension` by `model_type`."""
  if not model_type:
    return None
  if model_type in _EXTENSION_REGISTRY:
    return _EXTENSION_REGISTRY[model_type]
  _try_lazy_import_builtin(model_type)
  if model_type in _EXTENSION_REGISTRY:
    return _EXTENSION_REGISTRY[model_type]
  _load_entry_point_extensions()
  return _EXTENSION_REGISTRY.get(model_type)


def _extract_model_types(
    model_type_or_config: str | transformers.PretrainedConfig | None,
) -> list[str]:
  """Extracts candidate `model_type` strings from a config or string."""
  if model_type_or_config is None:
    return []
  if isinstance(model_type_or_config, str):
    return [model_type_or_config] if model_type_or_config else []
  candidates: list[str] = []
  top_mt = getattr(model_type_or_config, 'model_type', None)
  if top_mt:
    candidates.append(top_mt)
  text_cfg = getattr(model_type_or_config, 'text_config', None)
  if text_cfg is not None:
    text_mt = getattr(text_cfg, 'model_type', None)
    if text_mt and text_mt not in candidates:
      candidates.append(text_mt)
  return candidates


def resolve_model_extension(
    model_type_or_config: str | transformers.PretrainedConfig | None = None,
    export_config: (
        exportable_module_config.ExportableModuleConfig | None
    ) = None,
) -> ModelExtension | None:
  """Resolves the active `ModelExtension` for a model and export config.

  Resolution order:
  1. Loads any `export_config.extension_modules` if specified.
  2. Checks registered extensions for `model_config.model_type` (and
     `model_config.text_config.model_type` if present).
  3. If `export_config.model_extension` is provided:
     - If a registered extension was also found, composes
       `(registered_ext, explicit_ext)` so the explicit extension overrides any
       hooks it implements while inheriting others.
     - Otherwise returns the explicit `ModelExtension` (or
       `CompositeModelExtension` if a list/tuple was passed).

  Args:
    model_type_or_config: Either a `model_type` string or a `PretrainedConfig`.
    export_config: Optional `ExportableModuleConfig` carrying `model_extension`
      and/or `extension_modules`.

  Returns:
    A `ModelExtension` instance, or `None` if no extension applies.
  """
  if export_config is not None and getattr(
      export_config, 'extension_modules', None
  ):
    load_extension_modules(export_config.extension_modules)

  found_extensions: list[ModelExtension] = []
  for mt in _extract_model_types(model_type_or_config):
    ext = get_model_extension(mt)
    if ext is not None and ext not in found_extensions:
      found_extensions.append(ext)

  explicit_ext = (
      getattr(export_config, 'model_extension', None)
      if export_config is not None
      else None
  )
  if explicit_ext is not None:
    if isinstance(explicit_ext, (list, tuple)):
      explicit_composed = compose_extensions(*explicit_ext)
    else:
      explicit_composed = compose_extensions(explicit_ext)
    found_extensions.append(explicit_composed)

  if not found_extensions:
    return None
  return compose_extensions(*found_extensions)


def infer_model_type_from_extensions(
    model_path: str,
    task: Any = None,
    config_dict: dict[str, Any] | None = None,
    export_config: (
        exportable_module_config.ExportableModuleConfig | None
    ) = None,
) -> str | None:
  """Infers `model_type` for checkpoints without a standard HF `model_type`."""
  if export_config is not None and getattr(
      export_config, 'extension_modules', None
  ):
    load_extension_modules(export_config.extension_modules)
  _load_entry_point_extensions()
  explicit_ext = (
      getattr(export_config, 'model_extension', None)
      if export_config is not None
      else None
  )
  if explicit_ext is not None:
    ext = compose_extensions(*(
        explicit_ext
        if isinstance(explicit_ext, (list, tuple))
        else (explicit_ext,)
    ))
    if ext.matches_checkpoint(model_path, task=task, config_dict=config_dict):
      if ext.model_types:
        return ext.model_types[0]
  for model_type, ext in _EXTENSION_REGISTRY.items():
    if ext.matches_checkpoint(model_path, task=task, config_dict=config_dict):
      return model_type
  return None


def infer_tts_model_type_from_extensions(
    config_dict: dict[str, Any],
    model_path: str,
    export_config: (
        exportable_module_config.ExportableModuleConfig | None
    ) = None,
) -> str | None:
  """Infers TTS `model_type` from registered or explicit `ModelExtension`s."""
  return infer_model_type_from_extensions(
      model_path=model_path,
      task='text_to_speech',
      config_dict=config_dict,
      export_config=export_config,
  )


def update_export_config(
    export_config: exportable_module_config.ExportableModuleConfig,
    model_config: transformers.PretrainedConfig,
) -> exportable_module_config.ExportableModuleConfig:
  """Updates `export_config` using any resolved `ModelExtension`."""
  ext = resolve_model_extension(model_config, export_config)
  if ext is not None:
    return ext.update_export_config(export_config, model_config)
  return export_config
