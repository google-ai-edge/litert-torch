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
"""Exportable module dispatchers delegating to registered ModelExtensions."""

from typing import Any

from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core.speech import exportables as speech_exportables
from litert_torch.generative.export_hf.model_ext import builtin_extensions as _  # Registers built-in extensions.
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
import transformers


def get_prefill_decode_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig,
):
  """Gets prefill-decode exportables."""
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    return ext.get_prefill_decode_exportables(model_config, export_config)
  return None


def get_vision_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Gets vision exportables."""
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    ext_exportables = ext.get_vision_exportables(model_config, export_config)
    if ext_exportables is not None:
      return tuple(ext_exportables)
  raise ValueError(f'Unsupported model type: {model_config.model_type}')


def get_audio_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Gets audio exportables."""
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    ext_exportables = ext.get_audio_exportables(model_config, export_config)
    if ext_exportables is not None:
      return tuple(ext_exportables)
  raise ValueError(f'Unsupported model type: {model_config.model_type}')


def get_additional_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Gets additional exportables."""
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    return dict(ext.get_additional_exportables(model_config, export_config))
  return {}


def get_speech_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Gets speech (ASR) exportables."""
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    ext_exportables = ext.get_speech_exportables(model_config, export_config)
    if ext_exportables is not None:
      return ext_exportables
  return (
      speech_exportables.LiteRTExportableModuleForAsrEncode,
      speech_exportables.LiteRTExportableModuleForAsrDecode,
  )


def get_speech_model_cls(
    model_type: str,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Gets ASR model class by model type."""
  ext = extension_lib.resolve_model_extension(model_type, export_config)
  if ext is not None:
    cls = ext.get_speech_model_cls(model_type)
    if cls is not None:
      return cls
  raise ValueError(f'Unsupported speech model type: {model_type}')


def get_tts_model_cls(
    model_type: str,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Gets TTS model class by model type."""
  ext = extension_lib.resolve_model_extension(model_type, export_config)
  if ext is not None:
    cls = ext.get_tts_model_cls(model_type)
    if cls is not None:
      return cls
  raise ValueError(f'Unsupported TTS model type: {model_type}')


def infer_tts_model_type(
    config_dict: dict[str, Any],
    model_path: str,
    export_config: exportable_module.ExportableModuleConfig | None = None,
) -> str | None:
  """Infers the TTS model type of a checkpoint that has no `model_type`."""
  return extension_lib.infer_tts_model_type_from_extensions(
      config_dict, model_path, export_config
  )


def infer_model_type(
    model_path: str,
    task: Any = None,
    config_dict: dict[str, Any] | None = None,
    export_config: exportable_module.ExportableModuleConfig | None = None,
) -> str | None:
  """Infers `model_type` for checkpoints without a standard HF `model_type`."""
  return extension_lib.infer_model_type_from_extensions(
      model_path=model_path,
      task=task,
      config_dict=config_dict,
      export_config=export_config,
  )


def get_image_gen_model_cls(
    model_type: str,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Gets ImageGen model class by model type or pipeline class name."""
  ext = extension_lib.resolve_model_extension(model_type, export_config)
  if ext is not None:
    cls = ext.get_image_gen_model_cls(model_type)
    if cls is not None:
      return cls
  raise ValueError(f'Unsupported ImageGen model type: {model_type}')


def get_any2any_model_cls(
    model_type: str,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Gets Any2Any multimodal model class by model type."""
  ext = extension_lib.resolve_model_extension(model_type, export_config)
  if ext is not None:
    cls = ext.get_any2any_model_cls(model_type)
    if cls is not None:
      return cls
  raise ValueError(f'Unsupported Any2Any model type: {model_type}')
