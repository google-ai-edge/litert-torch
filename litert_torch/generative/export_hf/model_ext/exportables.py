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
"""Exportable modules for extended modules."""

from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core.speech import exportables as speech_exportables
from litert_torch.generative.export_hf.model_ext.gemma3 import vision_exportable as gemma3_vision_exportable
from litert_torch.generative.export_hf.model_ext.gemma3n import exportable_module as gemma3n_exportable
from litert_torch.generative.export_hf.model_ext.gemma3n import vision_exportable as gemma3n_vision_exportable
from litert_torch.generative.export_hf.model_ext.gemma4 import exportable_module as gemma4_exportable
from litert_torch.generative.export_hf.model_ext.gemma4 import split_cache_exportable_module as gemma4_split_cache_exportable_module
from litert_torch.generative.export_hf.model_ext.gemma4 import vision_exportable as gemma4_vision_exportable
from litert_torch.generative.export_hf.model_ext.gemma4_unified import vision_exportable as gemma4_unified_vision_exportable
from litert_torch.generative.export_hf.model_ext.lfm2_vl import vision_exportable as lfm2_vl_vision_exportable
from litert_torch.generative.export_hf.model_ext.moonshine import moonshine as moonshine_lib
from litert_torch.generative.export_hf.model_ext.parakeet import parakeet_ctc as parakeet_ctc_lib
from litert_torch.generative.export_hf.model_ext.parakeet import parakeet_tdt as parakeet_tdt_lib
from litert_torch.generative.export_hf.model_ext.qwen3 import qwen3_asr as qwen3_asr_lib
from litert_torch.generative.export_hf.model_ext.qwen3_5 import exportable_module as qwen3_5_exportable
from litert_torch.generative.export_hf.model_ext.qwen3_tts import qwen3_tts as qwen3_tts_lib
from litert_torch.generative.export_hf.model_ext.whisper import whisper as whisper_lib
import transformers


def get_prefill_decode_exportables(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig,
):
  """Gets prefill-decode exportables."""
  if model_config.model_type == 'gemma3n':
    assert (
        not export_config.split_cache
    ), 'Split cache is not supported for Gemma3N.'
    assert (
        export_config.externalize_embedder
    ), 'External embedder is required for Gemma3N.'
    print('Using Gemma3N exportables.')
    return (
        gemma3n_exportable.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder,
        gemma3n_exportable.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder,
    )
  elif (
      model_config.model_type == 'gemma4'
  ):
    if model_config.get_text_config().hidden_size_per_layer_input:
      assert (
          export_config.externalize_embedder
      ), 'External embedder is required for Gemma4.'
      print('Using Gemma4 exportables.')
      if export_config.split_cache:
        return (
            gemma4_split_cache_exportable_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill,
            gemma4_split_cache_exportable_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate,
        )
      else:
        return (
            gemma4_exportable.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder,
            gemma4_exportable.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder,
        )
    else:
      return None
  elif model_config.model_type in ('qwen3_5', 'qwen3_5_text'):
    print('Using Qwen3.5 exportables.')
    if export_config.split_cache:
      return (
          qwen3_5_exportable.LiteRTSplitCacheExportableModuleForQwen3_5Prefill,
          qwen3_5_exportable.LiteRTSplitCacheExportableModuleForQwen3_5Generate,
      )
    else:
      return (
          qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Prefill,
          qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Generate,
      )
  else:
    pass
  return None


def get_vision_exportables(
    model_config: transformers.PretrainedConfig,
):
  """Gets vision exportables."""
  if model_config.model_type == 'gemma3':
    return (
        gemma3_vision_exportable.LiteRTExportableModuleForGemma3VisionEncoder,
        gemma3_vision_exportable.LiteRTExportableModuleForGemma3VisionAdapter,
        None,
    )
  elif model_config.model_type == 'gemma3n':
    return (
        gemma3n_vision_exportable.LiteRTExportableModuleForGemma3nVisionEncoder,
        gemma3n_vision_exportable.LiteRTExportableModuleForGemma3nVisionAdapter,
        None,
    )
  elif model_config.model_type == 'gemma4':
    return (
        gemma4_vision_exportable.LiteRTExportableModuleForGemma4VisionEncoder,
        gemma4_vision_exportable.LiteRTExportableModuleForGemma4VisionAdapter,
        None,
    )
  elif model_config.model_type == 'gemma4_unified':
    return (
        gemma4_unified_vision_exportable.LiteRTExportableModuleForGemma4UnifiedVisionEncoder,
        None,
        gemma4_unified_vision_exportable.LiteRTExportableModuleForGemma4UnifiedEndOfImage,
    )
  elif model_config.model_type == 'lfm2_vl':
    return (
        lfm2_vl_vision_exportable.LiteRTExportableModuleForLFM2VisionEncoder,
        lfm2_vl_vision_exportable.LiteRTExportableModuleForLFM2VisionAdapter,
        None,
    )
  else:
    raise ValueError(f'Unsupported model type: {model_config.model_type}')


def get_additional_exportables(
    model_config: transformers.PretrainedConfig,
):
  """Gets additional exportables."""
  if model_config.model_type == 'gemma3n':
    return {
        'per_layer_embedder': (
            gemma3n_exportable.LiteRTExportableModuleForPerLayerEmbedder
        ),
    }
  elif (
      model_config.model_type == 'gemma4'
  ):
    if not model_config.text_config.hidden_size_per_layer_input:
      return {}
    return {
        'per_layer_embedder': (
            gemma4_exportable.LiteRTExportableModuleForPerLayerEmbedder
        ),
    }
  else:
    pass
  return {}


def get_speech_exportables(
    model_config: transformers.PretrainedConfig,
):
  """Gets speech (ASR) exportables."""
  if model_config.model_type == 'parakeet_ctc':
    return (speech_exportables.LiteRTExportableModuleForAsrEncode,)
  else:
    return (
        speech_exportables.LiteRTExportableModuleForAsrEncode,
        speech_exportables.LiteRTExportableModuleForAsrDecode,
    )


def get_speech_model_cls(model_type: str):
  """Gets ASR model class by model type."""
  if model_type == 'parakeet_ctc':
    return parakeet_ctc_lib.ParakeetCTC
  elif model_type == 'parakeet_tdt':
    return parakeet_tdt_lib.ParakeetTDT
  elif model_type == 'whisper':
    return whisper_lib.Whisper
  elif model_type == 'moonshine':
    return moonshine_lib.Moonshine
  elif model_type == 'qwen3_asr':
    return qwen3_asr_lib.Qwen3Asr
  else:
    raise ValueError(f'Unsupported speech model type: {model_type}')


def get_tts_model_cls(model_type: str):
  """Gets TTS model class by model type."""
  if model_type in ('qwen3_tts', 'qwen3_tts_talker', 'qwen3'):
    return qwen3_tts_lib.Qwen3Tts
  else:
    raise ValueError(f'Unsupported TTS model type: {model_type}')
