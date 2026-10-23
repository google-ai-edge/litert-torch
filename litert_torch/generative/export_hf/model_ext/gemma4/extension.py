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
"""Built-in ModelExtension for Gemma4."""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.gemma4 import audio_exportable as gemma4_audio_exportable
from litert_torch.generative.export_hf.model_ext.gemma4 import exportable_module as gemma4_exportable
from litert_torch.generative.export_hf.model_ext.gemma4 import metadata_builder as gemma4_metadata_builder
from litert_torch.generative.export_hf.model_ext.gemma4 import patch as _  # Registers gemma4 patches.
from litert_torch.generative.export_hf.model_ext.gemma4 import split_cache_exportable_module as gemma4_split_cache_exportable_module
from litert_torch.generative.export_hf.model_ext.gemma4 import vision_exportable as gemma4_vision_exportable


@extension_lib.register_model_extension(override=False)
class Gemma4Extension(extension_lib.ModelExtension):
  """Built-in ModelExtension for Gemma4."""

  model_types = (
      'gemma4',
      'gemma4_text',
  )

  def get_prefill_decode_exportables(self, model_config, export_config):
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
      return (
          gemma4_exportable.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder,
          gemma4_exportable.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder,
      )
    return None

  def get_vision_exportables(self, model_config, export_config=None):
    del model_config, export_config
    return extension_lib.VisionExportablesSpec(
        encoder_cls=(
            gemma4_vision_exportable.LiteRTExportableModuleForGemma4VisionEncoder
        ),
        adapter_cls=(
            gemma4_vision_exportable.LiteRTExportableModuleForGemma4VisionAdapter
        ),
        eoi_cls=(
            gemma4_vision_exportable.LiteRTExportableModuleForGemma4EndOfImage
        ),
    )

  def get_audio_exportables(self, model_config, export_config=None):
    del model_config, export_config
    return extension_lib.AudioExportablesSpec(
        encoder_cls=(
            gemma4_audio_exportable.LiteRTExportableModuleForGemma4AudioEncoder
        ),
        adapter_cls=(
            gemma4_audio_exportable.LiteRTExportableModuleForGemma4AudioAdapter
        ),
        eoa_cls=(
            gemma4_audio_exportable.LiteRTExportableModuleForGemma4EndOfAudio
        ),
    )

  def get_additional_exportables(self, model_config, export_config=None):
    del export_config
    if not model_config.get_text_config().hidden_size_per_layer_input:
      return {}
    return {
        'per_layer_embedder': (
            gemma4_exportable.LiteRTExportableModuleForPerLayerEmbedder
        ),
    }

  def build_llm_metadata(
      self,
      source_model_artifacts,
      export_config,
      exported_model_artifacts,
      llm_metadata,
  ):
    if (
        getattr(source_model_artifacts.model_config, 'model_type', None)
        == 'gemma4_text'
    ):
      return llm_metadata
    return gemma4_metadata_builder.build_llm_metadata(
        source_model_artifacts,
        export_config,
        exported_model_artifacts,
        llm_metadata,
    )
