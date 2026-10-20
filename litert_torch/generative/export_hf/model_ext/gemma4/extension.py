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
"""Built-in `ModelExtension` for Gemma 4 (`gemma4`, `gemma4_text`).

Full-featured multimodal extension illustrating:
- Custom architecture-specific export flag registration via `extra_kwargs_keys`
  (`'gemma4_vision_max_soft_tokens'`).
- Conditional custom prefill/decode exportables (both standard and split-cache)
  and `'per_layer_embedder'` auxiliary submodel when
  `hidden_size_per_layer_input` is active (`gemma4/exportable_module.py`,
  `gemma4/split_cache_exportable_module.py`).
- Vision encoder, adapter, and end-of-image (EOI) exportables
  (`gemma4/vision_exportable.py`).
- Audio encoder, adapter, and end-of-audio (EOA) exportables
  (`gemma4/audio_exportable.py`).
- Multimodal `LlmMetadata` builder (`gemma4/metadata_builder.py`) and runtime
  layer patches (`gemma4/patch.py`).
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.gemma4 import audio_exportable as gemma4_audio_exportable
from litert_torch.generative.export_hf.model_ext.gemma4 import exportable_module as gemma4_exportable
from litert_torch.generative.export_hf.model_ext.gemma4 import metadata_builder as gemma4_metadata_builder
# Registers gemma4 patches.
from litert_torch.generative.export_hf.model_ext.gemma4 import patch as _
from litert_torch.generative.export_hf.model_ext.gemma4 import split_cache_exportable_module as gemma4_split_cache_exportable_module
from litert_torch.generative.export_hf.model_ext.gemma4 import vision_exportable as gemma4_vision_exportable


@extension_lib.register_model_extension(override=False)
class Gemma4Extension(extension_lib.ModelExtension):
  """Built-in extension for Gemma 4 text, vision, and audio models."""

  model_types = (
      'gemma4',
      'gemma4_text',
  )
  extra_kwargs_keys = ('gemma4_vision_max_soft_tokens',)

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    vlm = config_dict.get('model_type') == 'gemma4' and task in (
        None,
        extension_lib.ExportTask.IMAGE_TEXT_TO_TEXT,
    )
    # E2B/E4B use per-layer embeddings and their own chat template.
    per_layer = extension_lib.text_config_dict(config_dict).get(
        'hidden_size_per_layer_input'
    )
    return extension_lib.ExportDefaults(
        task=(
            extension_lib.ExportTask.IMAGE_TEXT_TO_TEXT
            if vlm
            else extension_lib.ExportTask.TEXT_GENERATION
        ),
        export_kwargs={
            'quantization_recipe': 'dynamic_wi4b32_afp32',
            'externalize_embedder': True,
            'export_vision_encoder': vlm,
        },
        litert_lm_chat_template=(
            'gemma4/chat_template_e2b_e4b.jinja'
            if per_layer
            else 'gemma4/chat_template.jinja'
        ),
    )

  def get_prefill_decode_exportables(self, model_config, export_config):
    """See base class."""
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
    """See base class."""
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
    """See base class."""
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
    """See base class."""
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
    """See base class."""
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
