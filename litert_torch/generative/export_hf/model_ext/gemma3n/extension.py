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
"""Built-in `ModelExtension` for Gemma 3n (`gemma3n`).

Demonstrates an architecture with per-layer embeddings and vision inputs:
- Custom prefill/decode exportables accepting externalized token and per-layer
  embeddings (`gemma3n/exportable_module.py`).
- Auxiliary `'per_layer_embedder'` submodel via `get_additional_exportables`.
- MobileNet-based vision encoder and adapter (`gemma3n/vision_exportable.py`).
- Runtime layer patches (`gemma3n/patch.py`) and Gemma 3 family `LlmMetadata`
  (`gemma3/metadata_builder.py`).
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.gemma3 import metadata_builder as gemma3_metadata_builder
from litert_torch.generative.export_hf.model_ext.gemma3n import exportable_module as gemma3n_exportable
# Registers gemma3n patches.
from litert_torch.generative.export_hf.model_ext.gemma3n import patch as _
from litert_torch.generative.export_hf.model_ext.gemma3n import vision_exportable as gemma3n_vision_exportable


@extension_lib.register_model_extension(override=False)
class Gemma3nExtension(extension_lib.ModelExtension):
  """Built-in extension for Gemma 3n multimodal models."""

  model_types = ('gemma3n',)

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict
    # Gemma 3n always needs the external (per-layer) embedder.
    if task == extension_lib.ExportTask.TEXT_GENERATION:
      return extension_lib.ExportDefaults(
          task=extension_lib.ExportTask.TEXT_GENERATION,
          export_kwargs={
              'externalize_embedder': True,
              'export_vision_encoder': False,
          },
      )
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.IMAGE_TEXT_TO_TEXT,
        export_kwargs={
            'export_vision_encoder': True,
            'externalize_embedder': True,
            'vision_encoder_quantization_recipe': 'weight_only_wi8_afp32',
        },
    )

  def get_prefill_decode_exportables(self, model_config, export_config):
    """See base class."""
    del model_config
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

  def get_vision_exportables(self, model_config, export_config=None):
    """See base class."""
    del model_config, export_config
    return extension_lib.VisionExportablesSpec(
        encoder_cls=(
            gemma3n_vision_exportable.LiteRTExportableModuleForGemma3nVisionEncoder
        ),
        adapter_cls=(
            gemma3n_vision_exportable.LiteRTExportableModuleForGemma3nVisionAdapter
        ),
        eoi_cls=None,
    )

  def get_additional_exportables(self, model_config, export_config=None):
    """See base class."""
    del model_config, export_config
    return {
        'per_layer_embedder': (
            gemma3n_exportable.LiteRTExportableModuleForPerLayerEmbedder
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
    return gemma3_metadata_builder.build_llm_metadata(
        source_model_artifacts,
        export_config,
        exported_model_artifacts,
        llm_metadata,
    )
