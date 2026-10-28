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
"""Built-in `ModelExtension` for Gemma 3 (`gemma3`, `gemma3_text`).

Customizes Gemma 3 text-only and vision-language exports via:
- Runtime RMSNorm and QKV/MLP/RoPE fusion patches (`gemma3/patch.py`).
- SigLIP vision encoder and multimodal projector exportables
  (`gemma3/vision_exportable.py`), while reusing the generic core prefill/decode
  exportables for the text backbone.
- Multimodal token IDs and prompt formatting in `LlmMetadata`
  (`gemma3/metadata_builder.py`).
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.gemma3 import metadata_builder as gemma3_metadata_builder
# Registers gemma3 patches.
from litert_torch.generative.export_hf.model_ext.gemma3 import patch as _
from litert_torch.generative.export_hf.model_ext.gemma3 import vision_exportable as gemma3_vision_exportable


@extension_lib.register_model_extension(override=False)
class Gemma3Extension(extension_lib.ModelExtension):
  """Built-in extension for Gemma 3 text and vision-language models."""

  model_types = ('gemma3', 'gemma3_text')

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    if (
        config_dict.get('model_type') == 'gemma3_text'
        or task == extension_lib.ExportTask.TEXT_GENERATION
    ):
      return extension_lib.ExportDefaults(
          task=extension_lib.ExportTask.TEXT_GENERATION,
          export_kwargs={'export_vision_encoder': False},
      )
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.IMAGE_TEXT_TO_TEXT,
        export_kwargs={
            'export_vision_encoder': True,
            'externalize_embedder': True,
            'vision_encoder_quantization_recipe': 'weight_only_wi8_afp32',
        },
    )

  def get_vision_exportables(self, model_config, export_config=None):
    """See base class."""
    del model_config, export_config
    return extension_lib.VisionExportablesSpec(
        encoder_cls=(
            gemma3_vision_exportable.LiteRTExportableModuleForGemma3VisionEncoder
        ),
        adapter_cls=(
            gemma3_vision_exportable.LiteRTExportableModuleForGemma3VisionAdapter
        ),
        eoi_cls=None,
    )

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
