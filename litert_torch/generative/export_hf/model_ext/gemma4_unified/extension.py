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
"""Built-in `ModelExtension` for Gemma 4 Unified (`gemma4_unified`).

Illustrates a vision-language variant whose vision encoder already incorporates
feature projection (`adapter_cls=None`) while exporting a dedicated end-of-image
(EOI) module (`gemma4_unified/vision_exportable.py`) and reusing Gemma 4's
`LlmMetadata` builder (`gemma4/metadata_builder.py`).
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.gemma4 import metadata_builder as gemma4_metadata_builder
# Registers gemma4_unified patches.
from litert_torch.generative.export_hf.model_ext.gemma4_unified import patch as _
from litert_torch.generative.export_hf.model_ext.gemma4_unified import vision_exportable as gemma4_unified_vision_exportable


@extension_lib.register_model_extension(override=False)
class Gemma4UnifiedExtension(extension_lib.ModelExtension):
  """Built-in extension for Gemma 4 Unified vision-language models."""

  model_types = ('gemma4_unified',)
  extra_kwargs_keys = ('gemma4_vision_max_soft_tokens',)

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict
    vlm = task in (None, extension_lib.ExportTask.IMAGE_TEXT_TO_TEXT)
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.IMAGE_TEXT_TO_TEXT,
        export_kwargs={'export_vision_encoder': vlm},
    )

  def get_vision_exportables(self, model_config, export_config=None):
    """See base class."""
    del model_config, export_config
    return extension_lib.VisionExportablesSpec(
        encoder_cls=(
            gemma4_unified_vision_exportable.LiteRTExportableModuleForGemma4UnifiedVisionEncoder
        ),
        adapter_cls=None,
        eoi_cls=(
            gemma4_unified_vision_exportable.LiteRTExportableModuleForGemma4UnifiedEndOfImage
        ),
    )

  def build_llm_metadata(
      self,
      source_model_artifacts,
      export_config,
      exported_model_artifacts,
      llm_metadata,
  ):
    """See base class."""
    return gemma4_metadata_builder.build_llm_metadata(
        source_model_artifacts,
        export_config,
        exported_model_artifacts,
        llm_metadata,
    )
