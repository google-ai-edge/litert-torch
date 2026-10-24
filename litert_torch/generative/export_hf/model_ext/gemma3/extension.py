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
"""Built-in ModelExtension for Gemma3."""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.gemma3 import metadata_builder as gemma3_metadata_builder
from litert_torch.generative.export_hf.model_ext.gemma3 import patch as _  # Registers gemma3 patches.
from litert_torch.generative.export_hf.model_ext.gemma3 import vision_exportable as gemma3_vision_exportable


@extension_lib.register_model_extension(override=False)
class Gemma3Extension(extension_lib.ModelExtension):
  """Built-in ModelExtension for Gemma3."""

  model_types = ('gemma3', 'gemma3_text')

  def get_vision_exportables(self, model_config, export_config=None):
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
    return gemma3_metadata_builder.build_llm_metadata(
        source_model_artifacts,
        export_config,
        exported_model_artifacts,
        llm_metadata,
    )
