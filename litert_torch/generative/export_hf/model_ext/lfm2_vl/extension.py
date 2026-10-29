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
"""Built-in `ModelExtension` for LFM2-VL (`lfm2_vl`).

Adds vision encoder and multimodal adapter exportables
(`lfm2_vl/vision_exportable.py`) plus vision-tower runtime patches
(`lfm2_vl/patch.py`) on top of an LFM2 text backbone (`text_config.model_type ==
'lfm2'`), demonstrating automatic composition of top-level and `text_config`
extensions via `resolve_model_extension`.
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
# Registers lfm2_vl patches.
from litert_torch.generative.export_hf.model_ext.lfm2_vl import patch as _
from litert_torch.generative.export_hf.model_ext.lfm2_vl import vision_exportable as lfm2_vl_vision_exportable


@extension_lib.register_model_extension(override=False)
class Lfm2VlExtension(extension_lib.ModelExtension):
  """Built-in extension for LFM2-VL vision-language models."""

  model_types = ('lfm2_vl',)

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict
    vlm = task in (None, extension_lib.ExportTask.IMAGE_TEXT_TO_TEXT)
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.IMAGE_TEXT_TO_TEXT,
        export_kwargs={
            'quantization_recipe': 'dynamic_wi4b32_afp32',
            'export_vision_encoder': vlm,
        },
        litert_lm_model_family=None if vlm else 'lfm2',
    )

  def get_vision_exportables(self, model_config, export_config=None):
    """See base class."""
    del model_config, export_config
    return extension_lib.VisionExportablesSpec(
        encoder_cls=(
            lfm2_vl_vision_exportable.LiteRTExportableModuleForLFM2VisionEncoder
        ),
        adapter_cls=(
            lfm2_vl_vision_exportable.LiteRTExportableModuleForLFM2VisionAdapter
        ),
        eoi_cls=None,
    )
