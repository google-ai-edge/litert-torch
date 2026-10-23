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
"""Built-in ModelExtension for LFM2-VL."""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.lfm2_vl import patch as _  # Registers lfm2_vl patches.
from litert_torch.generative.export_hf.model_ext.lfm2_vl import vision_exportable as lfm2_vl_vision_exportable


@extension_lib.register_model_extension(override=False)
class Lfm2VlExtension(extension_lib.ModelExtension):
  """Built-in ModelExtension for LFM2-VL."""

  model_types = ('lfm2_vl',)

  def get_vision_exportables(self, model_config, export_config=None):
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
