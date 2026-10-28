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
"""Built-in `ModelExtension` for FLUX.2 Klein (`flux2_klein`).

Illustrates multi-stage diffusion section registration via
`get_litert_lm_section_types` (sharded text encoders and double/single DiT
blocks) and Diffusers `model_index.json` `_class_name` matching via
`match_checkpoint` (`flux2_klein/flux2_klein.py`).
"""

from typing import Any

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import extension as bonsai_flux2_ext
from litert_torch.generative.export_hf.model_ext.flux2_klein import flux2_klein as flux2_klein_lib

FLUX2_KLEIN_SECTION_TYPES = {
    **bonsai_flux2_ext.BONSAI_FLUX2_SECTION_TYPES,
    'text_encoder_0': 'tf_lite_text_encoder_0',
    'text_encoder_1': 'tf_lite_text_encoder_1',
    'text_encoder_2': 'tf_lite_text_encoder_2',
    'dit_initial': 'tf_lite_diffusion_transformer_initial',
    'dit_double_block_0': 'tf_lite_diffusion_transformer_double_block_0',
    'dit_double_block_1': 'tf_lite_diffusion_transformer_double_block_1',
    'dit_single_block_0': 'tf_lite_diffusion_transformer_single_block_0',
    'dit_single_block_1': 'tf_lite_diffusion_transformer_single_block_1',
    'dit_single_block_2': 'tf_lite_diffusion_transformer_single_block_2',
    'dit_single_block_3': 'tf_lite_diffusion_transformer_single_block_3',
    'dit_final': 'tf_lite_diffusion_transformer_final',
}


@extension_lib.register_model_extension(override=False)
class Flux2KleinExtension(extension_lib.ModelExtension):
  """Built-in extension for FLUX.2 Klein text-to-image Diffusers pipelines."""

  model_types = ('Flux2KleinPipeline', 'flux2_klein')

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict, task
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.TEXT_TO_IMAGE,
        export_kwargs={
            't2i_output_image_size': 512,
            'quantization_recipe': 'dynamic_wi4b32_afp32',
        },
        runtime_model_name='flux2-klein',
    )

  def get_image_gen_model_cls(self, model_type):
    """See base class."""
    del model_type
    return flux2_klein_lib.Flux2Klein

  def get_litert_lm_section_types(self):
    """See base class."""
    return dict(FLUX2_KLEIN_SECTION_TYPES)

  @classmethod
  def match_checkpoint(
      cls,
      model_path: str,
      task: Any = None,
      config_dict: dict[str, Any] | None = None,
  ) -> bool:
    """See base class."""
    task_name = getattr(task, 'value', task)
    if (
        task_name == 'text_to_image'
        and config_dict
        and config_dict.get('_class_name') == 'Flux2KleinPipeline'
        and 'bonsai' not in str(model_path).lower()
    ):
      return True
    return False
