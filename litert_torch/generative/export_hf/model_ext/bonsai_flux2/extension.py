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
"""Built-in `ModelExtension` for Bonsai FLUX.2 (`bonsai_flux2`).

Illustrates `text_to_image` extension hooks:
- `get_image_gen_model_cls`: Returns the `BonsaiFlux2` multi-component pipeline
  wrapper (`bonsai_flux2/bonsai_flux2.py`).
- `get_litert_lm_section_types`: Maps exported submodels (`text_encoder`,
  `dit`, `vae_decoder`, `vae_encoder`) to `.litertlm` container section names.
- `match_checkpoint`: Distinguishes Bonsai FLUX.2 checkpoints from standard
  `Flux2KleinPipeline` Diffusers directories.
"""

from typing import Any

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import bonsai_flux2 as bonsai_flux2_lib

BONSAI_FLUX2_SECTION_TYPES = {
    'text_encoder': 'tf_lite_text_encoder',
    'dit': 'tf_lite_image_denoiser',
    'vae_decoder': 'tf_lite_image_decoder',
    'vae_encoder': 'tf_lite_image_encoder',
}


@extension_lib.register_model_extension(override=False)
class BonsaiFlux2Extension(extension_lib.ModelExtension):
  """Built-in extension for Bonsai FLUX.2 text-to-image pipelines."""

  model_types = ('bonsai_flux2',)

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict, task
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.TEXT_TO_IMAGE,
        export_kwargs={
            't2i_output_image_size': 512,
            'quantization_recipe': 'dynamic_wi4b32_afp32',
        },
        runtime_model_name='bonsai-flux2',
    )

  def get_image_gen_model_cls(self, model_type):
    """See base class."""
    del model_type
    return bonsai_flux2_lib.BonsaiFlux2

  def get_litert_lm_section_types(self):
    """See base class."""
    return dict(BONSAI_FLUX2_SECTION_TYPES)

  @classmethod
  def match_checkpoint(
      cls,
      model_path: str,
      task: Any = None,
      config_dict: dict[str, Any] | None = None,
  ) -> bool:
    """See base class."""
    del config_dict
    # prism-ml/bonsai-image-ternary-4B-unpacked also sets
    # "_class_name": "Flux2KleinPipeline" in its model_index.json, so check
    # for "bonsai" in model_path before reading model_index.json.
    task_name = getattr(task, 'value', task)
    if task_name == 'text_to_image' and 'bonsai' in str(model_path).lower():
      return True
    return False
