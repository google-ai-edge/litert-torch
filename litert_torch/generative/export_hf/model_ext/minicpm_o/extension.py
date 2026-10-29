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
"""Built-in `ModelExtension` for MiniCPM-o (`minicpmo`, `minicpm_o`).

Illustrates `get_any2any_model_cls` for `any_to_any` omni-modal models
combining vision, audio, LLM, and speech synthesis submodels
(`minicpm_o/minicpm_o.py`).
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.minicpm_o import minicpm_o as minicpm_o_lib


@extension_lib.register_model_extension(override=False)
class MiniCpmOExtension(extension_lib.ModelExtension):
  """Built-in extension for MiniCPM-o Any2Any multimodal models."""

  model_types = (
      'minicpmo',
      'minicpm_o',
      'minicpm-o',
      'minicpm-o-4.5',
      'minicpm-o-4_5',
      'MiniCPM-o-4_5',
      'MiniCPMO',
  )

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict, task
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.ANY_TO_ANY,
    )

  def get_any2any_model_cls(self, model_type: str):
    """See base class."""
    del model_type
    return minicpm_o_lib.MiniCpmO
