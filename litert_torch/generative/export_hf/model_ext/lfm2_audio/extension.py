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
"""Built-in `ModelExtension` for LiquidAI LFM2.5-Audio (`lfm2_audio`)."""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.lfm2_audio import lfm2_audio as lfm2_audio_lib


@extension_lib.register_model_extension(override=False)
class Lfm2AudioExtension(extension_lib.ModelExtension):
  """Built-in extension for LFM2.5-Audio Any2Any multimodal models."""

  model_types = (
      'lfm2_audio',
      'lfm2-audio',
      'lfm2.5-audio-1.5b',
      'LFM2.5-Audio-1.5B',
  )

  @classmethod
  def match_checkpoint(
      cls,
      model_path: str,
      task=None,
      config_dict=None,
  ) -> bool:
    """Identifies LiquidAI LFM2.5-Audio checkpoints without top-level model_type."""
    task_name = getattr(task, 'value', task)
    if task_name not in (
        None,
        'auto',
        extension_lib.ExportTask.ANY_TO_ANY,
        extension_lib.ExportTask.ANY_TO_ANY.value,
    ):
      return False
    if config_dict:
      architectures = config_dict.get('architectures') or ()
      if 'Lfm2AudioForConditionalGeneration' in architectures:
        return True
      if 'lfm' in config_dict and 'depthformer' in config_dict:
        return True
    lowered_path = (model_path or '').lower()
    return (
        'lfm2.5-audio' in lowered_path
        or 'lfm2-audio' in lowered_path
        or 'lfm2_audio' in lowered_path
    )

  def get_export_defaults(self, config_dict, task=None, target='cpu_gpu'):
    """See base class."""
    del config_dict, task, target
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.ANY_TO_ANY,
    )

  def get_any2any_model_cls(self, model_type: str):
    """See base class."""
    del model_type
    return lfm2_audio_lib.Lfm2Audio
