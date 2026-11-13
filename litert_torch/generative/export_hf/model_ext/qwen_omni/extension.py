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
"""Built-in `ModelExtension` for Qwen2.5-Omni-3B Any2Any models."""

import re

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.qwen_omni import qwen_omni as qwen_omni_lib

_QWEN_OMNI_PATTERN = re.compile(r'qwen(?:2[._-]?5)?[._-]?omni', re.IGNORECASE)


@extension_lib.register_model_extension(override=False)
class QwenOmniExtension(extension_lib.ModelExtension):
  """Built-in extension for Qwen2.5-Omni Any2Any models."""

  model_types = (
      'qwen2_5_omni',
      'qwen2_5_omni_thinker',
      'qwen2.5-omni',
      'qwen2.5-omni-3b',
      'Qwen2.5-Omni-3B',
      'qwen_omni',
      'qwen-omni',
  )
  extra_kwargs_keys = (
      'generate_audio',
      'tts_max_cache_len',
      'tts_num_layers',
      'vocoder_chunk_frames',
      'vocoder_output_frames',
      'ode_steps',
      'ref_mel_frames',
      'vision_image_size',
      'vocoder_quantization_recipe',
      'speaker',
  )

  @classmethod
  def match_checkpoint(cls, model_path, task=None, config_dict=None) -> bool:
    """Matches Qwen2.5-Omni checkpoints by repo ID, `_name_or_path`, or arch."""
    task_val = getattr(task, 'value', task)
    if task_val not in (
        None,
        extension_lib.ExportTask.ANY_TO_ANY.value,
    ):
      return False
    cfg = config_dict or {}
    name_or_path = cfg.get('_name_or_path') or ''
    model_type = cfg.get('model_type') or ''
    architectures = ' '.join(str(a) for a in cfg.get('architectures') or ())
    return bool(
        _QWEN_OMNI_PATTERN.search(str(model_path or ''))
        or _QWEN_OMNI_PATTERN.search(str(name_or_path))
        or _QWEN_OMNI_PATTERN.search(str(model_type))
        or _QWEN_OMNI_PATTERN.search(architectures)
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
    return qwen_omni_lib.QwenOmni
