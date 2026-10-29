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
"""Built-in `ModelExtension`s for Qwen3 LLM and Qwen3 ASR.

Registers:
- `Qwen3Extension` (`qwen3`, `qwen3_text`): Applies Qwen3 RMSNorm and QKV/MLP
  composite patches (`qwen3/patch.py`) while using generic core prefill/decode
  exportables for text generation, and exposes `Qwen3Tts` when `qwen3` is used
  as a TTS checkpoint backbone.
- `Qwen3AsrExtension` (`qwen3_asr`): Provides the `Qwen3Asr` speech wrapper
  (`qwen3/qwen3_asr.py`) for `automatic_speech_recognition` export.
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
# Registers qwen3 and qwen3_asr patches.
from litert_torch.generative.export_hf.model_ext.qwen3 import patch as _
from litert_torch.generative.export_hf.model_ext.qwen3 import qwen3_asr as qwen3_asr_lib
from litert_torch.generative.export_hf.model_ext.qwen3_tts import qwen3_tts as qwen3_tts_lib


@extension_lib.register_model_extension(override=False)
class Qwen3Extension(extension_lib.ModelExtension):
  """Built-in extension for Qwen3 decoder-only LLMs (`qwen3`, `qwen3_text`)."""

  model_types = ('qwen3', 'qwen3_text')

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del task
    # LiteRT-LM export guide: int4 block-32 keeps quality for Qwen3-0.6B and
    # 4B+ on GPU and CPU; Qwen3-1.7B (hidden_size 2048) keeps int8 weights.
    hidden_size = extension_lib.text_config_dict(config_dict).get('hidden_size')
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.TEXT_GENERATION,
        export_kwargs={
            'quantization_recipe': (
                'dynamic_wi8_afp32'
                if hidden_size == 2048
                else 'dynamic_wi4b32_afp32'
            ),
        },
        litert_lm_model_family='qwen3',
    )

  def get_tts_model_cls(self, model_type):
    """See base class."""
    del model_type
    return qwen3_tts_lib.Qwen3Tts


@extension_lib.register_model_extension(override=False)
class Qwen3AsrExtension(extension_lib.ModelExtension):
  """Built-in extension for Qwen3 ASR (`qwen3_asr`)."""

  model_types = ('qwen3_asr',)

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict, task
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        export_kwargs={
            'input_sec': 5.0,
        },
        runtime_model_name='qwen3-asr',
    )

  def get_speech_model_cls(self, model_type):
    """See base class."""
    del model_type
    return qwen3_asr_lib.Qwen3Asr
