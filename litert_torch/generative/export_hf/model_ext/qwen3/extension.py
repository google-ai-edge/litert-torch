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
"""Built-in ModelExtensions for Qwen3 and Qwen3 ASR."""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.qwen3 import patch as _  # Registers qwen3 and qwen3_asr patches.
from litert_torch.generative.export_hf.model_ext.qwen3 import qwen3_asr as qwen3_asr_lib
from litert_torch.generative.export_hf.model_ext.qwen3_tts import qwen3_tts as qwen3_tts_lib


@extension_lib.register_model_extension(override=False)
class Qwen3Extension(extension_lib.ModelExtension):
  """Built-in ModelExtension for Qwen3 LLM."""

  model_types = ('qwen3', 'qwen3_text')

  def get_tts_model_cls(self, model_type):
    del model_type
    return qwen3_tts_lib.Qwen3Tts


@extension_lib.register_model_extension(override=False)
class Qwen3AsrExtension(extension_lib.ModelExtension):
  """Built-in ModelExtension for Qwen3 ASR."""

  model_types = ('qwen3_asr',)

  def get_speech_model_cls(self, model_type):
    del model_type
    return qwen3_asr_lib.Qwen3Asr
