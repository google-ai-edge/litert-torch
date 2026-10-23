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
"""Built-in ModelExtension for Kokoro TTS."""

from typing import Any

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.kokoro_tts import kokoro_tts as kokoro_tts_lib


@extension_lib.register_model_extension(override=False)
class KokoroTtsExtension(extension_lib.ModelExtension):
  """Built-in ModelExtension for Kokoro TTS."""

  model_types = ('kokoro', 'kokoro_tts', 'kokoro-82m', 'kokoro_82m')

  def get_tts_model_cls(self, model_type):
    del model_type
    return kokoro_tts_lib.KokoroTts

  @classmethod
  def is_matching_tts_checkpoint(
      cls, config_dict: dict[str, Any], model_path: str
  ) -> bool:
    return kokoro_tts_lib.KokoroTts.is_kokoro_checkpoint(
        config_dict, model_path
    )
