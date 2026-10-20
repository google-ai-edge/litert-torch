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
"""Built-in `ModelExtension` for Qwen3 TTS (`qwen3_tts`, `qwen3_tts_talker`).

Registers the `Qwen3Tts` wrapper (`qwen3_tts/qwen3_tts.py`) and runtime layer
patches (`qwen3_tts/patch.py`) for `text_to_speech` export.
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
# Registers qwen3_tts patches.
from litert_torch.generative.export_hf.model_ext.qwen3_tts import patch as _
from litert_torch.generative.export_hf.model_ext.qwen3_tts import qwen3_tts as qwen3_tts_lib


@extension_lib.register_model_extension(override=False)
class Qwen3TtsExtension(extension_lib.ModelExtension):
  """Built-in extension for Qwen3 text-to-speech checkpoints."""

  model_types = ('qwen3_tts', 'qwen3_tts_talker')

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict, task
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.TEXT_TO_SPEECH,
        runtime_model_name='qwen3-tts',
    )

  def get_tts_model_cls(self, model_type):
    """See base class."""
    del model_type
    return qwen3_tts_lib.Qwen3Tts
