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
"""Built-in `ModelExtension` for Whisper ASR (`whisper`).

Provides the `Whisper` speech wrapper (`whisper/whisper.py`) and attention
patches (`whisper/patch.py`) while using the default ASR encode + decode
exportables in `model_ext/exportables.py`.
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
# Registers whisper patches.
from litert_torch.generative.export_hf.model_ext.whisper import patch as _
from litert_torch.generative.export_hf.model_ext.whisper import whisper as whisper_lib


@extension_lib.register_model_extension(override=False)
class WhisperExtension(extension_lib.ModelExtension):
  """Built-in extension for OpenAI Whisper encoder-decoder ASR models."""

  model_types = ('whisper',)

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict, task
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        export_kwargs={
            'input_sec': 30.0,
        },
        runtime_model_name='whisper',
    )

  def get_speech2text_bundle_spec(self, model_config):
    """See base class."""
    del model_config
    return extension_lib.Speech2TextBundleSpec(
        asr_model_type='whisper',
        tflite_model_type='encoder_decoder',
        mel_frames_per_second=100,
    )

  def get_speech_model_cls(self, model_type):
    """See base class."""
    del model_type
    return whisper_lib.Whisper
