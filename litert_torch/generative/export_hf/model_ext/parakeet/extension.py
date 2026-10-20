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
"""Built-in `ModelExtension`s for Parakeet CTC and Parakeet TDT ASR.

Contrasts two speech export patterns:
- `ParakeetCtcExtension` (`parakeet_ctc`): Overrides `get_speech_exportables`
  to export only `LiteRTExportableModuleForAsrEncode` (encoder-only CTC).
- `ParakeetTdtExtension` (`parakeet_tdt`): Uses the default ASR encode + decode
  exportables with the `ParakeetTDT` wrapper (`parakeet/parakeet_tdt.py`).
"""

from litert_torch.generative.export_hf.core.speech import exportables as speech_exportables
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.parakeet import parakeet_ctc as parakeet_ctc_lib
from litert_torch.generative.export_hf.model_ext.parakeet import parakeet_tdt as parakeet_tdt_lib
# Registers parakeet patches.
from litert_torch.generative.export_hf.model_ext.parakeet import patch as _


@extension_lib.register_model_extension(override=False)
class ParakeetCtcExtension(extension_lib.ModelExtension):
  """Built-in extension for encoder-only Parakeet CTC ASR models."""

  model_types = ('parakeet_ctc',)

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict, task
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        export_kwargs={
            'input_sec': 5.0,
        },
        runtime_model_name='parakeet-ctc',
    )

  def get_speech2text_bundle_spec(self, model_config):
    """See base class."""
    del model_config
    return extension_lib.Speech2TextBundleSpec(
        asr_model_type='parakeet',
        tflite_model_type='audio_encoder',
        mel_frames_per_second=100,
    )

  def get_speech_exportables(self, model_config, export_config=None):
    """See base class."""
    del model_config, export_config
    return (speech_exportables.LiteRTExportableModuleForAsrEncode,)

  def get_speech_model_cls(self, model_type):
    """See base class."""
    del model_type
    return parakeet_ctc_lib.ParakeetCTC


@extension_lib.register_model_extension(override=False)
class ParakeetTdtExtension(extension_lib.ModelExtension):
  """Built-in extension for Parakeet Token-and-Duration Transducer (TDT) ASR."""

  model_types = ('parakeet_tdt',)

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict, task
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        export_kwargs={
            'input_sec': 5.0,
        },
        runtime_model_name='parakeet-tdt',
    )

  def get_speech2text_bundle_spec(self, model_config):
    """See base class."""
    del model_config
    return extension_lib.Speech2TextBundleSpec(
        asr_model_type='parakeet',
        tflite_model_type='encoder_decoder',
        mel_frames_per_second=100,
    )

  def get_speech_model_cls(self, model_type):
    """See base class."""
    del model_type
    return parakeet_tdt_lib.ParakeetTDT
