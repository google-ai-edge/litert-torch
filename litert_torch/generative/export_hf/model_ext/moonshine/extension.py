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
"""Built-in `ModelExtension` for Moonshine ASR (`moonshine`).

Provides the `Moonshine` speech wrapper (`moonshine/moonshine.py`) and runtime
patches (`moonshine/patch.py`) while inheriting the default ASR encode/decode
exportable pair from `model_ext/exportables.py`.
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.moonshine import moonshine as moonshine_lib
# Registers moonshine patches.
from litert_torch.generative.export_hf.model_ext.moonshine import patch as _


@extension_lib.register_model_extension(override=False)
class MoonshineExtension(extension_lib.ModelExtension):
  """Built-in extension for Moonshine encoder-decoder ASR models."""

  model_types = ('moonshine',)

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    del config_dict, task
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        export_kwargs={
            'input_sec': 5.0,
        },
        runtime_model_name='moonshine',
    )

  def get_speech2text_bundle_spec(self, model_config):
    """See base class."""
    del model_config
    return extension_lib.Speech2TextBundleSpec(
        asr_model_type='moonshine',
        tflite_model_type='encoder_decoder',
    )

  def get_speech_model_cls(self, model_type):
    """See base class."""
    del model_type
    return moonshine_lib.Moonshine
