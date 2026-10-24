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
"""Built-in ModelExtension for Moonshine."""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.moonshine import moonshine as moonshine_lib
from litert_torch.generative.export_hf.model_ext.moonshine import patch as _  # Registers moonshine patches.


@extension_lib.register_model_extension(override=False)
class MoonshineExtension(extension_lib.ModelExtension):
  """Built-in ModelExtension for Moonshine."""

  model_types = ('moonshine',)

  def get_speech_model_cls(self, model_type):
    del model_type
    return moonshine_lib.Moonshine
