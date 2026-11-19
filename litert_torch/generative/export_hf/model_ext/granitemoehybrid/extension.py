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
"""Built-in `ModelExtension` for Granite-4.0-H (`granitemoehybrid`).

Patch-only extension: Granite-4.0-H uses the generic core prefill and decode
exportables, keeps the Mamba2 conv and SSM states in `core/cache.py`
`LiteRTLMConvCacheLayer`, and needs only the patches registered in
`granitemoehybrid/patch.py` (the RMSNorm class and the Mamba2 mixers).
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
# Registers granitemoehybrid patches.
from litert_torch.generative.export_hf.model_ext.granitemoehybrid import patch as _


@extension_lib.register_model_extension(override=False)
class GraniteMoeHybridExtension(extension_lib.ModelExtension):
  """Built-in extension for Granite-4.0-H hybrid Mamba2 + attention LLMs."""

  model_types = ('granitemoehybrid',)
