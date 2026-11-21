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
"""Built-in `ModelExtension` for Zamba2 (`zamba2`).

Patch-only extension: Zamba2 uses the generic core prefill and decode
exportables, keeps the Mamba2 conv and SSM states in `core/cache.py`
`LiteRTLMConvCacheLayer` and gives its hybrid layers, which run a shared
attention block before the Mamba2 mixer, `LiteRTLMHybridCacheLayer`, and needs
only the patch registered in `zamba2/patch.py` (the Mamba2 mixers).
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
# Registers zamba2 patches.
from litert_torch.generative.export_hf.model_ext.zamba2 import patch as _


@extension_lib.register_model_extension(override=False)
class Zamba2Extension(extension_lib.ModelExtension):
  """Built-in extension for Zamba2 Mamba2 and shared attention block LLMs."""

  model_types = ('zamba2',)
