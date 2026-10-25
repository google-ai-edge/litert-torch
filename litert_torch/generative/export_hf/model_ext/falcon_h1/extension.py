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
"""Built-in `ModelExtension` for Falcon-H1 (`falcon_h1`).

Patch-only extension: Falcon-H1 uses the generic core prefill and decode
exportables, keeps the KV cache and the Mamba2 conv and SSM states of each
layer in `core/cache.py` `LiteRTLMHybridCacheLayer`, and needs only the patches
registered in `falcon_h1/patch.py`.
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
# Registers falcon_h1 patches.
from litert_torch.generative.export_hf.model_ext.falcon_h1 import patch as _


@extension_lib.register_model_extension(override=False)
class FalconH1Extension(extension_lib.ModelExtension):
  """Built-in extension for Falcon-H1 parallel attention and Mamba2 LLMs."""

  model_types = ('falcon_h1',)
