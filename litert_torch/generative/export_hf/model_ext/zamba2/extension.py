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
only the patches registered in `zamba2/patch.py` (the RMSNorms and the Mamba2
mixers). It refuses the export options it does not support.
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
# Registers zamba2 patches.
from litert_torch.generative.export_hf.model_ext.zamba2 import patch as _


@extension_lib.register_model_extension(override=False)
class Zamba2Extension(extension_lib.ModelExtension):
  """Built-in extension for Zamba2 Mamba2 and shared attention block LLMs."""

  model_types = ('zamba2',)

  def update_export_config(self, export_config, model_config):
    """Refuses the split cache, and `externalize_rope` without `use_mem_rope`.

    The split cache has no Mamba2 or hybrid layer, and an externalized RoPE
    replaces the model's rotary embedding, which Zamba2 builds only with
    `use_mem_rope`.

    Args:
      export_config: Current export config.
      model_config: Loaded model config.

    Returns:
      `export_config`, unchanged.

    Raises:
      ValueError: If `split_cache` is set, or `externalize_rope` is set while
        `use_mem_rope` is False.
    """
    # The split cache also sets externalize_rope, so it is checked first.
    if export_config.split_cache:
      raise ValueError(
          'Zamba2 does not support split_cache; export without it.'
      )
    if export_config.externalize_rope and not model_config.use_mem_rope:
      raise ValueError(
          'Zamba2 does not support externalize_rope when use_mem_rope is'
          ' False; export without it.'
      )
    return export_config
