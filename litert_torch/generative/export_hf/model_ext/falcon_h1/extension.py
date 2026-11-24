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
registered in `falcon_h1/patch.py`. It refuses the export options it does not
support.
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
# Registers falcon_h1 patches.
from litert_torch.generative.export_hf.model_ext.falcon_h1 import patch as _


@extension_lib.register_model_extension(override=False)
class FalconH1Extension(extension_lib.ModelExtension):
  """Built-in extension for Falcon-H1 parallel attention and Mamba2 LLMs."""

  model_types = ('falcon_h1',)

  def update_export_config(self, export_config, model_config):
    """Refuses the split cache and an externalized embedder.

    The split cache has no hybrid layer, and an externalized embedder would
    leave out the model's `embedding_multiplier`.

    Args:
      export_config: Current export config.
      model_config: Loaded model config.

    Returns:
      `export_config`, unchanged.

    Raises:
      ValueError: If `split_cache` or `externalize_embedder` is set.
    """
    del model_config
    if export_config.split_cache:
      raise ValueError(
          'Falcon-H1 does not support split_cache; export without it.'
      )
    if export_config.externalize_embedder:
      raise ValueError(
          'Falcon-H1 does not support externalize_embedder; export without it.'
      )
    return export_config
