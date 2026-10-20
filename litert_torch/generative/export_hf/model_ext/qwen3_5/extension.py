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
"""Built-in `ModelExtension` for Qwen3.5 (`qwen3_5`, `qwen3_5_text`).

Illustrates custom prefill/decode exportables for a hybrid full-attention +
linear-attention architecture across all three embedder/cache modes:
- Split-cache NPU export (`export_config.split_cache=True`).
- Externalized embedder GPU/CPU export (`export_config.externalize_embedder`).
- Bundled embedder prefill/decode export (`qwen3_5/exportable_module.py`).
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.qwen3_5 import exportable_module as qwen3_5_exportable
# Registers qwen3_5 patches.
from litert_torch.generative.export_hf.model_ext.qwen3_5 import patch as _


@extension_lib.register_model_extension(override=False)
class Qwen3_5Extension(extension_lib.ModelExtension):  # pylint: disable=invalid-name
  """Built-in extension for Qwen3.5 hybrid linear-attention LLMs."""

  model_types = ('qwen3_5', 'qwen3_5_text')

  def get_export_defaults(self, config_dict, task=None):
    """See base class."""
    # int4 exports can fail to load on the CPU backend (all-zero
    # int4 block scales), so Qwen3.5 defaults to int8 weights.
    del config_dict, task
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.TEXT_GENERATION,
        export_kwargs={
            'quantization_recipe': 'dynamic_wi8_afp32',
        },
        litert_lm_model_family='qwen3_5',
    )

  def get_prefill_decode_exportables(self, model_config, export_config):
    """See base class."""
    del model_config
    print('Using Qwen3.5 exportables.')
    if export_config.split_cache:
      return (
          qwen3_5_exportable.LiteRTSplitCacheExportableModuleForQwen3_5Prefill,
          qwen3_5_exportable.LiteRTSplitCacheExportableModuleForQwen3_5Generate,
      )
    if export_config.externalize_embedder:
      return (
          qwen3_5_exportable.LiteRTExportableModuleForQwen3_5PrefillExternalEmbedder,
          qwen3_5_exportable.LiteRTExportableModuleForQwen3_5GenerateExternalEmbedder,
      )
    return (
        qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Prefill,
        qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Generate,
    )
