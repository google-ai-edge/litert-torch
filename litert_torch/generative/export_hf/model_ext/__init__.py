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
"""Public entry point for `export_hf` pluggable model extensions.

Importing `litert_torch.generative.export_hf.model_ext`:
1. Eagerly registers all built-in model extensions via `builtin_extensions`
   (with `override=False` so user extensions registered via
   `@register_model_extension` or `model_extension=...` take precedence).
2. Registers LiteRT MoE expert forward implementations (`litert_moe`,
   `litert_moe_sequential`) with HuggingFace Transformers when available.
3. Re-exports the public `ModelExtension` base class, spec types, composition
   helpers, and registry functions from `model_ext.extension`.

See `model_ext/extension.py` for full hook documentation and
`model_ext/<arch>/extension.py` (e.g. `gemma3`, `gemma4`, `qwen3_5`, `whisper`,
`kokoro_tts`, `bonsai_flux2`, `minicpm_o`) for built-in implementations.
"""

from litert_torch.generative.export_hf.model_ext import builtin_extensions as _
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.layers import moe

try:
  from transformers.integrations import moe as transformers_moe

  transformers_moe.ALL_EXPERTS_FUNCTIONS.register(
      "litert_moe", moe.litert_moe_experts_forward
  )
  transformers_moe.ALL_EXPERTS_FUNCTIONS.register(
      "litert_moe_sequential", moe.litert_sequential_experts_forward
  )
except (ImportError, AttributeError):
  pass

AudioExportablesSpec = extension.AudioExportablesSpec
CompositeModelExtension = extension.CompositeModelExtension
ExportDefaults = extension.ExportDefaults
FusedWeightSpec = extension.FusedWeightSpec
ModelExtension = extension.ModelExtension
Speech2TextBundleSpec = extension.Speech2TextBundleSpec
VisionExportablesSpec = extension.VisionExportablesSpec
compose_extensions = extension.compose_extensions
get_model_extension = extension.get_model_extension
load_extension_modules = extension.load_extension_modules
register_model_extension = extension.register_model_extension
registered_model_types = extension.registered_model_types
resolve_model_extension = extension.resolve_model_extension
unregister_model_extension = extension.unregister_model_extension
