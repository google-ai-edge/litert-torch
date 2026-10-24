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
"""Export HF model extensions."""

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
ModelExtension = extension.ModelExtension
VisionExportablesSpec = extension.VisionExportablesSpec
compose_extensions = extension.compose_extensions
get_model_extension = extension.get_model_extension
load_extension_modules = extension.load_extension_modules
register_model_extension = extension.register_model_extension
registered_model_types = extension.registered_model_types
resolve_model_extension = extension.resolve_model_extension
unregister_model_extension = extension.unregister_model_extension
