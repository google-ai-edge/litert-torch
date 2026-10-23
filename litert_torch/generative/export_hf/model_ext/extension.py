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
"""Extension for HF integration."""

import dataclasses

from litert_torch.generative.export_hf.core import exportable_module
import transformers


def update_export_config(
    export_config: exportable_module.ExportableModuleConfig,
    model_config: transformers.PretrainedConfig,
) -> exportable_module.ExportableModuleConfig:
  """Updates export config."""
  if getattr(model_config, "model_type", None) == "flint":
    if export_config.moe_exports_implementation is None:
      export_config.moe_exports_implementation = (
          "litert_moe_sequential" if export_config.split_cache else "litert_moe"
      )
  return export_config
