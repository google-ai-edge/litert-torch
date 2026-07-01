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
"""Exportable modules for extended modules."""

from litert_torch.generative.export_hf.model_ext.gemma3 import metadata_builder as gemma3_metadata_builder
from litert_torch.generative.export_hf.model_ext.gemma4 import metadata_builder as gemma4_metadata_builder
import transformers


def get_metadata_builder(
    model_config: transformers.PretrainedConfig,
):
  """Gets metadata builder."""
  if model_config.model_type == 'gemma3':
    return gemma3_metadata_builder.build_llm_metadata
  elif model_config.model_type == 'gemma3n':
    return gemma3_metadata_builder.build_llm_metadata
  elif (
      model_config.model_type == 'gemma4'
  ):
    return gemma4_metadata_builder.build_llm_metadata
  else:
    return (
        lambda source_model_artifacts, export_config, exported_model_artifacts, llm_metadata: llm_metadata
    )
