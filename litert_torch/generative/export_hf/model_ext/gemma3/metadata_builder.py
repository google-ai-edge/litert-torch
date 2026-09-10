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
"""Metadata builder for Gemma3."""

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module

from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2


def build_llm_metadata(
    source_model_artifacts: export_lib.SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: export_lib.ExportedModelArtifacts,
    llm_metadata: llm_metadata_pb2.LlmMetadata,
) -> llm_metadata_pb2.LlmMetadata:
  """Builds LLM metadata."""
  llm_metadata.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(gemma3=llm_model_type_pb2.Gemma3())
  )
  if (
      export_config.task != 'image_text_to_text'
      or not export_config.export_vision_encoder
  ):
    return llm_metadata
  tokenizer = source_model_artifacts.tokenizer
  image_processor = source_model_artifacts.image_processor
  if exported_model_artifacts.vision_encoder_model_path:
    if not hasattr(tokenizer, 'special_tokens_map'):
      raise ValueError('Tokenizer does not have special_tokens_map.')
    token_map = tokenizer.special_tokens_map
    boi_token = token_map.get('boi_token', '')
    eoi_token = token_map.get('eoi_token', '')
    llm_metadata.llm_model_type.gemma3.start_of_image_token.token_str = (
        boi_token  # pyrefly: ignore[bad-assignment]
    )
    llm_metadata.llm_model_type.gemma3.end_of_image_token.token_str = eoi_token  # pyrefly: ignore[bad-assignment]
    llm_metadata.llm_model_type.gemma3.image_tensor_height = (
        image_processor.size['height']  # pyrefly: ignore[missing-attribute]
    )
    llm_metadata.llm_model_type.gemma3.image_tensor_width = (
        image_processor.size['width']  # pyrefly: ignore[missing-attribute]
    )
  return llm_metadata
