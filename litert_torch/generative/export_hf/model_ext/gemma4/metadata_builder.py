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
"""Metadata builder for Gemma4."""

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
      llm_model_type_pb2.LlmModelType(gemma4=llm_model_type_pb2.Gemma4())
  )
  tokenizer = source_model_artifacts.tokenizer
  if not hasattr(tokenizer, 'special_tokens_map'):
    raise ValueError('Tokenizer does not have special_tokens_map.')
  token_map = tokenizer.special_tokens_map
  assert token_map is not None, 'Tokenizer does not have special_tokens_map.'
  llm_metadata.llm_model_type.gemma4.code_fence_start = token_map.get(
      'stc_token', ''
  )
  llm_metadata.llm_model_type.gemma4.code_fence_end = token_map.get(
      'etc_token', ''
  )
  llm_metadata.llm_model_type.gemma4.open_quote = token_map.get(
      'escape_token', ''
  )
  llm_metadata.llm_model_type.gemma4.close_quote = token_map.get(
      'escape_token', ''
  )
  llm_metadata.llm_model_type.gemma4.function_response_start = token_map.get(
      'str_token', ''
  )
  llm_metadata.llm_model_type.gemma4.use_template_for_fc_format = True
  think_channel = llm_metadata.channels.add()
  think_channel.channel_name = 'thought'
  think_channel.start = f"{token_map.get('soc_token', '<|channel>')}thought\n"
  think_channel.end = token_map.get('eoc_token', '<channel|>')
  if exported_model_artifacts.vision_encoder_model_path:
    image_processor = source_model_artifacts.image_processor
    assert (
        image_processor is not None
    ), 'Image processor is required for Gemma4 vision export.'
    boi_token = token_map.get('boi_token', '')
    eoi_token = token_map.get('eoi_token', '')
    llm_metadata.llm_model_type.gemma4.start_of_image_token.token_str = (
        boi_token
    )
    llm_metadata.llm_model_type.gemma4.end_of_image_token.token_str = eoi_token
    llm_metadata.llm_model_type.gemma4.patch_width = image_processor.patch_size
    llm_metadata.llm_model_type.gemma4.patch_height = image_processor.patch_size
    # TODO(weiyiw): Update the max_num_patches to the actual value.
    # This is is hardcoded to 140 soft tokens for now.
    llm_metadata.llm_model_type.gemma4.max_num_patches = (
        3
        * 3
        * export_config.extra_kwargs.get('gemma4_vision_max_soft_tokens', 140)
    )
  return llm_metadata
