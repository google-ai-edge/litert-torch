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
"""Exportable module for externalized embedding."""

from litert_torch.generative.export_hf.core import exportable_module as base_exportable_module
import torch


class LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder(
    base_exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill
):
  """Exportable module for prefill with external embedder."""

  # pylint: disable=arguments-renamed
  def forward(
      self,
      embeddings,
      input_pos,
      kv_cache,
      mask,
  ):
    inputs = self.adapt_inputs(None, embeddings, input_pos, kv_cache, mask)
    output = self.model(**inputs)
    return {"kv_cache": output.past_key_values}

  def _get_input(
      self, batch_size, prefill_length, prefill_length_dim, model_config
  ):
    embeddings = {
        "embeddings": torch.ones(
            (batch_size, prefill_length, model_config.hidden_size),
            dtype=torch.float32,
        )
    }
    embeddings_dynamic_shape = (
        {"embeddings": {1: prefill_length_dim}} if prefill_length_dim else {}
    )
    return embeddings, embeddings_dynamic_shape


class LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder(
    base_exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate
):
  """Exportable module for generate with external embedder."""

  # pylint: disable=arguments-renamed
  def forward(
      self,
      embeddings,
      input_pos,
      kv_cache,
      mask,
  ):
    inputs = self.adapt_inputs(None, embeddings, input_pos, kv_cache, mask)
    output = self.model(**inputs)
    return {"kv_cache": output.past_key_values, "logits": output.logits}

  def _get_input(
      self, batch_size, decode_length, decode_length_dim, model_config
  ):
    embeddings = {
        "embeddings": torch.ones(
            (batch_size, decode_length, model_config.hidden_size),
            dtype=torch.float32,
        )
    }
    embeddings_dynamic_shape = {"embeddings": None} if decode_length_dim else {}
    return embeddings, embeddings_dynamic_shape


class LiteRTExportableModuleForEmbedder(torch.nn.Module):
  """Exportable module for embedder."""

  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(
      self,
      token_ids,
  ):
    token_ids = torch.maximum(token_ids, torch.tensor(0, dtype=torch.int32))
    output = self.model(token_ids)
    return {"embeddings": output}

  @classmethod
  def get_sample_inputs(
      cls,
      model_config,
      export_config: base_exportable_module.ExportableModuleConfig,
  ):
    """Gets sample inputs."""
    batch_size = export_config.batch_size
    prefill_length = export_config.prefill_lengths[0]
    prefill_length_dim = export_config.prefill_length_dim
    del model_config  # Unused.
    tokens = {"token_ids": torch.ones((batch_size, 1), dtype=torch.int32)}
    tokens_dynamic_shape = {"token_ids": {1: 1}} if prefill_length_dim else {}
    if export_config.single_token_embedder:
      return {"embedder": (tokens, tokens_dynamic_shape)}
    else:
      ret = {}
      ret["decode_embedder"] = (tokens, tokens_dynamic_shape)

      tokens = {
          "token_ids": torch.ones(
              (batch_size, prefill_length), dtype=torch.int32
          )
      }
      tokens_dynamic_shape = (
          {"token_ids": {1: prefill_length_dim}} if prefill_length_dim else {}
      )
      ret[f"prefill_embedder_{prefill_length}"] = (tokens, tokens_dynamic_shape)
      return ret
