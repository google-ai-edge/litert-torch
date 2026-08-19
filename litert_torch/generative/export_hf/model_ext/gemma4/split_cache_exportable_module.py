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

from litert_torch.generative.export_hf.core.split_cache import exportable_module as base_exportable_module
import torch


class LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill(
    base_exportable_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill
):
  """Exportable module for prefill with external embedder."""

  def adapt_inputs(  # pyrefly: ignore[bad-override]
      self,
      embeddings,
      per_layer_embeddings,
      pos_emb,
      mask,
      kv_cache,
  ):
    inputs = super().adapt_inputs(embeddings, pos_emb, mask, kv_cache)
    inputs["per_layer_inputs"] = per_layer_embeddings
    return inputs

  def _get_input(self, batch_size, input_length, cache_length):
    inputs = super()._get_input(batch_size, input_length, cache_length)
    inputs["per_layer_embeddings"] = torch.ones(  # pyrefly: ignore[no-matching-overload]
        (
            batch_size,
            input_length,
            self.model.config.text_config.num_hidden_layers,
            self.model.config.text_config.hidden_size_per_layer_input,
        ),
        dtype=torch.float32,
    )
    return inputs

  def forward(  # pyrefly: ignore[bad-override]
      self,
      embeddings,
      per_layer_embeddings,
      pos_emb,
      mask,
      kv_cache,
  ):
    inputs = self.adapt_inputs(
        embeddings,
        per_layer_embeddings,
        pos_emb,
        mask,
        kv_cache,
    )
    inputs["past_key_values"] = inputs[
        "past_key_values"
    ].insert_dummy_cache_layers(self.model.config.text_config)
    inputs |= self.attention_kwargs()
    output = self.model.model.language_model(**inputs)
    kv_cache = output.past_key_values
    kv_cache = kv_cache.remove_dummy_cache_layers(self.model.config.text_config)
    return self.post_process_kv_cache(kv_cache)


class LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate(
    base_exportable_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate
):
  """Exportable module for generate with external embedder."""

  def adapt_inputs(  # pyrefly: ignore[bad-override]
      self,
      embeddings,
      per_layer_embeddings,
      pos_emb,
      mask,
      kv_cache,
  ):
    inputs = super().adapt_inputs(embeddings, pos_emb, mask, kv_cache)
    inputs["per_layer_inputs"] = per_layer_embeddings
    return inputs

  def _get_input(self, batch_size, input_length, cache_length):
    inputs = super()._get_input(batch_size, input_length, cache_length)
    inputs["per_layer_embeddings"] = torch.ones(  # pyrefly: ignore[no-matching-overload]
        (
            batch_size,
            input_length,
            self.model.config.text_config.num_hidden_layers,
            self.model.config.text_config.hidden_size_per_layer_input,
        ),
        dtype=torch.float32,
    )
    return inputs

  def forward(  # pyrefly: ignore[bad-override]
      self,
      embeddings,
      per_layer_embeddings,
      pos_emb,
      mask,
      kv_cache,
  ):
    inputs = self.adapt_inputs(
        embeddings,
        per_layer_embeddings,
        pos_emb,
        mask,
        kv_cache,
    )
    inputs["past_key_values"] = inputs[
        "past_key_values"
    ].insert_dummy_cache_layers(self.model.config.text_config)
    inputs |= self.attention_kwargs()
    output = self.model.model.language_model(**inputs)
    hidden_states = output.last_hidden_state
    logits = self.model.lm_head(hidden_states)
    kv_cache = output.past_key_values
    kv_cache = kv_cache.remove_dummy_cache_layers(self.model.config.text_config)
    return self.post_process_kv_cache(kv_cache) | {"logits": logits}
