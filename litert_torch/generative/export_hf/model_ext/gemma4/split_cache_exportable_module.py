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
from litert_torch.generative.export_hf.model_ext.gemma4 import exportable_module as gemma4_exportable_module
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
    # Match the decoder compute dtype (fp16 with experimental_use_fp16) so that
    # per-layer inputs do not promote the residual stream back to fp32.
    inputs["per_layer_inputs"] = per_layer_embeddings.to(
        inputs["inputs_embeds"].dtype
    )
    return inputs

  def _get_input(self, batch_size, input_length, cache_length):
    inputs = super()._get_input(batch_size, input_length, cache_length)
    text_config = gemma4_exportable_module.get_text_config(self.model)
    inputs["per_layer_embeddings"] = torch.ones(  # pyrefly: ignore[no-matching-overload]
        (
            batch_size,
            input_length,
            text_config.num_hidden_layers,
            text_config.hidden_size_per_layer_input,
        ),
        dtype=torch.float32,
        device=self.device,
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
    text_config = gemma4_exportable_module.get_text_config(self.model)
    inputs["past_key_values"] = inputs[
        "past_key_values"
    ].insert_dummy_cache_layers(text_config)
    inputs |= self.attention_kwargs()
    output = gemma4_exportable_module.get_language_model(self.model)(**inputs)
    kv_cache = output.past_key_values
    kv_cache = kv_cache.remove_dummy_cache_layers(text_config)
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
    # Match the decoder compute dtype (fp16 with experimental_use_fp16) so that
    # per-layer inputs do not promote the residual stream back to fp32.
    inputs["per_layer_inputs"] = per_layer_embeddings.to(
        inputs["inputs_embeds"].dtype
    )
    return inputs

  def _get_input(self, batch_size, input_length, cache_length):
    inputs = super()._get_input(batch_size, input_length, cache_length)
    text_config = gemma4_exportable_module.get_text_config(self.model)
    inputs["per_layer_embeddings"] = torch.ones(  # pyrefly: ignore[no-matching-overload]
        (
            batch_size,
            input_length,
            text_config.num_hidden_layers,
            text_config.hidden_size_per_layer_input,
        ),
        dtype=torch.float32,
        device=self.device,
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
    text_config = gemma4_exportable_module.get_text_config(self.model)
    inputs["past_key_values"] = inputs[
        "past_key_values"
    ].insert_dummy_cache_layers(text_config)
    inputs |= self.attention_kwargs()
    output = gemma4_exportable_module.get_language_model(self.model)(**inputs)
    hidden_states = output.last_hidden_state
    logits = self.model.lm_head(hidden_states)
    logits = gemma4_exportable_module.apply_final_logit_softcapping(
        logits, text_config
    )
    # Logits are always emitted in fp32: the runtime samplers expect fp32.
    logits = logits.to(torch.float32)
    kv_cache = output.past_key_values
    kv_cache = kv_cache.remove_dummy_cache_layers(text_config)
    return self.post_process_kv_cache(kv_cache) | {"logits": logits}
