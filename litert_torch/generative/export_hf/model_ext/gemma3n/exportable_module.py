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

from typing import Any

from litert_torch.generative.export_hf.core import exportable_module as base_exportable_module
import torch


def get_language_model(model: Any) -> torch.nn.Module:
  """Returns the Gemma 3n text model from either multimodal or causal LM."""
  inner = model.model
  return getattr(inner, "language_model", inner)


def get_text_config(model: Any) -> Any:
  """Returns the Gemma 3n text config from either multimodal or causal LM."""
  config = model.config
  if hasattr(config, "get_text_config"):
    return config.get_text_config()
  return getattr(config, "text_config", config)


class LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder(
    base_exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill
):
  """Exportable module for prefill with external embedder."""

  def adapt_inputs(
      self,
      tokens,
      embeddings,
      per_layer_embeddings,
      input_pos,
      kv_cache,
      mask,
      **kwargs,
  ):
    inputs = super().adapt_inputs(
        tokens, embeddings, input_pos, kv_cache, mask, **kwargs
    )
    inputs["per_layer_inputs"] = per_layer_embeddings.to(
        inputs["inputs_embeds"].dtype
    )
    return inputs

  # pylint: disable=arguments-renamed
  def forward(  # pyrefly: ignore[bad-override]
      self,
      embeddings,
      per_layer_embeddings,
      input_pos,
      kv_cache,
      mask,
      **kwargs,
  ):
    kwargs.update(self.composite_flags())
    inputs = self.adapt_inputs(
        None,
        embeddings,
        per_layer_embeddings,
        input_pos,
        kv_cache,
        mask,
        use_bool_mask=self.export_config.extra_kwargs.get(
            "use_bool_mask", False
        ),
        **kwargs,
    )
    text_config = get_text_config(self.model)
    inputs["past_key_values"] = inputs[
        "past_key_values"
    ].insert_dummy_cache_layers(text_config)
    inputs |= self.attention_kwargs()
    output = get_language_model(self.model)(**inputs)
    kv_cache = output.past_key_values
    kv_cache = kv_cache.remove_dummy_cache_layers(text_config)
    return {"kv_cache": kv_cache}

  def _get_input(
      self, batch_size, prefill_length, prefill_length_dim, model_config
  ):
    device = self.device
    embeddings = {
        "embeddings": torch.ones(
            (batch_size, prefill_length, model_config.hidden_size),
            dtype=torch.float32,
            device=device,
        )
    }
    embeddings_dynamic_shape = (
        {"embeddings": {1: prefill_length_dim}} if prefill_length_dim else {}
    )
    embeddings |= {
        "per_layer_embeddings": torch.ones(
            (
                batch_size,
                prefill_length,
                model_config.num_hidden_layers,
                model_config.hidden_size_per_layer_input,
            ),
            dtype=torch.float32,
            device=device,
        )
    }
    embeddings_dynamic_shape |= (
        {"per_layer_embeddings": {1: prefill_length_dim}}
        if prefill_length_dim
        else {}
    )
    return embeddings, embeddings_dynamic_shape


class LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder(
    base_exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate
):
  """Exportable module for generate with external embedder."""

  def adapt_inputs(
      self,
      tokens,
      embeddings,
      per_layer_embeddings,
      input_pos,
      kv_cache,
      mask,
      **kwargs,
  ):
    inputs = super().adapt_inputs(
        tokens, embeddings, input_pos, kv_cache, mask, **kwargs
    )
    inputs["per_layer_inputs"] = per_layer_embeddings.to(
        inputs["inputs_embeds"].dtype
    )
    return inputs

  # pylint: disable=arguments-renamed
  def forward(  # pyrefly: ignore[bad-override]
      self,
      embeddings,
      per_layer_embeddings,
      input_pos,
      kv_cache,
      mask,
      **kwargs,
  ):
    kwargs.update(self.composite_flags())
    inputs = self.adapt_inputs(
        None,
        embeddings,
        per_layer_embeddings,
        input_pos,
        kv_cache,
        mask,
        use_bool_mask=self.export_config.extra_kwargs.get(
            "use_bool_mask", False
        ),
        **kwargs,
    )
    text_config = get_text_config(self.model)
    inputs["past_key_values"] = inputs[
        "past_key_values"
    ].insert_dummy_cache_layers(text_config)
    inputs |= self.attention_kwargs()
    output = get_language_model(self.model)(**inputs)
    hidden_states = output.last_hidden_state
    logits = self.model.lm_head(hidden_states).to(torch.float32)
    kv_cache = output.past_key_values
    kv_cache = kv_cache.remove_dummy_cache_layers(text_config)
    return {"kv_cache": kv_cache, "logits": logits}

  def _get_input(
      self, batch_size, decode_length, decode_length_dim, model_config
  ):
    device = self.device
    embeddings = {
        "embeddings": torch.ones(
            (batch_size, decode_length, model_config.hidden_size),
            dtype=torch.float32,
            device=device,
        )
    }
    embeddings_dynamic_shape = {"embeddings": None} if decode_length_dim else {}
    embeddings |= {
        "per_layer_embeddings": torch.ones(
            (
                batch_size,
                decode_length,
                model_config.num_hidden_layers,
                model_config.hidden_size_per_layer_input,
            ),
            dtype=torch.float32,
            device=device,
        )
    }
    embeddings_dynamic_shape |= (
        {"per_layer_embeddings": None} if decode_length_dim else {}
    )
    return embeddings, embeddings_dynamic_shape


class LiteRTExportableModuleForPerLayerEmbedder(torch.nn.Module):
  """Exportable module for embedder."""

  def __init__(self, model):
    super().__init__()
    self.model = model

  @property
  def device(self) -> torch.device:
    """Device of the wrapped model's parameters (`meta` for V2 export)."""
    return next(self.model.parameters(), torch.empty(0)).device

  def forward(
      self,
      token_ids,
  ):
    output = get_language_model(self.model).get_per_layer_inputs(token_ids)
    return {"embeddings": output.to(torch.float32)}

  def get_sample_inputs(
      self,
      model_config,
      export_config: base_exportable_module.ExportableModuleConfig,
      **kwargs,
  ):
    """Gets sample inputs."""
    del kwargs  # Unused.
    del model_config  # Unused.
    batch_size = export_config.batch_size
    device = self.device
    prefill_length_dim = export_config.prefill_length_dim
    tokens = {
        "token_ids": torch.ones(
            (batch_size, 1), dtype=torch.int32, device=device
        )
    }
    tokens_dynamic_shape = {"token_ids": {1: 1}} if prefill_length_dim else {}
    if export_config.single_token_embedder:
      return {"per_layer_embedder": (tokens, tokens_dynamic_shape)}
    else:
      ret = {}
      ret["decode_per_layer_embedder"] = (tokens, tokens_dynamic_shape)

      for prefill_length in export_config.prefill_lengths:
        tokens = {
            "token_ids": torch.ones(
                (batch_size, prefill_length), dtype=torch.int32, device=device
            )
        }
        tokens_dynamic_shape = (
            {"token_ids": {1: prefill_length_dim}} if prefill_length_dim else {}
        )
        ret[f"prefill_per_layer_embedder_{prefill_length}"] = (
            tokens,
            tokens_dynamic_shape,
        )
      return ret
