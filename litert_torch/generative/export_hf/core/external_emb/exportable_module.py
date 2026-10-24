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
from litert_torch.generative.export_hf.core import lora as lora_lib
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
      local_mask=None,
      **kwargs,
  ):
    if self.export_config.apply_gpu_composites:
      kwargs["apply_gpu_composites"] = True
    if self.export_config.use_sdpa_composite:
      kwargs["use_sdpa_composite"] = True
    inputs = self.adapt_inputs(
        None,
        embeddings,
        input_pos,
        kv_cache,
        mask,
        local_mask=local_mask,
        use_bool_mask=self.export_config.extra_kwargs.get(
            "use_bool_mask", False
        ),
        **kwargs,
    )
    inputs |= self.attention_kwargs()
    with lora_lib.bind_lora_weights(self.model, kwargs.get("lora")):
      output = self.model(**inputs)
    return {"kv_cache": output.past_key_values}

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
      local_mask=None,
      **kwargs,
  ):
    if self.export_config.apply_gpu_composites:
      kwargs["apply_gpu_composites"] = True
    if self.export_config.use_sdpa_composite:
      kwargs["use_sdpa_composite"] = True
    inputs = self.adapt_inputs(
        None,
        embeddings,
        input_pos,
        kv_cache,
        mask,
        local_mask=local_mask,
        use_bool_mask=self.export_config.extra_kwargs.get(
            "use_bool_mask", False
        ),
        **kwargs,
    )
    inputs |= self.attention_kwargs()
    with lora_lib.bind_lora_weights(self.model, kwargs.get("lora")):
      output = self.model(**inputs, output_hidden_states=self.export_verifier)
    if self.export_verifier:
      extra_outputs = {"activations": output["hidden_states"][-1]}
    else:
      extra_outputs = {}
    return {
        "kv_cache": output.past_key_values,
        "logits": output.logits.to(torch.float32),
        **extra_outputs,
    }

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
    return embeddings, embeddings_dynamic_shape


class LiteRTExportableModuleForEmbedder(torch.nn.Module):
  """Exportable module for embedder."""

  def __init__(self, model):
    super().__init__()
    self.model = model

  @property
  def device(self) -> torch.device:
    """Device of the wrapped embedding's parameters (`meta` for V2 export)."""
    return next(self.model.parameters(), torch.empty(0)).device

  def forward(
      self,
      token_ids,
  ):
    # Keep the zero as a 0-d CPU tensor: it becomes a lifted constant, and a
    # constant created on `meta` (V2 export) has no recoverable value.
    token_ids = torch.maximum(token_ids, torch.tensor(0, dtype=torch.int32))
    output = self.model(token_ids)
    return {"embeddings": output.to(torch.float32)}

  def get_sample_inputs(
      self,
      model_config,
      export_config: base_exportable_module.ExportableModuleConfig,
      **kwargs,
  ):
    """Gets sample inputs."""
    del kwargs  # Unused.
    batch_size = export_config.batch_size
    del model_config  # Unused.
    device = self.device
    prefill_length_dim = export_config.prefill_length_dim
    tokens = {
        "token_ids": torch.ones(
            (batch_size, 1), dtype=torch.int32, device=device
        )
    }
    tokens_dynamic_shape = {"token_ids": {1: 1}} if prefill_length_dim else {}
    if export_config.single_token_embedder:
      return {"embedder": (tokens, tokens_dynamic_shape)}
    else:
      ret = {}
      ret["decode_embedder"] = (tokens, tokens_dynamic_shape)

      for prefill_length in export_config.prefill_lengths:
        tokens = {
            "token_ids": torch.ones(
                (batch_size, prefill_length), dtype=torch.int32, device=device
            )
        }
        tokens_dynamic_shape = (
            {"token_ids": {1: prefill_length_dim}} if prefill_length_dim else {}
        )
        ret[f"prefill_embedder_{prefill_length}"] = (
            tokens,
            tokens_dynamic_shape,
        )
      return ret
