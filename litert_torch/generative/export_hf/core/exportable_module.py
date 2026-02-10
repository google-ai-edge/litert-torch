# Copyright 2025 The LiteRT Torch Authors.
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
"""Exportable modules."""

import abc
from litert_torch.generative.export_hf.core import cache as _
from litert_torch.generative.export_hf.core import cache_base as kv_cache_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import utils
import torch


ExportableModuleConfig = exportable_module_config.ExportableModuleConfig


class ExportableModuleBase(torch.nn.Module, abc.ABC):
  """Base class for exportable modules."""

  def __init__(self, export_config: ExportableModuleConfig):
    super().__init__()
    self._export_config = export_config

  @property
  def export_config(self) -> ExportableModuleConfig:
    return self._export_config

  def attention_kwargs(self):
    k_ts_idx = self.export_config.k_ts_idx
    v_ts_idx = self.export_config.v_ts_idx
    return {"k_ts_idx": k_ts_idx, "v_ts_idx": v_ts_idx}

  @abc.abstractmethod
  def get_sample_inputs(
      self, model_config
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    ...


class LiteRTExportableModuleForDecoderOnlyLM(ExportableModuleBase):
  """Base class for exportable modules for decoder-only LM."""

  def __init__(
      self, model: torch.nn.Module, export_config: ExportableModuleConfig
  ):
    super().__init__(export_config)
    self.model = model

  def adapt_inputs(
      self,
      tokens,
      embeddings,
      input_pos,
      kv_cache,
      mask,
  ):
    sliding_window = getattr(self.model.config, "sliding_window", None)
    # TODO(weiyiw): This is a hack to check if it's Mistral.
    is_mistral = getattr(self.model.config, "model_type", "") == "mistral"
    if sliding_window is not None:
      layer_types = getattr(self.model.config, "layer_types", None)
      masks = {
          "full_attention": mask,
      }
      need_sliding_mask = (
          layer_types is not None and "sliding_attention" in layer_types
      ) or is_mistral
      if need_sliding_mask:
        masks["sliding_attention"] = (
            utils.create_sliding_mask(
                input_pos.clone().unsqueeze(0),
                kv_cache.get_max_cache_shape(),
                self.model.config.sliding_window,
            )
            + mask
        )
      if is_mistral:
        masks = masks["sliding_attention"]
    else:
      masks = mask

    ret = {}
    if embeddings is not None:
      ret["inputs_embeds"] = embeddings
    else:
      ret["input_ids"] = tokens

    ret.update({
        "position_ids": input_pos.clone().unsqueeze(0),
        "past_key_values": kv_cache,
        "cache_position": input_pos,
        "attention_mask": masks,
        # Other common settings
        "use_cache": True,
    })
    return ret

  def get_sample_kv_cache(self, model_config):
    """Returns the input sample KV cache for the model."""
    export_config = self.export_config
    num_layers = model_config.num_hidden_layers
    kv_cache = kv_cache_lib.CACHE_REGISTRY[
        export_config.cache_implementation
    ].create_from_config(
        model_config,
        export_config,
        k_ts_idx=export_config.k_ts_idx,
        v_ts_idx=export_config.v_ts_idx,
        batch_size=export_config.batch_size,
        cache_length=export_config.cache_length,
    )
    inputs = {"kv_cache": kv_cache}
    if export_config.cache_length_dim is not None:
      all_k_shapes = tuple(
          {2: export_config.cache_length_dim} for _ in range(num_layers)
      )
      all_v_shapes = tuple(
          {3: export_config.cache_length_dim} for _ in range(num_layers)
      )
      dynamic_shapes = {
          "kv_cache": {
              "k": all_k_shapes,
              "v": all_v_shapes,
          }
      }
      return inputs, dynamic_shapes
    else:
      return inputs, {}


class LiteRTExportableModuleForDecoderOnlyLMPrefill(
    LiteRTExportableModuleForDecoderOnlyLM
):
  """Exportable module for prefill."""

  def forward(
      self,
      tokens,
      input_pos,
      kv_cache,
      mask,
  ):
    inputs = self.adapt_inputs(tokens, None, input_pos, kv_cache, mask)
    inputs |= self.attention_kwargs()
    output = self.model(**inputs)
    return {"kv_cache": output.past_key_values}

  def _get_input(
      self, batch_size, prefill_length, prefill_length_dim, model_config
  ):
    del model_config  # Unused.
    tokens = {
        "tokens": torch.ones((batch_size, prefill_length), dtype=torch.int32)
    }
    tokens_dynamic_shape = (
        {"tokens": {1: prefill_length_dim}} if prefill_length_dim else {}
    )
    return tokens, tokens_dynamic_shape

  def get_sample_inputs(self, model_config):
    export_config = self.export_config
    kv_cache_inputs, kv_cache_dynamic_shapes = self.get_sample_kv_cache(
        model_config
    )
    batch_size = export_config.batch_size
    cache_length = export_config.cache_length
    sample_inputs = {}
    for prefill_length in export_config.prefill_lengths:
      tokens, tokens_dynamic_shape = self._get_input(
          batch_size,
          prefill_length,
          export_config.prefill_length_dim,
          model_config,
      )
      inputs = {
          **tokens,
          "input_pos": torch.ones((prefill_length), dtype=torch.int32),
          "mask": torch.ones(
              (1, 1, prefill_length, cache_length), dtype=torch.float32
          ),
      }
      inputs.update(kv_cache_inputs)
      if export_config.prefill_length_dim is not None:
        dynamic_shapes = {
            **tokens_dynamic_shape,
            "mask": {
                2: export_config.prefill_length_dim,
                3: export_config.cache_length_dim,
            },
            "input_pos": {0: export_config.prefill_length_dim},
        }
        dynamic_shapes.update(kv_cache_dynamic_shapes)
        sample_inputs[f"prefill_{prefill_length}"] = (inputs, dynamic_shapes)
      else:
        sample_inputs[f"prefill_{prefill_length}"] = (inputs, {})
    return sample_inputs


class LiteRTExportableModuleForDecoderOnlyLMGenerate(
    LiteRTExportableModuleForDecoderOnlyLM
):
  """Exportable module for generate / decode."""

  def forward(
      self,
      tokens,
      input_pos,
      kv_cache,
      mask,
  ):
    inputs = self.adapt_inputs(tokens, None, input_pos, kv_cache, mask)
    inputs |= self.attention_kwargs()
    output = self.model(**inputs)
    return {"kv_cache": output.past_key_values, "logits": output.logits}

  def _get_input(
      self, batch_size, decode_length, decode_length_dim, model_config
  ):
    del model_config  # Unused.
    tokens = {
        "tokens": torch.ones((batch_size, decode_length), dtype=torch.int32)
    }
    tokens_dynamic_shape = {"tokens": None} if decode_length_dim else {}
    return tokens, tokens_dynamic_shape

  def get_sample_inputs(self, model_config):
    export_config = self.export_config
    kv_cache_inputs, kv_cache_dynamic_shapes = self.get_sample_kv_cache(
        model_config
    )
    batch_size = export_config.batch_size
    cache_length = export_config.cache_length
    tokens, tokens_dynamic_shape = self._get_input(
        batch_size,
        1,
        export_config.prefill_length_dim,
        model_config,
    )
    inputs = {
        **tokens,
        "input_pos": torch.ones((1), dtype=torch.int32),
        "mask": torch.ones((1, 1, 1, cache_length), dtype=torch.float32),
    }
    inputs.update(kv_cache_inputs)
    if export_config.cache_length_dim is not None:
      decode_dynamic_shapes = {
          **tokens_dynamic_shape,
          "mask": {3: export_config.cache_length_dim},
          "input_pos": None,
      }
      decode_dynamic_shapes.update(kv_cache_dynamic_shapes)
    else:
      decode_dynamic_shapes = {}
    return {"decode": (inputs, decode_dynamic_shapes)}
