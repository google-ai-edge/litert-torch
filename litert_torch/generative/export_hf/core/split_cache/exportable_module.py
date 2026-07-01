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
"""Exportable module for split cache attention models."""

import copy
from litert_torch.generative.export_hf.core import cache as base_cache_lib
from litert_torch.generative.export_hf.core import exportable_module as base_exportable_module
from litert_torch.generative.export_hf.core import utils
from litert_torch.generative.export_hf.core.split_cache import attention_mask
from litert_torch.generative.export_hf.core.split_cache import cache as kv_cache_lib
import torch
from torch import nn


class LiteRTSplitCacheExportableModuleForDecoderOnlyLM(
    base_exportable_module.LiteRTExportableModuleForDecoderOnlyLM
):
  """Exportable module for decoder only LM."""

  def adapt_inputs(
      self,
      embeddings,
      pos_emb,
      mask,
      kv_cache,
  ):
    mask_global = mask['global']
    if 'local' in mask:
      mask_local = mask['local']
    else:
      mask_local = None
    pos_emb_cos = pos_emb['cos']
    pos_emb_sin = pos_emb['sin']
    if 'local_cos' in pos_emb:
      pos_emb_local_cos = pos_emb['local_cos']
      pos_emb_local_sin = pos_emb['local_sin']
    else:
      pos_emb_local_cos = None
      pos_emb_local_sin = None
    sliding_window = getattr(self.model.config, 'sliding_window', None)
    if not sliding_window:
      try:
        sliding_window = getattr(
            self.model.config.text_config, 'sliding_window'
        )
      except AttributeError:
        sliding_window = None

    if sliding_window is not None:
      masks = {
          'full_attention': mask_global,
      }
      if utils.has_sliding_attention(self.model):
        masks['sliding_attention'] = mask_local
    else:
      masks = mask_global

    ret = {}
    ret['inputs_embeds'] = embeddings

    cache_runtime_args = {'cache_position': torch.arange(embeddings.shape[1])}
    kv_cache.set_cache_runtime_args(cache_runtime_args)

    ret.update({
        'position_ids': torch.arange(embeddings.shape[1])[None, :],
        'past_key_values': kv_cache,
        'cache_position': torch.arange(embeddings.shape[1]),
        'attention_mask': masks,
        # Other common settings
        'use_cache': True,
    })

    language_model = self.model.model
    if hasattr(self.model.model, 'language_model'):
      language_model = self.model.model.language_model

    assert language_model.original_rotary_emb is not None
    language_model.rotary_emb.data = (
        pos_emb_cos.permute(0, 2, 1, 3).squeeze(0),
        pos_emb_sin.permute(0, 2, 1, 3).squeeze(0),
    )
    if utils.has_local_rope(self.model):
      assert pos_emb_local_cos is not None
      assert pos_emb_local_sin is not None
      rope_data = {
          'full_attention': language_model.rotary_emb.data,
          'sliding_attention': (
              pos_emb_local_cos.permute(0, 2, 1, 3).squeeze(0),
              pos_emb_local_sin.permute(0, 2, 1, 3).squeeze(0),
          ),
      }
      language_model.rotary_emb.data = rope_data

    return ret

  def post_process_kv_cache(self, output_cache):
    assert isinstance(output_cache, kv_cache_lib.LiteRTLMSplitCache)
    k_slices = []
    v_slices = []
    for layer in output_cache.layers:
      k_slices.append(layer.keys[1])
      v_slices.append(layer.values[1])
    assert all(x is not None for x in k_slices)
    assert all(x is not None for x in v_slices)
    return {'kv_slice_k': k_slices, 'kv_slice_v': v_slices}

  def _get_input(self, batch_size, input_length, cache_length):

    model_config = self.model.model.config
    if hasattr(model_config, 'text_config'):
      model_config = model_config.text_config
    embed_size_per_head = (
        getattr(model_config, 'head_dim', None)
        or model_config.hidden_size // model_config.num_attention_heads
    )
    if hasattr(model_config, 'global_head_dim'):
      global_embed_size_per_head = (
          model_config.global_head_dim or embed_size_per_head
      )
    else:
      global_embed_size_per_head = embed_size_per_head

    sample_inputs = {
        'embeddings': torch.ones(
            (batch_size, input_length, model_config.hidden_size),
            dtype=torch.float32,
        ),
    }
    pos_emb = {
        'cos': torch.ones(
            (1, input_length, 1, global_embed_size_per_head),
            dtype=torch.float32,
        ),
        'sin': torch.ones(
            (1, input_length, 1, global_embed_size_per_head),
            dtype=torch.float32,
        ),
    }
    if utils.has_local_rope(self.model):
      pos_emb.update({
          'local_cos': torch.ones(
              (1, input_length, 1, embed_size_per_head), dtype=torch.float32
          ),
          'local_sin': torch.ones(
              (1, input_length, 1, embed_size_per_head), dtype=torch.float32
          ),
      })

    mask_shape = (1, 1, input_length, cache_length + input_length)

    mask = {
        'global': torch.ones(mask_shape, dtype=torch.float32),
    }
    if utils.has_sliding_attention(self.model):
      mask.update({
          'local': torch.ones(mask_shape, dtype=torch.float32),
      })
    sample_inputs.update({
        'mask': mask,
        'pos_emb': pos_emb,
    })
    return sample_inputs


class LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill(
    LiteRTSplitCacheExportableModuleForDecoderOnlyLM
):
  """Exportable module for decoder only LM."""

  def forward(
      self,
      embeddings,
      pos_emb,
      mask,
      kv_cache,
  ):
    inputs = self.adapt_inputs(
        embeddings,
        pos_emb,
        mask,
        kv_cache,
    )
    inputs |= self.attention_kwargs()
    output = self.model(**inputs)
    output_cache = output.past_key_values
    return self.post_process_kv_cache(output_cache)

  def get_sample_inputs(
      self,
      model_config,
      **kwargs,
  ):
    export_config = self.export_config
    kv_cache_inputs, _ = self.get_sample_kv_cache(model_config)

    sample_inputs = {}
    for prefill_length in export_config.prefill_lengths:
      inputs = {
          **kv_cache_inputs,
          **self._get_input(
              export_config.batch_size,
              prefill_length,
              export_config.cache_length,
          ),
      }
      sample_inputs[f'prefill_{prefill_length}'] = (inputs, {})
    return sample_inputs


class LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate(
    LiteRTSplitCacheExportableModuleForDecoderOnlyLM
):
  """Exportable module for decoder only LM."""

  def forward(
      self,
      embeddings,
      pos_emb,
      mask,
      kv_cache,
  ):
    inputs = self.adapt_inputs(
        embeddings,
        pos_emb,
        mask,
        kv_cache,
    )
    inputs |= self.attention_kwargs()
    output = self.model(**inputs)
    output_cache = output.past_key_values
    ret = self.post_process_kv_cache(output_cache)
    ret['logits'] = output.logits
    return ret

  def get_sample_inputs(
      self,
      model_config,
      **kwargs,
  ):
    export_config = self.export_config
    kv_cache_inputs, _ = self.get_sample_kv_cache(model_config)
    sample_inputs = {
        **kv_cache_inputs,
        **self._get_input(
            export_config.batch_size,
            1,
            export_config.cache_length,
        ),
    }
    return {'decode': (sample_inputs, {})}


class SplitAttentionMaskBuilder(nn.Module):
  """Split attention mask builder."""

  def __init__(
      self,
      context_size: int,
      sliding_window_sizes: list[int | None] | None = None,
  ):
    super().__init__()
    if sliding_window_sizes is None:
      sliding_window_sizes = [None]
    local_masks = {}
    self.global_mask = attention_mask.SplitAttentionMask(context_size, None)
    for sliding_window_size in sliding_window_sizes:
      if sliding_window_size is not None:
        local_masks[sliding_window_size] = attention_mask.SplitAttentionMask(
            context_size, sliding_window_size
        )
      else:
        self.global_mask = attention_mask.SplitAttentionMask(context_size, None)
    self.local_masks = local_masks

  def forward(
      self,
      input_tokens: torch.Tensor,
      time_step: torch.Tensor,
      valid_mask: torch.Tensor,
  ) -> dict[str, attention_mask.SplitMask]:
    if self.global_mask is None:
      global_mask = None
    else:
      global_mask = self.global_mask(input_tokens, time_step, valid_mask)
    local_masks = {
        window_size: builder(input_tokens, time_step, valid_mask)
        for window_size, builder in self.local_masks.items()
    }

    if len(local_masks) == 1:
      local_masks = list(local_masks.values())[0]
    elif not len(local_masks):
      local_masks = None
    return {
        'mask': attention_mask.SplitMask(
            mask=global_mask, local_masks=local_masks
        )
    }

  @classmethod
  def get_sample_inputs(
      cls,
      model_config,
      export_config: base_exportable_module.ExportableModuleConfig,
  ):
    """Gets sample inputs."""
    del model_config
    sample_inputs = {}
    for prefill_length in export_config.prefill_lengths:
      inputs = {
          'input_tokens': torch.full((1, prefill_length), 0, dtype=torch.int32),
          'time_step': torch.tensor(1, dtype=torch.int32),
          'valid_mask': torch.ones((1, prefill_length), dtype=torch.bool),
      }
      sample_inputs[f'prefill_mask_{prefill_length}'] = (inputs, {})
    decode_inputs = {
        'input_tokens': torch.full((1, 1), 0, dtype=torch.int32),
        'time_step': torch.tensor(1, dtype=torch.int32),
        'valid_mask': torch.ones((1, 1), dtype=torch.bool),
    }
    sample_inputs['decode_mask'] = (decode_inputs, {})
    return sample_inputs


class CacheUpdate(torch.nn.Module):
  """Exportable module for decoder only LM running on NPU."""

  def forward(
      self,
      kv_slice: base_cache_lib.LiteRTLMCache,
      kv_cache: base_cache_lib.LiteRTLMCache,
      input_pos: torch.Tensor,
  ):
    assert len(kv_slice.layers) == len(kv_cache.layers)
    num_layers = len(kv_slice.layers)

    cache_kwargs = {'cache_position': input_pos, 'kv_slice_preprocessed': True}
    kv_cache.set_cache_runtime_args(cache_kwargs)

    for i in range(num_layers):
      # TODO(weiyiw): Fix for linear attention layers.
      assert hasattr(kv_slice.layers[i], 'keys')
      assert hasattr(kv_slice.layers[i], 'values')
      k_slice = kv_slice.layers[i].keys
      v_slice = kv_slice.layers[i].values
      kv_cache.update(k_slice, v_slice, i)

    return {'kv_cache': kv_cache}

  @classmethod
  def _get_input(
      cls,
      model_config,
      input_length,
      export_config: base_exportable_module.ExportableModuleConfig,
  ):
    """Gets sample inputs for the model."""
    kv_cache = base_cache_lib.LiteRTLMCache.create_from_config(
        model_config, export_config
    )
    slice_export_config = copy.deepcopy(export_config)
    slice_export_config.cache_length = input_length
    kv_slice = base_cache_lib.LiteRTLMCache.create_from_config(
        model_config, slice_export_config
    )
    return {
        'kv_cache': kv_cache,
        'kv_slice': kv_slice,
        'input_pos': torch.ones((input_length), dtype=torch.int32),
    }

  @classmethod
  def get_sample_inputs(
      cls,
      model_config,
      export_config: base_exportable_module.ExportableModuleConfig,
  ):
    """Gets sample inputs."""
    sample_inputs = {}
    for prefill_length in export_config.prefill_lengths:
      inputs = cls._get_input(
          model_config,
          prefill_length,
          export_config,
      )
      sample_inputs[f'prefill_cache_update_{prefill_length}'] = (inputs, {})
    decode_inputs = cls._get_input(
        model_config,
        1,
        export_config,
    )
    sample_inputs['decode_cache_update'] = (decode_inputs, {})
    return sample_inputs
