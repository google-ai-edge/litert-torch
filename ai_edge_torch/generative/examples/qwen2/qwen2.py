# Copyright 2024 The AI Edge Torch Authors.
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
# Example of building the qwen2 model.

import os
from pathlib import Path
from typing import Optional, Tuple
import copy
from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.attention_utils as attn_utils
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn

# Below are weight mappings from the model definition code
# you can go the modeling code and find the tensor names
# inside the class, for example for the first 3 below
# they are defined in class Qwen2MLP:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L213
TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.layers.{}.self_attn.q_proj",# from Qwen2Attention class
    attn_key_proj="model.layers.{}.self_attn.k_proj",# from Qwen2Attention class
    attn_value_proj="model.layers.{}.self_attn.v_proj",# from Qwen2Attention class
    attn_output_proj="model.layers.{}.self_attn.o_proj",# from Qwen2Attention class
    pre_attn_norm="model.layers.{}.input_layernorm", # from Qwen2DecoderLayer class'
    post_attn_norm="model.layers.{}.post_attention_layernorm",# from Qwen2DecoderLayer class'
    embedding="model.embed_tokens",#from Qwen2Model model
    final_norm="model.norm",# from Qwen2Model model
)


class Qwen2Model(nn.Module):

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    self.config = config
    # Construct model layers.
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.lm_head = nn.Linear(
        config.embedding_dim,
        config.vocab_size,
        bias=config.lm_head_use_bias,
    )
    if config.lm_head_share_weight_with_embedding:
      self.lm_head.weight.data = self.tok_embedding.weight.data
    self.transformer_blocks = nn.ModuleList(
        attention.TransformerBlock(config.block_config(idx), config)
        for idx in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    # ROPE parameters for all attn_configs are the same. Take the first one.
    attn_config = config.block_config(0).attn_config
    self.rope_cache = attn_utils.build_rope_cache(
        size=config.kv_cache_max,
        dim=int(attn_config.rotary_percentage * attn_config.head_dim),
        base=attn_config.rotary_base,
    )
    self.mask_cache = attn_utils.build_causal_mask_cache(
        size=config.kv_cache_max,
    )
    self.config = config

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
  ) -> dict:
    _, seq_len = tokens.size()
    assert self.config.max_seq_len >= seq_len, (
        f"Cannot forward sequence of length {seq_len}, max seq length is only"
        f" {self.config.max_seq_len}"
    )
    assert len(self.transformer_blocks) == len(kv_cache.caches), (
        "The number of transformer blocks and the number of KV cache entries"
        " must be the same."
    )

    cos, sin = self.rope_cache
    cos = cos.index_select(0, input_pos)
    sin = sin.index_select(0, input_pos)
    mask = self.mask_cache.index_select(2, input_pos)
    mask = mask[:, :, :, :seq_len]

    # token embeddings of shape (b, t, n_embd)
    x = self.tok_embedding(tokens)
    if self.config.embedding_scale is not None:
      x = x * self.config.embedding_scale

    updated_kv_entries = []
    for i, block in enumerate(self.transformer_blocks):
      kv_entry = kv_cache.caches[i] if kv_cache else None
      output = block(x, (cos, sin), mask, input_pos, kv_entry)
      if isinstance(output, tuple):
          x, kv_entry = output 
      else:
          x = output 
      if kv_entry:
       updated_kv_entries.append(kv_entry)
    updated_kv_cache = kv_utils.KVCache(tuple(updated_kv_entries))

    x = self.final_norm(x)
    logits = self.lm_head(x)  # (b, t, vocab_size)
    return {"logits": logits, "kv_cache": updated_kv_cache}


def build_qwen2_model(
    checkpoint_path: str,
    config: cfg.ModelConfig,
    tensor_names: loading_utils.ModelLoader.TensorNames,
) -> Qwen2Model:
  model = Qwen2Model(config)
  loader = loading_utils.ModelLoader(checkpoint_path, tensor_names)
  loader.load(
      model, strict=not config.lm_head_share_weight_with_embedding
  )
  model.eval()
  return model


def get_qwen2_model_config(kv_cache_max_len: int = 1024) -> cfg.ModelConfig:

  attn_config = cfg.AttentionConfig(
      num_heads=14,
      head_dim=64,
      num_query_groups=2,
      rotary_base=1000000,
      rotary_percentage=1.0,
      qkv_use_bias=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=4864,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-06,
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=151936,
      num_layers=24,
      max_seq_len=32768,
      embedding_dim=896,
      kv_cache_max_len=kv_cache_max_len,
      block_configs=block_config,
      final_norm_config=norm_config,
      enable_hlfb=True,
  )
  return config

def get_fake_model_config(**kwargs) -> cfg.ModelConfig:
  config = get_qwen2_model_config(**kwargs)
  config.vocab_size = 128
  config.num_layers = 2
  config.block_config(0).ff_config.intermediate_size = 64
  return config

def build_0_5b_model(
    checkpoint_path: str, **kwargs
) -> Qwen2Model:
  return build_qwen2_model(
      checkpoint_path=checkpoint_path,
      config=get_qwen2_model_config(**kwargs),
      tensor_names=TENSOR_NAMES,
  )


