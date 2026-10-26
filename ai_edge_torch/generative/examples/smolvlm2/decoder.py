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

"""Example of building a Decoder for SmolVLM2 model based on a Llama model."""

from functools import partial
from typing import Callable, Dict, Optional

from ai_edge_torch.generative.examples.llama import llama
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import lora as lora_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import export_config as export_cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn


TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    # Embedding
    embedding="model.text_model.embed_tokens",
    # Transformer blocks
    ff_up_proj="model.text_model.layers.{}.mlp.up_proj",
    ff_down_proj="model.text_model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.text_model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.text_model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.text_model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.text_model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.text_model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.text_model.layers.{}.input_layernorm",
    post_attn_norm="model.text_model.layers.{}.post_attention_layernorm",
    # Final norm
    final_norm="model.text_model.norm",
    # LM head
    lm_head="lm_head",
)


class Decoder(model_builder.DecoderOnlyModel):
  """Decoder for the SmolVLM2 model, based on DecoderOnlyModel."""

  def __init__(self, config: cfg.ModelConfig, mask_cache_size: int = 0):
    super().__init__(config, mask_cache_size)

    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=2
    )

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      input_embeds: Optional[torch.Tensor] = None,
      mask: Optional[torch.Tensor] = None,
      lora: Optional[lora_utils.LoRA] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    if input_embeds is None:
      input_embeds = self.tok_embedding(tokens)

    _, seq_len = tokens.size()
    assert self.config.max_seq_len >= seq_len, (
        f"Cannot forward sequence of length {seq_len}, max seq length is only"
        f" {self.config.max_seq_len}"
    )

    # ROPE parameters for all attn_configs are the same. Take the first one.
    attn_config = self.config.block_config(0).attn_config
    n_elem = int(attn_config.rotary_percentage * attn_config.head_dim)
    rope = self.config.build_rope(input_pos, n_elem, attn_config.rotary_base)

    if mask is None:
      assert self.mask_cache is not None, "Mask cache must be built."
      assert kv_cache is not None, "KV cache must be provided."
      mask = self.mask_cache.index_select(2, input_pos)
      mask = mask[:, :, :, :kv_cache.get_max_seq_len()]

    return super()._forward_with_embeds(
        input_embeds, rope, mask, input_pos, kv_cache, lora, export_config
    )


def get_decoder_config() -> cfg.ModelConfig:
  """Returns the model config for a SmolVLM2 Decoder model."""
  attn_config = cfg.AttentionConfig(
      num_heads=32,  # hidden_size / head_dim2 = 2048 / 64
      head_dim=64,
      num_query_groups=32,
      rotary_base=130000,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=8192,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      enable_hlfb=False
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )

  max_seq_len = 8192
  build_rope = partial(
      llama._build_llama3_rope_cache,
      dtype=torch.float32,
      device=torch.device("cpu"),
      condense_ratio=1,
      factor=32.0,
      low_freq_factor=1.0,
      high_freq_factor=4.0,
      max_seq_len=max_seq_len,
  )

  config = cfg.ModelConfig(
      vocab_size=49280,
      num_layers=24,
      max_seq_len=max_seq_len,
      embedding_dim=2048,
      block_configs=block_config,
      final_norm_config=norm_config,
      build_rope=build_rope,
      lm_head_share_weight_with_embedding=False,
      enable_hlfb=False,
  )
  return config


def build_decoder(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  """Builds a Smolvlm2 Decoder from the checkpoint path."""
  return model_builder.build_decoder_only_model(
      checkpoint_path=checkpoint_path,
      config=get_decoder_config(),
      tensor_names=TENSOR_NAMES,
      model_class=Decoder,
      custom_loader=custom_loader,
      mask_cache_size=mask_cache_size,
  )
