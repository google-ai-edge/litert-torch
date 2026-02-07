# Copyright 2024 The LiteRT Torch Authors.
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

"""Example of building a Decoder for Gemma3 model."""

from typing import Callable, Dict, List, Optional, Tuple

from litert_torch.generative.layers import attention
from litert_torch.generative.layers import builder
from litert_torch.generative.layers import kv_cache as kv_utils
import litert_torch.generative.layers.attention_utils as attn_utils
import litert_torch.generative.layers.model_config as cfg
import litert_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
from litert_torch.generative.utilities import export_config as export_cfg
from litert_torch.generative.utilities import model_builder
import litert_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn


TENSOR_NAMES_SEP_QKV = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    attn_query_norm="model.layers.{}.self_attn.q_norm",
    attn_key_norm="model.layers.{}.self_attn.k_norm",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    pre_ff_norm="model.layers.{}.pre_feedforward_layernorm",
    post_ff_norm="model.layers.{}.post_feedforward_layernorm",
    embedding="model.embed_tokens",
    final_norm="model.norm",
    lm_head=None,
)


TENSOR_NAMES_FUSED_QKV = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.layers.{}.mlp.up_proj",
    ff_down_proj="model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.layers.{}.mlp.gate_proj",
    attn_fused_qkv_proj="model.layers.{}.self_attn.qkv_proj",
    attn_output_proj="model.layers.{}.self_attn.o_proj",
    attn_query_norm="model.layers.{}.self_attn.query_norm",
    attn_key_norm="model.layers.{}.self_attn.key_norm",
    pre_attn_norm="model.layers.{}.input_layernorm",
    post_attn_norm="model.layers.{}.post_attention_layernorm",
    pre_ff_norm="model.layers.{}.pre_feedforward_layernorm",
    post_ff_norm="model.layers.{}.post_feedforward_layernorm",
    embedding="embedder",
    final_norm="model.norm",
    lm_head=None,
)

TENSOR_NAMES_HF_4B = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="language_model.model.layers.{}.mlp.up_proj",
    ff_down_proj="language_model.model.layers.{}.mlp.down_proj",
    ff_gate_proj="language_model.model.layers.{}.mlp.gate_proj",
    attn_query_proj="language_model.model.layers.{}.self_attn.q_proj",
    attn_key_proj="language_model.model.layers.{}.self_attn.k_proj",
    attn_value_proj="language_model.model.layers.{}.self_attn.v_proj",
    attn_output_proj="language_model.model.layers.{}.self_attn.o_proj",
    attn_query_norm="language_model.model.layers.{}.self_attn.q_norm",
    attn_key_norm="language_model.model.layers.{}.self_attn.k_norm",
    pre_attn_norm="language_model.model.layers.{}.input_layernorm",
    post_attn_norm="language_model.model.layers.{}.post_attention_layernorm",
    pre_ff_norm="language_model.model.layers.{}.pre_feedforward_layernorm",
    post_ff_norm="language_model.model.layers.{}.post_feedforward_layernorm",
    embedding="language_model.model.embed_tokens",
    final_norm="language_model.model.norm",
    lm_head=None,
)

TENSOR_NAMES_DICT = {
    "safetensors": TENSOR_NAMES_SEP_QKV,
    "kaggle": TENSOR_NAMES_FUSED_QKV,
    "hf_4b": TENSOR_NAMES_HF_4B,
}


class DecoderBlock(attention.TransformerBlock):
  """A Gemma3 decoder block built from the Edge Generative API layers."""

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: kv_utils.KVCacheEntry = None,
  ) -> Tuple[torch.Tensor, Optional[kv_utils.KVCacheEntry]]:
    """Forward function of the Gemma3Block.

    Exactly the same as TransformerBlock but we call the post-attention norm
    immediately after attention and not after the residual pointwise addition.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): the optional kv cache entry.

    Returns:
      output activation from this transformer block, and updated kv cache (if
      passed in).
    """

    x_norm = self.pre_atten_norm(x)
    attn_out, kv = self.atten_func(x_norm, rope, mask, input_pos, kv_cache)
    attn_out_norm = self.post_atten_norm(attn_out)
    x = x + attn_out_norm
    output = x + self.ff(x)
    return output, kv


class Decoder(nn.Module):
  """A Gemma3 decoder model built from the Edge Generative API layers."""

  def __init__(self, config: cfg.ModelConfig, mask_cache_size: int = 0):
    super().__init__()

    # Construct model layers.
    self.tok_embedding = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )
    self.lm_head = nn.Linear(
        config.embedding_dim, config.vocab_size, bias=config.lm_head_use_bias
    )
    # Gemma3 re-uses the embedding as the head projection layer.
    self.lm_head.weight.data = self.tok_embedding.weight.data
    self.transformer_blocks = nn.ModuleList(
        DecoderBlock(config.block_config(idx), config)
        for idx in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim, config.final_norm_config
    )
    self.config = config
    self.build_mask_cache(mask_cache_size)

  def build_mask_cache(self, mask_cache_size: int):
    assert (
        mask_cache_size <= self.config.max_seq_len
    ), "Mask cache size must be less than or equal to the max seq length."
    if mask_cache_size <= 0:
      self.mask_cache = None
    else:
      self.mask_cache = attn_utils.build_causal_mask_cache(mask_cache_size)

  def get_local_global_attention_mask(
      self,
      attention_mask: torch.Tensor,
      attn_type: cfg.AttentionType,
      segment_pos: torch.Tensor,
      sliding_window_size: int,
  ) -> torch.Tensor:
    """Returns the attention mask for the current batch (PyTorch)."""
    if attn_type == cfg.AttentionType.LOCAL_SLIDING:
      sliding_mask = self.create_sliding_mask(
          segment_pos=segment_pos,
          cache_len=attention_mask.shape[-1],
          sliding_window_size=sliding_window_size,
      )
      # Expand sliding_mask to match attention_mask's dimensions
      # (e.g., [B, 1, seq_len, cache_len]).
      # Assuming the head dimension is dim 1 for attention_mask.
      expanded_sliding_mask = sliding_mask.unsqueeze(1)
      # Combine masks using logical AND (min ensures -inf propagates).
      combined_mask = torch.min(attention_mask, expanded_sliding_mask)
      return combined_mask
    return attention_mask

  def create_sliding_mask(
      self,
      segment_pos: torch.Tensor,  # [B, L]
      cache_len: int,
      sliding_window_size: int,
  ) -> torch.Tensor:
    """Creates mask for sliding window attention (PyTorch)."""
    # Use torch.arange to create a tensor with a range of integers in a
    # Dynamo-friendly way.
    cache_positions = torch.arange(cache_len, dtype=torch.int32)
    cache_positions = cache_positions.view(1, 1, -1)  # [1, 1, cache_len]
    segment_pos_expanded = segment_pos.clone().unsqueeze(-1)  # [B, seq_len, 1]

    # Create boolean masks for window boundaries.
    left_boundary = cache_positions > segment_pos_expanded - sliding_window_size
    right_boundary = (
        cache_positions < segment_pos_expanded + sliding_window_size
    )

    # Combine boolean masks (AND).
    sliding_mask_bool = left_boundary & right_boundary

    # Convert boolean mask to float mask with 0 and -inf.
    sliding_mask = torch.where(
        sliding_mask_bool,
        torch.zeros_like(sliding_mask_bool, dtype=torch.float),
        torch.full_like(
            sliding_mask_bool, self.config.causal_mask_value, dtype=torch.float
        ),
    )

    return sliding_mask

  def compose_mask(
      self,
      mask: torch.Tensor,
      pixel_mask: torch.Tensor,
      attn_type: cfg.AttentionType,
  ) -> torch.Tensor:
    mask = mask == 0
    if attn_type == cfg.AttentionType.LOCAL_SLIDING:
      mask = torch.logical_and(mask, pixel_mask)
    else:
      mask = torch.logical_or(mask, pixel_mask)
    mask = torch.where(mask, 0, self.config.causal_mask_value)
    return mask

  def build_pixel_mask(self, image_indices: torch.Tensor, max_seq_len: int):
    pixel_mask = image_indices >= 0
    if pixel_mask.size(1) < max_seq_len:
      pixel_mask = torch.cat(
          [
              pixel_mask,
              torch.zeros(
                  (pixel_mask.size(0), max_seq_len - pixel_mask.size(1))
              ),
          ],
          dim=1,
      )
    pixel_mask = torch.logical_and(
        pixel_mask.unsqueeze(1), pixel_mask.unsqueeze(-1)
    )
    return pixel_mask.unsqueeze(1)

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      input_embeds: Optional[torch.Tensor] = None,
      mask: Optional[torch.Tensor] = None,
      image_indices: Optional[torch.Tensor] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    if input_embeds is None:
      # token embeddings of shape (b, t, n_embd)
      input_embeds = self.tok_embedding(tokens)
      if self.config.embedding_scale is not None:
        input_embeds = input_embeds * self.config.embedding_scale

    # RoPE parameters are the same for all blocks. Use the first layer.
    attn_config = self.config.block_config(0).attn_config
    # Different rotary base for global and local attention
    # based on attention pattern
    rope = [
        rotary_pos_emb.build_rope(
            input_pos,
            attn_config.head_dim,
            self.config.block_config(i).attn_config.rotary_base,
        )
        for i in range(self.config.num_layers)
    ]

    if mask is None:
      assert self.mask_cache is not None, "Mask cache must be built."
      assert kv_cache is not None, "KV cache must be provided."
      kv_cache_max_len = kv_cache.get_max_seq_len()
      mask = self.mask_cache.index_select(2, input_pos)
      mask = mask[:, :, :, :kv_cache_max_len]
    else:
      kv_cache_max_len = mask.size(3)

    pixel_mask = None
    if image_indices is not None:
      pixel_mask = self.build_pixel_mask(image_indices, kv_cache_max_len)

    return self._forward_with_embeds(
        input_embeds, rope, mask, input_pos, kv_cache, pixel_mask, export_config
    )

  def _forward_with_embeds(
      self,
      input_embeds: torch.Tensor,
      rope: List[Tuple[torch.Tensor, torch.Tensor]],
      mask: torch.Tensor | List[torch.Tensor],
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      pixel_mask: Optional[torch.Tensor] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    """Forwards the model with input embeddings."""
    assert len(self.transformer_blocks) == len(kv_cache.caches), (
        "The number of transformer blocks and the number of KV cache entries"
        " must be the same."
    )

    x = input_embeds

    if pixel_mask is None:
      mask = [
          self.get_local_global_attention_mask(
              mask[i] if isinstance(mask, list) else mask,
              self.config.block_config(i).attn_config.attn_type,
              input_pos,
              self.config.block_config(i).attn_config.sliding_window_size,
          )
          for i in range(self.config.num_layers)
      ]
    else:
      pixel_mask = pixel_mask.index_select(2, input_pos)
      mask = [
          self.compose_mask(
              mask[i] if isinstance(mask, list) else mask,
              pixel_mask,
              self.config.block_config(i).attn_config.attn_type,
          )
          for i in range(self.config.num_layers)
      ]
    updated_kv_entries = []
    for i, block in enumerate(self.transformer_blocks):
      mask_entry = mask[i] if isinstance(mask, list) else mask
      kv_entry = kv_cache.caches[i] if kv_cache else None
      x, kv_entry = block(x, rope[i], mask_entry, input_pos, kv_entry)
      if kv_entry:
        updated_kv_entries.append(kv_entry)
    updated_kv_cache = kv_utils.KVCache(tuple(updated_kv_entries))

    if export_config is not None:
      if (
          torch.numel(input_pos) > 1
          and not export_config.output_logits_on_prefill
      ):
        return {"kv_cache": updated_kv_cache}

    x = self.final_norm(x)
    res = self.lm_head(x)  # (b, t, vocab_size)

    return {"logits": res, "kv_cache": updated_kv_cache}


def get_decoder_config_1b() -> cfg.ModelConfig:
  """Returns the model config for a Gemma3 1B model."""
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6, zero_centered=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=6 * 1152,
      pre_ff_norm_config=norm_config,
      post_ff_norm_config=norm_config,
  )

  def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
    attn_config = cfg.AttentionConfig(
        num_heads=4,
        head_dim=256,
        num_query_groups=1,
        rotary_base=1_000_000 if (idx + 1) % 6 == 0 else 10_000,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        logit_softcap=None,
        sliding_window_size=512,
        attn_type=(
            cfg.AttentionType.GLOBAL
            if (idx + 1) % 6 == 0
            else cfg.AttentionType.LOCAL_SLIDING
        ),
    )
    return cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )

  num_layers = 26
  embedding_dim = 1152
  config = cfg.ModelConfig(
      vocab_size=262_144,
      num_layers=num_layers,
      max_seq_len=32_768,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      final_logit_softcap=None,
  )
  return config


def get_decoder_config_270m() -> cfg.ModelConfig:
  """Returns the model config for a Gemma3 270M model."""
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6, zero_centered=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=2048,
      pre_ff_norm_config=norm_config,
      post_ff_norm_config=norm_config,
  )

  def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
    attn_config = cfg.AttentionConfig(
        num_heads=4,
        head_dim=256,
        num_query_groups=1,
        rotary_base=1_000_000 if (idx + 1) % 6 == 0 else 10_000,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        logit_softcap=None,
        sliding_window_size=512,
        attn_type=(
            cfg.AttentionType.GLOBAL
            if (idx + 1) % 6 == 0
            else cfg.AttentionType.LOCAL_SLIDING
        ),
    )
    return cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )

  num_layers = 18
  embedding_dim = 640
  config = cfg.ModelConfig(
      vocab_size=262_144,
      num_layers=num_layers,
      max_seq_len=32_768,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      final_logit_softcap=None,
  )
  return config


def get_decoder_config_4b() -> cfg.ModelConfig:
  """Returns the model config for a Gemma3 4B model."""
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6, zero_centered=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=10240,
      pre_ff_norm_config=norm_config,
      post_ff_norm_config=norm_config,
  )

  def get_block_config(idx: int) -> cfg.TransformerBlockConfig:
    attn_config = cfg.AttentionConfig(
        num_heads=8,
        head_dim=256,
        num_query_groups=1,
        rotary_base=1_000_000 if (idx + 1) % 6 == 0 else 10_000,
        rotary_percentage=1.0,
        qkv_transpose_before_split=True,
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        logit_softcap=None,
        sliding_window_size=1024,
        attn_type=(
            cfg.AttentionType.GLOBAL
            if (idx + 1) % 6 == 0
            else cfg.AttentionType.LOCAL_SLIDING
        ),
    )
    return cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )

  num_layers = 34
  embedding_dim = 2560
  config = cfg.ModelConfig(
      vocab_size=262_208,
      num_layers=num_layers,
      max_seq_len=32_768,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      block_configs=[get_block_config(i) for i in range(num_layers)],
      final_norm_config=norm_config,
      lm_head_use_bias=False,
      final_logit_softcap=None,
  )
  return config


def get_fake_decoder_config_1b() -> cfg.ModelConfig:
  """Returns a fake model config for a Gemma3 1B model."""
  config = get_decoder_config_1b()
  config.vocab_size = 128
  config.num_layers = 2
  config.max_seq_len = 256
  config.embedding_dim = 128
  config.embedding_scale = config.embedding_dim**0.5
  config.block_configs = config.block_configs[: config.num_layers]
  for block_config in config.block_configs:
    block_config.attn_config.num_heads = 4
    block_config.attn_config.head_dim = 64
    block_config.attn_config.sliding_window_size = 64
    block_config.ff_config.intermediate_size = 128
  return config


def build_model_1b(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  # TODO(b/403644647): Better error handling for loading checkpoints with
  # different tensor names.
  for tensor_names in TENSOR_NAMES_DICT.values():
    try:
      return model_builder.build_decoder_only_model(
          checkpoint_path=checkpoint_path,
          config=get_decoder_config_1b(),
          tensor_names=tensor_names,
          model_class=Decoder,
          custom_loader=custom_loader,
          mask_cache_size=mask_cache_size,
      )
    except KeyError as ke:
      continue
  raise RuntimeError(
      f"Failed to build model from checkpoint at {checkpoint_path}. "
      "None of the known tensor name mappings matched the checkpoint."
  )


def build_model_270m(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  """Builds a Gemma3 270M model."""
  # TODO(b/403644647): Better error handling for loading checkpoints with
  # different tensor names.
  for tensor_names in TENSOR_NAMES_DICT.values():
    try:
      return model_builder.build_decoder_only_model(
          checkpoint_path=checkpoint_path,
          config=get_decoder_config_270m(),
          tensor_names=tensor_names,
          model_class=Decoder,
          custom_loader=custom_loader,
          mask_cache_size=mask_cache_size,
      )
    except KeyError as _:
      continue
  raise RuntimeError(
      f"Failed to build model from checkpoint at {checkpoint_path}. "
      "None of the known tensor name mappings matched the checkpoint."
  )


def build_model_4b(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:
  """Builds a Gemma3 4B model."""
  # TODO(b/403644647): Better error handling for loading checkpoints with
  # different tensor names.
  for tensor_names in TENSOR_NAMES_DICT.values():
    try:
      return model_builder.build_decoder_only_model(
          checkpoint_path=checkpoint_path,
          config=get_decoder_config_4b(),
          tensor_names=tensor_names,
          model_class=Decoder,
          custom_loader=custom_loader,
          mask_cache_size=mask_cache_size,
      )
    except KeyError as _:
      continue
  raise RuntimeError(
      f"Failed to build model from checkpoint at {checkpoint_path}. "
      "None of the known tensor name mappings matched the checkpoint."
  )
