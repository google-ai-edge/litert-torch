##########################################
# generative/layers/whisper/model_config.py
##########################################

import dataclasses
import enum

from ai_edge_torch.generative.layers.model_config import (AttentionConfig,
    FeedForwardConfig,
    NormalizationConfig,
    ActivationConfig,
    ModelConfig,
    TransformerBlockConfig)


@dataclasses.dataclass
class AudioEncoderBlockConfig:
    # Configuration of the attention layers in the encoder block
    attn_config: AttentionConfig
    # configuration of the feedforward layers in the encoder block
    ff_config: FeedForwardConfig
    embedding_dim: int
    # normalization applied before self attention
    pre_attention_norm_config: NormalizationConfig = dataclasses.field(
        default_factory=NormalizationConfig
    )
    # normalization applied before feedforward
    pre_ff_norm_config: NormalizationConfig = dataclasses.field(
        default_factory=NormalizationConfig
    )
    post_attention_norm_config: NormalizationConfig = dataclasses.field(
        default_factory=NormalizationConfig
    )

@dataclasses.dataclass
class TextDecoderBlockConfig:
    # configuration for attention blocks
    attn_config: AttentionConfig
    # configuration for cross attention blocks
    cross_attn_config: AttentionConfig
    # configuration for feed forward layers
    ff_config: FeedForwardConfig
    embedding_dim: int
    # norm applied before the self attention
    pre_self_attention_norm_config: NormalizationConfig = dataclasses.field(
        default_factory=NormalizationConfig
    )
    # norm applied before the cross attention
    pre_cross_attention_norm_config: NormalizationConfig = dataclasses.field(
        default_factory=NormalizationConfig
    )
    # norm applied before the feed forward layers
    pre_ff_norm_config: NormalizationConfig = dataclasses.field(
        default_factory=NormalizationConfig
    )

@dataclasses.dataclass
class AudioEncoderConfig:
    block_config: AudioEncoderBlockConfig
    vocab_size: int
    max_source_positions: int
    num_layers: int
    embedding_dim: int
    enable_hlfb: bool
    final_norm_config: NormalizationConfig = dataclasses.field(
        default_factory=NormalizationConfig
    )


@dataclasses.dataclass
class TextDecoderConfig:
    block_config: TextDecoderBlockConfig
    vocab_size: int
    num_layers: int
    embedding_dim: int
    max_seq_len: int
    enable_hlfb: bool
    final_norm_config: NormalizationConfig = dataclasses.field(
        default_factory=NormalizationConfig
    )

