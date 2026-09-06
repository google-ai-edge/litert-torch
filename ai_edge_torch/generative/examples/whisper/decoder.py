from ai_edge_torch.generative.layers import attention
import ai_edge_torch.generative.layers.builder as builder
from ai_edge_torch.generative.layers import scaled_dot_product_attention as sdpa


class WhisperDecoderBlock(nn.Module):

    def __init__(self, config: TextDecoderBlockConfig,
                 model_config: TextDecoderConfig,
                 block_index: Optional[int] = None):
        """Initialize an instance of the WhisperDecoderBlock.

        Args:
            config (whisper_cfg.TextDecoderBlockConfig): the configuration of this block.
            model_config (whisper_cfg.TextDecoderConfig): the configuration of the text decoder model.
            block_index (Optional[int], optional): the index of this block. Defaults to None.
        """
        super().__init__()
        self.config = config
        self.pre_atten_norm = builder.build_norm(
            config.embedding_dim, config.pre_self_attention_norm_config
        )

        self.pre_cross_atten_norm = builder.build_norm(
            config.embedding_dim, config.pre_cross_attention_norm_config
        )
        self.block_index = block_index
        self.atten_func = attention.CausalSelfAttention(
            config.embedding_dim,
            config.attn_config,
            model_config.enable_hlfb,
        )

        self.cross_atten_func = CrossAttentionWithStaticCache(
            config.embedding_dim,
            config.embedding_dim,
            config.embedding_dim,
            config.embedding_dim,
            config.attn_config,
            model_config.enable_hlfb,
        )

        self.ff = builder.build_ff(
            config.embedding_dim, config.ff_config
        )

    def forward(self,
                hidden_states: torch.Tensor,
                cross_attn_kv_cache: Tuple[torch.Tensor]):

        residual = hidden_states
        hidden_states = self.pre_self_atten_norm(hidden_states)
        hidden_states = self.self_atten_func(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_cross_atten_norm(hidden_states)
        hidden_states = self.cross_atten_func(hidden_states, cross_attn_kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class EdgeTorchWhisperDecoder(nn.Module):
    def __init__(self,
                 config: TextDecoderConfig,
                 enable_hlfb = True):
        super().__init__()

        self.config = config
        self.embed_dim = config.embedding_dim
        self.num_heads = config.block_config.attn_config.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.decoder_ffn_dim = config.block_config.ff_config.intermediate_size

        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.embedding_dim
        )

        self.embed_positions = nn.Embedding(
            self.config.max_seq_len, self.config.embedding_dim
        )

        self.transformer_blocks = nn.ModuleList(
            WhisperDecoderBlock(self.config.block_config, self.config, idx)
                for idx in range(self.config.num_layers)
        )

        self.layer_norm = builder.build_norm(self.embed_dim, self.config.final_norm_config)


    @torch.no_grad()
    def forward(self,
                input_ids: torch.Tensor,
                cross_attn_kv_cache: Tuple[torch.Tensor],
                past_key_values: Optional[Tuple[torch.Tensor]] = None,
                cache_position: Optional[int] = None,
                **kwargs):

        input_shape = input_ids.shape

        input_ids = input_ids.view(-1, input_shape[-1])

        input_embeds = self.embed_tokens(input_ids)

        past_seq_len_offset = 0

        if cache_position is not None:
            past_seq_len_offset = cache_position

        input_positions = self.embed_positions(torch.arange(past_seq_len_offset, past_seq_len_offset + input_shape[-1]))

        hidden_states = input_embeds + input_positions

        for idx, layer in enumerate(self.transformer_blocks):
            hidden_states, = layer(hidden_states, cross_attn_kv_cache)

        out = self.layer_norm(hidden_states)

        return out

def get_decoder_config(checkpoint_path: str) -> TextDecoderConfig:
    whisper_model = pipeline("automatic-speech-recognition", model=checkpoint_path, tokenizer=checkpoint_path)
    decoder_config = whisper_model.model.get_decoder().config

    num_heads = decoder_config.decoder_attention_heads
    head_dim = decoder_config.d_model // decoder_config.decoder_attention_heads
    embed_dim = decoder_config.d_model
    decoder_ffn_dim = decoder_config.decoder_ffn_dim

    # per block norms - pre self attn, pre cross attn, pre ff
    pre_self_attention_norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.LAYER_NORM,
    )

    pre_cross_attention_norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.LAYER_NORM,
    )

    pre_ff_norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.LAYER_NORM,
    )

    # per model norm - final_norm
    final_norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.LAYER_NORM,
    )

    # per block causal self attention configuration
    attn_config = AttentionConfig(
        num_heads = num_heads,
        head_dim = head_dim,
        num_query_groups = num_heads,
        enable_kv_cache = True, # need to change to true for decoder...
        qkv_use_bias = True,
        qkv_transpose_before_split = True,
        output_proj_use_bias = True,
    )

    # per block ff config
    ff_config = cfg.FeedForwardConfig(
        type=cfg.FeedForwardType.SEQUENTIAL,
        activation=cfg.ActivationConfig(cfg.ActivationType.GELU),
        intermediate_size=decoder_ffn_dim,
        use_bias = True,
        pre_ff_norm_config=pre_ff_norm_config
    )

    # per block cross attention config (with static kv cross projections)
    # -- the configuration attributes match the self attn
    decoder_block_config = TextDecoderBlockConfig(
        attn_config=attn_config,
        cross_attn_config=attn_config,
        ff_config=ff_config,
        embedding_dim=embed_dim,
        pre_self_attention_norm_config=pre_self_attention_norm_config,
        pre_cross_attention_norm_config=pre_cross_attention_norm_config,
        pre_ff_norm_config=pre_ff_norm_config
    )

    model_configuration = TextDecoderConfig(
        block_config=decoder_block_config,
        vocab_size = decoder_config.vocab_size,
        num_layers = decoder_config.num_hidden_layers,
        max_seq_len = decoder_config.max_target_positions,
        embedding_dim = decoder_config.d_model,
        enable_hlfb = True,
        final_norm_config=final_norm_config
    )

    return model_configuration