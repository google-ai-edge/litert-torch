from ai_edge_torch.generative.layers import attention
import ai_edge_torch.generative.layers.builder as builder

N_MEL_BINS = 80

MAX_SOURCE_SEQUENCE_LENGTH = 1500


class EdgeTorchWhisperEncoder(nn.Module):
    def __init__(self,
                 config: AudioEncoderConfig,
                 enable_hlfb = True,
                 pre_transpose_k_cross = True):

        super().__init__()

        self.config = config
        self.embed_dim = config.embedding_dim
        self.num_heads = config.block_config.attn_config.num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.embed_positions = nn.Embedding(
            config.max_source_positions, self.embed_dim
        )

        # VERIFY: always 80

        self.conv1 = nn.Conv1d(N_MEL_BINS, self.embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)
        self.transformer_blocks = nn.ModuleList(
            attention.TransformerBlock(self.config.block_config, self.config)
                for _ in range(self.config.num_layers)
        )

        self.layer_norm = builder.build_norm(self.embed_dim, self.config.final_norm_config)
        self.k_cross_modules = nn.ModuleList(nn.Linear(self.embed_dim, self.embed_dim, bias = False) for _ in range(self.config.num_layers))
        self.v_cross_modules = nn.ModuleList(nn.Linear(self.embed_dim, self.embed_dim, bias = True) for _ in range(self.config.num_layers))

        self.k_cross_cache = None
        self.v_cross_cache = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @torch.no_grad()
    def forward(self, input_features):

        x = nn.functional.gelu(self.conv1(input_features))
        x = nn.functional.gelu(self.conv2(x))

        input_embeds = x.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight

        x = input_embeds + embed_pos

        full_self_attention_mask = torch.zeros(1, 1, MAX_SOURCE_SEQUENCE_LENGTH, MAX_SOURCE_SEQUENCE_LENGTH)

        for _, block in enumerate(self.transformer_blocks):

            x = block(x, mask = full_self_attention_mask)

        encoder_latents = self.layer_norm(x)

        for i in range(len(self.k_cross_modules)):
            if self.k_cross_cache is None:
                self.k_cross_cache = self._shape(self.k_cross_modules[i](encoder_latents), -1, 1)
                self.v_cross_cache = self._shape(self.v_cross_modules[i](encoder_latents), -1, 1)
            else:
                self.k_cross_cache = torch.cat((self.k_cross_cache, self._shape(self.k_cross_modules[i](encoder_latents), -1, 1)), dim = 0)
                self.v_cross_cache = torch.cat((self.v_cross_cache, self._shape(self.v_cross_modules[i](encoder_latents), -1, 1)), dim = 0)

        return self.k_cross_cache, self.v_cross_cache
