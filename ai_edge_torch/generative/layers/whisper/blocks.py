##########################################
# generative/layers/whisper/blocks.py
##########################################

import torch.nn as nn

import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.layers import lora as lora_utils
from typing import Optional, Tuple


class CrossAttentionWithStaticCache(nn.Module):

    def __init__(
        self,
        query_dim: int,
        cross_dim: int,
        hidden_dim: int,
        output_dim: int,
        config: cfg.AttentionConfig,
        enable_hlfb: bool):

        super().__init__()
        self.config = config
        self.n_heads = config.num_heads
        self.q_projection = nn.Linear(
            query_dim, hidden_dim, bias=True
        )
        self.output_projection = nn.Linear(
            hidden_dim, output_dim, bias=True
        )

        self.sdpa_func = (
            sdpa.scaled_dot_product_attention_with_hlfb
            if enable_hlfb
            else sdpa.scaled_dot_product_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        cross_attention_projections: Tuple[torch.Tensor],
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        lora: Optional[lora_utils.LoRAEntry] = None,
    ):
        batch_size = x.size()[0]
        target_seq_len = x.size()[1]

        # k_cache is *not* pre-transposed for this. (vs QUIC exported models)
        k = cross_attention_projections[0]
        v = cross_attention_projections[1]

        source_seq_len = k.size()[1]

        q = self.q_projection(x)

        if lora is not None:
            q += lora_utils.apply_lora(x, lora.attention.query, shape=q.shape)
            k += lora_utils.apply_lora(x, lora.attention.key, shape=k.shape)
            v += lora_utils.apply_lora(x, lora.attention.value, shape=v.shape)

        interim_shape = (batch_size, -1, self.n_heads, self.config.head_dim)
        q = q.view(interim_shape)

        if mask is None:
            mask = torch.zeros(
                (batch_size, 1, target_seq_len, source_seq_len), dtype=torch.float32
            )
        y = self.sdpa_func(q, k, v, self.config.head_dim, mask=mask)
        y = y.reshape(batch_size, target_seq_len, -1)

        # Compute the output projection.
        y = self.output_projection(y)
        if lora is not None:
            y += lora_utils.apply_lora(y, lora.attention.output)

        return y