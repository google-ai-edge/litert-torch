from typing import Optional, Tuple, Union
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import lora as lora_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.utilities import export_config as export_cfg
import torch
from torch import nn
from ai_edge_torch.generative.layers import sdpa_with_kv_update


def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
  x = x.transpose(1, 2)
  x = (x * cos) + (rotate_half(x) * sin)
  return x.transpose(1, 2)


def apply_rope_inline(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Computes rotary positional embedding inline for a query and key.

  Args:
  q: the query tensor.
  k: the key tensor.
  cos: the cosine tensor.
  sin: the sine tensor.

  Returns:
  output the RoPE'd query and key.
  """
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_roped = apply_rope(q, cos, sin)
  k_roped = apply_rope(k, cos, sin)
  return q_roped.type_as(q), k_roped.type_as(k)


class CustomTransformerBlock(attention.TransformerBlock):

  def __init__(
      self,
      config: cfg.TransformerBlockConfig,
      model_config: cfg.ModelConfig,
  ) -> None:
    super().__init__(config, model_config)
    self.atten_func = CustomCausalSelfAttention(
        model_config.embedding_dim,
        config.attn_config,
        model_config.enable_hlfb,
    )


class CustomCausalSelfAttention(attention.CausalSelfAttention):
  """Causal self attention layer implementation."""

  def forward(
      self,
      x: torch.Tensor,
      rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      mask: Optional[torch.Tensor] = None,
      input_pos: Optional[torch.Tensor] = None,
      kv_cache: Optional[kv_utils.KVCacheEntry] = None,
      lora: Optional[lora_utils.LoRAEntry] = None,
  ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
    """Forward function of the CausalSelfAttention layer, which can support

       MQA, GQA and MHA.

    Args:
      x (torch.Tensor): the input tensor.
      rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
      mask (torch.Tensor): the optional mask tensor.
      input_pos (torch.Tensor): the optional input position tensor.
      kv_cache (KVCacheEntry): the KV cache entry corresponding to this module.
      lora (LoRAEntry): the optional lora entry.

    Returns:
      output activation from this self attention layer, and the updated
        KV Cach Entry (if passed in).
    """
    # Batch size, sequence length, embedding dimensionality.
    B, T, _ = x.size()
    qkv = self.qkv_projection(x)

    # Assemble into a number of query groups to support MHA, MQA and GQA.
    q_per_kv = self.config.num_heads // self.config.num_query_groups
    # Each group has >=1 queries, 1 key, and 1 value.
    if self.config.qkv_transpose_before_split:
      qkv = qkv.view(B, T, -1, self.config.head_dim)
      q, k, v = qkv.split(
          (
              q_per_kv * self.config.num_query_groups,
              self.config.num_query_groups,
              self.config.num_query_groups,
          ),
          dim=-2,
      )
    else:
      qkv = qkv.view(B, T, self.config.num_query_groups, -1)
      q, k, v = qkv.split(
          (
              q_per_kv * self.config.head_dim,
              self.config.head_dim,
              self.config.head_dim,
          ),
          dim=-1,
      )

    if lora is not None:
      q += lora_utils.apply_lora(x, lora.attention.query, shape=q.shape)
      k += lora_utils.apply_lora(x, lora.attention.key, shape=k.shape)
      v += lora_utils.apply_lora(x, lora.attention.value, shape=v.shape)

    q = self.query_norm(q)
    k = self.key_norm(k)
    v = self.value_norm(v)

    q = q.reshape(B, T, -1, self.config.head_dim)
    k = k.reshape(B, T, -1, self.config.head_dim)
    v = v.reshape(B, T, -1, self.config.head_dim)

    if rope is not None:
      # Compute rotary positional embedding for query and key.
      cos, sin = rope
      q, k = apply_rope_inline(q, k, cos, sin)

    sdpa_out, kv_cache = sdpa_with_kv_update.sdpa_with_kv_update(
        q, k, v, kv_cache, input_pos, mask, self.config, self.enable_hlfb
    )

    # Compute the output projection.
    y = self.output_projection(sdpa_out)
    if lora is not None:
      y += lora_utils.apply_lora(sdpa_out, lora.attention.output)

    return y if kv_cache is None else (y, kv_cache)


class DecoderOnlyModel(model_builder.DecoderOnlyModel):

  def __init__(self, config: cfg.ModelConfig, mask_cache_size: int = 0):
    super().__init__(config, mask_cache_size)

    self.transformer_blocks = nn.ModuleList(
        CustomTransformerBlock(config.block_config(idx), config)
        for idx in range(config.num_layers)
    )

  def get_rope(self, position_ids: torch.Tensor):

    base = 100000.0
    dim = float(64.0)

    timescale = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )

    radians = position_ids.clone().unsqueeze(0).unsqueeze(
        -1
    ) * timescale.unsqueeze(0).unsqueeze(0)
    radians = torch.cat((radians, radians), dim=-1).squeeze(0)
    cos = torch.cos(radians)
    sin = torch.sin(radians)
    return cos, sin

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      mask: Optional[torch.Tensor] = None,
      lora: Optional[lora_utils.LoRA] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    _, seq_len = tokens.size()
    assert self.config.max_seq_len >= seq_len, (
        f"Cannot forward sequence of length {seq_len}, max seq length is only"
        f" {self.config.max_seq_len}"
    )

    # token embeddings of shape (b, t, n_embd)
    input_embeds = self.tok_embedding(tokens)

    rope = self.get_rope(input_pos.view(1, -1))

    if mask is None:
      assert self.mask_cache is not None, "Mask cache must be built."
      assert kv_cache is not None, "KV cache must be provided."
      mask = self.mask_cache.index_select(2, input_pos)
      mask = mask[:, :, :, : kv_cache.get_max_seq_len()]

    return self._forward_with_embeds(
        input_embeds, rope, mask, input_pos, kv_cache, lora, export_config
    )
