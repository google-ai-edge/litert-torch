from typing import Callable, Dict, Optional, Tuple
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import model_builder
import ai_edge_torch.generative.utilities.loader as loading_utils
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.utilities import export_config as export_cfg
import torch
from torch import nn

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.text_model.layers.{}.mlp.up_proj",
    ff_down_proj="model.text_model.layers.{}.mlp.down_proj",
    ff_gate_proj="model.text_model.layers.{}.mlp.gate_proj",
    attn_query_proj="model.text_model.layers.{}.self_attn.q_proj",
    attn_key_proj="model.text_model.layers.{}.self_attn.k_proj",
    attn_value_proj="model.text_model.layers.{}.self_attn.v_proj",
    attn_output_proj="model.text_model.layers.{}.self_attn.o_proj",
    pre_attn_norm="model.text_model.layers.{}.input_layernorm",
    post_attn_norm="model.text_model.layers.{}.post_attention_layernorm",
    embedding="model.text_model.embed_tokens",
    final_norm="model.text_model.norm",
    lm_head="lm_head",
)


from decoder_only_model import DecoderOnlyModel


class SmolVLMText(DecoderOnlyModel):
  """A SmolVLMText model built from the Edge Generative API layers."""

  @torch.inference_mode
  def forward_embeds(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      input_embeds: torch.Tensor = None,
      rope: Tuple[torch.Tensor, torch.Tensor] = None,
      mask: Optional[torch.Tensor] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:

    if input_embeds is None:
      _, seq_len = tokens.size()
      assert self.config.max_seq_len >= seq_len, (
          f"Cannot forward sequence of length {seq_len}, max seq length is only"
          f" {self.config.max_seq_len}"
      )
      # token embeddings of shape (b, t, n_embd)
      input_embeds = self.tok_embedding(tokens)

    if mask is None:
      assert kv_cache is not None, "KV cache must be provided."
      mask = self.mask_cache.index_select(2, input_pos)
      mask = mask[:, :, :, : kv_cache.get_max_seq_len()]

    return self._forward_with_embeds(
        input_embeds,
        rope,
        mask,
        input_pos,
        kv_cache,
        export_config=export_config,
    )


def get_model_config() -> cfg.ModelConfig:
  """Returns the model config for a SmolVLM 256M model.

  Returns:
    The model config for a SmolVLM model.
  """
  attn_config = cfg.AttentionConfig(
      num_heads=9,
      head_dim=64,
      num_query_groups=3,
      rotary_base=100000,
      rotary_percentage=1.0,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
      intermediate_size=1536,
  )
  norm_config = cfg.NormalizationConfig(type=cfg.NormalizationType.RMS_NORM)
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=49280,
      num_layers=30,
      max_seq_len=8192,
      embedding_dim=576,
      block_configs=block_config,
      final_norm_config=norm_config,
      lm_head_share_weight_with_embedding=False,
  )
  return config


def build_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> nn.Module:

  transformer = SmolVLMText(get_model_config(), mask_cache_size=mask_cache_size)
  loader = loading_utils.ModelLoader(
      checkpoint_path, TENSOR_NAMES, custom_loader
  )
  loader.load(
      transformer,
      strict=False,
  )
  transformer.eval()
  return transformer


if __name__ == "__main__":  # TODO delete
  model = build_model("./models/SmolVLM-256M-Instruct", mask_cache_size=1024)
  print(model)
