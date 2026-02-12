from ai_edge_torch.generative.layers import attention
from ai_edge_torch.generative.layers import builder
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn

TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    ff_up_proj="model.vision_model.encoder.layers.{}.mlp.fc1",
    ff_down_proj="model.vision_model.encoder.layers.{}.mlp.fc2",
    attn_query_proj="model.vision_model.encoder.layers.{}.self_attn.q_proj",
    attn_key_proj="model.vision_model.encoder.layers.{}.self_attn.k_proj",
    attn_value_proj="model.vision_model.encoder.layers.{}.self_attn.v_proj",
    attn_output_proj="model.vision_model.encoder.layers.{}.self_attn.out_proj",
    pre_attn_norm="model.vision_model.encoder.layers.{}.layer_norm1",
    post_attn_norm="model.vision_model.encoder.layers.{}.layer_norm2",
    embedding="model.vision_model.embeddings.patch_embedding",
    embedding_position=(
        "model.vision_model.embeddings.position_embedding.weight"
    ),
    final_norm="model.vision_model.post_layernorm",
)


class SiglipVisionEncoder(nn.Module):
  """Signlip vision encoder from the Edge Generative API."""

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()

    # Construct model layers.
    self.tok_embedding = nn.Conv2d(
        in_channels=config.image_embedding.channels,
        out_channels=config.embedding_dim,
        kernel_size=config.image_embedding.patch_size,
        stride=config.image_embedding.patch_size,
        padding=0,
        bias=config.embedding_use_bias,
    )
    num_patches = (
        config.image_embedding.image_size // config.image_embedding.patch_size
    ) ** 2
    self.tok_embedding_position = nn.Parameter(
        torch.zeros((num_patches, config.embedding_dim)),
        requires_grad=False,
    )

    self.transformer_blocks = nn.ModuleList(
        attention.TransformerBlock(config.block_config(idx), config)
        for idx in range(config.num_layers)
    )
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )
    self.config = config

  @torch.inference_mode
  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    # Embed the image according to SiplipVisionEmbeddings.

    x = self.tok_embedding(pixel_values)
    x = x.flatten(2).transpose(1, 2) + self.tok_embedding_position

    # Pass a dummy mask because SDPA attention impl expects non-None mask.
    mask = torch.zeros(x.shape[:2]).unsqueeze(1).unsqueeze(2)

    for _, block in enumerate(self.transformer_blocks):
      x = block(x, mask=mask)
    return self.final_norm(x)


def get_image_encoder_config() -> cfg.ModelConfig:
  """Returns the model config for the image encoder of a SmolVLM-256M model.

  Returns:
    The model config for the image encoder of a SmolVLM-256M model.
  """
  image_embedding_config = cfg.ImageEmbeddingConfig(
      channels=3,
      image_size=512,
      patch_size=16,
  )
  attn_config = cfg.AttentionConfig(
      num_heads=12,
      head_dim=64,
      num_query_groups=12,
      qkv_use_bias=True,
      output_proj_use_bias=True,
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.SEQUENTIAL,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=3072,
      use_bias=True,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.LAYER_NORM, epsilon=1e-6
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=0,
      num_layers=12,
      max_seq_len=0,
      embedding_dim=768,
      embedding_use_bias=True,
      image_embedding=image_embedding_config,
      block_configs=block_config,
      final_norm_config=norm_config,
  )
  return config


def build_image_encoder(checkpoint_path: str) -> SiglipVisionEncoder:
  config = get_image_encoder_config()
  encoder = SiglipVisionEncoder(config)
  loader = loading_utils.ModelLoader(checkpoint_path, TENSOR_NAMES)
  loader.load(encoder, strict=False)
  encoder.eval()
  return encoder


if __name__ == "__main__":
  model = build_image_encoder("./models/SmolVLM-256M-Instruct")
  print(model)
