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

"""Example of building an image encoder of Smolvlm2 model which is Siglip.

File includes a connector class which is used to project the image tokens into
image embeddings for concatenation with text embeddings.
"""

from dataclasses import dataclass
from typing import Callable, Dict

from ai_edge_torch.generative.examples.paligemma import image_encoder
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import export_config as export_cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn


CONNECTOR_TENSOR_NAME = "model.connector.modality_projection.proj"


@dataclass
class ConnectorConfig:
  """SmolVLM2 connector configurations."""

  scale_factor: int
  vision_hidden_size: int
  text_hidden_size: int


class SmolVLMConnector(nn.Module):
  """SmolVLM2 connector from the Edge Generative API."""

  def __init__(self, config: ConnectorConfig):
    super().__init__()
    self.scale_factor = config.scale_factor
    input_size = config.vision_hidden_size * (self.scale_factor**2)
    self.modality_projection = nn.Linear(
        input_size, config.text_hidden_size, bias=False
    )

  def pixel_shuffle(self, x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    bsz, seq, embed_dim = x.size()
    height = width = int(seq**0.5)
    if height * width != seq:
      raise ValueError(
          f"Invalid sequence length: {seq}. It must be a perfect square."
      )
    x = x.view(bsz, height, width, embed_dim)
    x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(
        bsz,
        int(width / scale_factor),
        int(height / scale_factor),
        embed_dim * (scale_factor**2),
    )
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(
        bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2)
    )
    return x

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.pixel_shuffle(x, self.scale_factor)
    x = self.modality_projection(x)
    return x


def get_connector_config() -> ConnectorConfig:
  return ConnectorConfig(
      scale_factor=3,
      vision_hidden_size=1152,
      text_hidden_size=2048,
  )


TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
    # Embeddings
    embedding="model.vision_model.embeddings.patch_embedding",
    embedding_position=(
        "model.vision_model.embeddings.position_embedding.weight"
    ),
    # Transformer blocks
    ff_up_proj="model.vision_model.encoder.layers.{}.mlp.fc1",
    ff_down_proj="model.vision_model.encoder.layers.{}.mlp.fc2",
    attn_query_proj="model.vision_model.encoder.layers.{}.self_attn.q_proj",
    attn_key_proj="model.vision_model.encoder.layers.{}.self_attn.k_proj",
    attn_value_proj="model.vision_model.encoder.layers.{}.self_attn.v_proj",
    attn_output_proj="model.vision_model.encoder.layers.{}.self_attn.out_proj",
    pre_attn_norm="model.vision_model.encoder.layers.{}.layer_norm1",
    post_attn_norm="model.vision_model.encoder.layers.{}.layer_norm2",
    # Final norm
    final_norm="model.vision_model.post_layernorm",
)


class FullVisionEncoder(nn.Module):
  """Siglip vision encoder and connector from the Edge Generative API."""

  def __init__(
      self,
      encoder_config: cfg.ModelConfig,
      connector_config: ConnectorConfig,
  ):
    super().__init__()
    self.siglip_encoder = image_encoder.SiglipVisionEncoder(encoder_config)
    self.connector = SmolVLMConnector(connector_config)
    self.encoder_config = encoder_config
    self.connector_config = connector_config

  @torch.inference_mode
  def forward(
      self,
      pixel_values: torch.Tensor,
      export_config: export_cfg.ExportConfig = None,
  ) -> torch.Tensor:
    x = self.siglip_encoder(pixel_values)
    x = self.connector(x)
    return x


class VisionEncoder(nn.Module):
  """Siglip vision encoder from the Edge Generative API."""

  def __init__(self, config: cfg.ModelConfig):
    super().__init__()
    self.siglip_encoder = image_encoder.SiglipVisionEncoder(config)
    self.config = config

  def forward(self, pixel_values: torch.Tensor = None) -> torch.Tensor:
    x = self.siglip_encoder(pixel_values)
    return x


def get_image_encoder_config() -> cfg.ModelConfig:
  """Returns the model config for the Full Image Encoder of a Smolvlm2 2B model.

  Returns:
    The model config for the image encoder of a Gemma3 4B model.
  """
  image_embedding_config = cfg.ImageEmbeddingConfig(
      channels=3,
      image_size=378,
      patch_size=14,
  )
  attn_config = cfg.AttentionConfig(
      num_heads=16,
      head_dim=72,
      num_query_groups=16,
      qkv_use_bias=True,
      output_proj_use_bias=True,
  )
  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.LAYER_NORM, epsilon=1e-6
  )
  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.SEQUENTIAL,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=4304,
      use_bias=True,
  )
  block_config = cfg.TransformerBlockConfig(
      attn_config=attn_config,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
  )
  config = cfg.ModelConfig(
      vocab_size=0,
      num_layers=27,
      max_seq_len=0,
      embedding_dim=1152,
      embedding_use_bias=True,
      image_embedding=image_embedding_config,
      block_configs=block_config,
      final_norm_config=norm_config,
      # num_mm_tokens_per_image=81,
      # enable_hlfb=False
  )
  return config


def build_image_encoder(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
) -> FullVisionEncoder:
  """Builds a FullVisionEncoder from the checkpoint path."""
  encoder_config = get_image_encoder_config()
  connector_config = get_connector_config()
  encoder = FullVisionEncoder(encoder_config, connector_config)
  loader = loading_utils.ModelLoader(
      checkpoint_path, TENSOR_NAMES, custom_loader
  )
  loader.load(encoder.siglip_encoder, strict=False)

  loader = loading_utils.ModelLoader(checkpoint_path, None, custom_loader)
  state = loader.get_state()
  converted_state = dict()
  converted_state["modality_projection.weight"] = state.pop(
      f"{CONNECTOR_TENSOR_NAME}.weight"
  )
  encoder.connector.load_state_dict(converted_state)

  encoder.eval()
  return encoder
