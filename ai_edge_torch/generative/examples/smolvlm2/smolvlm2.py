# Copyright 2025 The AI Edge Torch Authors.
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

"""Example of building a SmolVLM2 model."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ai_edge_torch.generative.examples.smolvlm2 import decoder
from ai_edge_torch.generative.examples.smolvlm2 import vision_encoder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import export_config as export_cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn


@dataclass
class SmolVLM2Config:
  """SmolVLM2 model configurations."""

  encoder_config: cfg.ModelConfig
  connector_config: vision_encoder.ConnectorConfig
  decoder_config: cfg.ModelConfig
  pad_token_id: int
  image_token_id: int


def get_model_config() -> SmolVLM2Config:
  """Returns the model config for a SmolVLM2 model."""
  return SmolVLM2Config(
      encoder_config=vision_encoder.get_image_encoder_config(),
      connector_config=vision_encoder.get_connector_config(),
      decoder_config=decoder.get_decoder_config(),
      pad_token_id=2,
      image_token_id=49190
  )


class SmolVLM2(nn.Module):
  """SmolVLM2 multimodal model built from the Edge Generative API layers."""

  def __init__(self, config: SmolVLM2Config, mask_cache_size: int = 0):
    super().__init__()
    self.config = config
    self.vocab_size = config.decoder_config.vocab_size
    image_embedding_config = config.encoder_config.image_embedding
    if image_embedding_config is None:
      raise ValueError("encoder_config needs to be initialized")
    self.num_patches = (
        image_embedding_config.image_size // image_embedding_config.patch_size
    ) ** 2
    self.image_seq_len = int(
        self.num_patches / (config.connector_config.scale_factor**2)
    )
    self.image_token_id = self.config.image_token_id
    self.padding_idx = self.config.pad_token_id

    self.vision_encoder = vision_encoder.VisionEncoder(config.encoder_config)
    self.connector = vision_encoder.SmolVLMConnector(config.connector_config)
    self.decoder = decoder.Decoder(config.decoder_config, mask_cache_size)
    self.config = config

  def inputs_merger(
      self,
      input_ids: torch.LongTensor,
      inputs_embeds: torch.Tensor,
      image_hidden_states: torch.Tensor,
  ):
    """Merges image hidden states into text embeddings based on image tokens."""
    _, patch_size, _ = image_hidden_states.shape

    image_mask = (input_ids == self.image_token_id)
    num_image_tokens = image_mask.sum(dim=1)

    assert torch.all(num_image_tokens % patch_size == 0)

    blocks_per_sample = num_image_tokens // patch_size

    offsets = torch.nn.functional.pad(
        blocks_per_sample.cumsum(dim=0), (1, 0), value=0
    )
    block_offset = offsets[:-1]
    row_cum = image_mask.cumsum(dim=-1)
    chunk_idx = (row_cum - 1) // patch_size
    local_idx = (row_cum - 1) % patch_size
    block_idx = block_offset.unsqueeze(1) + chunk_idx

    image_embeds = torch.zeros_like(inputs_embeds)
    image_embeds[image_mask] = image_hidden_states[
        block_idx[image_mask], local_idx[image_mask], :
    ]
    merged_embeds = torch.where(
        image_mask.unsqueeze(-1), image_embeds, inputs_embeds
    )
    return merged_embeds

  def get_image_features(
      self,
      pixel_values: torch.FloatTensor,
  ):
    """Get image features from pixel values."""
    image_encoded = self.vision_encoder(pixel_values=pixel_values)
    image_hidden_states = self.connector(image_encoded)
    return image_hidden_states

  @torch.inference_mode
  def forward(
      self,
      tokens: Optional[torch.Tensor],  # input_ids
      input_pos: torch.Tensor,  # position_ids
      kv_cache: kv_utils.KVCache,
      input_embeds: Optional[torch.Tensor] = None,
      pixel_values: torch.Tensor = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    if tokens is not None:
      _, seq_length = tokens.shape
    elif input_embeds is not None:
      _, seq_length, _ = input_embeds.shape
    else:
      raise ValueError("Must specify either `input_ids` or `inputs_embeds`.")
    assert self.config.decoder_config.max_seq_len >= seq_length, (
        f"Cannot forward sequence of length {seq_length}, max seq length is "
        f"only {self.config.decoder_config.max_seq_len}"
    )

    if input_embeds is None:
      input_embeds = self.decoder.tok_embedding(tokens)

    image_hidden_states = None
    if pixel_values is not None:
      image_hidden_states = self.get_image_features(pixel_values)

    if input_embeds is not None and image_hidden_states is not None:
      input_embeds = self.inputs_merger(
          input_ids=tokens,
          inputs_embeds=input_embeds,
          image_hidden_states=image_hidden_states,
      )

    outputs = self.decoder(
        tokens=tokens,
        input_pos=input_pos,
        kv_cache=kv_cache,
        input_embeds=input_embeds,
        export_config=export_config,
    )
    return outputs


def build_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> SmolVLM2:
  """Builds a SmolVLM2 model from the checkpoint path."""
  config = get_model_config()
  model = SmolVLM2(config, mask_cache_size)

  # Load the parameters of image encoder.
  loader = loading_utils.ModelLoader(
      checkpoint_path, vision_encoder.TENSOR_NAMES, custom_loader
  )
  loader.load(model.vision_encoder, strict=False)

  # Load the parameters of decoder.
  loader = loading_utils.ModelLoader(
      checkpoint_path, decoder.TENSOR_NAMES, custom_loader
  )
  loader.load(model.decoder, strict=False)

  # Load the parameters of connector.
  loader = loading_utils.ModelLoader(checkpoint_path, None, custom_loader)
  state = loader.get_state()
  converted_state = dict()
  converted_state["modality_projection.weight"] = state.pop(
      f"{vision_encoder.CONNECTOR_TENSOR_NAME}.weight"
  )
  model.connector.load_state_dict(converted_state)

  model.eval()
  return model

