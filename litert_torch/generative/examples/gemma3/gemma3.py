# Copyright 2025 The LiteRT Torch Authors.
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

"""Example of building a Gemma3 gpu model."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from litert_torch.generative.examples.gemma3 import decoder
from litert_torch.generative.examples.gemma3 import image_encoder
from litert_torch.generative.layers import builder
from litert_torch.generative.layers import kv_cache as kv_utils
import litert_torch.generative.layers.model_config as cfg
from litert_torch.generative.utilities import export_config as export_cfg
import litert_torch.generative.utilities.loader as loading_utils
import json
import os
import torch
from torch import nn


PROJECTION_TENSOR_NAME = "multi_modal_projector.linear"


def detect_model_size(checkpoint_path: str) -> Optional[str]:
  """Attempts to detect the model size from config.json in the checkpoint path.

  Args:
    checkpoint_path: Path to the checkpoint directory.

  Returns:
    '270m', '1b', or None if detection fails or config.json is missing.
  """
  config_path = os.path.join(checkpoint_path, "config.json")
  if not os.path.exists(config_path):
    return None

  try:
    with open(config_path, "r") as f:
      config = json.load(f)

    num_layers = config.get("num_hidden_layers")
    hidden_size = config.get("hidden_size")

    if num_layers == 18 or hidden_size == 640:
      return "270m"
    if num_layers == 26 or hidden_size == 1152:
      return "1b"
  except Exception:
    return None

  return None


@dataclass
class Gemma3MMConfig:
  """Gemma3 model configurations."""

  image_encoder_config: cfg.ModelConfig
  decoder_config: cfg.ModelConfig
  mm_norm_config: cfg.NormalizationConfig
  mm_extra_tokens: int
  image_token_id: int
  image_projection_scale: float
  image_projection_use_bias: bool = False


class Gemma3MM(nn.Module):
  """A Gemma3 multimodal model built from the Edge Generative API layers."""

  def __init__(self, config: Gemma3MMConfig, mask_cache_size: int = 0):
    super().__init__()

    self.image_encoder = image_encoder.SiglipVisionEncoderWithExit(
        config.image_encoder_config
    )
    self.decoder = decoder.Decoder(config.decoder_config, mask_cache_size)
    self.mm_norm = builder.build_norm(
        config.image_encoder_config.embedding_dim,
        config.mm_norm_config,
    )
    self.extra_embedding = nn.Embedding(
        config.mm_extra_tokens, config.image_encoder_config.embedding_dim
    )
    self.image_projection = nn.Linear(
        config.image_encoder_config.embedding_dim,
        config.decoder_config.embedding_dim,
        bias=config.image_projection_use_bias,
    )
    image_embedding_config = config.image_encoder_config.image_embedding
    self.num_patches = (
        image_embedding_config.image_size // image_embedding_config.patch_size
    ) ** 2
    self.config = config

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      image_indices: Optional[torch.Tensor] = None,
      image_feat_indices: Optional[torch.Tensor] = None,
      pixel_values: torch.Tensor = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    _, seq_len = tokens.size()
    assert self.config.decoder_config.max_seq_len >= seq_len, (
        f"Cannot forward sequence of length {seq_len}, max seq length is only"
        f" {self.config.decoder_config.max_seq_len}"
    )
    if pixel_values is None:
      return self.decoder(
          tokens=tokens,
          input_pos=input_pos,
          kv_cache=kv_cache,
          input_embeds=None,
          export_config=export_config,
      )
    vocab_size = self.config.decoder_config.vocab_size
    input_embeds = self.decoder.tok_embedding(
        torch.clip(tokens, 0, vocab_size - 1)
    )
    if self.decoder.config.embedding_scale is not None:
      input_embeds = input_embeds * self.decoder.config.embedding_scale

    # TODO: Identify embedding path for hard tokens if required.
    # extra_embeds = self.extra_embedding(
    #     torch.clip(tokens - vocab_size, 0, self.config.mm_extra_tokens - 1)
    # )
    # extra_embeds = self.image_projection(extra_embeds)
    # input_embeds = torch.where(tokens < self.config.decoder_config.vocab_size,
    #                            input_embeds, extra_embeds)
    # alternate method of implementation
    # rows, cols = torch.where(tokens >= self.config.vocab_size)
    # ext_embeds = self.ext_embedding(
    #     tokens[rows, cols] - self.config.vocab_size
    # )
    # ext_embeds = self.mm_projection(extra_embeds)
    # input_embeds[rows, cols, :] = extra_embeds

    # Shape of pixel_values: (b, n, c, h, w)
    batch_size, num_media, c, h, w = pixel_values.size()
    pixel_values = pixel_values.view(-1, c, h, w)
    image_encoded = self.image_encoder(pixel_values=pixel_values)
    image_encoded = self.mm_norm(image_encoded)
    image_encoded = self.image_projection(image_encoded)
    _, num_patches, num_channels = image_encoded.size()
    image_encoded = image_encoded.view(
        batch_size, num_media, num_patches, num_channels
    )

    # Interleave the image soft embeddings with the text embeddings
    for b in range(tokens.shape[0]):
      unbatched_image_encoded = image_encoded[b]
      image_features = unbatched_image_encoded[
          image_indices[b], image_feat_indices[b]
      ]
      index_to_copy = torch.where(image_indices[b] >= 0)[0]
      input_embeds[b] = torch.index_copy(
          input_embeds[b], 0, index_to_copy, image_features[index_to_copy]
      )
    return self.decoder(
        tokens=None,
        input_pos=input_pos,
        kv_cache=kv_cache,
        input_embeds=input_embeds,
        image_indices=image_indices,
        export_config=export_config,
    )


def get_fake_model_config() -> Gemma3MMConfig:
  return Gemma3MMConfig(
      image_encoder_config=image_encoder.get_fake_image_encoder_config(),
      decoder_config=decoder.get_fake_decoder_config_1b(),
      image_token_id=127,
      image_projection_scale=128**0.5,
      image_projection_use_bias=False,
      mm_norm_config=cfg.NormalizationConfig(
          type=cfg.NormalizationType.LAYER_NORM, epsilon=1e-6
      ),
      mm_extra_tokens=32,
  )


def build_model_1b(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> decoder.Decoder:
  if checkpoint_path:
    model = decoder.build_model_1b(
        checkpoint_path, custom_loader, mask_cache_size
    )
  else:
    config = decoder.get_decoder_config_1b()
    model = decoder.Decoder(config, mask_cache_size)
  # TODO: Load the parameters of decoder from checkpoint.
  model.eval()
  return model


def build_model_270m(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> decoder.Decoder:
  """Builds a Gemma3 270M model."""
  if checkpoint_path:
    model = decoder.build_model_270m(
        checkpoint_path, custom_loader, mask_cache_size
    )
  else:
    config = decoder.get_decoder_config_270m()
    model = decoder.Decoder(config, mask_cache_size)
  # TODO: Load the parameters of decoder from checkpoint.
  model.eval()
  return model


def build_model_4b(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> decoder.Decoder:
  """Builds a Gemma3 4B model."""
  if checkpoint_path:
    model = decoder.build_model_4b(
        checkpoint_path, custom_loader, mask_cache_size
    )
  else:
    config = decoder.get_decoder_config_4b()
    model = decoder.Decoder(config, mask_cache_size)
  # TODO: Load the parameters of decoder from checkpoint.
  model.eval()
  return model
