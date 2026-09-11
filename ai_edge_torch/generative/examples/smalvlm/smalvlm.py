import dataclasses
from typing import Callable, Dict, Optional

import ai_edge_torch.generative.layers.kv_cache as kv_utils
import ai_edge_torch.generative.layers.model_config as cfg
from ai_edge_torch.generative.utilities import export_config as export_cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import torch
from torch import nn

import image_encoder
import text_model

PROJECTION_TENSOR_NAME = "model.connector.modality_projection.proj"


@dataclasses.dataclass
class SmalVLMConfig:
  """SmalVLM model configurations."""

  image_encoder_config: cfg.ModelConfig
  decoder_config: cfg.ModelConfig

  image_token_id: int
  image_projection_use_bias: bool = False


class SmalVLM(nn.Module):
  """SmalVLM 256M model from the Edge Generative API."""

  def __init__(
      self,
      config: SmalVLMConfig,
      decoder_class: nn.Module,
      mask_cache_size: int = 0,
  ):
    super().__init__()

    self.image_encoder = image_encoder.SiglipVisionEncoder(
        config.image_encoder_config
    )
    self.image_projection = nn.Linear(
        12288,
        576,
        bias=False,
    )
    self.decoder = decoder_class(config.decoder_config, mask_cache_size)
    image_embedding_config = config.image_encoder_config.image_embedding
    self.num_patches = (
        image_embedding_config.image_size // image_embedding_config.patch_size
    ) ** 2
    self.config = config
    self.scale_factor = 4

  def get_image_features(self, pixel_values: torch.FloatTensor):
    batch_size, num_patches, num_channels, height, width = pixel_values.shape

    pixel_values = pixel_values.view(
        batch_size * num_patches, num_channels, height, width
    )
    image_encoded = self.image_encoder(pixel_values=pixel_values)
    return image_encoded

  def apply_connector(self, x: torch.FloatTensor):
    bsz, seq, embed_dim = x.size()
    height = width = int(seq**0.5)
    x = x.view(bsz, height, width, embed_dim)
    x = x.view(
        bsz,
        height,
        int(width / self.scale_factor),
        embed_dim * self.scale_factor,
    )
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(
        bsz,
        int(width / self.scale_factor),
        int(height / self.scale_factor),
        embed_dim * (self.scale_factor**2),
    )
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(
        bsz,
        int(seq / (self.scale_factor**2)),
        embed_dim * (self.scale_factor**2),
    )
    return self.image_projection(x)

  def inputs_merger(
      self,
      input_ids: torch.LongTensor,
      inputs_embeds: Optional[torch.Tensor],
      image_hidden_states: Optional[torch.Tensor],
  ):

    # All code below is simple operation, to replace dummy image embeddings with real ones:
    # special_image_token_mask = input_ids == self.config.image_token_id
    # inputs_embeds[special_image_token_mask] = image_hidden_states

    # inputs_embeds.shape: [1, 72, 576]
    # image_hidden_states.shape: [64, 576]
    # special_image_token_mask.shape: [1, 72]
    special_image_token_mask = input_ids == self.config.image_token_id

    # Create expanded mask for broadcasting: [1, 72, 1] -> [1, 72, 576]
    mask_expanded = special_image_token_mask.unsqueeze(-1).expand_as(
        inputs_embeds
    )

    # Create cumulative sum to map positions to image embedding indices
    # This gives us 0, 1, 2, ... for each True position in the mask
    cumsum_mask = special_image_token_mask.float().cumsum(dim=-1) - 1
    # Zero out positions where mask is False
    cumsum_mask = cumsum_mask * special_image_token_mask.float()

    # Convert to long for indexing
    cumsum_indices = cumsum_mask.long()

    # Expand image_hidden_states to match input sequence length
    # Use gather to select appropriate embeddings for each position
    batch_size, seq_len, embed_dim = inputs_embeds.shape
    num_images = image_hidden_states.shape[0]

    # Clamp indices to valid range to avoid out-of-bounds access
    cumsum_indices_clamped = torch.clamp(cumsum_indices, 0, num_images - 1)

    # Use advanced indexing to select embeddings - this should be JAX-compatible
    # since we're using integer indices rather than boolean masks
    selected_embeddings = image_hidden_states[
        cumsum_indices_clamped.view(-1)
    ].view(batch_size, seq_len, embed_dim)

    # Use torch.where to conditionally select between original and image embeddings
    result = torch.where(mask_expanded, selected_embeddings, inputs_embeds)

    return result

  @torch.inference_mode
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      kv_cache: kv_utils.KVCache,
      mask: Optional[torch.Tensor] = None,
      pixel_values: Optional[torch.Tensor] = None,
      export_config: Optional[export_cfg.ExportConfig] = None,
  ) -> dict[torch.Tensor, kv_utils.KVCache]:
    if pixel_values is None:
      return self.decoder.forward(
          tokens=tokens,
          input_pos=input_pos,
          kv_cache=kv_cache,
          mask=mask,
          export_config=export_config,
      )

    input_embeds = self.decoder.tok_embedding(tokens)

    image_encoded = self.get_image_features(pixel_values)
    image_hidden_states = self.apply_connector(image_encoded)
    image_hidden_states = image_hidden_states.view(
        -1, image_hidden_states.shape[-1]
    )

    input_embeds = self.inputs_merger(tokens, input_embeds, image_hidden_states)
    position_embeddings = self.decoder.get_rope(input_pos.view(1, -1))

    return self.decoder.forward_embeds(
        tokens=tokens,
        input_pos=input_pos,
        rope=position_embeddings,
        kv_cache=kv_cache,
        mask=mask,
        input_embeds=input_embeds,
        export_config=export_config,
    )


def get_model_config(get_decoder_config) -> SmalVLMConfig:
  """Returns the model config for a SmalVLM model.

  Returns:
    The model config for a SmalVLM model.
  """
  return SmalVLMConfig(
      image_encoder_config=image_encoder.get_image_encoder_config(),
      decoder_config=get_decoder_config(),
      image_token_id=49190,
      image_projection_use_bias=False,
  )


def build_model(
    checkpoint_path: str,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] = None,
    mask_cache_size: int = 0,
) -> SmalVLM:

  decoder_class = text_model.SmolVLMText
  decoder_tensor_names = text_model.TENSOR_NAMES
  get_decoder_config = text_model.get_model_config

  image_encoder_tensor_names = image_encoder.TENSOR_NAMES

  config = get_model_config(get_decoder_config)
  model = SmalVLM(config, decoder_class, mask_cache_size)

  # Load the parameters of image encoder.
  loader = loading_utils.ModelLoader(
      checkpoint_path, image_encoder_tensor_names, custom_loader
  )
  loader.load(model.image_encoder, strict=False)

  # Load the parameters of decoder.
  loader = loading_utils.ModelLoader(
      checkpoint_path, decoder_tensor_names, custom_loader
  )
  loader.load(model.decoder, strict=False)

  # Load the parameters of image projection.
  loader = loading_utils.ModelLoader(checkpoint_path, None, custom_loader)
  state = loader.get_state()
  converted_state = dict()
  converted_state["weight"] = state.pop(f"{PROJECTION_TENSOR_NAME}.weight")
  if config.image_projection_use_bias:
    converted_state["bias"] = state.pop(f"{PROJECTION_TENSOR_NAME}.bias")
  model.image_projection.load_state_dict(converted_state, strict=True)

  model.eval()
  return model


if __name__ == "__main__":
  model = build_model(
      checkpoint_path="./models/SmolVLM-256M-Instruct",
      mask_cache_size=1024,
  )
  print(model)
