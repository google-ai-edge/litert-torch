# Copyright 2026 The LiteRT Torch Authors.
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
"""Exportable modules for Gemma4 vision encoder and adapter."""

from litert_torch.generative.export_hf.core import attention as attention_lib
from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
import torch


class LiteRTExportableModuleForGemma4VisionEncoder(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma4 vision encoder."""

  def __init__(self, model: torch.nn.Module, export_config):
    super().__init__(export_config)
    self.model = model
    if export_config.use_sdpa_composite_for_vision:
      # The vision encoder attention is bidirectional over all patches and has
      # no KV cache, so it can use the fused SDPA composite. Only the vision
      # tower is switched; `LiteRTGemma4VisionEncoder` then calls attention
      # without a mask.
      model.model.vision_tower.set_attn_implementation(
          attention_lib.BIDIRECTIONAL_SDPA_ATTENTION
      )

  def forward(
      self,
      images,
      positions_xy,
  ):
    pixel_values = images
    pixel_position_ids = positions_xy
    vision_tower = self.model.model.vision_tower

    pooling_kernel_size = vision_tower.config.pooling_kernel_size
    output_length = pixel_values.shape[-2] // (
        pooling_kernel_size * pooling_kernel_size  # pyrefly: ignore[unsupported-operation]
    )

    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    inputs_embeds = vision_tower.patch_embedder(
        pixel_values, pixel_position_ids, padding_positions
    )
    output = vision_tower.encoder(
        inputs_embeds=inputs_embeds,
        attention_mask=~padding_positions,
        pixel_position_ids=pixel_position_ids,
    )

    hidden_states, pooler_mask = vision_tower.pooler(
        hidden_states=output.last_hidden_state,
        pixel_position_ids=pixel_position_ids,
        padding_positions=padding_positions,
        output_length=output_length,
    )

    if vision_tower.config.standardize:
      hidden_states = (
          hidden_states - vision_tower.std_bias
      ) * vision_tower.std_scale

    return {'features': hidden_states, 'mask': pooler_mask}

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    # Currently we only support batch size = 1.
    image_processor = kwargs.get('image_processor', None)
    if image_processor is None:
      raise ValueError(
          'Image processor is required for Exporting Gemma4 vision encoder.'
      )
    num_soft_tokens = kwargs.get('gemma4_vision_max_soft_tokens', 140)
    dummy_image = image_processor(
        images=[torch.zeros((1, 3, 224, 224))],
        max_soft_tokens=num_soft_tokens,
        return_tensors='pt',
    )
    inputs = {
        'images': dummy_image.pixel_values,
        'positions_xy': dummy_image.image_position_ids.int(),
    }
    return {f'vision_{num_soft_tokens}': (inputs, {})}


class LiteRTExportableModuleForGemma4VisionAdapter(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma4 vision adapter."""

  def __init__(self, model: torch.nn.Module, export_config, tokenizer):
    super().__init__(export_config)
    self.model = model
    self.tokenizer = tokenizer

  def forward(
      self,
      soft_tokens,
  ):
    mm_embedding = self.model.model.embed_vision(inputs_embeds=soft_tokens)
    return {'mm_embedding': mm_embedding}

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    # Currently we only support batch size = 1.
    image_processor = kwargs.get('image_processor', None)
    if image_processor is None:
      raise ValueError(
          'Image processor is required for Exporting Gemma4 vision adapter.'
      )
    vision_output_length = kwargs.get('gemma4_vision_max_soft_tokens', 140)
    dummy_image = image_processor(
        images=[torch.zeros((1, 3, 224, 224))],
        max_soft_tokens=vision_output_length,
        return_tensors='pt',
    ).pixel_values
    hidden_size = dummy_image.shape[-1]
    features = torch.zeros(
        (1, vision_output_length, hidden_size), dtype=torch.float32
    )
    inputs = {'soft_tokens': features}
    return {f'vision_adapter_{vision_output_length}': (inputs, {})}


class LiteRTExportableModuleForGemma4EndOfImage(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma4 end of image token."""

  def __init__(self, model: torch.nn.Module, export_config, tokenizer):
    super().__init__(export_config)
    token_ids = tokenizer.encode(
        tokenizer.special_tokens_map['eoi_token'],
        add_special_tokens=False,
    )
    self.register_buffer(
        'eoi_embedding',
        exportable_module_base.embed_tokens_on_cpu(model, token_ids),
        persistent=False,
    )

  def forward(self):
    return {'eoi_embedding': self.eoi_embedding}

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    del model_config, kwargs
    return {'eoi': (dict(), {})}
