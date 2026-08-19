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
"""Exportable modules for LFM2 vision encoder and adapter."""

from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
import torch


class LiteRTExportableModuleForLFM2VisionEncoder(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for LFM2 vision encoder."""

  def __init__(self, model: torch.nn.Module, export_config):
    super().__init__(export_config)
    self.model = model

  def forward(
      self,
      images,
  ):
    pixel_attention_mask = torch.ones([1, 1024], dtype=torch.int32)
    spatial_shapes = torch.tensor([[32, 32]], dtype=torch.int32)
    return {
        'features': (
            self.model.model.vision_tower(
                pixel_values=images,
                spatial_shapes=spatial_shapes,
                pixel_attention_mask=pixel_attention_mask,
                return_dict=True,
            ).last_hidden_state
        )
    }

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    # Currently we only support batch size = 1.
    image_processor = kwargs.get('image_processor', None)
    if image_processor is None:
      raise ValueError(
          'Image processor is required for Exporting LFM2 vision encoder.'
      )
    dummy_image = image_processor(
        images=[torch.zeros((1, 3, 512, 512))],
        return_tensors='pt',
    ).pixel_values
    inputs = {'images': dummy_image}
    return {f'vision_{dummy_image.shape[-1]}': (inputs, {})}


class LiteRTExportableModuleForLFM2VisionAdapter(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for LFM2 vision adapter."""

  def __init__(self, model: torch.nn.Module, export_config, tokenizer):
    super().__init__(export_config)
    self.model = model
    self.tokenizer = tokenizer

  def forward(
      self,
      soft_tokens,
  ):
    soft_tokens = soft_tokens.reshape((1, 32, 32, -1))
    image_features = self.model.model.multi_modal_projector(soft_tokens)
    image_features = image_features.reshape(1, -1, image_features.size(-1))
    return {'mm_embedding': image_features}

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    # Currently we only support batch size = 1.
    image_processor = kwargs.get('image_processor', None)
    if image_processor is None:
      raise ValueError(
          'Image processor is required for Exporting LFM2-VL vision adapter.'
      )
    dummy_image = image_processor(
        images=[torch.zeros((1, 3, 512, 512))],
        return_tensors='pt',
    ).pixel_values
    pixel_attention_mask = torch.ones([1, 1024], dtype=torch.int32)
    spatial_shapes = torch.tensor([[32, 32]], dtype=torch.int32)
    with torch.device('meta'):
      features = self.model.model.vision_tower(
          pixel_values=dummy_image,
          spatial_shapes=spatial_shapes,
          pixel_attention_mask=pixel_attention_mask,
      ).last_hidden_state
    inputs = {'soft_tokens': torch.zeros_like(features, dtype=torch.float32)}
    return {'vision_adapter': (inputs, {})}
