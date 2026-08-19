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
"""Exportable modules for Gemma4 Unified vision encoder and adapter."""

from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
import torch


class LiteRTExportableModuleForGemma4UnifiedVisionEncoder(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma4 unified vision encoder."""

  def __init__(self, model: torch.nn.Module, export_config):
    super().__init__(export_config)
    self.model = model

  def forward(
      self,
      images,
      positions_xy,
  ):
    pixel_values = images
    pixel_position_ids = positions_xy
    vision_embedder = self.model.model.vision_embedder
    embed_vision = self.model.model.embed_vision

    embedded = vision_embedder(pixel_values, pixel_position_ids)
    projected = embed_vision(embedded)
    padding_mask = ~((pixel_position_ids == -1).all(dim=-1))
    return {'features': projected, 'mask': padding_mask}

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


class LiteRTExportableModuleForGemma4UnifiedEndOfImage(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma4 end of image token."""

  def __init__(self, model: torch.nn.Module, export_config, tokenizer):
    super().__init__(export_config)
    self.model = model
    self.tokenizer = tokenizer

  def forward(self):
    return {
        'eoi_embedding': self.model.get_input_embeddings()(
            torch.tensor(
                [
                    self.tokenizer.encode(
                        self.tokenizer.special_tokens_map['eoi_token'],
                        add_special_tokens=False,
                    )
                ],
                dtype=torch.int32,
            )
        )
    }

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    return {'eoi': (dict(), {})}
