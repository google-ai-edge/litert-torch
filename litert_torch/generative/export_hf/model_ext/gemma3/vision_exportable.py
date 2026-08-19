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
"""Exportable modules for Gemma3 vision encoder and adapter."""

from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
import torch


class LiteRTExportableModuleForGemma3VisionEncoder(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma3 vision encoder."""

  def __init__(self, model: torch.nn.Module, export_config):
    super().__init__(export_config)
    self.model = model

  def forward(
      self,
      images,
  ):
    images = images.permute((0, 3, 1, 2))  # to NCHW
    return {
        'features': (
            self.model.model.vision_tower(pixel_values=images).last_hidden_state
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
          'Image processor is required for Exporting Gemma3 vision encoder.'
      )
    dummy_image = image_processor(
        images=[torch.zeros((1, 3, 224, 224))],
        return_tensors='pt',
    ).pixel_values
    inputs = {'images': dummy_image.permute((0, 2, 3, 1))}
    return {f'vision_{dummy_image.shape[-1]}': (inputs, {})}


class LiteRTExportableModuleForGemma3VisionAdapter(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma3 vision adapter."""

  def __init__(self, model: torch.nn.Module, export_config, tokenizer):
    super().__init__(export_config)
    self.model = model
    self.tokenizer = tokenizer

  def forward(
      self,
      soft_tokens,
  ):
    image_features = self.model.model.multi_modal_projector(soft_tokens)
    eoi = self.tokenizer.encode(
        self.tokenizer.special_tokens_map['eoi_token'], add_special_tokens=False
    )
    eoi_emb = self.model.get_input_embeddings()(torch.tensor(eoi)[None, :])

    mm_embedding = torch.concat([image_features, eoi_emb], axis=1)  # pyrefly: ignore[unexpected-keyword]
    return {'mm_embedding': mm_embedding}

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    # Currently we only support batch size = 1.
    image_processor = kwargs.get('image_processor', None)
    if image_processor is None:
      raise ValueError(
          'Image processor is required for Exporting Gemma3 vision encoder.'
      )
    dummy_image = image_processor(
        images=[torch.zeros((1, 3, 224, 224))],
        return_tensors='pt',
    ).pixel_values
    with torch.device('meta'):
      features = self.model.model.vision_tower(
          pixel_values=dummy_image
      ).last_hidden_state
    inputs = {'soft_tokens': torch.zeros_like(features, dtype=torch.float32)}
    return {'vision_adapter': (inputs, {})}
