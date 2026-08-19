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

from litert_torch.generative.export_hf.model_ext.gemma3 import vision_exportable as gemma3_vision_exportable
import torch


class LiteRTExportableModuleForGemma3nVisionEncoder(
    gemma3_vision_exportable.LiteRTExportableModuleForGemma3VisionEncoder
):
  """Exportable module for Gemma3n vision encoder."""

  def forward(
      self,
      images,
  ):
    images = images.permute((0, 3, 1, 2))  # to NCHW
    vision_outputs = self.model.model.vision_tower(
        pixel_values=images
    ).last_hidden_state
    vision_outputs = vision_outputs.reshape(
        vision_outputs.shape[0],
        self.model.config.vision_config.hidden_size,
        self.model.config.vision_soft_tokens_per_image,
    ).permute(0, 2, 1)
    # Normalize and embed the soft tokens into language model space.
    vision_outputs *= self.model.config.vision_config.hidden_size**0.5  # pyrefly: ignore[unsupported-operation]
    return {'features': vision_outputs}


class LiteRTExportableModuleForGemma3nVisionAdapter(
    gemma3_vision_exportable.LiteRTExportableModuleForGemma3VisionAdapter
):
  """Exportable module for Gemma3n vision adapter."""

  def forward(
      self,
      soft_tokens,
  ):
    image_features = self.model.model.embed_vision(inputs_embeds=soft_tokens)
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
          'Image processor is required for Exporting Gemma3n vision encoder.'
      )
    dummy_image = image_processor(
        images=[torch.zeros((1, 3, 224, 224))],
        return_tensors='pt',
    ).pixel_values
    with torch.device('meta'):
      features = self.model.model.vision_tower(
          pixel_values=dummy_image
      ).last_hidden_state
      features = features.reshape(
          features.shape[0],
          self.model.config.vision_config.hidden_size,
          self.model.config.vision_soft_tokens_per_image,
      ).permute(0, 2, 1)
    inputs = {'soft_tokens': torch.zeros_like(features, dtype=torch.float32)}
    return {'vision_adapter': (inputs, {})}
