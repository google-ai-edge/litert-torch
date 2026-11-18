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
from litert_torch.generative.export_hf.model_ext.gemma4 import vision_budgets
import torch


class LiteRTExportableModuleForGemma4UnifiedVisionEncoder(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for Gemma4 unified vision encoder.

  Gemma 4 Unified has no vision tower: `model.model.embed_vision`
  (`Gemma4UnifiedVisionEmbedder`) projects merged raw pixel patches straight
  into the language model space (LN -> Dense -> LN -> + factorized 2D position
  embedding -> LN -> RMSNorm -> Linear).
  """

  def __init__(self, model: torch.nn.Module, export_config):
    super().__init__(export_config)
    self.model = model

  def forward(
      self,
      images,
      positions_xy,
  ):
    embed_vision = self.model.model.embed_vision
    pixel_values = images.to(embed_vision.patch_dense.weight.dtype)
    hidden_states = embed_vision.patch_ln1(pixel_values)
    hidden_states = embed_vision.patch_dense(hidden_states)
    hidden_states = embed_vision.patch_ln2(hidden_states)

    # Factorized 2D position embedding; padding patches are (-1, -1).
    clamped = positions_xy.clamp(min=0).long()
    valid = (positions_xy != -1).to(embed_vision.pos_embedding.dtype)
    pos_x = embed_vision.pos_embedding[clamped[..., 0], 0]
    pos_y = embed_vision.pos_embedding[clamped[..., 1], 1]
    pos_embs = pos_x * valid[..., 0:1] + pos_y * valid[..., 1:2]
    hidden_states = embed_vision.pos_norm(hidden_states + pos_embs)

    projected = embed_vision.multimodal_embedder(hidden_states)
    padding_mask = ~((positions_xy == -1).all(dim=-1))
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
    sample_inputs = {}
    for num_soft_tokens in vision_budgets.get_vision_token_budgets(kwargs):
      dummy_image = image_processor(
          images=[torch.zeros((1, 3, 224, 224))],
          max_soft_tokens=num_soft_tokens,
          return_tensors='pt',
      )
      inputs = {
          'images': dummy_image.pixel_values,
          'positions_xy': dummy_image.image_position_ids.int(),
      }
      sample_inputs[f'vision_{num_soft_tokens}'] = (inputs, {})
    return sample_inputs


class LiteRTExportableModuleForGemma4UnifiedEndOfImage(
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
    return {'eoi': (dict(), {})}
