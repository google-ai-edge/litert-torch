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
"""Patches for Gemma4 unified model for on-device deployment."""

import contextlib
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.layers import normalization
import torch


class Gemma4UnifiedRMSNorm(torch.nn.Module):
  """RMSNorm Layer."""

  def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
    """RMSNorm Layer."""
    super().__init__()
    self.with_scale = with_scale

    if self.with_scale:
      self.weight = torch.nn.Parameter(torch.ones(dim), requires_grad=True)
    else:
      self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    self.variance_epsilon = eps
    self.hidden_size = dim

  def forward(self, hidden_states):
    return normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight
        if self.with_scale
        else torch.ones((self.hidden_size,), dtype=torch.float32),
        self.variance_epsilon,
        torch.ones((self.hidden_size,), dtype=torch.float32),
    )

  def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# pytype: disable=import-error
@patches_lib.register_patch(["gemma4_unified"])
@contextlib.contextmanager
def gemma4_litert_patch():
  """Gemma4 unified patch."""
  print("Gemma4 unified patch applied.")
  from transformers.models.gemma4_unified import modeling_gemma4_unified  # pylint: disable=g-import-not-at-top

  original_norm = modeling_gemma4_unified.Gemma4UnifiedRMSNorm
  modeling_gemma4_unified.Gemma4UnifiedRMSNorm = Gemma4UnifiedRMSNorm

  try:
    yield
  finally:
    modeling_gemma4_unified.Gemma4UnifiedRMSNorm = original_norm


# pytype: enable=import-error


@patches_lib.register_model_patch(["gemma4_unified111"])
@contextlib.contextmanager
def gemma4_unified_litert_model_patch(model, export_config):
  """Patches Gemma4 unified model instance for export."""
  del export_config  # Unused.
  embed_tokens = None
  if (
      hasattr(model, "model")
      and hasattr(model.model, "language_model")
      and hasattr(model.model.language_model, "embed_tokens")
  ):
    embed_tokens = model.model.language_model.embed_tokens

  if embed_tokens is not None:
    model_config = model.config
    text_config = getattr(model_config, "text_config", model_config)
    eoi_token_id = getattr(text_config, "eoi_token_id", 258882)
    eoa_token_id = getattr(text_config, "eoa_token_id", 258883)

    with torch.no_grad():
      # Grab source row of PAD
      pad_row = embed_tokens.weight[0].clone()
      # Save original rows
      orig_eoi_row = embed_tokens.weight[eoi_token_id].clone()
      orig_eoa_row = embed_tokens.weight[eoa_token_id].clone()
      # Copy PAD row to the target indices
      embed_tokens.weight[[eoi_token_id, eoa_token_id], :] = pad_row

    try:
      yield
    finally:
      with torch.no_grad():
        # Restore target indices rows
        embed_tokens.weight[eoi_token_id] = orig_eoi_row
        embed_tokens.weight[eoa_token_id] = orig_eoa_row
  else:
    yield
