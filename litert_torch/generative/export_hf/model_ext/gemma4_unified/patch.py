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
    x_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    ret = normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight
        if self.with_scale
        else torch.ones((self.hidden_size,), dtype=torch.float32),
        self.variance_epsilon,
        torch.ones((self.hidden_size,), dtype=torch.float32),
    )
    return ret.to(x_dtype)

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
