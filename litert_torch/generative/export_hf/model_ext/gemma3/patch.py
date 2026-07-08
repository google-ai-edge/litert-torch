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
"""Patches for Gemma3."""

import contextlib
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.layers import normalization
import torch
from transformers.models.gemma3 import modeling_gemma3


class Gemma3RMSNorm(torch.nn.Module):
  """RMSNorm Layer."""

  def __init__(self, dim: int, eps: float = 1e-6):
    """RMSNorm Layer."""
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(dim))
    self.variance_epsilon = eps
    self.hidden_size = dim

  def forward(self, hidden_states):
    return normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight + 1.0,
        self.variance_epsilon,
        torch.ones((self.hidden_size,), dtype=torch.float32),
    )

  def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@patches_lib.register_patch(["gemma3", "gemma3_text"])
@contextlib.contextmanager
def gemma3_litert_patch():
  print("Gemma3 patch applied.")
  original_norm = modeling_gemma3.Gemma3RMSNorm
  modeling_gemma3.Gemma3RMSNorm = Gemma3RMSNorm  # pyrefly: ignore[bad-assignment]

  try:
    yield
  finally:
    modeling_gemma3.Gemma3RMSNorm = original_norm
