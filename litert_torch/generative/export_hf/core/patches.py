# Copyright 2025 The LiteRT Torch Authors.
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
"""Patches for transformers."""

import transformers
from litert_torch.generative.layers import normalization
import torch


class RMSNorm(torch.nn.Module):
  """RMSNorm Layer."""

  def __init__(self, hidden_size, eps=1e-6):
    """RMSNorm Layer."""
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.hidden_size = hidden_size

  def forward(self, hidden_states):
    return normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight,
        self.variance_epsilon,
        torch.ones((self.hidden_size,), dtype=torch.float32),
    )

  def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# TODO(weiyiw): Add model specific patches to cover more models.
def _use_kernel_forward_from_hub(layer_name):

  def decorator(cls):
    if layer_name == "RMSNorm":
      return RMSNorm
    return cls

  return decorator


original_use_kernel_forward_from_hub = (
    transformers.integrations.use_kernel_forward_from_hub
)

transformers.integrations.use_kernel_forward_from_hub = (
    _use_kernel_forward_from_hub
)
