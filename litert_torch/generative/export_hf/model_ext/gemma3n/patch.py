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
"""Patches for Gemma3n."""

import contextlib
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
import transformers

# from litert_torch.generative.layers import normalization
import torch
from torch import nn

modeling_gemma3n = transformers.models.gemma3n.modeling_gemma3n


# https://github.com/huggingface/transformers/issues/43412
class Gemma3nTextMLP(modeling_gemma3n.Gemma3nTextMLP):
  """Patched Gemma3nTextMLP."""

  def __init__(
      self, config: modeling_gemma3n.Gemma3nTextConfig, layer_idx: int = 0
  ):
    super().__init__(config, layer_idx)
    if self.activation_sparsity > 0.0:
      normal_dist = torch.distributions.normal.Normal(0, 1)
      std_multiplier = normal_dist.icdf(
          torch.tensor(self.activation_sparsity, dtype=torch.float32)
      )
      self.register_buffer("_std_multiplier", std_multiplier, persistent=False)

  def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
    std_multiplier = self._std_multiplier.to(inputs.dtype)
    inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
    inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
    cutoff_x = inputs_mean + inputs_std * std_multiplier
    return nn.functional.relu(inputs - cutoff_x)


class Gemma3nTextAltUp(modeling_gemma3n.Gemma3nTextAltUp):
  """Patched Gemma3nTextAltUp."""

  def correct(
      self, predictions: torch.Tensor, activated: torch.Tensor
  ) -> torch.Tensor:
    """Corrects the predictions relative to the

    Args:
        predictions: A 4D tensor of shape `[num_altup_inputs, batch_size,
          num_tokens, hidden_size]` derived by stacking the input embeddings and
          preprocessing the last `num_altup_inputs - 1` matrices.
        activated: A 3D tensor of shape `[batch_size, num_tokens, hidden_size]`
          containing the activated inputs.

    Returns:
        A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens,
        hidden_size]` correcting the original
            predictions relative to the activated input embeddings.
    """
    modalities = self.compute_router_modalities(activated)
    innovation = (
        activated - predictions[self.config.altup_active_idx]
    )  # (batch, num_tokens, hidden_size)
    innovation = innovation.repeat(
        self.config.altup_num_inputs, 1, 1, 1
    )  # Repeat on dim0 to match predictions

    # === Change start ===
    if self.training and self.config.altup_coef_clip is not None:
      self.correction_coefs.weight.data.clamp_(
          -self.config.altup_coef_clip, self.config.altup_coef_clip
      )
    # === Change end ===

    # all_coefs adapted from jax.numpy.einsum("...p,pi->...i", ...)
    # Permute to (altup_num_inputs, batch_size, num_tokens) as the last dim is
    # a scalar applied to each altup input and expand on dim1 for
    # broadcastability
    all_coefs: torch.Tensor = self.correction_coefs(modalities) + 1.0
    all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)

    corrected = torch.mul(innovation, all_coefs)
    corrected += predictions  # add the original input
    return corrected.contiguous().type_as(activated)


# TODO(weiyiw): Add patch for RMSNorm.


@patches_lib.register_patch(["gemma3n", "gemma3n_text", "gemma3n_vision"])
@contextlib.contextmanager
def gemma3n_litert_patch():
  """Patches for Gemma3n."""
  print("Gemma3n patch applied.")
  original_mlp = modeling_gemma3n.Gemma3nTextMLP
  modeling_gemma3n.Gemma3nTextMLP = Gemma3nTextMLP
  original_altup = modeling_gemma3n.Gemma3nTextAltUp
  modeling_gemma3n.Gemma3nTextAltUp = Gemma3nTextAltUp
  try:
    yield
  finally:
    modeling_gemma3n.Gemma3nTextMLP = original_mlp
    modeling_gemma3n.Gemma3nTextAltUp = original_altup
