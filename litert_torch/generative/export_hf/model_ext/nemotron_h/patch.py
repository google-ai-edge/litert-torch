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
"""Patch for NemotronH (Mamba2, MLP and attention layers).

`NemotronHMamba2Mixer` has the attribute names and the forward of the
GraniteMoeHybrid Mamba2 mixer, so its layers are exported with the same
`mamba2_mixer_forward`; see `granitemoehybrid/patch.py`. The differences are
read from the mixer itself: the time step limit is `(time_step_min, inf)` and
the gated norm is grouped. The MLP and MoE layers keep no state and get a
`LiteRTLMStatelessCacheLayer` in the cache.

`NemotronHBlock` builds its mixer from the module-level `MIXER_TYPES` dict, so
the forward is bound to the mixer instances of the loaded model rather than
swapped on the module attribute. In transformers 5.16.1 the block calls the
mixer without the keyword arguments of the model call, so the prefill
`valid_mask` reaches the mixer through the hooks of `patch_mamba2_mixers`.
"""

import contextlib
from typing import Any

from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.granitemoehybrid import patch as granitemoehybrid_patch
from litert_torch.generative.layers import normalization
import torch
from transformers.models.nemotron_h import configuration_nemotron_h
from transformers.models.nemotron_h import modeling_nemotron_h


class NemotronHRMSNorm(torch.nn.Module):
  """RMSNorm with the HLFB `odml.rms_norm` composite, computed in float32."""

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.hidden_size = hidden_size

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    original_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    return normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight,
        self.variance_epsilon,
        torch.ones(
            (self.hidden_size,),
            dtype=torch.float32,
            device=hidden_states.device,
        ),
    ).to(original_dtype)

  def extra_repr(self):
    return f"{self.hidden_size}, eps={self.variance_epsilon}"


def _check_config() -> configuration_nemotron_h.NemotronHConfig:
  return configuration_nemotron_h.NemotronHConfig(
      hidden_size=64,
      layers_block_type=["linear_attention"],
      mamba_num_heads=8,
      mamba_head_dim=16,
      ssm_state_size=16,
      n_groups=2,
      conv_kernel=4,
      chunk_size=8,
  )


@patches_lib.register_patch(["nemotron_h"])
@contextlib.contextmanager
def nemotron_h_litert_patch():
  """Builds the model with `NemotronHRMSNorm` above.

  The same swap as `granitemoehybrid_litert_patch`, for the same reason: the
  HLFB RMSNorm of `core/patches.py` reaches only modeling modules imported
  after it.
  """
  print("NemotronH patch applied.")
  original_norm = modeling_nemotron_h.NemotronHRMSNorm
  modeling_nemotron_h.NemotronHRMSNorm = NemotronHRMSNorm  # pyrefly: ignore[bad-assignment]
  try:
    yield
  finally:
    modeling_nemotron_h.NemotronHRMSNorm = original_norm


@patches_lib.register_model_patch(["nemotron_h"])
@contextlib.contextmanager
def patch_nemotron_h_model(model: torch.nn.Module, export_config: Any):
  """Model patch for NemotronH export."""
  del export_config
  rel = granitemoehybrid_patch.check_mamba2_mixer_forward(
      modeling_nemotron_h.NemotronHMamba2Mixer, _check_config()
  )
  with granitemoehybrid_patch.patch_mamba2_mixers(
      model, modeling_nemotron_h.NemotronHMamba2Mixer
  ) as mixers:
    print(
        f"NemotronH model patch applied to {len(mixers)} Mamba2 layers"
        f" (check vs. transformers: relative max |difference| {rel:.1e})."
    )
    yield
