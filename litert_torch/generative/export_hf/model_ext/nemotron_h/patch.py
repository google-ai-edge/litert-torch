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
import torch
from transformers.models.nemotron_h import configuration_nemotron_h
from transformers.models.nemotron_h import modeling_nemotron_h


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
