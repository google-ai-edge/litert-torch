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
"""Patch for Zamba2 (Mamba2 layers and a shared attention block).

Zamba2 stacks Mamba2 layers ("linear_attention") and "hybrid" layers, in which
a transformer block shared by the hybrid layers runs before the layer's Mamba2
mixer. Each hybrid layer keeps its KV cache and its Mamba2 states in a
`LiteRTLMHybridCacheLayer` at its layer index. The shared block needs no
patch: its weights are tied, and the adapters of a layer are picked by the
layer index, a Python int at export time. The mixers are exported with
`mamba2_mixer_forward`; see `granitemoehybrid/patch.py`. The RMSNorms are built
as `Zamba2RMSNorm` below.
"""

import contextlib
from typing import Any

from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.granitemoehybrid import patch as granitemoehybrid_patch
from litert_torch.generative.layers import normalization
import torch
from transformers.models.zamba2 import configuration_zamba2
from transformers.models.zamba2 import modeling_zamba2


class Zamba2RMSNorm(torch.nn.Module):
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


def _check_config() -> configuration_zamba2.Zamba2Config:
  return configuration_zamba2.Zamba2Config(
      hidden_size=64,
      num_hidden_layers=1,
      layers_block_type=["linear_attention"],
      num_attention_heads=4,
      n_mamba_heads=8,
      mamba_d_state=16,
      mamba_ngroups=2,
      chunk_size=8,
  )


@patches_lib.register_patch(["zamba2"])
@contextlib.contextmanager
def zamba2_litert_patch():
  """Builds the model with `Zamba2RMSNorm` above.

  The transformers `Zamba2RMSNorm` does not take the hub kernel decorator that
  `core/patches.py` uses to put in its HLFB RMSNorm, so this patch swaps it.
  """
  print("Zamba2 patch applied.")
  original_norm = modeling_zamba2.Zamba2RMSNorm
  modeling_zamba2.Zamba2RMSNorm = Zamba2RMSNorm  # pyrefly: ignore[bad-assignment]
  try:
    yield
  finally:
    modeling_zamba2.Zamba2RMSNorm = original_norm


@patches_lib.register_model_patch(["zamba2"])
@contextlib.contextmanager
def patch_zamba2_model(model: torch.nn.Module, export_config: Any):
  """Model patch for Zamba2 export."""
  del export_config
  rel = granitemoehybrid_patch.check_mamba2_mixer_forward(
      modeling_zamba2.Zamba2MambaMixer, _check_config()
  )
  with granitemoehybrid_patch.patch_mamba2_mixers(
      model, modeling_zamba2.Zamba2MambaMixer
  ) as mixers:
    print(
        f"Zamba2 model patch applied to {len(mixers)} Mamba2 mixers"
        f" (check vs. transformers: relative max |difference| {rel:.1e})."
    )
    yield
