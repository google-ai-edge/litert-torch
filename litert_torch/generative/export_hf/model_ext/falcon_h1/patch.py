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
"""Patch for Falcon-H1 (attention and a Mamba2 mixer in every layer).

Every FalconH1 decoder layer runs attention and a Mamba2 mixer side by side on
the same input. The layers are "hybrid": both keep their states in one
`LiteRTLMHybridCacheLayer` per layer (`kv_cache_k_{i}`, `kv_cache_v_{i}`,
`kv_cache_c_{i}` and `kv_cache_r_{i}`). The mixers are exported with
`mamba2_mixer_forward` (see `granitemoehybrid/patch.py`), which also applies
the FalconH1 input and projection multipliers and the optional gated norm.

`FalconH1Model` in transformers 5.16.1 does not pass the keyword arguments of
the model call on to its decoder layers, so the patch passes them on: the
attention needs the exporter's `k_ts_idx` and `v_ts_idx`.
"""

import contextlib
import inspect
from typing import Any

from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.granitemoehybrid import patch as granitemoehybrid_patch
import torch
from transformers.models.falcon_h1 import configuration_falcon_h1
from transformers.models.falcon_h1 import modeling_falcon_h1


class _FalconH1Mixer(modeling_falcon_h1.FalconH1Mixer):
  """`FalconH1Mixer` with the `mup_vector` buffer that `FalconH1Model` adds."""

  def __init__(self, config: configuration_falcon_h1.FalconH1Config, layer_idx):
    super().__init__(config, layer_idx=layer_idx)
    self.register_buffer(
        "mup_vector",
        modeling_falcon_h1.compute_mup_vector(config),
        persistent=False,
    )


def _check_config(model_config: Any) -> configuration_falcon_h1.FalconH1Config:
  """A small mixer config with the gated norm settings of `model_config`."""
  return configuration_falcon_h1.FalconH1Config(
      hidden_size=64,
      num_hidden_layers=1,
      num_attention_heads=4,
      num_key_value_heads=2,
      mamba_d_ssm=128,
      mamba_n_heads=8,
      mamba_d_state=16,
      mamba_n_groups=2,
      mamba_chunk_size=8,
      mamba_rms_norm=model_config.mamba_rms_norm,
      mamba_norm_before_gate=model_config.mamba_norm_before_gate,
      ssm_in_multiplier=0.8,
      ssm_multipliers=[0.6, 0.8, 1.2, 1.4, 0.7],
  )


@contextlib.contextmanager
def _pass_model_kwargs_to_layers(model: torch.nn.Module):
  """Passes the keyword arguments of the base model call to its layers.

  A forward pre-hook on the base model keeps the keyword arguments of the call
  that its forward does not name, and a forward pre-hook on each decoder layer
  adds the ones that the layer call lacks. Where the base model passes them on
  itself, the layer call already has them.

  Args:
    model: The model to patch.

  Yields:
    None.
  """
  base_model = getattr(model, "base_model", model)
  params = inspect.signature(base_model.forward).parameters.values()
  named = {p.name for p in params if p.kind != p.VAR_KEYWORD}
  # The keyword arguments of the base model call in progress. They are cleared
  # after every call, so no traced tensor outlives the call that made it.
  call_kwargs: dict[str, Any] = {}

  def keep_kwargs(module, args, kwargs):
    del module, args
    call_kwargs.update(
        (key, value) for key, value in kwargs.items() if key not in named
    )

  def clear_kwargs(module, args, output):
    del module, args, output
    call_kwargs.clear()

  def add_kwargs(module, args, kwargs):
    del module
    missing = {k: v for k, v in call_kwargs.items() if k not in kwargs}
    if not missing:
      return None
    return args, {**kwargs, **missing}

  handles = [
      base_model.register_forward_pre_hook(keep_kwargs, with_kwargs=True),
      base_model.register_forward_hook(clear_kwargs, always_call=True),
  ]
  for layer in base_model.modules():
    if isinstance(layer, modeling_falcon_h1.FalconH1DecoderLayer):
      handles.append(
          layer.register_forward_pre_hook(add_kwargs, with_kwargs=True)
      )
  try:
    yield
  finally:
    for handle in handles:
      handle.remove()
    call_kwargs.clear()


@patches_lib.register_model_patch(["falcon_h1"])
@contextlib.contextmanager
def patch_falcon_h1_model(model: torch.nn.Module, export_config: Any):
  """Model patch for Falcon-H1 export."""
  del export_config
  config = getattr(model.config, "text_config", model.config)
  rel = granitemoehybrid_patch.check_mamba2_mixer_forward(
      _FalconH1Mixer, _check_config(config)
  )
  with granitemoehybrid_patch.patch_mamba2_mixers(
      model, modeling_falcon_h1.FalconH1Mixer
  ) as mixers, _pass_model_kwargs_to_layers(model):
    print(
        f"FalconH1 model patch applied to {len(mixers)} Mamba2 mixers"
        f" (check vs. transformers: relative max |difference| {rel:.1e})."
    )
    yield
