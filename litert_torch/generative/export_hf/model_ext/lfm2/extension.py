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
"""Built-in `ModelExtension` for Liquid Foundation Model 2 (`lfm2`).

Minimal patch-only extension example: LFM2 reuses the generic core prefill and
decode exportables and `model_ext/metadata_builder.py` `conv` state buffers,
needing only runtime RMSNorm, short-convolution, and QKV/SwiGLU composite
patches registered in `lfm2/patch.py`, plus a `fused_weight_specs` entry so
Converter V2 can stream the fused `gate_up_proj` from the checkpoint.
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
# Registers lfm2 patches.
from litert_torch.generative.export_hf.model_ext.lfm2 import patch as _


@extension_lib.register_model_extension(override=False)
class Lfm2Extension(extension_lib.ModelExtension):
  """Built-in extension for LFM2 hybrid attention + short-conv LLMs."""

  model_types = ('lfm2',)
  # `lfm2/patch.py` `FusedLfm2MLP` fuses the MLP gate / up projections, which
  # LFM2 checkpoints name `w1` / `w3`, into `gate_up_proj`.
  fused_weight_specs = (
      extension_lib.FusedWeightSpec('gate_up_proj', ('w1', 'w3'), dim=0),
  )
