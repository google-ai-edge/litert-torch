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
"""Patch for LFM2."""

import contextlib
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.lfm2 import short_conv as short_conv_lib
import transformers

modeling_lfm2 = transformers.models.lfm2.modeling_lfm2


@patches_lib.register_patch(["lfm2"])
@contextlib.contextmanager
def lfm2_litert_patch():
  print("LFM2 patch applied.")
  original_short_conv = modeling_lfm2.Lfm2ShortConv
  modeling_lfm2.Lfm2ShortConv = short_conv_lib.Lfm2ShortConv

  try:
    yield
  finally:
    modeling_lfm2.Lfm2ShortConv = original_short_conv
