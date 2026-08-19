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

"""Patches for Whisper ASR model."""

import contextlib
from litert_torch.generative.export_hf.core.speech import asr_model
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
import transformers


@patches_lib.register_patch(["whisper"])
@contextlib.contextmanager
def whisper_litert_patch():
  print("Whisper ASR patch applied.")
  attn_funs = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
  original_sdpa = attn_funs.get("sdpa")
  attn_funs["sdpa"] = asr_model._sdpa

  # TODO: b/524681030 - Move more patches like normalization replacements from
  # each model wrapper to here.

  try:
    yield
  finally:
    attn_funs["sdpa"] = original_sdpa
