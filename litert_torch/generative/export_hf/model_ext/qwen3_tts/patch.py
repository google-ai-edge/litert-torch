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
"""Patches for Qwen3-TTS models for LiteRT export."""

import contextlib

from litert_torch.generative.export_hf.core.speech import asr_model
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.layers import normalization
import torch
from torch import nn
import transformers


class Qwen3TTSRMSNormHLFB(nn.Module):
  """RMSNorm layer using LiteRT High-Level Function Binding."""

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.hidden_size = hidden_size

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight,
        self.variance_epsilon,
        torch.ones(
            (self.hidden_size,),
            dtype=torch.float32,
            device=hidden_states.device,
        ),
    )


@patches_lib.register_patch(["qwen3_tts", "qwen3_tts_talker"])
@contextlib.contextmanager
def qwen3_tts_litert_patch():
  """Applies static LiteRT-friendly patches to Qwen3-TTS modules during export."""
  print("Qwen3-TTS LiteRT patch applied.")
  original_sdpa = None
  attn_funs = None
  try:
    attn_funs = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
    original_sdpa = attn_funs.get("sdpa", None)
    attn_funs["sdpa"] = asr_model._sdpa  # pylint: disable=protected-access
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  try:
    yield
  finally:
    if original_sdpa is not None and attn_funs is not None:
      attn_funs["sdpa"] = original_sdpa
    print("Qwen3-TTS LiteRT patch reverted.")
