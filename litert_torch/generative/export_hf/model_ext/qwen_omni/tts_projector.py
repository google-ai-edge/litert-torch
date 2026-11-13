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
"""TTS conditioning projector wrapper for Qwen2.5-Omni."""

from __future__ import annotations

import torch
from torch import nn


class QwenOmniTtsProjectorWrapper(nn.Module):
  """Projects Thinker `(hidden_state + token_embed)` into Talker hidden space.

  Upstream HuggingFace has no standalone projector module:
  `Qwen2_5OmniForConditionalGeneration.generate()` imperatively sums Thinker
  hidden states with Thinker text-token embeddings (zeroing text embeddings at
  multimodal positions and hidden states at special BOS/EOS/PAD tokens) before
  `Qwen2_5OmniTalkerForConditionalGeneration.forward()` applies
  `thinker_to_talker_proj`. The LiteRT C++ runtime (`QwenOmniTtsStage`)
  executes this step as a dedicated `tts_projector` submodel.

  Reuses upstream `thinker.get_input_embeddings()` and
  `talker.thinker_to_talker_proj` directly with zero parameter copies.
  Alphabetical `a_`/`b_` input prefixes fix TFLite signature buffer order for
  C++ `QwenOmniTtsStage`.
  """

  def __init__(
      self,
      thinker_embed_tokens: nn.Embedding,
      thinker_to_talker_proj: nn.Linear,
  ):
    super().__init__()
    self.thinker_embed_tokens = thinker_embed_tokens
    self.thinker_to_talker_proj = thinker_to_talker_proj

  def forward(
      self, a_llm_hidden: torch.Tensor, b_token_id: torch.Tensor
  ) -> torch.Tensor:
    """Projects Thinker conditioning into Talker hidden space.

    Custom `forward()` is needed because TFLite has no optional tensor inputs
    and `nn.Embedding` rejects negative indices: `b_token_id < 0` clamps the
    lookup index and zeros the text embedding for multimodal prompt positions.
    Applying `thinker_to_talker_proj` (with bias) once here lets `tts_decoder`
    add weight-only codec projections without double-adding `proj.bias`.

    Args:
      a_llm_hidden: Thinker hidden states, `float32[1, S, thinker_hidden_size]`.
      b_token_id: Thinker token IDs, `int32[1, S]` (`< 0` omits text embedding).

    Returns:
      Projected conditioning tensor, `float32[1, S, talker_hidden_size]`.
    """
    safe_id = torch.clamp(b_token_id, min=0).to(torch.long)
    has_token = (b_token_id >= 0).unsqueeze(-1).to(a_llm_hidden.dtype)
    tok_emb = self.thinker_embed_tokens(safe_id)
    return self.thinker_to_talker_proj(a_llm_hidden + tok_emb * has_token)
