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
"""Static-shape export wrapper for HF `Qwen2_5OmniAudioEncoder`."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as hf_qwen25


class QwenOmniAudioEncoderWrapper(nn.Module):
  """Static-shape export wrapper around HF `Qwen2_5OmniAudioEncoder`.

  Matches C++ `QwenOmniAudioStage` tensor contract:
    Input:  `mel_features: float32[1, num_mel_bins, chunk_frames]`
    Output: `float32[1, num_audio_tokens, output_dim]`

  Why a wrapper is needed (cannot export `Qwen2_5OmniAudioEncoder` directly):
    1. I/O contract: C++ `QwenOmniAudioStage` supplies 3D
       `mel_features: [1, num_mel_bins, chunk_frames]` and expects 3D
       `[1, num_audio_tokens, output_dim]`, whereas HF takes 2D
       `[num_mel_bins, total_frames]` plus dynamic `feature_lens` and returns a
       `BaseModelOutputWithPooling` with 2D `last_hidden_state`.
    2. Untraceable length-dependent preprocessing: upstream
       `chunk_and_pad_features`, `get_valid_indices`, `get_pool_indices`, and
       `get_audio_cu_seqlens` call `.tolist()`, `.nonzero()`, and
       `pad_sequence`, which fail `torch.export`. Precomputing them for fixed
       `chunk_frames` and passing them via HF v5's precomputed-kwarg hooks lets
       `audio_tower(...)` run verbatim (convs, positional embeddings, all
       transformer layers, stride-2 pooling, `ln_post`, and `proj`).
    3. Untraceable `cu_seqlens.tolist()` in `Qwen2_5OmniAudioAttention.forward`:
       the non-flash path splits Q/K/V using `(cu_seqlens[1:] - cu_seqlens[:-1])
       .tolist()`. Passing `cu_seqlens` as a static NumPy array resolves that to
       Python ints at trace time so upstream attention layers export without any
       re-implementation.
  """

  def __init__(
      self,
      audio_tower: hf_qwen25.Qwen2_5OmniAudioEncoder,
      chunk_frames: int,
  ):
    super().__init__()
    audio_tower.set_attn_implementation("eager")
    self.audio_tower = audio_tower
    self.chunk_frames = chunk_frames

    max_chunk = int(audio_tower.n_window) * 2
    full_chunks, rem = divmod(chunk_frames, max_chunk)
    self._chunk_lens: list[int] = [max_chunk] * full_chunks + (
        [rem] if rem else []
    )
    self._max_chunk_len = max(self._chunk_lens)

    chunk_lengths = torch.tensor(self._chunk_lens, dtype=torch.long)
    feature_lens = torch.tensor([chunk_frames], dtype=torch.long)
    pool_indices = hf_qwen25.get_pool_indices(feature_lens)
    self.num_audio_tokens = int(pool_indices.numel())

    # Keep `cu_seqlens` as a NumPy array so upstream `Qwen2_5OmniAudioAttention`
    # resolves `(cu_seqlens[1:] - cu_seqlens[:-1]).tolist()` to static Python
    # ints during `torch.export` tracing.
    self._cu_seqlens: Any = hf_qwen25.get_audio_cu_seqlens(
        chunk_lengths
    ).numpy()
    self.register_buffer("_chunk_lengths", chunk_lengths, persistent=False)
    self.register_buffer(
        "_valid_indices",
        hf_qwen25.get_valid_indices(chunk_lengths),
        persistent=False,
    )
    self.register_buffer("_pool_indices", pool_indices, persistent=False)

  def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
    """Runs HF `audio_tower` with static chunking and precomputed index kwargs.

    Cannot call `audio_tower(mel_features, feature_lens)` directly because HF's
    default path uses untraceable `.tolist()`/`.nonzero()` and expects 2D I/O.

    Args:
      mel_features: Mel spectrogram tensor of shape
        `[1, num_mel_bins, chunk_frames]`.

    Returns:
      Audio feature embeddings of shape `[1, num_audio_tokens, output_dim]`.
    """
    if len(self._chunk_lens) == 1:
      padded = mel_features
    else:
      chunks = torch.split(mel_features.squeeze(0), self._chunk_lens, dim=1)
      padded = torch.stack(
          [
              F.pad(c, (0, self._max_chunk_len - c.shape[1]))
              if c.shape[1] < self._max_chunk_len
              else c
              for c in chunks
          ],
          dim=0,
      )
    out = self.audio_tower(
        padded_feature=padded,
        chunk_lengths=self._chunk_lengths,
        valid_indices=self._valid_indices,
        pool_indices=self._pool_indices,
        cu_seqlens=self._cu_seqlens,
    )
    return out.last_hidden_state.unsqueeze(0)
