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
"""Static-shape export wrapper for HF `Qwen2_5OmniVisionEncoder`."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import vision_utils
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as hf_qwen25


def _as_int(val: Any) -> int:
  """Extracts an int from a scalar or `(h, w)` config tuple/list."""
  return int(val[0] if isinstance(val, (list, tuple)) else val)


class QwenOmniVisionEncoderWrapper(nn.Module):
  """Static-shape export wrapper around HF `Qwen2_5OmniVisionEncoder`.

  Matches C++ `QwenOmniImageStage` tensor contract:
    Input:  `pixel_values: float32[1, 3, image_size, image_size]`
    Output: `float32[1, num_merged_tokens, out_hidden_size]`

  Why a wrapper is needed (cannot export `Qwen2_5OmniVisionEncoder` directly):
    1. I/O contract: C++ `QwenOmniImageStage` supplies a square RGB image tensor
       `pixel_values: [1, 3, image_size, image_size]` and expects
       `[1, num_merged_tokens, out_hidden_size]`, whereas HF expects
       pre-patchified 2D `[num_patches, C*T*P*P]` plus dynamic `grid_thw` and
       returns a `BaseModelOutputWithPooling` with 2D `pooler_output`.
    2. `dynamic_wi8_afp32` quantization compatibility: upstream
       `visual.patch_embed.proj` is a stride-`(T, P, P)` `nn.Conv3d`, which
       fails TFLite CPU hybrid quantization (`CONV_3D` rejects float32
       activations with int8 weights). Running the identical projection via
       `F.linear` lowers to `FULLY_CONNECTED`, quantizes cleanly, and shrinks
       the model ~4x.
    3. Untraceable `grid_thw` metadata, `cu_seqlens.tolist()`, and `argsort`:
       upstream `get_vision_*` helpers call `.tolist()`/`repeat_interleave`,
       `visual.forward` computes `torch.argsort(window_index)` at runtime, and
       `Qwen2_5OmniVisionAttention.forward` calls `(cu_seqlens[1:] -
       cu_seqlens[:-1]).tolist()` for `torch.split`. Precomputing
       `window_index`, `reverse_indices`, and `rotary_pos_emb` as static
       buffers and passing `cu_seqlens`/`cu_window_seqlens` as static NumPy
       arrays lets every upstream `Qwen2_5OmniVisionBlock` and
       `Qwen2_5OmniPatchMerger` run verbatim without re-implementing attention,
       RoPE, norms, or MLP.
  """

  def __init__(
      self,
      visual: hf_qwen25.Qwen2_5OmniVisionEncoder,
      image_size: int,
  ):
    super().__init__()
    visual.set_attn_implementation("eager")
    self.visual = visual
    self.image_size = image_size
    self.patch_size = _as_int(visual.patch_size)
    self.temporal_patch_size = _as_int(visual.patch_embed.temporal_patch_size)
    self.spatial_merge_size = int(visual.spatial_merge_size)
    self.spatial_merge_unit = int(visual.spatial_merge_unit)
    self.fullatt_block_indexes = set(visual.fullatt_block_indexes)

    grid_h = image_size // self.patch_size
    grid_w = image_size // self.patch_size
    self.num_merged_tokens = (grid_h // self.spatial_merge_size) * (
        grid_w // self.spatial_merge_size
    )
    grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)

    position_ids = vision_utils.get_vision_position_ids(
        grid_thw, self.spatial_merge_size
    )
    cu_seqlens, _ = vision_utils.get_vision_attention_seqlens(
        grid_thw, visual.config
    )
    window_index, cu_window_seqlens = vision_utils.get_vision_window_index(
        grid_thw,
        spatial_merge_size=self.spatial_merge_size,
        window_size=visual.window_size,
        patch_size=self.patch_size,
    )
    # Keep `cu_*seqlens` as NumPy arrays so `Qwen2_5OmniVisionAttention`
    # resolves `(cu_seqlens[1:] - cu_seqlens[:-1]).tolist()` to static Python
    # ints during `torch.export` tracing.
    self._cu_seqlens: Any = cu_seqlens.numpy()
    self._cu_window_seqlens: Any = cu_window_seqlens.numpy()
    self.register_buffer("_window_index", window_index, persistent=False)
    self.register_buffer(
        "_reverse_indices", torch.argsort(window_index), persistent=False
    )
    with torch.no_grad():
      pos_dev = visual.rotary_pos_emb.inv_freq.device
      rotary_pos_emb = (
          visual.rotary_pos_emb(position_ids.to(pos_dev))
          .reshape(self.num_merged_tokens, self.spatial_merge_unit, -1)[
              window_index.to(pos_dev), :, :
          ]
          .reshape(grid_h * grid_w, -1)
      )
    self.register_buffer("_rotary_pos_emb", rotary_pos_emb, persistent=False)

  def compute_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
    """Converts `[B, 3, H, W]` image tensor to Qwen2.5 merged patch order."""
    bsz, channels, height, width = pixel_values.shape
    grid_h = height // self.patch_size
    grid_w = width // self.patch_size
    merge = self.spatial_merge_size
    frames = pixel_values.unsqueeze(1).expand(
        bsz, self.temporal_patch_size, channels, height, width
    )
    patches = frames.reshape(
        bsz,
        self.temporal_patch_size,
        channels,
        grid_h // merge,
        merge,
        self.patch_size,
        grid_w // merge,
        merge,
        self.patch_size,
    )
    return patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(
        bsz * grid_h * grid_w,
        channels * self.temporal_patch_size * self.patch_size * self.patch_size,
    )

  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    """Patchifies `pixel_values` and runs HF vision blocks + patch merger.

    Cannot call `self.visual(...)` directly because `patch_embed.proj` uses
    `Conv3d` (breaks TFLite CPU `dynamic_wi8_afp32` quantization) and
    `visual.forward` runs `torch.argsort(window_index)` at runtime.

    Args:
      pixel_values: Square RGB image tensor of shape
        `[1, 3, image_size, image_size]`.

    Returns:
      Merged visual token embeddings of shape
      `[1, num_merged_tokens, out_hidden_size]`.
    """
    patches = self.compute_patches(pixel_values)
    # `patch_embed.proj` is a stride-(T, P, P) Conv3d over flattened patch
    # vectors, mathematically identical to a Linear projection. Using
    # `F.linear` lowers to FULLY_CONNECTED, avoiding TFLite CPU CONV_3D
    # hybrid-quantization failures under `dynamic_wi8_afp32`.
    proj = self.visual.patch_embed.proj
    hidden = F.linear(
        patches.to(dtype=proj.weight.dtype),
        proj.weight.flatten(1),
        proj.bias,
    )
    seq_len = hidden.shape[0]
    hidden = hidden.reshape(
        self.num_merged_tokens, self.spatial_merge_unit, -1
    )[self._window_index, :, :].reshape(seq_len, -1)

    for layer_idx, blk in enumerate(self.visual.blocks):
      cu_seqlens = (
          self._cu_seqlens
          if layer_idx in self.fullatt_block_indexes
          else self._cu_window_seqlens
      )
      hidden = blk(
          hidden,
          cu_seqlens=cu_seqlens,
          position_embeddings=self._rotary_pos_emb,
      )

    return self.visual.merger(hidden)[self._reverse_indices, :].unsqueeze(0)
