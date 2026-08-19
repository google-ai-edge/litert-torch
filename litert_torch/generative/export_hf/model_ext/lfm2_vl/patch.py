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
from litert_torch.generative.export_hf.model_ext.lfm2 import patch as lfm2_patch_lib
from litert_torch.generative.export_hf.model_ext.lfm2 import short_conv as short_conv_lib
import torch
from transformers.models.lfm2 import modeling_lfm2
from transformers.models.siglip2 import modeling_siglip2

TARGET_PATCH_SIZE = (32, 32)


class PatchedSiglip2VisionEmbeddings(modeling_siglip2.Siglip2VisionEmbeddings):

  @staticmethod
  def resize_positional_embeddings(
      positional_embeddings: torch.Tensor,
      spatial_shapes: torch.LongTensor,
      max_length: int,
  ) -> torch.Tensor:
    """Resize positional embeddings to image-specific size and pad to a fixed size.

    Args:
        positional_embeddings (`torch.Tensor`): Position embeddings of shape
          (height, width, embed_dim)
        spatial_shapes (`torch.LongTensor`): Spatial shapes of shape
          (batch_size, 2) to resize the positional embeddings to
        max_length (`int`): Maximum length of the positional embeddings to pad
          resized positional embeddings to

    Returns:
        `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
    """
    batch_size = spatial_shapes.shape[0]
    embed_dim = positional_embeddings.shape[-1]
    source_dtype = positional_embeddings.dtype

    resulted_positional_embeddings = torch.empty(
        (batch_size, max_length, embed_dim),
        device=positional_embeddings.device,
        dtype=source_dtype,
    )

    # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
    positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

    # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
    if positional_embeddings.device.type == "cpu":
      positional_embeddings = positional_embeddings.to(torch.float32)

    for i in range(batch_size):
      # (1, dim, height, width) -> (1, dim, target_height, target_width)
      height, width = TARGET_PATCH_SIZE
      resized_embeddings = torch.nn.functional.interpolate(
          positional_embeddings,
          size=(height, width),
          mode="bilinear",
          align_corners=False,
          # antialias=True,
      )

      # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
      resized_embeddings = resized_embeddings.reshape(
          embed_dim, height * width
      ).transpose(0, 1)

      # Cast to original dtype
      resized_embeddings = resized_embeddings.to(source_dtype)

      resulted_positional_embeddings[i, : height * width] = resized_embeddings
      resulted_positional_embeddings[i, height * width :] = resized_embeddings[
          0
      ]

    return resulted_positional_embeddings


@patches_lib.register_patch(["lfm2_vl"])
@contextlib.contextmanager
def lfm2_vl_litert_patch():
  print("LFM2 VL patch applied.")
  original_short_conv = modeling_lfm2.Lfm2ShortConv
  modeling_lfm2.Lfm2ShortConv = short_conv_lib.Lfm2ShortConv

  original_decoder_layer = modeling_lfm2.Lfm2DecoderLayer
  modeling_lfm2.Lfm2DecoderLayer = lfm2_patch_lib.PatchedLfm2DecoderLayer

  original_siglip2_vision_embeddings = modeling_siglip2.Siglip2VisionEmbeddings
  modeling_siglip2.Siglip2VisionEmbeddings = PatchedSiglip2VisionEmbeddings

  try:
    yield
  finally:
    modeling_lfm2.Lfm2ShortConv = original_short_conv
    modeling_lfm2.Lfm2DecoderLayer = original_decoder_layer
    modeling_siglip2.Siglip2VisionEmbeddings = (
        original_siglip2_vision_embeddings
    )
