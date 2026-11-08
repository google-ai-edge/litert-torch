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
# Buffer name of the precomputed positional-embedding table, see
# `fold_positional_embeddings`.
FOLDED_POSITIONAL_EMBEDDINGS_BUFFER = "litert_folded_positional_embeddings"


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

  def forward(
      self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor
  ) -> torch.Tensor:
    """Adds the folded positional embeddings when they were precomputed.

    `fold_positional_embeddings` stores the resized table as a buffer after
    the weights are loaded, so the exported graph carries a constant instead
    of a RESIZE_BILINEAR whose only input is a constant. GPU delegates reject
    that op (it expects one runtime input), which made the exported vision
    encoder fail to compile on GPU backends. Without the buffer this falls
    back to the original computation.
    """
    folded = getattr(self, FOLDED_POSITIONAL_EMBEDDINGS_BUFFER, None)
    if folded is None or folded.shape[1] != pixel_values.shape[1]:
      return super().forward(pixel_values, spatial_shapes)
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
    return patch_embeds + folded.to(dtype=patch_embeds.dtype)


def fold_positional_embeddings(
    module: torch.nn.Module, max_length: int | None = None
) -> int:
  """Precomputes the resized positional embeddings of every SigLIP2 embedding.

  The resize depends only on the loaded position embedding weight and the
  fixed TARGET_PATCH_SIZE, so it is computed once here (same function, same
  numerics) and registered as a non-persistent buffer that the patched
  forward adds directly. Call it after the weights are loaded and before
  export.

  Args:
    module: the model (or any parent module) holding the vision embeddings.
    max_length: rows of the table (padding follows the original function);
      defaults to TARGET_PATCH_SIZE height * width.

  Returns:
    The number of embedding modules folded.
  """
  height, width = TARGET_PATCH_SIZE
  max_length = max_length or height * width
  spatial_shapes = torch.tensor([[height, width]], dtype=torch.int64)
  folded = 0
  for child in module.modules():
    if not isinstance(child, modeling_siglip2.Siglip2VisionEmbeddings):
      continue
    with torch.no_grad():
      positional_embeddings = child.position_embedding.weight.reshape(
          child.position_embedding_size, child.position_embedding_size, -1
      )
      table = PatchedSiglip2VisionEmbeddings.resize_positional_embeddings(
          positional_embeddings, spatial_shapes, max_length=max_length
      )
    child.register_buffer(
        FOLDED_POSITIONAL_EMBEDDINGS_BUFFER,
        table.detach().clone(),
        persistent=False,
    )
    folded += 1
  return folded


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
