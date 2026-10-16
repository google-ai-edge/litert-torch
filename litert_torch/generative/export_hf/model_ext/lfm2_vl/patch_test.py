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
"""Tests for the LFM2-VL export patch."""

import litert_torch
import litert_torch.generative.export_hf  # pylint: disable=unused-import
from litert_torch.generative.export_hf.model_ext.lfm2_vl import patch
import torch
from transformers.models.siglip2 import configuration_siglip2
from transformers.models.siglip2 import modeling_siglip2

from absl.testing import absltest as googletest


def _get_dummy_siglip2_config():
  return configuration_siglip2.Siglip2VisionConfig(
      hidden_size=64,
      patch_size=16,
      num_channels=3,
      num_patches=256,
      num_hidden_layers=1,
      num_attention_heads=4,
      intermediate_size=128,
  )


class _EmbeddingsWrapper(torch.nn.Module):
  """Fixes spatial_shapes so the embeddings can be exported with one input."""

  def __init__(self, embeddings):
    super().__init__()
    self.embeddings = embeddings

  def forward(self, pixel_values):
    spatial_shapes = torch.tensor(
        [list(patch.TARGET_PATCH_SIZE)], dtype=torch.int64
    )
    return self.embeddings(pixel_values, spatial_shapes)


def _resize_ops(exported_program):
  return sorted(
      str(node.target)
      for node in exported_program.graph.nodes
      if node.op == "call_function" and "upsample" in str(node.target)
  )


class PatchTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    self.config = _get_dummy_siglip2_config()
    height, width = patch.TARGET_PATCH_SIZE
    self.num_tokens = height * width
    self.patch_dim = self.config.num_channels * self.config.patch_size**2
    self.spatial_shapes = torch.tensor([[height, width]], dtype=torch.int64)

  def test_fold_positional_embeddings_matches_original(self):
    with patch.lfm2_vl_litert_patch():
      embeddings = modeling_siglip2.Siglip2VisionEmbeddings(self.config).eval()
    self.assertIsInstance(embeddings, patch.PatchedSiglip2VisionEmbeddings)
    pixel_values = torch.randn(1, self.num_tokens, self.patch_dim)

    with torch.no_grad():
      expected = embeddings(pixel_values, self.spatial_shapes)
      folded = patch.fold_positional_embeddings(embeddings)
      actual = embeddings(pixel_values, self.spatial_shapes)

    self.assertEqual(folded, 1)
    self.assertEqual(
        tuple(
            getattr(embeddings, patch.FOLDED_POSITIONAL_EMBEDDINGS_BUFFER).shape
        ),
        (1, self.num_tokens, self.config.hidden_size),
    )
    self.assertTrue(torch.equal(expected, actual))

  def test_fold_positional_embeddings_padded_length_matches_original(self):
    with patch.lfm2_vl_litert_patch():
      embeddings = modeling_siglip2.Siglip2VisionEmbeddings(self.config).eval()
    max_length = self.num_tokens + 76
    pixel_values = torch.randn(1, max_length, self.patch_dim)

    with torch.no_grad():
      expected = embeddings(pixel_values, self.spatial_shapes)
      patch.fold_positional_embeddings(embeddings, max_length=max_length)
      actual = embeddings(pixel_values, self.spatial_shapes)

    self.assertTrue(torch.equal(expected, actual))

  def test_fold_positional_embeddings_falls_back_on_other_length(self):
    with patch.lfm2_vl_litert_patch():
      embeddings = modeling_siglip2.Siglip2VisionEmbeddings(self.config).eval()
      reference = modeling_siglip2.Siglip2VisionEmbeddings(self.config).eval()
    reference.load_state_dict(embeddings.state_dict())
    patch.fold_positional_embeddings(embeddings)
    # A length the folded table does not cover uses the original computation.
    pixel_values = torch.randn(1, self.num_tokens + 1, self.patch_dim)

    with torch.no_grad():
      expected = reference(pixel_values, self.spatial_shapes)
      actual = embeddings(pixel_values, self.spatial_shapes)

    self.assertTrue(torch.equal(expected, actual))

  def test_folded_export_has_no_resize_op(self):
    pixel_values = torch.randn(1, self.num_tokens, self.patch_dim)
    with patch.lfm2_vl_litert_patch():
      embeddings = modeling_siglip2.Siglip2VisionEmbeddings(self.config).eval()
    unfolded = torch.export.export(
        _EmbeddingsWrapper(embeddings).eval(), (pixel_values,), strict=False
    )
    self.assertNotEmpty(_resize_ops(unfolded))

    patch.fold_positional_embeddings(embeddings)
    folded = torch.export.export(
        _EmbeddingsWrapper(embeddings).eval(), (pixel_values,), strict=False
    )
    self.assertEmpty(_resize_ops(folded))


if __name__ == "__main__":
  googletest.main()
