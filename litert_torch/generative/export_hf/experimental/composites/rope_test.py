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
"""Tests for RoPE composite op."""

from litert_torch.generative.export_hf.experimental.composites import rope
from litert_torch.generative.layers import rotary_position_embedding as rotary_pos_emb
import torch

from absl.testing import absltest as googletest


class RopeTest(googletest.TestCase):

  def test_apply_mldrift_compatible_rope_numerical(self):
    batch_size = 2
    num_heads = 4
    seq_len = 8
    head_dim = 16
    base = 10000.0

    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    position = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    cos, sin = rotary_pos_emb.build_rope(
        position[0], n_elem=head_dim, base=int(base)
    )
    if cos is not None and cos.ndim == 3:
      cos = cos.unsqueeze(2)
      sin = sin.unsqueeze(2)

    expected = rotary_pos_emb.apply_rope(x, cos, sin)
    actual = rope.apply_mldrift_compatible_rope(
        x, position, base=base, head_dim=head_dim
    )

    self.assertEqual(actual.shape, expected.shape)
    self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
  googletest.main()
