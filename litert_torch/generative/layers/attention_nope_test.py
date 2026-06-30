# Copyright 2024 The LiteRT Torch Authors.
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

"""Tests for the NoPE (AttentionConfig.enable_rope=False) path."""

from litert_torch.generative.layers import attention
from litert_torch.generative.layers import model_config as cfg
from litert_torch.generative.layers import rotary_position_embedding as rope_lib
import torch

from absl.testing import absltest


class NoPETest(absltest.TestCase):

  def _build_layer(self):
    config = cfg.AttentionConfig(
        num_heads=4,
        head_dim=8,
        num_query_groups=4,
        rotary_base=10000,
        rotary_percentage=1.0,
    )
    layer = attention.CausalSelfAttention(
        dim=config.num_heads * config.head_dim, config=config, enable_hlfb=False
    )
    return config, layer

  def _forward(self, layer, x, rope, input_pos, mask):
    out = layer(x, rope=rope, mask=mask, input_pos=input_pos)
    return out[0] if isinstance(out, tuple) else out

  def test_enable_rope_false_matches_no_rope(self):
    torch.manual_seed(0)
    config, layer = self._build_layer()
    b, t = 1, 4
    x = torch.randn(b, t, config.num_heads * config.head_dim)
    input_pos = torch.arange(t)
    n_elem = int(config.rotary_percentage * config.head_dim)
    rope = rope_lib.build_rope(input_pos, n_elem, config.rotary_base)
    mask = torch.zeros(b, 1, t, t)

    # Reference: no RoPE supplied at all.
    baseline = self._forward(layer, x, None, input_pos, mask)

    # enable_rope=False must be identical to supplying no RoPE.
    config.enable_rope = False
    nope_out = self._forward(layer, x, rope, input_pos, mask)
    torch.testing.assert_close(nope_out, baseline)

    # Sanity: with RoPE enabled the output must differ.
    config.enable_rope = True
    rope_out = self._forward(layer, x, rope, input_pos, mask)
    self.assertFalse(torch.allclose(rope_out, baseline))


if __name__ == '__main__':
  absltest.main()
