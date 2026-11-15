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
"""Tests for Qwen3.5 Gated Delta Rule."""

import importlib.metadata

orig_version = importlib.metadata.version


def patched_version(distribution_name: str) -> str:
  if distribution_name == "torchao":
    return "0.4.0"
  return orig_version(distribution_name)


importlib.metadata.version = patched_version

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from absl.testing import parameterized  # pylint: disable=g-import-not-at-top
from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-import-not-at-top
import litert_torch  # pylint: disable=g-import-not-at-top
from litert_torch.generative.export_hf.model_ext.qwen3_5 import gated_delta_rule  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top

_NUM_K_HEADS = 2
_HEAD_K_DIM = 8
_HEAD_V_DIM = 8


def _make_inputs(g_ratio: int, seq_len: int):
  num_v_heads = _NUM_K_HEADS * g_ratio
  query = torch.randn(1, seq_len, _NUM_K_HEADS, _HEAD_K_DIM)
  key = torch.randn(1, seq_len, _NUM_K_HEADS, _HEAD_K_DIM)
  value = torch.randn(1, seq_len, num_v_heads, _HEAD_V_DIM)
  beta = torch.rand(1, seq_len, num_v_heads)
  g = -torch.rand(1, seq_len, num_v_heads)
  state = torch.randn(1, num_v_heads, _HEAD_K_DIM, _HEAD_V_DIM)
  return query, key, value, beta, g, state


class GatedDeltaRuleTest(parameterized.TestCase):

  @parameterized.product(g_ratio=(1, 2, 3), seq_len=(1, 4))
  def test_gqa_head_repeat_matches_repeat_interleave(
      self, g_ratio: int, seq_len: int
  ):
    torch.manual_seed(0)
    query, key, value, beta, g, state = _make_inputs(g_ratio, seq_len)

    actual = gated_delta_rule.apply_gated_delta_rule(
        query, key, value, beta, g, state, use_fused_gdn=False
    )

    q_t, k_t, v_t, beta_t, g_t = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (
            query.repeat_interleave(g_ratio, dim=2),
            key.repeat_interleave(g_ratio, dim=2),
            value,
            beta,
            g,
        )
    ]
    expected = gated_delta_rule.recurrent_gated_delta_rule(
        q_t, k_t, v_t, beta_t, g_t, state
    )
    for a, e in zip(actual, expected):
      torch.testing.assert_close(a, e, rtol=0, atol=0)

  def test_gqa_head_repeat_lowers_without_broadcast_to(self):

    class DecomposedGatedDeltaRule(torch.nn.Module):

      def forward(self, query, key, value, beta, g, state):
        return gated_delta_rule.apply_gated_delta_rule(
            query, key, value, beta, g, state, use_fused_gdn=False
        )

    torch.manual_seed(0)
    args = _make_inputs(g_ratio=2, seq_len=1)
    edge_model = litert_torch.convert(DecomposedGatedDeltaRule().eval(), args)

    interpreter = tfl_interpreter.Interpreter(
        model_content=edge_model.model_content()
    )
    # pylint: disable=protected-access
    op_names = [op["op_name"] for op in interpreter._get_ops_details()]
    self.assertIn("CONCATENATION", op_names)
    self.assertNotIn("BROADCAST_TO", op_names)


if __name__ == "__main__":
  absltest.main()
