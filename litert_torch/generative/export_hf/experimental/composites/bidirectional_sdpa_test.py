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
"""Tests for the bidirectional SDPA composite."""

from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.experimental.composites import bidirectional_sdpa
import numpy as np
import torch
import torch.nn.functional as F

from absl.testing import absltest as googletest


def _reference(query, key, value, scale):
  """torch SDPA reference with BTNH inputs/outputs."""
  g = query.shape[2] // key.shape[2]
  q = query.transpose(1, 2)
  k = key.transpose(1, 2).repeat_interleave(g, dim=1)
  v = value.transpose(1, 2).repeat_interleave(g, dim=1)
  out = F.scaled_dot_product_attention(q, k, v, scale=scale)
  return out.transpose(1, 2)


class _Wrapper(torch.nn.Module):

  def __init__(self, scale: float):
    super().__init__()
    self.scale = scale

  def forward(self, query, key, value):
    return bidirectional_sdpa.bidirectional_sdpa(
        query, key, value, scale=self.scale
    )


class BidirectionalSdpaTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="mha", num_heads=4, num_kv_heads=4, scale=1.0),
      dict(testcase_name="gqa", num_heads=4, num_kv_heads=2, scale=1.0),
      dict(testcase_name="mqa", num_heads=4, num_kv_heads=1, scale=1.0),
      dict(testcase_name="scaled", num_heads=4, num_kv_heads=2, scale=0.25),
  )
  def test_matches_reference(self, num_heads, num_kv_heads, scale):
    torch.manual_seed(0)
    query = torch.randn(1, 12, num_heads, 16)
    key = torch.randn(1, 12, num_kv_heads, 16)
    value = torch.randn(1, 12, num_kv_heads, 16)
    actual = bidirectional_sdpa.bidirectional_sdpa(
        query, key, value, scale=scale
    )
    torch.testing.assert_close(
        actual, _reference(query, key, value, scale), rtol=1e-5, atol=1e-5
    )

  def test_emits_unmasked_composite(self):
    inputs = (
        torch.randn(1, 12, 4, 16),
        torch.randn(1, 12, 2, 16),
        torch.randn(1, 12, 2, 16),
    )
    exported = torch.export.export(_Wrapper(scale=0.5), inputs)
    marks = [
        node
        for node in exported.graph.nodes
        if node.op == "call_function"
        and "mark_tensor" in str(node.target)
        and bidirectional_sdpa.COMPOSITE_NAME in (
            str(node.args) + str(node.kwargs)
        )
    ]
    # 3 inputs (Q, K, V; no mask) + 1 output.
    self.assertLen(marks, 4)
    self.assertIn("('scale', 0.5)", str([n.args for n in marks]))

  def test_tflite_conversion(self):
    torch.manual_seed(0)
    inputs = (
        torch.randn(1, 12, 4, 16),
        torch.randn(1, 12, 2, 16),
        torch.randn(1, 12, 2, 16),
    )
    module = _Wrapper(scale=1.0).eval()
    edge_model = litert_torch.convert(module, inputs)
    self.assertIn(
        bidirectional_sdpa.COMPOSITE_NAME.encode(), edge_model.model_content()
    )
    actual = np.asarray(edge_model(*inputs))
    np.testing.assert_allclose(
        actual, module(*inputs).numpy(), rtol=1e-5, atol=1e-5
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="not_4d",
          shapes=((12, 4, 16), (1, 12, 4, 16), (1, 12, 4, 16)),
      ),
      dict(
          testcase_name="kv_mismatch",
          shapes=((1, 12, 4, 16), (1, 12, 2, 16), (1, 12, 4, 16)),
      ),
      dict(
          testcase_name="head_dim_mismatch",
          shapes=((1, 12, 4, 8), (1, 12, 2, 16), (1, 12, 2, 16)),
      ),
      dict(
          testcase_name="heads_not_divisible",
          shapes=((1, 12, 3, 16), (1, 12, 2, 16), (1, 12, 2, 16)),
      ),
  )
  def test_rejects_invalid_shapes(self, shapes):
    query, key, value = (torch.randn(*s) for s in shapes)
    with self.assertRaises(ValueError):
      bidirectional_sdpa.bidirectional_sdpa(query, key, value, scale=1.0)


if __name__ == "__main__":
  googletest.main()
