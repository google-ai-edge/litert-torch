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
"""Tests for QKV NormRoPE composite op."""

import typing
from typing import Any

from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.experimental.composites import qkv_norm_rope
import torch

from absl.testing import absltest as googletest


class QkvNormRopeModule(torch.nn.Module):

  def __init__(self, head_dim: int, num_heads: int, num_kv_heads: int):
    super().__init__()
    self.head_dim = head_dim
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.q_weight = torch.nn.Parameter(torch.ones(head_dim))
    self.k_weight = torch.nn.Parameter(torch.ones(head_dim))

  def forward(self, qkv, position):
    return qkv_norm_rope.apply_qkv_norm_rope(
        qkv,
        position,
        self.q_weight,
        self.k_weight,
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        base=1000000.0,
        eps=1e-6,
    )


class QkvNormRopeTest(parameterized.TestCase):

  def test_apply_qkv_norm_rope(self):
    batch_size = 2
    seq_len = 4
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(batch_size, seq_len, total_dim)
    position = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    module = QkvNormRopeModule(head_dim, num_heads, num_kv_heads).eval()
    q_out, k_out, v_out = module(qkv, position)

    self.assertEqual(q_out.shape, (batch_size, num_heads, seq_len, head_dim))
    self.assertEqual(k_out.shape, (batch_size, num_kv_heads, seq_len, head_dim))
    self.assertEqual(v_out.shape, (batch_size, num_kv_heads, seq_len, head_dim))

  def test_convert_qkv_norm_rope(self):
    batch_size = 1
    seq_len = 4
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(batch_size, seq_len, total_dim)
    position = torch.arange(seq_len)

    module = QkvNormRopeModule(head_dim, num_heads, num_kv_heads).eval()
    edge_model = litert_torch.convert(module, (qkv, position))
    self.assertIsNotNone(edge_model)

  def test_trailing_channels_are_ignored(self):
    batch_size = 1
    seq_len = 4
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    extra_channels = num_heads * head_dim  # e.g. a fused output gate

    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(batch_size, seq_len, qkv_dim)
    wide = torch.cat(
        [qkv, torch.randn(batch_size, seq_len, extra_channels)], -1
    )
    position = torch.arange(seq_len).unsqueeze(0)

    module = QkvNormRopeModule(head_dim, num_heads, num_kv_heads).eval()
    for out, wide_out in zip(module(qkv, position), module(wide, position)):
      torch.testing.assert_close(wide_out, out, rtol=0, atol=0)

    edge_model = litert_torch.convert(module, (wide, position))
    self.assertIn(b"odml.qkv_norm_rope", edge_model.model_content())
    tfl_outs = typing.cast(Any, edge_model(wide, position.to(torch.int32)))
    for i, out in enumerate(module(wide, position)):
      torch.testing.assert_close(
          torch.as_tensor(tfl_outs[i]), out, rtol=1e-5, atol=1e-5
      )

  def test_apply_qkv_norm_rope_with_v_norm_and_proportion(self):
    batch_size = 1
    seq_len = 4
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.randn(batch_size, seq_len, total_dim)
    position = torch.arange(seq_len).unsqueeze(0)
    q_weight = torch.ones(head_dim)
    k_weight = torch.ones(head_dim)

    q_out, k_out, v_out = qkv_norm_rope.apply_qkv_norm_rope(
        qkv,
        position,
        q_weight,
        k_weight,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        base=1000000.0,
        eps=1e-6,
        proportion=0.25,
        has_v_norm=True,
    )
    self.assertEqual(q_out.shape, (batch_size, num_heads, seq_len, head_dim))
    self.assertEqual(k_out.shape, (batch_size, num_kv_heads, seq_len, head_dim))
    self.assertEqual(v_out.shape, (batch_size, num_kv_heads, seq_len, head_dim))

  def test_apply_q_only_norm_rope(self):
    batch_size = 1
    seq_len = 4
    num_heads = 4
    head_dim = 16

    q = torch.randn(batch_size, seq_len, num_heads * head_dim)
    position = torch.arange(seq_len).unsqueeze(0)
    q_weight = torch.ones(head_dim)

    q_out = qkv_norm_rope.apply_q_norm_rope(
        q,
        position,
        q_weight,
        num_heads=num_heads,
        head_dim=head_dim,
        base=10000.0,
        eps=1e-6,
    )
    self.assertEqual(q_out.shape, (batch_size, num_heads, seq_len, head_dim))


if __name__ == "__main__":
  googletest.main()
