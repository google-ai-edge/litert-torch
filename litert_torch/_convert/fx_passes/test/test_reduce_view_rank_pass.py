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
"""Tests for ReduceViewRankPass."""

from litert_torch import fx_infra
from litert_torch._convert import fx_passes
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from absl.testing import absltest as googletest


aten = torch.ops.aten


def _export_and_decompose(
    module: torch.nn.Module, export_args
) -> torch.export.ExportedProgram:
  exported_program = torch.export.export(module.eval(), export_args)
  return fx_infra.safe_run_decompositions(
      exported_program,
      fx_infra.decomp.pre_convert_decomp(),
  )


def _max_rank(exported_program: torch.export.ExportedProgram) -> int:
  max_rank = 0
  for node in exported_program.graph.nodes:
    val = node.meta.get("val")
    vals = val if isinstance(val, (list, tuple)) else [val]
    for v in vals:
      if v is not None and hasattr(v, "dim"):
        max_rank = max(max_rank, v.dim())
  return max_rank


class _SelfAttention(torch.nn.Module):

  def __init__(self, embed_dim=64, num_heads=4, batch_first=True):
    super().__init__()
    self.attn = torch.nn.MultiheadAttention(
        embed_dim, num_heads, batch_first=batch_first
    )

  def forward(self, x):
    y, _ = self.attn(x, x, x, need_weights=False)
    return y


class TestReduceViewRankPass(googletest.TestCase):
  """Tests for ReduceViewRankPass."""

  def test_mha_packed_qkv_drops_to_rank_4(self):
    module = _SelfAttention()
    args = (torch.randn(1, 16, 64),)

    # Stock nn.MultiheadAttention emits transient rank-5 view tensors.
    before = _export_and_decompose(module, args)
    self.assertGreater(_max_rank(before), 4)

    after = fx_infra.run_passes(
        _export_and_decompose(module, args),
        [fx_passes.ReduceViewRankPass()],
    )
    self.assertLessEqual(_max_rank(after), 4)

  def test_mha_is_numerically_unchanged(self):
    module = _SelfAttention()
    args = (torch.randn(1, 16, 64),)

    after = fx_infra.run_passes(
        _export_and_decompose(module, args),
        [fx_passes.ReduceViewRankPass()],
    )

    with torch.no_grad():
      expected = module(*args)
      actual = after.module()(*args)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

  def test_batch_first_false(self):
    module = _SelfAttention(embed_dim=32, num_heads=4, batch_first=False)
    args = (torch.randn(10, 1, 32),)

    after = fx_infra.run_passes(
        _export_and_decompose(module, args),
        [fx_passes.ReduceViewRankPass()],
    )
    self.assertLessEqual(_max_rank(after), 4)
    with torch.no_grad():
      expected = module(*args)
      actual = after.module()(*args)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

  def test_multi_unit_dim_squeeze(self):
    # A unit-dim unsqueeze whose inserted dim is removed alongside another
    # unit dim by the squeeze: the pass must fold to permute + squeeze and
    # stay within rank 4.
    def f(x):  # x: (2, 1, 3, 4)
      y = torch.unsqueeze(x, 0)  # (1, 2, 1, 3, 4)
      y = torch.permute(y, [0, 2, 1, 3, 4])  # (1, 1, 2, 3, 4)
      return torch.squeeze(y, [0, 1])  # (2, 3, 4)

    x = torch.randn(2, 1, 3, 4)
    graph_module = make_fx(f, tracing_mode="fake")(x)

    result = fx_passes.ReduceViewRankPass()(graph_module)
    self.assertTrue(result.modified)
    max_rank = 0
    for node in result.graph_module.graph.nodes:
      val = node.meta.get("val")
      if val is not None and hasattr(val, "dim"):
        max_rank = max(max_rank, val.dim())
    self.assertLessEqual(max_rank, 4)
    torch.testing.assert_close(result.graph_module(x), f(x))

  def test_permute_with_second_user_is_not_folded(self):
    # The permute output forks to a second consumer besides the squeeze, so
    # the rank-5 chain must stay alive for that branch; the pass backs off.
    def f(x):  # x: (2, 1, 3, 4)
      y = torch.unsqueeze(x, 0)  # (1, 2, 1, 3, 4)
      y = torch.permute(y, [1, 0, 2, 3, 4])  # (2, 1, 1, 3, 4)
      z = torch.squeeze(y, 1)  # (2, 1, 3, 4)
      return z, torch.sum(y)  # second user of the permute

    x = torch.randn(2, 1, 3, 4)
    graph_module = make_fx(f, tracing_mode="fake")(x)
    targets_before = [n.target for n in graph_module.graph.nodes]

    result = fx_passes.ReduceViewRankPass()(graph_module)

    self.assertFalse(result.modified)
    targets_after = [n.target for n in result.graph_module.graph.nodes]
    self.assertEqual(targets_before, targets_after)
    torch.testing.assert_close(result.graph_module(x), f(x))

  def test_provenance_meta_copied_to_new_nodes(self):
    # The folded permute/squeeze nodes must carry the debug provenance of the
    # squeeze node they replace.
    def f(x):  # x: (2, 1, 3, 4)
      y = torch.unsqueeze(x, 0)  # (1, 2, 1, 3, 4)
      y = torch.permute(y, [0, 2, 1, 3, 4])  # (1, 1, 2, 3, 4)
      return torch.squeeze(y, [0, 1])  # (2, 3, 4)

    x = torch.randn(2, 1, 3, 4)
    graph_module = make_fx(f, tracing_mode="fake")(x)

    provenance = {
        "stack_trace": "test_stack_trace",
        "nn_module_stack": {"attn": ("attn", torch.nn.MultiheadAttention)},
        "source_fn_stack": [("squeeze", torch.squeeze)],
        "from_node": ["squeeze_1"],
    }
    for node in graph_module.graph.nodes:
      if node.target == aten.squeeze.dims:
        node.meta.update(provenance)

    result = fx_passes.ReduceViewRankPass()(graph_module)
    self.assertTrue(result.modified)

    # After DCE the only permute / squeeze left are the ones the pass created.
    new_nodes = [
        n
        for n in result.graph_module.graph.nodes
        if n.target in (aten.permute.default, aten.squeeze.dims)
    ]
    self.assertLen(new_nodes, 2)
    for node in new_nodes:
      for key, value in provenance.items():
        self.assertEqual(node.meta.get(key), value)

  def test_no_op_when_already_rank_4(self):
    # A plain MLP has no rank-5 chain; the pass must leave it untouched.
    class Mlp(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 16)

      def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

    args = (torch.randn(2, 4, 16),)
    exported_program = _export_and_decompose(Mlp(), args)
    graph_module = exported_program.graph_module
    targets_before = [n.target for n in graph_module.graph.nodes]

    result = fx_passes.ReduceViewRankPass()(graph_module)
    self.assertFalse(result.modified)
    targets_after = [n.target for n in result.graph_module.graph.nodes]
    self.assertEqual(targets_before, targets_after)


if __name__ == "__main__":
  googletest.main()
