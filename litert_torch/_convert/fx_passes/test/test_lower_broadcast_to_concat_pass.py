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
"""Tests for LowerBroadcastToConcatPass."""

from litert_torch import fx_infra
from litert_torch._convert import fx_passes
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from absl.testing import absltest as googletest

aten = torch.ops.aten

_BROADCASTING_COPIES = (
    aten.expand.default,
    aten.expand_copy.default,
    aten.repeat.default,
)


def _export_and_decompose(
    module: torch.nn.Module, export_args
) -> torch.export.ExportedProgram:
  exported_program = torch.export.export(module.eval(), export_args)
  return fx_infra.safe_run_decompositions(
      exported_program,
      fx_infra.decomp.pre_convert_decomp(),
  )


def _copying_expands(graph: torch.fx.Graph) -> list[torch.fx.Node]:
  """Returns expand/repeat nodes that actually copy data."""
  result = []
  for node in graph.nodes:
    if node.op != "call_function" or node.target not in _BROADCASTING_COPIES:
      continue
    if tuple(node.args[0].meta["val"].shape) != tuple(node.meta["val"].shape):
      result.append(node)
  return result


def _count(graph: torch.fx.Graph, target) -> int:
  return sum(
      1 for n in graph.nodes if n.op == "call_function" and n.target == target
  )


class _RepeatKV(torch.nn.Module):
  """transformers' `repeat_kv` for grouped-query attention."""

  def forward(self, q, k):
    b, kv, s, d = k.shape
    n_rep = q.shape[1] // kv
    k = k[:, :, None, :, :].expand(b, kv, n_rep, s, d)
    k = k.reshape(b, kv * n_rep, s, d)
    return q @ k.transpose(-1, -2)


class _RepeatInterleave(torch.nn.Module):

  def forward(self, q, k):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    return q @ k.transpose(-1, -2)


class _ClsToken(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.cls = torch.nn.Parameter(torch.randn(1, 1, 16))

  def forward(self, x):
    return torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)


class TestLowerBroadcastToConcatPass(googletest.TestCase):
  """Tests for LowerBroadcastToConcatPass."""

  def _assert_lowered(self, module, args):
    before = _export_and_decompose(module, args)
    self.assertNotEmpty(_copying_expands(before.graph))

    after = fx_infra.run_passes(
        _export_and_decompose(module, args),
        [fx_passes.LowerBroadcastToConcatPass()],
    )
    self.assertEmpty(_copying_expands(after.graph))
    self.assertGreater(_count(after.graph, aten.cat.default), 0)
    with torch.no_grad():
      expected = module(*args)
      actual = after.module()(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

  def test_repeat_kv(self):
    self._assert_lowered(
        _RepeatKV(), (torch.randn(1, 8, 16, 32), torch.randn(1, 2, 16, 32))
    )

  def test_repeat_interleave(self):
    self._assert_lowered(
        _RepeatInterleave(),
        (torch.randn(1, 8, 16, 32), torch.randn(1, 2, 16, 32)),
    )

  def test_cls_token_expand_before_cat(self):
    self._assert_lowered(_ClsToken(), (torch.randn(2, 10, 16),))

  def test_repeat_with_extra_leading_dims(self):
    def f(x):  # x: (2, 3)
      return x.repeat(2, 1, 3)  # (2, 2, 9)

    x = torch.randn(2, 3)
    graph_module = make_fx(f, tracing_mode="fake")(x)

    result = fx_passes.LowerBroadcastToConcatPass()(graph_module)
    self.assertTrue(result.modified)
    self.assertEqual(_count(result.graph_module.graph, aten.repeat.default), 0)
    torch.testing.assert_close(result.graph_module(x), f(x), atol=0, rtol=0)

  def test_expand_multiple_axes_and_new_leading_dim(self):
    def f(x):  # x: (3, 1)
      return x.expand(2, 3, 4).reshape(24)

    x = torch.randn(3, 1)
    graph_module = make_fx(f, tracing_mode="fake")(x)

    result = fx_passes.LowerBroadcastToConcatPass()(graph_module)
    self.assertTrue(result.modified)
    self.assertEmpty(_copying_expands(result.graph_module.graph))
    torch.testing.assert_close(result.graph_module(x), f(x), atol=0, rtol=0)

  def test_large_repeat_count_uses_doubling(self):
    def f(x):  # x: (1, 4)
      return x.expand(13, 4).reshape(52)

    x = torch.randn(1, 4)
    graph_module = make_fx(f, tracing_mode="fake")(x)

    result = fx_passes.LowerBroadcastToConcatPass()(graph_module)
    self.assertTrue(result.modified)
    cats = [
        n
        for n in result.graph_module.graph.nodes
        if n.op == "call_function" and n.target == aten.cat.default
    ]
    # 13 = 8 + 4 + 1: three doublings plus one final concatenation.
    self.assertLen(cats, 4)
    self.assertTrue(all(len(n.args[0]) <= 3 for n in cats))
    torch.testing.assert_close(result.graph_module(x), f(x), atol=0, rtol=0)

  def test_expand_into_elementwise_op_is_left_for_the_converter(self):
    def f(x, y):  # x: (1, 1, 8, 8), y: (1, 4, 8, 8)
      return x.expand(1, 4, 8, 8) + y

    args = (torch.randn(1, 1, 8, 8), torch.randn(1, 4, 8, 8))
    graph_module = make_fx(f, tracing_mode="fake")(*args)

    result = fx_passes.LowerBroadcastToConcatPass()(graph_module)
    self.assertFalse(result.modified)

  def test_rank_above_five_is_declined(self):
    def f(x):  # x: (1, 2, 1, 2, 1, 2)
      return x.expand(3, 2, 3, 2, 3, 2).reshape(-1)

    x = torch.randn(1, 2, 1, 2, 1, 2)
    graph_module = make_fx(f, tracing_mode="fake")(x)

    result = fx_passes.LowerBroadcastToConcatPass()(graph_module)
    self.assertFalse(result.modified)

  def test_provenance_meta_copied_to_new_nodes(self):
    def f(x):
      return x.expand(3, 4).reshape(12)

    x = torch.randn(1, 4)
    graph_module = make_fx(f, tracing_mode="fake")(x)
    for node in graph_module.graph.nodes:
      if node.target == aten.expand.default:
        node.meta["stack_trace"] = "model.py:1"

    result = fx_passes.LowerBroadcastToConcatPass()(graph_module)
    self.assertTrue(result.modified)
    cat = next(
        n
        for n in result.graph_module.graph.nodes
        if n.op == "call_function" and n.target == aten.cat.default
    )
    self.assertEqual(cat.meta.get("stack_trace"), "model.py:1")


if __name__ == "__main__":
  googletest.main()
