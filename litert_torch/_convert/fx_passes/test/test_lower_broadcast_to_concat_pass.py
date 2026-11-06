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

from absl.testing import absltest as googletest

aten = torch.ops.aten

_BROADCASTING_COPIES = (
    aten.expand.default,
    aten.expand_copy.default,
    aten.repeat.default,
)


class _Fn(torch.nn.Module):

  def __init__(self, fn):
    super().__init__()
    self._fn = fn

  def forward(self, *args):
    return self._fn(*args)


def _export(module_or_fn, args) -> torch.export.ExportedProgram:
  module = (
      module_or_fn
      if isinstance(module_or_fn, torch.nn.Module)
      else _Fn(module_or_fn)
  )
  return torch.export.export(module.eval(), args, strict=False)


def _export_and_decompose(module, args) -> torch.export.ExportedProgram:
  return fx_infra.safe_run_decompositions(
      _export(module, args), fx_infra.decomp.pre_convert_decomp()
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


def _cats(graph: torch.fx.Graph) -> list[torch.fx.Node]:
  return [
      n
      for n in graph.nodes
      if n.op == "call_function" and n.target == aten.cat.default
  ]


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
  """A parameter expanded along the batch axis before a concatenation."""

  def __init__(self):
    super().__init__()
    self.cls = torch.nn.Parameter(torch.randn(1, 1, 16))

  def forward(self, x):
    return torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)


class _ScaledClsToken(_ClsToken):
  """A parameter that reaches the expand through another constant op."""

  def forward(self, x):
    cls = self.cls * 2.0
    return torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1)


class TestLowerBroadcastToConcatPass(googletest.TestCase):
  """Tests for LowerBroadcastToConcatPass."""

  def _run(self, exported_program):
    return fx_passes.LowerBroadcastToConcatPass()(exported_program)

  def _assert_lowered(self, module_or_fn, args, decompose=True):
    export = _export_and_decompose if decompose else _export
    before = export(module_or_fn, args)
    self.assertNotEmpty(_copying_expands(before.graph))

    result = self._run(export(module_or_fn, args))
    self.assertTrue(result.modified)
    after = result.exported_program
    self.assertEmpty(_copying_expands(after.graph))
    self.assertGreater(_count(after.graph, aten.cat.default), 0)
    with torch.no_grad():
      expected = before.module()(*args)
      actual = after.module()(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    return after

  def _assert_untouched(self, module_or_fn, args, decompose=True):
    export = _export_and_decompose if decompose else _export
    before = export(module_or_fn, args)
    self.assertNotEmpty(_copying_expands(before.graph))
    result = self._run(export(module_or_fn, args))
    self.assertFalse(result.modified)

  def test_repeat_kv(self):
    self._assert_lowered(
        _RepeatKV(), (torch.randn(1, 8, 16, 32), torch.randn(1, 2, 16, 32))
    )

  def test_repeat_interleave(self):
    self._assert_lowered(
        _RepeatInterleave(),
        (torch.randn(1, 8, 16, 32), torch.randn(1, 2, 16, 32)),
    )

  def test_expand_of_activation_before_cat(self):
    def f(cls, x):  # cls: (1, 1, 16), x: (2, 10, 16)
      return torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1)

    self._assert_lowered(f, (torch.randn(1, 1, 16), torch.randn(2, 10, 16)))

  def test_expand_of_parameter_is_left_for_the_converter(self):
    # The converter would constant-fold the concatenation and the model would
    # then store one copy of the parameter per repeat.
    self._assert_untouched(_ClsToken(), (torch.randn(2, 10, 16),))

  def test_expand_of_constant_expression_is_left_for_the_converter(self):
    self._assert_untouched(_ScaledClsToken(), (torch.randn(2, 10, 16),))

  def test_repeat_with_extra_leading_dims(self):
    def f(x):  # x: (2, 3)
      return x.repeat(2, 1, 3)  # (2, 2, 9)

    self._assert_lowered(f, (torch.randn(2, 3),), decompose=False)

  def test_expand_multiple_axes_and_new_leading_dim(self):
    def f(x):  # x: (3, 1)
      return x.expand(2, 3, 4).reshape(24)

    self._assert_lowered(f, (torch.randn(3, 1),), decompose=False)

  def test_large_repeat_count_uses_doubling(self):
    def f(x):  # x: (1, 4)
      return x.expand(13, 4).reshape(52)

    after = self._assert_lowered(f, (torch.randn(1, 4),), decompose=False)
    cats = _cats(after.graph)
    # 13 = 8 + 4 + 1: three doublings plus one final concatenation.
    self.assertLen(cats, 4)
    self.assertTrue(all(len(n.args[0]) <= 3 for n in cats))

  def test_expand_into_elementwise_op_is_left_for_the_converter(self):
    def f(x, y):  # x: (1, 1, 8, 8), y: (1, 4, 8, 8)
      return x.expand(1, 4, 8, 8) + y

    args = (torch.randn(1, 1, 8, 8), torch.randn(1, 4, 8, 8))
    self._assert_untouched(f, args, decompose=False)

  def test_expand_into_elementwise_op_that_still_broadcasts_is_lowered(self):
    # The converter only folds BROADCAST_TO into `mul` when the other operand
    # already has the output shape; here `scale` does not, so the
    # BROADCAST_TO would merely be reordered past the `mul` and survive.
    def f(x, scale):  # x: (1, 1, 8, 8), scale: (1, 1, 1, 8)
      return (x.expand(1, 4, 8, 8) * scale).reshape(4, 64)

    args = (torch.randn(1, 1, 8, 8), torch.randn(1, 1, 1, 8))
    self._assert_lowered(f, args, decompose=False)

  def test_expand_into_bmm_is_left_for_the_converter(self):
    def f(x, y):  # x: (1, 8, 4), y: (3, 4, 8)
      return torch.bmm(x.expand(3, 8, 4), y)

    args = (torch.randn(1, 8, 4), torch.randn(3, 4, 8))
    self._assert_untouched(f, args, decompose=False)

  def test_two_expands_into_one_add_are_both_lowered(self):
    # The converter folds one operand at a time; after folding the first
    # BROADCAST_TO it sees add(x, broadcast_to(y)) with x at (1, 8), which does
    # not reach (4, 8) on its own, so the second BROADCAST_TO survives.
    def f(x, y):  # x, y: (1, 8)
      return (x.expand(4, 8) + y.expand(4, 8)).reshape(32)

    args = (torch.randn(1, 8), torch.randn(1, 8))
    after = self._assert_lowered(f, args, decompose=False)
    self.assertEmpty(_copying_expands(after.graph))

  def test_expand_next_to_constant_expand_is_lowered(self):
    # The sibling expand of a parameter is left alone (constant-folded by the
    # converter), so this expand is checked against the parameter's own shape.
    class M(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(1, 8))

      def forward(self, x):  # x: (1, 8)
        return (x.expand(4, 8) + self.bias.expand(4, 8)).reshape(32)

    module = M()
    args = (torch.randn(1, 8),)
    before = _export(module, args)
    self.assertLen(_copying_expands(before.graph), 2)
    result = self._run(_export(module, args))
    self.assertTrue(result.modified)
    after = result.exported_program
    # Only the activation expand is lowered; the parameter one stays.
    remaining = _copying_expands(after.graph)
    self.assertLen(remaining, 1)
    self.assertEqual(remaining[0].args[0].op, "placeholder")
    self.assertGreater(_count(after.graph, aten.cat.default), 0)
    with torch.no_grad():
      torch.testing.assert_close(
          after.module()(*args), before.module()(*args), atol=0, rtol=0
      )

  def test_expand_into_rank_five_comparison_is_lowered(self):
    # ConvertResultsBroadcastableShapeOp handles comparisons only up to rank 4.
    def f(x, y):  # x: (1, 1, 2, 2, 2), y: (1, 3, 2, 2, 2)
      return torch.lt(x.expand(1, 3, 2, 2, 2), y)

    args = (torch.randn(1, 1, 2, 2, 2), torch.randn(1, 3, 2, 2, 2))
    self._assert_lowered(f, args, decompose=False)

  def test_expand_into_rank_five_add_is_left_for_the_converter(self):
    # ...but add/sub/mul fold up to rank 6.
    def f(x, y):  # x: (1, 1, 2, 2, 2), y: (1, 3, 2, 2, 2)
      return x.expand(1, 3, 2, 2, 2) + y

    args = (torch.randn(1, 1, 2, 2, 2), torch.randn(1, 3, 2, 2, 2))
    self._assert_untouched(f, args, decompose=False)

  def test_expand_into_where_is_left_when_other_operands_fill_output(self):
    def f(c, x, y):  # c: (1, 8) bool, x: (4, 8), y: (4, 8)
      return torch.where(c.expand(4, 8), x, y)

    args = (torch.rand(1, 8) > 0.5, torch.randn(4, 8), torch.randn(4, 8))
    self._assert_untouched(f, args, decompose=False)

  def test_expand_into_where_is_lowered_when_it_sets_output_shape(self):
    # SELECT_V2's fold needs the *other two* operands alone to produce the
    # result shape; here only the expanded condition does.
    def f(c, x, y):  # c: (1, 8) bool, x: (1, 8), y: (1, 8)
      return torch.where(c.expand(4, 8), x, y)

    args = (torch.rand(1, 8) > 0.5, torch.randn(1, 8), torch.randn(1, 8))
    self._assert_lowered(f, args, decompose=False)

  def test_repeat_kv_with_dynamic_sequence_length(self):
    # Only the copied (head) axis must be static; the sequence axis may be
    # symbolic.
    module = _RepeatKV()
    args = (torch.randn(1, 8, 16, 32), torch.randn(1, 2, 16, 32))
    seq = torch.export.Dim("seq", min=2, max=64)
    dynamic_shapes = ({2: seq}, {2: seq})
    exported = torch.export.export(module, args, dynamic_shapes=dynamic_shapes)
    exported = exported.run_decompositions()
    self.assertNotEmpty(_copying_expands(exported.graph))
    result = self._run(exported)
    self.assertTrue(result.modified)
    after = result.exported_program
    self.assertEmpty(_copying_expands(after.graph))
    for a in (args, (torch.randn(1, 8, 40, 32), torch.randn(1, 2, 40, 32))):
      with torch.no_grad():
        torch.testing.assert_close(
            after.module()(*a), module(*a), atol=0, rtol=0
        )

  def test_expand_of_matrix_dim_into_bmm_is_lowered(self):
    # BATCH_MATMUL only broadcasts batch dims; copying a matrix dim survives.
    def f(x, y):  # x: (3, 1, 4), y: (3, 4, 8)
      return torch.bmm(x.expand(3, 8, 4), y)

    args = (torch.randn(3, 1, 4), torch.randn(3, 4, 8))
    self._assert_lowered(f, args, decompose=False)

  def test_zero_copy_count_is_declined(self):
    def f(x):  # x: (1, 4)
      return x.repeat(0, 1).reshape(0)

    self._assert_untouched(f, (torch.randn(1, 4),), decompose=False)

  def test_rank_above_five_is_declined(self):
    def f(x):  # x: (1, 2, 1, 2, 1, 2)
      return x.expand(3, 2, 3, 2, 3, 2).reshape(-1)

    self._assert_untouched(f, (torch.randn(1, 2, 1, 2, 1, 2),), decompose=False)

  def test_provenance_meta_copied_to_new_nodes(self):
    def f(x):
      return x.expand(3, 4).reshape(12)

    exported_program = _export(f, (torch.randn(1, 4),))
    for node in exported_program.graph.nodes:
      if node.target == aten.expand.default:
        node.meta["stack_trace"] = "model.py:1"

    result = self._run(exported_program)
    self.assertTrue(result.modified)
    cat = _cats(result.exported_program.graph)[0]
    self.assertEqual(cat.meta.get("stack_trace"), "model.py:1")


if __name__ == "__main__":
  googletest.main()
