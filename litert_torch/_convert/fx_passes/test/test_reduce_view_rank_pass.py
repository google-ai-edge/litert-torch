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

import math
import random

from litert_torch import fx_infra
from litert_torch._convert import fx_passes
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from absl.testing import absltest as googletest


aten = torch.ops.aten


def _export_and_decompose(
    module: torch.nn.Module, export_args, dynamic_shapes=None
) -> torch.export.ExportedProgram:
  exported_program = torch.export.export(
      module.eval(), export_args, dynamic_shapes=dynamic_shapes
  )
  return fx_infra.safe_run_decompositions(
      exported_program,
      fx_infra.decomp.pre_convert_decomp(),
  )


def _max_rank(graph) -> int:
  """Returns the highest rank of any tensor flowing through `graph`."""
  if isinstance(graph, torch.export.ExportedProgram):
    graph = graph.graph
  elif isinstance(graph, torch.fx.GraphModule):
    graph = graph.graph
  max_rank = 0
  for node in graph.nodes:
    val = node.meta.get("val")
    vals = val if isinstance(val, (list, tuple)) else [val]
    for v in vals:
      if v is not None and hasattr(v, "dim"):
        max_rank = max(max_rank, v.dim())
  return max_rank


def _trace(fn, *args) -> torch.fx.GraphModule:
  return make_fx(fn, tracing_mode="fake")(*args)


class _SelfAttention(torch.nn.Module):

  def __init__(self, embed_dim=64, num_heads=4, batch_first=True):
    super().__init__()
    self.attn = torch.nn.MultiheadAttention(
        embed_dim, num_heads, batch_first=batch_first
    )

  def forward(self, x):
    y, _ = self.attn(x, x, x, need_weights=False)
    return y


class _WindowPartition(torch.nn.Module):
  """The Swin / Twins window partition idiom, which transiently hits rank 6."""

  def __init__(self, window: int = 2):
    super().__init__()
    self.window = window

  def forward(self, x):  # x: (B, H, W, C)
    b, h, w, c = x.shape
    m = self.window
    y = x.view(b, h // m, m, w // m, m, c)
    y = y.permute(0, 1, 3, 2, 4, 5)
    return y.reshape(-1, m * m, c)


class PackedQkvTest(googletest.TestCase):
  """The `unsqueeze -> permute -> squeeze` chain from nn.MultiheadAttention."""

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

  def test_dynamic_batch_is_declined_safely(self):
    # Symbolic extents are out of scope for now. The chain is left exactly as
    # it was -- the transient rank-5 tensor survives -- and the model still
    # runs correctly at other batch sizes.
    module = _SelfAttention()
    args = (torch.randn(2, 16, 64),)
    batch = torch.export.Dim("batch", min=1, max=64)

    after = fx_infra.run_passes(
        _export_and_decompose(module, args, dynamic_shapes=({0: batch},)),
        [fx_passes.ReduceViewRankPass()],
    )

    self.assertGreater(_max_rank(after), 4)
    other = (torch.randn(5, 16, 64),)
    with torch.no_grad():
      expected = module(*other)
      actual = after.module()(*other)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

  def test_unit_dim_chain_collapses_to_a_bare_permute(self):
    # The source tensor has no unit axes, so the whole rank-5 chain is just a
    # reordering of its axes and no reshape is needed at all.
    def f(x):  # x: (2, 3, 4)
      y = torch.unsqueeze(x, 0)  # (1, 2, 3, 4)
      y = torch.unsqueeze(y, 0)  # (1, 1, 2, 3, 4)
      y = torch.permute(y, [0, 1, 4, 2, 3])  # (1, 1, 4, 2, 3)
      return torch.squeeze(y, [0, 1])  # (4, 2, 3)

    x = torch.randn(2, 3, 4)
    result = fx_passes.ReduceViewRankPass()(_trace(f, x))

    self.assertTrue(result.modified)
    self.assertLessEqual(_max_rank(result.graph_module), 4)
    targets = [
        n.target
        for n in result.graph_module.graph.nodes
        if n.op == "call_function"
    ]
    self.assertEqual(targets, [aten.permute.default])
    torch.testing.assert_close(result.graph_module(x), f(x))

  def test_multi_unit_dim_squeeze(self):
    # A unit-dim unsqueeze whose inserted axis is removed alongside another
    # unit axis by the squeeze.
    def f(x):  # x: (2, 1, 3, 4)
      y = torch.unsqueeze(x, 0)  # (1, 2, 1, 3, 4)
      y = torch.permute(y, [0, 2, 1, 3, 4])  # (1, 1, 2, 3, 4)
      return torch.squeeze(y, [0, 1])  # (2, 3, 4)

    x = torch.randn(2, 1, 3, 4)
    result = fx_passes.ReduceViewRankPass()(_trace(f, x))

    self.assertTrue(result.modified)
    self.assertLessEqual(_max_rank(result.graph_module), 4)
    torch.testing.assert_close(result.graph_module(x), f(x))


class BlockTransposeTest(googletest.TestCase):
  """Split-permute-flatten chains from windowed vision transformers."""

  def test_window_partition_drops_to_rank_4(self):
    module = _WindowPartition(window=2)
    args = (torch.randn(2, 8, 8, 4),)

    before = _export_and_decompose(module, args)
    self.assertGreater(_max_rank(before), 4)

    after = fx_infra.run_passes(
        _export_and_decompose(module, args),
        [fx_passes.ReduceViewRankPass()],
    )
    self.assertLessEqual(_max_rank(after), 4)
    with torch.no_grad():
      expected = module(*args)
      actual = after.module()(*args)
    torch.testing.assert_close(actual, expected)

  def test_dynamic_batch_is_declined_safely(self):
    # Symbolic extents are out of scope for now, so the chain must be left
    # exactly as it was rather than rewritten on a guess. The rank-6 tensor
    # therefore survives, and the model still runs at other batch sizes.
    module = _WindowPartition(window=2)
    args = (torch.randn(2, 8, 8, 4),)
    batch = torch.export.Dim("batch", min=1, max=64)

    after = fx_infra.run_passes(
        _export_and_decompose(module, args, dynamic_shapes=({0: batch},)),
        [fx_passes.ReduceViewRankPass()],
    )

    self.assertGreater(_max_rank(after), 4)
    other = (torch.randn(5, 8, 8, 4),)
    with torch.no_grad():
      expected = module(*other)
      actual = after.module()(*other)
    torch.testing.assert_close(actual, expected)

  def test_interior_axis_runs_are_merged(self):
    # `perm` has no fixed point at either edge, so an edge-only merge rule
    # finds nothing. The axes still group into two runs that stay contiguous
    # in the result, so the chain is expressible at rank 4.
    def f(x):  # x: (6, 20, 6)
      y = x.view(2, 3, 4, 5, 6)
      y = y.permute(3, 4, 0, 1, 2)
      return y.reshape(30, 24)

    x = torch.randn(6, 20, 6)
    result = fx_passes.ReduceViewRankPass()(_trace(f, x))

    self.assertTrue(result.modified)
    self.assertLessEqual(_max_rank(result.graph_module), 4)
    torch.testing.assert_close(result.graph_module(x), f(x))

  def test_grid_partition_needs_rank_5(self):
    # MaxViT's grid partition interleaves both split axes, so no two source
    # axes stay adjacent and rank 5 is genuinely the minimum. It is left alone
    # under the default cap and rewritten under a cap of 5.
    def f(x):  # x: (1, 8, 8, 4)
      y = x.view(1, 2, 4, 2, 4, 4)
      y = y.permute(0, 2, 4, 1, 3, 5)
      return y.reshape(-1, 4, 4)

    x = torch.randn(1, 8, 8, 4)

    self.assertFalse(fx_passes.ReduceViewRankPass()(_trace(f, x)).modified)

    result = fx_passes.ReduceViewRankPass(max_rank=5)(_trace(f, x))
    self.assertTrue(result.modified)
    self.assertLessEqual(_max_rank(result.graph_module), 5)
    torch.testing.assert_close(result.graph_module(x), f(x))

  def test_chain_through_a_contiguous_copy(self):
    # Decomposing a reshape of a permuted tensor inserts a copy mid-chain,
    # which must be stepped over rather than ending the chain.
    def f(x):  # x: (2, 8, 8, 4)
      y = x.view(2, 4, 2, 4, 2, 4)
      y = y.permute(0, 1, 3, 2, 4, 5)
      return y.contiguous().reshape(-1, 4, 4)

    x = torch.randn(2, 8, 8, 4)
    result = fx_passes.ReduceViewRankPass()(_trace(f, x))

    self.assertTrue(result.modified)
    self.assertLessEqual(_max_rank(result.graph_module), 4)
    torch.testing.assert_close(result.graph_module(x), f(x))

  def test_transpose_is_understood(self):
    # `transpose` is just a two-axis permute and participates in a chain.
    def f(x):  # x: (6, 20, 6)
      y = x.view(2, 3, 4, 5, 6)
      y = y.transpose(0, 3)
      return y.reshape(-1, 6)

    x = torch.randn(6, 20, 6)
    result = fx_passes.ReduceViewRankPass()(_trace(f, x))

    self.assertTrue(result.modified)
    self.assertLessEqual(_max_rank(result.graph_module), 4)
    torch.testing.assert_close(result.graph_module(x), f(x))

  def test_permutes_compose_across_a_reshape(self):
    # MaxViT: an NHWC -> NCHW permute feeds straight into a window partition.
    # Taken alone, the partition's permute is irreducible below rank 5: none of
    # (C, h1, h2, w1, w2) stay adjacent. Modeled from the NHWC source, w2 and C
    # are adjacent both in the source and in the result, so the whole chain
    # fits in rank 4 with a single transpose. A matcher that only looks at one
    # transpose at a time (e.g. coremltools'
    # `expand_high_rank_reshape_and_transpose`) cannot see this.
    def partition(y):  # y: (1, C, H, W)
      y = y.contiguous().view(1, 4, 2, 4, 2, 4)
      y = y.permute(0, 2, 4, 3, 5, 1)  # (1, h1, w1, h2, w2, C)
      return y.contiguous().view(1, 4, 16, 4)

    def f(x):  # x: (1, H, W, C) = (1, 8, 8, 4)
      return partition(x.permute(0, 3, 1, 2))

    nchw = torch.randn(1, 4, 8, 8)
    self.assertFalse(
        fx_passes.ReduceViewRankPass()(_trace(partition, nchw)).modified
    )

    x = torch.randn(1, 8, 8, 4)
    result = fx_passes.ReduceViewRankPass()(_trace(f, x))

    self.assertTrue(result.modified)
    self.assertLessEqual(_max_rank(result.graph_module), 4)
    permutes = [
        n
        for n in result.graph_module.graph.nodes
        if n.target == aten.permute.default
    ]
    self.assertLen(permutes, 1)
    torch.testing.assert_close(result.graph_module(x), f(x))


class SafetyTest(googletest.TestCase):
  """Cases the pass must decline or leave untouched."""

  def test_no_op_when_already_within_the_cap(self):
    # A plain MLP has no high-rank chain; the pass must leave it untouched.
    class Mlp(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 16)

      def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

    args = (torch.randn(2, 4, 16),)
    graph_module = _export_and_decompose(Mlp(), args).graph_module
    targets_before = [n.target for n in graph_module.graph.nodes]

    result = fx_passes.ReduceViewRankPass()(graph_module)
    self.assertFalse(result.modified)
    targets_after = [n.target for n in result.graph_module.graph.nodes]
    self.assertEqual(targets_before, targets_after)

  def test_declines_when_the_high_rank_tensor_escapes(self):
    # The rank-6 intermediate is consumed by a real op, so it cannot be
    # deleted and rewriting the chain around it would buy nothing.
    def f(x):  # x: (2, 8, 8, 4)
      y = x.view(2, 4, 2, 4, 2, 4)
      y = y.permute(0, 1, 3, 2, 4, 5)
      return y.reshape(-1, 4, 4), torch.relu(y)

    x = torch.randn(2, 8, 8, 4)
    result = fx_passes.ReduceViewRankPass()(_trace(f, x))
    self.assertFalse(result.modified)

  def test_declines_when_the_result_is_over_the_cap(self):
    # The chain ends at rank 5, so there is no rank-4 form to emit.
    def f(x):  # x: (6, 20, 6)
      y = x.view(2, 3, 4, 5, 6)
      y = y.permute(3, 4, 0, 1, 2)
      return y.reshape(5, 6, 2, 3, 4)

    x = torch.randn(6, 20, 6)
    result = fx_passes.ReduceViewRankPass()(_trace(f, x))
    self.assertFalse(result.modified)

  def test_provenance_meta_copied_to_new_nodes(self):
    # Every node the pass emits must carry the debug provenance of the chain
    # output it replaces, so failures still point back at the model source.
    def f(x):  # x: (2, 1, 3, 4)
      y = torch.unsqueeze(x, 0)  # (1, 2, 1, 3, 4)
      y = torch.permute(y, [0, 2, 1, 3, 4])  # (1, 1, 2, 3, 4)
      return torch.squeeze(y, [0, 1])  # (2, 3, 4)

    x = torch.randn(2, 1, 3, 4)
    graph_module = _trace(f, x)

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

    emitted = [
        n for n in result.graph_module.graph.nodes if n.op == "call_function"
    ]
    self.assertNotEmpty(emitted)
    for node in emitted:
      for key, value in provenance.items():
        self.assertEqual(node.meta.get(key), value)

  def test_trailing_copy_is_preserved(self):
    # `_in_projection_packed` ends its view chain with `.contiguous()`, and a
    # later `view` relies on that layout. The copy must survive the rewrite:
    # absorbing it would leave a strided view behind and the `view` would fail
    # at runtime even though the values are right.
    def f(x):  # x: (2, 3, 4, 5)
      y = torch.unsqueeze(x, 0)  # (1, 2, 3, 4, 5)
      y = torch.permute(y, [0, 4, 1, 2, 3])  # (1, 5, 2, 3, 4)
      y = torch.squeeze(y, 0).contiguous()  # (5, 2, 3, 4)
      return y[0].view(6, 4)

    x = torch.randn(2, 3, 4, 5)
    result = fx_passes.ReduceViewRankPass()(_trace(f, x))

    self.assertTrue(result.modified)
    self.assertLessEqual(_max_rank(result.graph_module), 4)
    torch.testing.assert_close(result.graph_module(x), f(x))

  def test_pass_is_idempotent(self):
    # A second run must find nothing left to do.
    module = _WindowPartition(window=2)
    args = (torch.randn(2, 8, 8, 4),)

    after = fx_infra.run_passes(
        _export_and_decompose(module, args),
        [fx_passes.ReduceViewRankPass()],
    )
    again = fx_passes.ReduceViewRankPass()(after.graph_module)
    self.assertFalse(again.modified)


def _random_shape(rng: random.Random, factors: list[int]) -> tuple[int, ...]:
  """Groups `factors` into 1 to 6 random axis sizes with the same product."""
  num_axes = rng.randint(1, 6)
  shape = [1] * num_axes
  for factor in factors:
    shape[rng.randrange(num_axes)] *= factor
  return tuple(shape)


def _random_view_chain(seed: int):
  """Returns (fn, input_shape) for a randomly generated view chain."""
  rng = random.Random(seed)
  factors = [rng.choice([2, 3, 5]) for _ in range(rng.randint(4, 6))]
  shape = tuple(factors)

  steps = []
  current = shape
  for _ in range(rng.randint(2, 5)):
    kind = rng.choice(
        ["reshape", "permute", "permute", "unsqueeze", "squeeze", "contiguous"]
    )
    if kind == "reshape":
      target = _random_shape(rng, factors)
      steps.append(("reshape", target))
      current = target
    elif kind == "permute":
      order = list(range(len(current)))
      rng.shuffle(order)
      steps.append(("permute", order))
      current = tuple(current[i] for i in order)
    elif kind == "unsqueeze":
      dim = rng.randint(0, len(current))
      steps.append(("unsqueeze", dim))
      current = current[:dim] + (1,) + current[dim:]
    elif kind == "squeeze":
      unit_dims = [i for i, d in enumerate(current) if d == 1]
      if not unit_dims:
        steps.append(("contiguous", None))
        continue
      dim = rng.choice(unit_dims)
      steps.append(("squeeze", dim))
      current = current[:dim] + current[dim + 1 :]
    else:
      steps.append(("contiguous", None))

  def fn(x):
    for kind, arg in steps:
      if kind == "reshape":
        x = torch.reshape(x, arg)
      elif kind == "permute":
        x = torch.permute(x, arg)
      elif kind == "unsqueeze":
        x = torch.unsqueeze(x, arg)
      elif kind == "squeeze":
        x = torch.squeeze(x, arg)
      else:
        x = x.contiguous()
    return x

  return fn, shape


def _element_order(graph_module, shape) -> list[int]:
  """Returns where each source element lands, as a flat list."""
  x = torch.arange(math.prod(shape)).reshape(shape)
  return graph_module(x).reshape(-1).tolist()


class RandomChainTest(googletest.TestCase):
  """Differential test against the un-rewritten graph."""

  def test_rewrite_preserves_element_order(self):
    for seed in range(200):
      with self.subTest(seed=seed):
        fn, shape = _random_view_chain(seed)
        graph_module = _trace(fn, torch.randn(shape))
        expected_order = _element_order(graph_module, shape)
        rank_before = _max_rank(graph_module)

        result = fx_passes.ReduceViewRankPass()(graph_module)

        self.assertEqual(
            _element_order(result.graph_module, shape), expected_order
        )
        self.assertLessEqual(_max_rank(result.graph_module), rank_before)


if __name__ == "__main__":
  googletest.main()
