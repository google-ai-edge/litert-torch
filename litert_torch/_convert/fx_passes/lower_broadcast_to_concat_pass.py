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
"""Pass to lower broadcasting copies to concatenation for the GPU delegate.

`aten.expand` and `aten.repeat` both lower to TFLite `BROADCAST_TO`. The TFLite
converter folds a `BROADCAST_TO` into a consumer that broadcasts implicitly
(`ADD`, `MUL`, `SELECT_V2`, ...), but when the consumer is a reshape, a
concatenation or a matmul the op survives into the model -- and the ML Drift GPU
delegate has no `BROADCAST_TO` kernel, so every one of them splits the graph.

The common sources are GQA key/value head expansion (`repeat_kv`,
`repeat_interleave`), cls-token and position-table expansion in ViTs, and
`Tensor.repeat`. All of them copy a size-1 axis `n` times, which is exactly
`cat([x] * n, dim)`: a `CONCATENATION` that the delegate accepts up to rank 5.

This pass performs that rewrite for static shapes. An `expand` whose every user
is an implicitly broadcasting elementwise op is left alone, since the converter
already folds it and materializing the copies would only cost memory.
"""

from litert_torch import fx_infra
import torch

aten = torch.ops.aten

# ML Drift's CONCATENATION accepts up to rank 5; beyond that the rewrite would
# only trade one rejected op for another.
_MAX_RANK = 5

# Above this many copies, build the repeat by doubling rather than with one
# wide CONCATENATION.
_MAX_CAT_INPUTS = 8

# Elementwise ops the TFLite converter fuses a preceding BROADCAST_TO into
# (see `optimize_broadcast_like_patterns.td`). An expand used only by these
# never reaches the model as a BROADCAST_TO.
_IMPLICITLY_BROADCASTING_OPS = frozenset({
    aten.add.Tensor,
    aten.sub.Tensor,
    aten.mul.Tensor,
    aten.div.Tensor,
    aten.div.Tensor_mode,
    aten.maximum.default,
    aten.minimum.default,
    aten.pow.Tensor_Tensor,
    aten.eq.Tensor,
    aten.ne.Tensor,
    aten.lt.Tensor,
    aten.le.Tensor,
    aten.gt.Tensor,
    aten.ge.Tensor,
    aten.where.self,
    aten.logical_and.default,
    aten.logical_or.default,
})

_EXPAND_OPS = (aten.expand.default, aten.expand_copy.default)
_REPEAT_OPS = (aten.repeat.default,)

# Debug-info meta fields that map a node back to its source model code.
_PROVENANCE_META_KEYS = (
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "from_node",
)


def _static_shape(node: torch.fx.Node) -> list[int] | None:
  val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
  if not isinstance(val, torch.Tensor):
    return None
  if not all(isinstance(s, int) for s in val.shape):
    return None
  return list(val.shape)


class _Builder:
  """Inserts nodes before `anchor`, carrying its provenance and a fake `val`."""

  def __init__(self, graph: torch.fx.Graph, anchor: torch.fx.Node):
    self._graph = graph
    self._anchor = anchor
    self._fake_mode = anchor.meta["val"].fake_mode

  def __call__(self, target, *args) -> torch.fx.Node:
    with self._graph.inserting_before(self._anchor):
      node = self._graph.call_function(target, args)
    for key in _PROVENANCE_META_KEYS:
      if key in self._anchor.meta:
        node.meta[key] = self._anchor.meta[key]
    fake_args = torch.fx.node.map_arg(args, lambda n: n.meta["val"])
    with self._fake_mode:
      node.meta["val"] = target(*fake_args)
    return node


def _repeat_along(
    build: _Builder, x: torch.fx.Node, dim: int, n: int
) -> torch.fx.Node:
  """Returns `x` concatenated with itself `n` times along `dim`."""
  if n <= _MAX_CAT_INPUTS:
    return build(aten.cat.default, [x] * n, dim)
  # Binary decomposition: x, x2, x4, ... and concatenate the set bits of n.
  parts = []
  power = x
  while True:
    if n & 1:
      parts.append(power)
    n >>= 1
    if not n:
      break
    power = build(aten.cat.default, [power, power], dim)
  return parts[0] if len(parts) == 1 else build(aten.cat.default, parts, dim)


def _tile(
    build: _Builder, x: torch.fx.Node, rank: int, multiples: list[int]
) -> torch.fx.Node:
  """Prepends unit dims to reach `len(multiples)`, then repeats each axis."""
  for _ in range(len(multiples) - rank):
    x = build(aten.unsqueeze.default, x, 0)
  for dim, n in enumerate(multiples):
    if n > 1:
      x = _repeat_along(build, x, dim, n)
  return x


def _multiples(node: torch.fx.Node) -> list[int] | None:
  """Per-axis copy counts for an expand/repeat node, or None to decline."""
  in_shape = _static_shape(node.args[0])
  out_shape = _static_shape(node)
  if in_shape is None or out_shape is None or len(out_shape) > _MAX_RANK:
    return None
  if node.target in _REPEAT_OPS:
    return [int(r) for r in node.args[1]]
  # expand: a copy count only exists where a size-1 axis grows.
  padded = [1] * (len(out_shape) - len(in_shape)) + in_shape
  multiples = []
  for i, o in zip(padded, out_shape):
    if i != o and i != 1:
      return None
    multiples.append(o // i if i == 1 else 1)
  return multiples


def _is_folded_by_converter(node: torch.fx.Node) -> bool:
  return all(
      user.op == "call_function" and user.target in _IMPLICITLY_BROADCASTING_OPS
      for user in node.users
  )


class LowerBroadcastToConcatPass(fx_infra.PassBase):
  """Rewrites static expand/repeat copies as concatenation for the GPU."""

  def call(self, graph_module: torch.fx.GraphModule):
    graph = graph_module.graph
    modified = False
    for node in list(graph.nodes):
      if node.op != "call_function":
        continue
      if node.target not in _EXPAND_OPS + _REPEAT_OPS:
        continue
      if node.target in _EXPAND_OPS and _is_folded_by_converter(node):
        continue
      multiples = _multiples(node)
      if multiples is None or all(n == 1 for n in multiples):
        continue

      x = node.args[0]
      result = _tile(_Builder(graph, node), x, x.meta["val"].dim(), multiples)
      result.meta["val"] = node.meta["val"]
      if "tensor_meta" in node.meta:
        result.meta["tensor_meta"] = node.meta["tensor_meta"]
      node.replace_all_uses_with(result)
      modified = True

    if modified:
      graph.eliminate_dead_code()
      graph.lint()
      graph_module.recompile()
    return fx_infra.PassResult(graph_module, modified)
