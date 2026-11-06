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

This pass performs that rewrite for static shapes. Two cases are left alone:

  * an `expand` the converter already folds into its consumers (an implicitly
    broadcasting elementwise op or a batch matmul whose output shape does not
    depend on the expand), since materializing the copies would only cost
    memory;
  * an input that does not depend on any user input (a parameter, buffer or
    lifted constant, possibly through further constant ops), since the
    converter would constant-fold the concatenation and the model would then
    store `n` copies of the constant.
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
# (see `optimize_broadcast_like_patterns.td`), with the maximum result rank
# `ConvertResultsBroadcastableShapeOp` handles for each (`IsRankSupported`):
# 6 for add/sub/mul, 5 for div, 4 for the rest. An expand used only by these,
# within rank, never reaches the model as a BROADCAST_TO.
_IMPLICITLY_BROADCASTING_OPS = {
    aten.add.Tensor: 6,
    aten.sub.Tensor: 6,
    aten.mul.Tensor: 6,
    aten.div.Tensor: 5,
    aten.div.Tensor_mode: 5,
    aten.maximum.default: 4,
    aten.minimum.default: 4,
    aten.pow.Tensor_Tensor: 4,
    aten.eq.Tensor: 4,
    aten.ne.Tensor: 4,
    aten.lt.Tensor: 4,
    aten.le.Tensor: 4,
    aten.gt.Tensor: 4,
    aten.ge.Tensor: 4,
    aten.logical_and.default: 4,
    aten.logical_or.default: 4,
}

# `where` lowers to SELECT_V2, whose BROADCAST_TO fusion requires the *other
# two* operands alone to produce the result shape.
_SELECT_OPS = frozenset({aten.where.self})
_SELECT_MAX_RANK = 4

# Batch matmuls the converter fuses a preceding BROADCAST_TO into
# (`ConvertResultsBroadcastableBatchMatMulShapeOp`), as long as only the batch
# dims are broadcast. `aten.matmul` is not listed: it is decomposed before
# lowering, and for rank >= 4 that decomposition is expand -> reshape -> bmm,
# where the BROADCAST_TO feeds the reshape and survives.
_BATCH_MATMUL_OPS = frozenset({aten.bmm.default})

_EXPAND_OPS = (aten.expand.default, aten.expand_copy.default)
_REPEAT_OPS = (aten.repeat.default,)

_CONSTANT_INPUT_KINDS = (
    torch.export.graph_signature.InputKind.PARAMETER,
    torch.export.graph_signature.InputKind.BUFFER,
    torch.export.graph_signature.InputKind.CONSTANT_TENSOR,
)

# Debug-info meta fields that map a node back to its source model code.
_PROVENANCE_META_KEYS = (
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "from_node",
)


def _shape(node: torch.fx.Node) -> list[object] | None:
  """A node's tensor shape; entries are ints or symbolic sizes."""
  val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
  if not isinstance(val, torch.Tensor):
    return None
  return list(val.shape)


def _static_shape(node: torch.fx.Node) -> list[int] | None:
  shape = _shape(node)
  if shape is None or not all(isinstance(s, int) for s in shape):
    return None
  return shape


def _same_size(a, b) -> bool:
  """True if two (possibly symbolic) sizes are known to be equal."""
  if isinstance(a, int) and isinstance(b, int):
    return a == b
  try:
    return bool(a == b)
  except Exception:  # pylint: disable=broad-except
    return False


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
  """Per-axis copy counts for an expand/repeat node, or None to decline.

  Only the copied axes need static sizes: concatenation along an axis is
  well-defined whatever the other (possibly symbolic) axes are, so a
  `repeat_kv` exported with a dynamic sequence length still qualifies as
  long as the head-count axis it copies is static.
  """
  in_shape = _shape(node.args[0])
  out_shape = _shape(node)
  if in_shape is None or out_shape is None or len(out_shape) > _MAX_RANK:
    return None
  if node.target in _REPEAT_OPS:
    if not all(isinstance(r, int) for r in node.args[1]):
      return None
    multiples = [int(r) for r in node.args[1]]
  else:
    # expand: a copy count only exists where a size-1 axis grows.
    padded = [1] * (len(out_shape) - len(in_shape)) + in_shape
    multiples = []
    for i, o in zip(padded, out_shape):
      if _same_size(i, o):
        multiples.append(1)
      elif isinstance(i, int) and i == 1 and isinstance(o, int):
        multiples.append(o)
      else:
        # A non-unit or symbolic source axis that changes size, or a copy to
        # a symbolic size: not expressible as a static concatenation.
        return None
  # A zero count (`repeat(0, ...)`, expand to a size-0 axis) has no
  # concatenation equivalent; leave it to the converter.
  if any(m < 1 for m in multiples):
    return None
  return multiples


def _unexpanded_shape(
    arg: torch.fx.Node, constants: set[torch.fx.Node]
) -> list[object] | None:
  """The shape the converter will see for `arg` when it fuses a sibling.

  `ConvertResultsBroadcastableShapeOp` folds one operand at a time, so a
  sibling expand that this pass also lowers (or leaves alone because its input
  is constant) is seen at its *input* shape, not its expanded shape.
  """
  if (
      isinstance(arg, torch.fx.Node)
      and arg.op == "call_function"
      and arg.target in _EXPAND_OPS
      and (arg.args[0] in constants or _multiples(arg) is not None)
  ):
    return _shape(arg.args[0])
  return _shape(arg)


def _other_input_shapes(
    user: torch.fx.Node, node: torch.fx.Node, constants: set[torch.fx.Node]
) -> list[tuple[object, ...]] | None:
  """Shapes of `user`'s tensor inputs other than `node`."""
  shapes = []
  for arg in user.all_input_nodes:
    if arg is node:
      continue
    shape = _unexpanded_shape(arg, constants)
    if shape is None:
      return None
    shapes.append(tuple(shape))
  return shapes


def _broadcasts_to(shapes, target) -> bool:
  try:
    result = torch.broadcast_shapes(*shapes)
  except RuntimeError:
    return False
  return len(result) == len(target) and all(
      _same_size(r, t) for r, t in zip(result, target)
  )


def _is_folded_by_converter(
    node: torch.fx.Node, in_shape: list[object], constants: set[torch.fx.Node]
) -> bool:
  """True if the converter would fold this expand into every consumer.

  The converter only replaces `op(broadcast_to(x), y)` with `op(x, y)` when
  implicitly broadcasting `x` against `y` already yields the op's result shape
  (`ConvertResultsBroadcastableShapeOp`), and only up to a per-op result rank.
  Otherwise the `BROADCAST_TO` is merely reordered and survives --
  `expand(k) * scale -> reshape` still ends up as `BROADCAST_TO -> RESHAPE` --
  so such an expand is worth lowering here.
  """
  for user in node.users:
    if user.op != "call_function":
      return False
    user_shape = _shape(user)
    others = _other_input_shapes(user, node, constants)
    if user_shape is None or others is None:
      return False
    if user.target in _IMPLICITLY_BROADCASTING_OPS:
      if len(user_shape) > _IMPLICITLY_BROADCASTING_OPS[user.target]:
        return False
      if not _broadcasts_to([tuple(in_shape), *others], user_shape):
        return False
    elif user.target in _SELECT_OPS:
      # SELECT_V2's fusion needs the other two operands alone to produce the
      # result shape; the broadcast operand does not count.
      if len(user_shape) > _SELECT_MAX_RANK or len(others) != 2:
        return False
      if not _broadcasts_to(others, user_shape):
        return False
    elif user.target in _BATCH_MATMUL_OPS:
      # BATCH_MATMUL broadcasts batch dims only; the matrix dims of the
      # unexpanded input must already match the expanded ones.
      out_shape = _shape(node)
      if len(in_shape) < 2 or not all(
          _same_size(a, b) for a, b in zip(in_shape[-2:], out_shape[-2:])
      ):
        return False
      batch_shapes = [tuple(in_shape[:-2])] + [o[:-2] for o in others]
      if not _broadcasts_to(batch_shapes, user_shape[:-2]):
        return False
    else:
      return False
  return True


def _constant_nodes(
    exported_program: torch.export.ExportedProgram,
) -> set[torch.fx.Node]:
  """Nodes that depend on no user input and would be constant-folded."""
  constant_placeholders = {
      spec.arg.name
      for spec in exported_program.graph_signature.input_specs
      if spec.kind in _CONSTANT_INPUT_KINDS
  }
  constants = set()
  for node in exported_program.graph.nodes:
    if node.op == "get_attr" or (
        node.op == "placeholder" and node.name in constant_placeholders
    ):
      constants.add(node)
    elif node.op == "call_function" and all(
        arg in constants for arg in node.all_input_nodes
    ):
      constants.add(node)
  return constants


class LowerBroadcastToConcatPass(fx_infra.ExportedProgramPassBase):
  """Rewrites static expand/repeat copies as concatenation for the GPU."""

  def call(self, exported_program: torch.export.ExportedProgram):
    graph_module = exported_program.graph_module
    graph = graph_module.graph
    constants = _constant_nodes(exported_program)

    # Decide every candidate before rewriting any: the fold check looks at
    # sibling expands, and a sibling already rewritten to a concatenation
    # would be seen at its full shape.
    plan = []
    for node in graph.nodes:
      if node.op != "call_function":
        continue
      if node.target not in _EXPAND_OPS + _REPEAT_OPS:
        continue
      x = node.args[0]
      if x in constants:
        continue
      multiples = _multiples(node)
      if multiples is None or all(n == 1 for n in multiples):
        continue
      if node.target in _EXPAND_OPS and _is_folded_by_converter(
          node, _shape(x), constants
      ):
        continue
      plan.append((node, x, multiples))

    for node, x, multiples in plan:
      result = _tile(_Builder(graph, node), x, x.meta["val"].dim(), multiples)
      result.meta["val"] = node.meta["val"]
      if "tensor_meta" in node.meta:
        result.meta["tensor_meta"] = node.meta["tensor_meta"]
      node.replace_all_uses_with(result)

    modified = bool(plan)
    if modified:
      graph.eliminate_dead_code()
      graph.lint()
      graph_module.recompile()
    return fx_infra.ExportedProgramPassResult(exported_program, modified)
