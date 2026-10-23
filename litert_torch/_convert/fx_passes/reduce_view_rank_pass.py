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
"""Pass to rewrite high-rank view chains within the GPU delegate's rank cap.

The ML Drift GPU delegate caps `RESHAPE` and `TRANSPOSE` at rank 5 and
`FULLY_CONNECTED` at rank 4. Common PyTorch idioms transiently exceed those
caps inside chains of pure view ops:

  * Packed QKV projection (`torch.nn.functional._in_projection_packed`, used by
    `nn.MultiheadAttention`), which unpacks Q, K, V via
    `unsqueeze -> permute -> squeeze` at rank 5.
  * Window partition in vision transformers (Swin, MaxViT, Twins, XCiT):
    `x.view(B, H // M, M, W // M, M, C).permute(0, 1, 3, 2, 4, 5).reshape(...)`,
    which transiently hits rank 6.

When every intermediate in a chain of view ops (`reshape`, `view`, `permute`,
`transpose`, `squeeze`, `unsqueeze`, and mid-chain `clone`/`contiguous`) has a
single user, the chain's only effect is to reorder and regroup the source
tensor's axes. This pass models the whole chain and re-emits it as at most three
ops within `max_rank`:

    reshape(pre_shape) -> permute(perm) -> reshape(out_shape)

omitting any step that is an identity.

Worked example
--------------
Consider Swin's window partition with `max_rank = 4`:

    x: (1, 56, 56, 96)
    -> view(1, 8, 7, 8, 7, 96)
    -> permute(0, 1, 3, 2, 4, 5)
    -> reshape(64, 49, 96)

1. **Simulate the chain with atoms.** Start with one `_Atom` per non-unit source
   axis: `[A(56), B(56), C(96)]` (unit axes are represented as empty lists
   `[]`).
   * `view(1, 8, 7, 8, 7, 96)` splits `A(56)` into `A1(8), A2(7)` and `B(56)`
     into `B1(8), B2(7)`.
   * `permute(0, 1, 3, 2, 4, 5)` reorders the axes to
     `[[], [A1(8)], [B1(8)], [A2(7)], [B2(7)], [C(96)]]`.
   * `reshape(64, 49, 96)` regroups them into
     `[[A1(8), B1(8)], [A2(7), B2(7)], [C(96)]]`.
   The final flattened atom order is `[A1, B1, A2, B2, C]`.

2. **Split source axes where separated by the permutation.** Because `A1` and
   `A2` are no longer adjacent in the output (and likewise `B1` and `B2`), they
   must be separate axes before the permute:
   `[[], [A1(8)], [A2(7)], [B1(8)], [B2(7)], [C(96)]]` (rank 6).

3. **Merge adjacent axes down to `max_rank`.** Two adjacent pre-permute axes can
   be merged into one axis whenever their atoms are also adjacent, in the same
   order, in the output:
   * Absorb the leading unit axis `[]` into `[A1(8)]` (rank 5).
   * `B2(7)` and `C(96)` are adjacent in both source and output, so merge them
     into `[B2(7), C(96)]` of size `7 * 96 = 672` (rank 4).
   This gives `pre_shape = (8, 7, 8, 672)` and `perm = [0, 2, 1, 3]`, followed
   by `reshape(64, 49, 96)`.

Scope
-----
Static shapes only. Any chain containing a symbolic dimension is left unchanged.
Trailing `clone`/`contiguous` ops at the end of a chain are preserved so that
downstream ops that require a contiguous buffer still receive one.
"""

import collections
from collections.abc import Callable
import math

from litert_torch import fx_infra
import torch

aten = torch.ops.aten

# Debug-info meta fields that map a node back to its source model code.
_PROVENANCE_META_KEYS = (
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "from_node",
)

_RESHAPE_TARGETS = (
    aten.reshape.default,
    aten.view.default,
    aten._unsafe_view.default,
)
_PERMUTE_TARGETS = (aten.permute.default,)
_TRANSPOSE_TARGETS = (aten.transpose.int,)
_UNSQUEEZE_TARGETS = (aten.unsqueeze.default,)
_SQUEEZE_TARGETS = (
    aten.squeeze.default,
    aten.squeeze.dim,
    aten.squeeze.dims,
)
# Decomposing a reshape of a non-contiguous tensor inserts a contiguous copy
# before the view. A copy in the middle of a chain can be absorbed; a copy at
# the end of a chain may be required by downstream view ops and is kept in
# place (see `_rewrite_chain`).
_COPY_TARGETS = (
    aten.clone.default,
    aten.contiguous.default,
)

_VIEW_TARGETS = frozenset(
    _RESHAPE_TARGETS
    + _PERMUTE_TARGETS
    + _TRANSPOSE_TARGETS
    + _UNSQUEEZE_TARGETS
    + _SQUEEZE_TARGETS
    + _COPY_TARGETS
)


def _static_shape(val) -> tuple[int, ...] | None:
  """Returns `val`'s shape if all dimensions are positive concrete ints.

  `torch.SymInt` is not an `int` subclass, so the `isinstance(dim, int)` check
  rejects symbolic dimensions before any comparison `dim <= 0` that could
  install a shape guard.
  """
  if val is None or not hasattr(val, "shape"):
    return None
  shape = tuple(val.shape)
  if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
    return None
  return shape


class _Atom:
  """A factor of a source tensor dimension tracked through view operations.

  Each non-unit source dimension starts as a single `_Atom`. If a later reshape
  splits that dimension across two target axes, `split()` subdivides the atom
  into `(head, tail)` children so `leaves()` can recover all pieces of the
  original source dimension in source element order.

  Atoms are compared by object identity because two distinct dimensions may have
  the same numerical size.
  """

  __slots__ = ("size", "_children")

  def __init__(self, size: int):
    self.size = size
    self._children: tuple[_Atom, _Atom] | None = None

  def split(self, head_size: int) -> tuple["_Atom", "_Atom"]:
    """Subdivides this atom into two child atoms of size `head_size` and the remainder."""
    head = _Atom(head_size)
    tail = _Atom(self.size // head_size)
    self._children = (head, tail)
    return head, tail

  def leaves(self) -> list["_Atom"]:
    """Returns the leaf atoms this atom was split into, in source order."""
    if self._children is None:
      return [self]
    head, tail = self._children
    return head.leaves() + tail.leaves()


def _axis_size(axis: list[_Atom]) -> int:
  """Returns the dimension size of `axis` (1 for an empty/unit axis)."""
  return math.prod(atom.size for atom in axis)


def _take_axis(
    queue: collections.deque[_Atom], target_dim: int
) -> list[_Atom] | None:
  """Pulls atoms from the front of `queue` whose sizes multiply to `target_dim`."""
  axis: list[_Atom] = []
  remaining = target_dim
  while remaining > 1:
    if not queue:
      return None
    atom = queue.popleft()
    if atom.size <= remaining:
      if remaining % atom.size != 0:
        return None
      axis.append(atom)
      remaining //= atom.size
    else:
      # The target axis ends partway through `atom`: split `atom` and push the
      # unused tail back onto the front of the queue for the next axis.
      if atom.size % remaining != 0:
        return None
      head, tail = atom.split(remaining)
      axis.append(head)
      queue.appendleft(tail)
      remaining = 1
  return axis


def _reshape_axes(
    axes: list[list[_Atom]], target_shape: tuple[int, ...]
) -> list[list[_Atom]] | None:
  """Re-groups the flattened atoms of `axes` to match `target_shape`."""
  queue = collections.deque(atom for axis in axes for atom in axis)
  new_axes: list[list[_Atom]] = []
  for dim in target_shape:
    axis = _take_axis(queue, dim)
    if axis is None:
      return None
    new_axes.append(axis)
  if queue:
    return None
  return new_axes


def _simulate_chain(
    chain: list[torch.fx.Node],
    shapes: list[tuple[int, ...]],
    initial_axes: list[list[_Atom]],
) -> list[list[_Atom]] | None:
  """Simulates `chain` on `initial_axes` and returns the final axis grouping."""
  axes = list(initial_axes)
  for node, shape in zip(chain, shapes):
    rank = len(axes)
    if node.target in _PERMUTE_TARGETS:
      perm = [p % rank for p in node.args[1]]
      if sorted(perm) != list(range(rank)):
        return None
      axes = [axes[p] for p in perm]
    elif node.target in _TRANSPOSE_TARGETS:
      dim0 = node.args[1] % rank
      dim1 = node.args[2] % rank
      axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    else:
      axes = _reshape_axes(axes, shape)
      if axes is None:
        return None
  return axes


def _group_contiguous_runs(
    atoms: list[_Atom], are_adjacent: Callable[[_Atom, _Atom], bool]
) -> list[list[_Atom]]:
  """Groups `atoms` into maximal runs where consecutive atoms satisfy `are_adjacent`."""
  runs = [[atoms[0]]]
  for atom in atoms[1:]:
    if are_adjacent(runs[-1][-1], atom):
      runs[-1].append(atom)
    else:
      runs.append([atom])
  return runs


def _can_merge_axes(
    left: list[_Atom], right: list[_Atom], out_pos: dict[_Atom, int]
) -> bool:
  """Returns True if two adjacent source axes stay adjacent in the output."""
  if not left or not right:
    # A unit axis carries no elements and can be merged with any neighbor.
    return True
  return out_pos[right[0]] == out_pos[left[-1]] + 1


def _merge_first_adjacent_pair(
    axes: list[list[_Atom]], out_pos: dict[_Atom, int]
) -> bool:
  """Merges the first mergeable pair of adjacent axes in `axes` in place."""
  for i in range(len(axes) - 1):
    if _can_merge_axes(axes[i], axes[i + 1], out_pos):
      axes[i] = axes[i] + axes[i + 1]
      del axes[i + 1]
      return True
  return False


def _plan_rewrite(
    chain: list[torch.fx.Node],
    shapes: list[tuple[int, ...]],
    src_shape: tuple[int, ...],
    max_rank: int,
) -> tuple[tuple[int, ...], list[int]] | None:
  """Plans a `(pre_shape, perm)` rewrite of `chain` within `max_rank`.

  Returns:
    `(pre_shape, perm)`, where `pre_shape` is the shape to reshape the source
    into before permuting with `perm`, or None if no rewrite within `max_rank`
    exists.
  """
  # 1. Create one root atom per non-unit source dimension and simulate the
  #    chain to see how source dimensions get split and reordered.
  src_roots = [_Atom(dim) if dim > 1 else None for dim in src_shape]
  initial_axes = [[root] if root is not None else [] for root in src_roots]
  out_axes = _simulate_chain(chain, shapes, initial_axes)
  if out_axes is None:
    return None

  out_atoms = [atom for axis in out_axes for atom in axis]
  out_pos = {atom: i for i, atom in enumerate(out_atoms)}

  def are_adjacent_in_output(first: _Atom, second: _Atom) -> bool:
    return out_pos[second] == out_pos[first] + 1

  # 2. Build the initial pre-permute axes from the source dimensions. If a
  #    source dimension was split into atoms that are no longer adjacent in the
  #    output, those pieces must stay separate axes; pieces that stayed
  #    adjacent in the output can remain merged as one axis.
  axes: list[list[_Atom]] = []
  for root in src_roots:
    if root is None:
      axes.append([])
    else:
      axes.extend(_group_contiguous_runs(root.leaves(), are_adjacent_in_output))

  # 3. If the axis count still exceeds `max_rank`, merge adjacent source axes
  #    that also stay adjacent in the output until we fit within `max_rank`.
  while len(axes) > max_rank:
    if not _merge_first_adjacent_pair(axes, out_pos):
      return None

  # 4. Compute the permutation that reorders `axes` into output order. Unit
  #    axes carry no elements; placing them at the end keeps `perm` a valid
  #    permutation and the trailing reshape restores any output unit axes.
  non_unit_axes = [i for i, axis in enumerate(axes) if axis]
  unit_axes = [i for i, axis in enumerate(axes) if not axis]
  non_unit_axes.sort(key=lambda i: out_pos[axes[i][0]])
  perm = non_unit_axes + unit_axes

  pre_shape = tuple(_axis_size(axis) for axis in axes)
  return pre_shape, perm


def _copy_provenance_meta(src: torch.fx.Node, dst: torch.fx.Node):
  for key in _PROVENANCE_META_KEYS:
    if key in src.meta:
      dst.meta[key] = src.meta[key]


def _emit_view_op(
    graph: torch.fx.Graph,
    template_node: torch.fx.Node,
    target,
    input_node: torch.fx.Node,
    arg,
) -> torch.fx.Node:
  """Inserts `target(input_node, arg)` before `template_node` with metadata."""
  input_val = input_node.meta["val"]
  with input_val.fake_mode:
    out_val = target(input_val, arg)
  with graph.inserting_before(template_node):
    node = graph.call_function(target, (input_node, arg))
  node.meta["val"] = out_val
  _copy_provenance_meta(template_node, node)
  return node


def _is_view_node(node) -> bool:
  return (
      isinstance(node, torch.fx.Node)
      and node.op == "call_function"
      and node.target in _VIEW_TARGETS
  )


def _collect_chain(node: torch.fx.Node) -> list[torch.fx.Node]:
  """Extends a chain forward while each node feeds into a single view op."""
  chain = [node]
  while len(chain[-1].users) == 1:
    user = next(iter(chain[-1].users))
    if not _is_view_node(user):
      break
    chain.append(user)
  return chain


def _rewrite_chain(
    graph_module: torch.fx.GraphModule,
    chain: list[torch.fx.Node],
    max_rank: int,
) -> bool:
  """Re-emits `chain` at lower rank. Returns True if the graph was rewritten."""
  # Keep any trailing copy ops outside the rewrite so downstream ops that rely
  # on a contiguous layout still get one.
  while chain and chain[-1].target in _COPY_TARGETS:
    chain = chain[:-1]
  if not chain:
    return False

  src = chain[0].args[0]
  if not isinstance(src, torch.fx.Node):
    return False
  src_shape = _static_shape(src.meta.get("val"))
  last_shape = _static_shape(chain[-1].meta.get("val"))
  if src_shape is None or last_shape is None:
    return False

  shapes = [_static_shape(node.meta.get("val")) for node in chain]
  if any(shape is None for shape in shapes):
    return False

  old_peak_rank = max([len(src_shape)] + [len(shape) for shape in shapes])
  if old_peak_rank <= max_rank:
    return False
  if len(src_shape) > max_rank or len(last_shape) > max_rank:
    # The chain's endpoints already exceed `max_rank`, so rewriting only the
    # interior cannot bring the chain within the cap.
    return False

  plan = _plan_rewrite(chain, shapes, src_shape, max_rank)
  if plan is None:
    return False
  pre_shape, perm = plan

  new_peak_rank = max(len(src_shape), len(pre_shape), len(last_shape))
  if new_peak_rank >= old_peak_rank:
    return False

  perm_shape = tuple(pre_shape[i] for i in perm)
  need_pre_reshape = pre_shape != src_shape
  need_permute = perm != list(range(len(perm)))
  need_post_reshape = perm_shape != last_shape
  if not (need_pre_reshape or need_permute or need_post_reshape):
    return False

  graph = graph_module.graph
  last = chain[-1]
  current = src
  if need_pre_reshape:
    current = _emit_view_op(
        graph, last, aten.reshape.default, current, list(pre_shape)
    )
  if need_permute:
    current = _emit_view_op(graph, last, aten.permute.default, current, perm)
  if need_post_reshape:
    current = _emit_view_op(
        graph, last, aten.reshape.default, current, list(last_shape)
    )

  current.meta["val"] = last.meta["val"]
  if "tensor_meta" in last.meta:
    current.meta["tensor_meta"] = last.meta["tensor_meta"]

  last.replace_all_uses_with(current)
  return True


class ReduceViewRankPass(fx_infra.PassBase):
  """Re-emits pure view chains at a rank the GPU delegate accepts."""

  def __init__(self, max_rank: int = 4):
    super().__init__()
    self._max_rank = max_rank

  def call(self, graph_module: torch.fx.GraphModule):
    modified = False
    visited = set()
    for node in list(graph_module.graph.nodes):
      if node in visited or not _is_view_node(node):
        continue
      chain = _collect_chain(node)
      visited.update(chain)
      if _rewrite_chain(graph_module, chain, self._max_rank):
        modified = True

    if modified:
      graph_module.graph.eliminate_dead_code()
      graph_module.graph.lint()
      graph_module.recompile()
    return fx_infra.PassResult(graph_module, modified)
