# Copyright 2024 The LiteRT Torch Authors.
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
"""ExportedProgram.run_decompositions wrapper to handle unexpected export behavior."""

import operator
from typing import Any, Callable
from litert_torch import progress
import torch


# A dummy decomp table for running ExportedProgram.run_decompositions without
# any op decompositions but just aot_export_module. Due to the check in
# run_decompositions, if None or an empty dict is passed as decomp_table,
# it will run the default aten-coreaten decompositions. Therefore a non-empty
# dummy decomp table is needed.
# Ref: https://github.com/pytorch/pytorch/blob/db895ace1d36726e64781774f53b3d3098206116/torch/export/exported_program.py#L543
_DUMMY_DECOMP_TABLE = {
    torch._ops.OperatorBase(): lambda: None,
}

_BUILTIN_OPERATORS = {
    getattr(operator, name)
    for name in dir(operator)
    if not name.startswith("_")
}


def _require_decomp(
    exported_program: torch.export.ExportedProgram, decomp_table
):
  """Checks if the exported program requires decompositions."""
  for node in exported_program.graph.nodes:
    if "call_" not in str(node.op):
      continue

    op = node.target
    if isinstance(op, torch._ops.OpOverloadPacket):
      op = op.default

    if op in decomp_table:
      return True

    if (
        not isinstance(op, (torch._ops.OpOverload, torch._ops.OperatorBase))
        and op not in _BUILTIN_OPERATORS
    ):
      # Python function that requires to be retraced via run_decompositions.
      return True

  return False


_FORCE_DECOMP_ATTR = "_litert_torch_force_decomp"


def annotate_force_decomp(decomp: Callable[..., Any]):
  """Annotates a decomp to force it to be run (at least shallowly) in safe_run_decompositions."""
  setattr(decomp, _FORCE_DECOMP_ATTR, _FORCE_DECOMP_ATTR)
  return decomp


def _erase_assert_tensor_metadata_nodes(exported_program):
  """Removes aten._assert_tensor_metadata nodes, which are not convertible."""
  if not hasattr(torch.ops.aten, "_assert_tensor_metadata"):
    return
  for node in list(exported_program.graph.nodes):
    if (
        node.target == torch.ops.aten._assert_tensor_metadata.default
        and not node.users
    ):
      exported_program.graph.erase_node(node)


@progress.task("ExportedProgram Run Decompositions")
def safe_run_decompositions(exported_program, decomp_table=None, can_skip=True):
  """Wrapper for ExportedProgram.run_decompositions to handle unexpected export behavior."""

  if decomp_table is not None and not decomp_table:
    # Empty decomp table means no op decompositions. Use dummy decomp table
    # instead for backward compatibility.
    decomp_table = _DUMMY_DECOMP_TABLE

  if can_skip and not _require_decomp(exported_program, decomp_table):
    return exported_program

  for node in exported_program.graph.nodes:
    if node.target == torch.ops.aten.view.default:
      # Passes or torch.export may generate aten.view nodes not respecting the
      # tensor memory format. Changes all the aten.view to torch.reshape
      # for retracing. If the input memory format is already contiguous,
      # retracing in run_decomposition below would decompose torch.reshape
      # back to one aten.view.
      node.target = lambda self, size: torch.reshape(self.contiguous(), size)

    # Torch may skip some decompositions even if target is in decomp_table.
    # The following ensures the target is always run through the decompositions
    # shallowly if it has _FORCE_DECOMP_ATTR.
    if decomp_table and node.target in decomp_table:
      decomp = decomp_table[node.target]
      if hasattr(decomp, _FORCE_DECOMP_ATTR):
        node.target = decomp

  # Strip the asserts before retracing as well: run_decompositions can choke on
  # them when the traced dtypes differ from the recorded metadata.
  _erase_assert_tensor_metadata_nodes(exported_program)
  exported_program.graph_module.recompile()

  exported_program = exported_program.run_decompositions(decomp_table)

  _erase_assert_tensor_metadata_nodes(exported_program)

  exported_program.graph.eliminate_dead_code()
  exported_program.graph_module.recompile()

  return exported_program
