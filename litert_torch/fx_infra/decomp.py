# Copyright 2025 The LiteRT Torch Authors.
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
import torch

# Decompositions to be run after torch.export before conversion and any
# passes. Remove ops from this decomposition table if they need to be preserved
# in passes.
_pre_convert_decomp = torch._decomp.core_aten_decompositions().copy()


# Decompositions to be run after conversion before backend passes and
# lowerings.
_pre_lower_decomp = torch._decomp.core_aten_decompositions().copy()


def _get_ops(op):
  if isinstance(op, torch._ops.OpOverloadPacket):
    return [getattr(op, overload) for overload in op.overloads()]
  else:
    return [op]


def pre_convert_decomp():
  """Decompositions to be run after torch.export before conversion and any passes."""
  return _pre_convert_decomp.copy()


def pre_lower_decomp():
  """Decompositions to be run after conversion before backend passes and lowerings."""

  return _pre_lower_decomp.copy()


def remove_pre_lower_decomp(op):
  # Also remove from pre_convert_decomp which always run before pre_lower_
  # decomp.
  remove_pre_convert_decomp(op)

  for op_ in _get_ops(op):
    _pre_lower_decomp.pop(op_, None)


def remove_pre_convert_decomp(op):
  for op_ in _get_ops(op):
    _pre_convert_decomp.pop(op_, None)


def add_pre_convert_decomp(op, decomp):
  # Also add decomp to pre_lower_decomp which runs after pre_convert_decomp.
  add_pre_lower_decomp(op, decomp)

  for op_ in _get_ops(op):
    _pre_convert_decomp[op_] = decomp


def add_pre_lower_decomp(op, decomp):
  for op_ in _get_ops(op):
    _pre_lower_decomp[op_] = decomp


def update_pre_convert_decomp(decomps):
  for op, decomp in decomps.items():
    add_pre_convert_decomp(op, decomp)


def update_pre_lower_decomp(decomps):
  for op, decomp in decomps.items():
    add_pre_lower_decomp(op, decomp)
