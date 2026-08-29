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
"""Tests for how strided/non-contiguous tensor access is lowered.

Non-contiguous access converts correctly for every stride pattern we support,
but it lowers into one of two very different shapes:

  * Rectangular access -- a permutation, a slice with a step, or a broadcast --
    lowers to a view-like TFL op (`TRANSPOSE`, `STRIDED_SLICE`, `BROADCAST_TO`)
    whose cost is independent of the tensor size.

  * Everything else -- `as_strided` with arbitrary strides, `diagonal`,
    `unfold` -- lowers through the JAX bridge's `_aten_as_strided`, which
    computes `flattened[ind]` where `ind` is materialized at trace time from
    the concrete sizes. That becomes a `GATHER_ND` against an index constant
    baked into the flatbuffer, at 4 bytes per output element.

The second form costs model size, costs latency, and is not supported by
XNNPACK, so it splits the graph into separate delegate partitions.

These tests pin the current behavior. If a change routes the general case onto
a view-like op, the lowering assertions here are expected to fail and should be
updated -- that is the intended direction.
"""

import litert_torch
from litert_torch.testing import model_coverage
import torch
from torch import nn

from absl.testing import absltest as googletest
from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import


_N = 128


class RectangularStride(nn.Module):
  """Reads every other column with a rectangular slice."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x[:, ::2] * 2.0 + 1.0)


class GeneralStride(nn.Module):
  """Reads the same elements via `as_strided`.

  `as_strided(x, (N, N // 2), (N, 2))` selects element `(i, j)` from flat
  offset `i * N + 2 * j`, i.e. exactly `x[i, 2 * j]`. The output is identical
  to `RectangularStride`; only the lowering differs.
  """

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    strided = torch.as_strided(x, (_N, _N // 2), (_N, 2))
    return torch.relu(strided * 2.0 + 1.0)


def _op_names(model_content: bytes) -> list[str]:
  """Returns the interpreter's node list, with XNNPACK applied."""
  interpreter = tfl_interpreter.Interpreter(model_content=model_content)
  interpreter.allocate_tensors()
  return [op["op_name"] for op in interpreter._get_ops_details()]  # pylint: disable=protected-access


def _delegate_partitions(model_content: bytes) -> int:
  """Counts the delegated subgraphs XNNPACK carved out of the model."""
  return _op_names(model_content).count("DELEGATE")


class TestStridedAccess(googletest.TestCase):
  """Tests lowering and delegation of strided tensor access."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    self.args = (torch.randn(_N, _N),)

  def test_both_paths_are_the_same_computation(self):
    """The two modules must agree in eager, or the comparison is meaningless."""
    with torch.no_grad():
      rectangular = RectangularStride().eval()(*self.args)
      general = GeneralStride().eval()(*self.args)
    self.assertTrue(torch.equal(rectangular, general))

  def test_rectangular_stride_numerics(self):
    """A rectangular slice converts and matches torch."""
    model = RectangularStride().eval()
    edge_model = litert_torch.convert(model, self.args)
    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, model, self.args)
    )

  def test_general_stride_numerics(self):
    """Arbitrary strides convert and match torch."""
    model = GeneralStride().eval()
    edge_model = litert_torch.convert(model, self.args)
    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, model, self.args)
    )

  def test_rectangular_stride_lowers_to_a_view_op(self):
    """A rectangular slice becomes `STRIDED_SLICE`, not a gather."""
    edge_model = litert_torch.convert(RectangularStride().eval(), self.args)
    op_names = _op_names(edge_model.model_content())
    self.assertIn("STRIDED_SLICE", op_names)
    self.assertNotIn("GATHER_ND", op_names)

  def test_general_stride_lowers_to_a_gather(self):
    """Arbitrary strides fall back to `GATHER_ND`.

    This documents a limitation rather than a desired property. Routing this
    case onto `tfl.strided_slice` would be an improvement, and would require
    updating this assertion.
    """
    edge_model = litert_torch.convert(GeneralStride().eval(), self.args)
    op_names = _op_names(edge_model.model_content())
    self.assertIn("GATHER_ND", op_names)

  def test_rectangular_stride_stays_in_one_xnnpack_partition(self):
    """A view-like op does not interrupt delegation."""
    edge_model = litert_torch.convert(RectangularStride().eval(), self.args)
    self.assertEqual(_delegate_partitions(edge_model.model_content()), 1)

  def test_general_stride_splits_the_xnnpack_partition(self):
    """XNNPACK has no `GATHER_ND` kernel, so the graph is cut in two.

    The gather runs on CPU between two delegated regions, which costs a
    delegate boundary crossing on top of the gather itself.
    """
    edge_model = litert_torch.convert(GeneralStride().eval(), self.args)
    self.assertEqual(_delegate_partitions(edge_model.model_content()), 2)

  def test_gather_index_constant_dominates_model_size(self):
    """The baked index tensor is 4 bytes per output element.

    The two models describe the same computation over the same data, so any
    size difference is the index constant.
    """
    rectangular = litert_torch.convert(RectangularStride().eval(), self.args)
    general = litert_torch.convert(GeneralStride().eval(), self.args)

    rectangular_size = len(rectangular.model_content())
    general_size = len(general.model_content())
    index_bytes = _N * (_N // 2) * 4

    self.assertGreater(general_size - rectangular_size, index_bytes * 0.9)


if __name__ == "__main__":
  googletest.main()
