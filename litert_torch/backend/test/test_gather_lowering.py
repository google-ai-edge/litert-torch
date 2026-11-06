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
"""Tests that index_select / embedding gathers lower without OOB machinery.

`jnp.take`'s default `mode="fill"` emits NumPy-compat out-of-bounds handling
(compare/select/reduce_all around the gather). aten.index_select and
aten.embedding require in-range indices, so the lowerings use `mode="clip"`,
which is StableHLO gather's native semantics and adds no extra TFLite ops.
"""

import litert_torch
from litert_torch import testing
import numpy as np
import torch

from absl.testing import absltest as googletest

# Ops emitted by jnp.take's mode="fill" out-of-bounds handling that must not
# appear around a plain gather.
_OOB_OPS = frozenset({"REDUCE_ALL", "SELECT_V2", "LOGICAL_AND"})


class IndexSelect(torch.nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x, index):
    return torch.index_select(x, self.dim, index)


class Embedding(torch.nn.Module):

  def __init__(self, padding_idx=None):
    super().__init__()
    self.embedding = torch.nn.Embedding(10, 4, padding_idx=padding_idx)

  def forward(self, index):
    return self.embedding(index)


def _op_names(edge_model) -> list[str]:
  interpreter = edge_model._get_interpreter()  # pylint: disable=protected-access
  return [op["op_name"] for op in interpreter._get_ops_details()]  # pylint: disable=protected-access


@testing.parameterized_class(testing.V1_V2_PARAMETERS)
class TestGatherLowering(testing.V1V2TestCase):
  """index_select / embedding convert to a bare gather with matching numerics."""

  def _assert_lowers_without_oob_ops(self, model, args):
    edge_model = litert_torch.convert(model.eval(), args)
    op_names = _op_names(edge_model)
    self.assertEmpty(
        _OOB_OPS.intersection(op_names),
        f"Unexpected out-of-bounds handling ops in {op_names}",
    )
    # The converter downcasts int64 inputs to int32, so feed the TFLite model
    # int32 copies of any int64 args.
    tfl_args = tuple(
        a.to(torch.int32) if a.dtype == torch.int64 else a for a in args
    )
    np.testing.assert_allclose(
        edge_model(*tfl_args), model(*args).detach().numpy(), atol=1e-5
    )

  def test_index_select_int32_index(self):
    x = torch.randn(2, 16, 8)
    index = torch.tensor([3, 0, 15, 7], dtype=torch.int32)
    self._assert_lowers_without_oob_ops(IndexSelect(dim=1), (x, index))

  def test_index_select_int64_index(self):
    x = torch.randn(2, 16, 8)
    index = torch.tensor([3, 0, 15, 7], dtype=torch.int64)
    self._assert_lowers_without_oob_ops(IndexSelect(dim=1), (x, index))

  def test_embedding(self):
    index = torch.tensor([[1, 9, 0], [4, 4, 2]], dtype=torch.int64)
    self._assert_lowers_without_oob_ops(Embedding(), (index,))

  def test_embedding_with_padding_idx(self):
    index = torch.tensor([[1, 9, 0], [4, 4, 2]], dtype=torch.int64)
    self._assert_lowers_without_oob_ops(Embedding(padding_idx=0), (index,))


if __name__ == "__main__":
  googletest.main()
