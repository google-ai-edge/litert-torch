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
"""Tests for aten.triu_indices and aten.tril_indices lowerings."""

import litert_torch
from litert_torch import backend
from litert_torch import testing
import numpy as np
import torch

from absl.testing import absltest as googletest
from absl.testing import parameterized


class DlrmInteraction(torch.nn.Module):
  """DLRM-style pairwise feature interaction using torch.triu_indices."""

  def __init__(self, offset: int = 1):
    super().__init__()
    self.offset = offset

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    z = torch.bmm(t, t.transpose(1, 2))
    iu = torch.triu_indices(t.shape[1], t.shape[1], offset=self.offset)
    return z[:, iu[0], iu[1]]


class TriluGather(torch.nn.Module):
  """Gathers matrix entries selected by triu_indices or tril_indices."""

  def __init__(
      self,
      fn,
      row: int,
      col: int,
      offset: int = 0,
      dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.fn = fn
    self.row = row
    self.col = col
    self.offset = offset
    self.dtype = dtype

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    kwargs = {} if self.dtype is None else {"dtype": self.dtype}
    idx = self.fn(self.row, self.col, offset=self.offset, **kwargs)
    return x[idx[0], idx[1]]


@testing.parameterized_class(testing.V1_V2_PARAMETERS)
class TestTriluIndicesConversion(testing.V1V2TestCase, parameterized.TestCase):
  """End-to-end V1/V2 conversion tests for triu_indices and tril_indices."""

  def test_dlrm_triu_indices_repro(self):
    torch.manual_seed(0)
    model = DlrmInteraction(offset=1).eval()
    sample_input = (torch.randn(1, 5, 16),)
    edge_model = litert_torch.convert(model, sample_input, enable_x64=False)
    expected = model(*sample_input).detach().numpy()
    actual = edge_model(*[x.numpy() for x in sample_input])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("triu_square_offset0", torch.triu_indices, 4, 4, 0, None),
      ("triu_square_offset1", torch.triu_indices, 5, 5, 1, None),
      ("triu_rect_wide_neg_offset", torch.triu_indices, 3, 5, -1, None),
      ("triu_rect_tall_pos_offset", torch.triu_indices, 5, 3, 1, torch.int32),
      ("tril_square_offset0", torch.tril_indices, 4, 4, 0, None),
      ("tril_square_neg_offset", torch.tril_indices, 5, 5, -1, None),
      ("tril_rect_wide_pos_offset", torch.tril_indices, 3, 5, 1, torch.int32),
      ("tril_rect_tall_neg_offset", torch.tril_indices, 5, 3, -2, torch.int64),
  )
  def test_trilu_indices_gather(self, fn, row, col, offset, dtype):
    torch.manual_seed(0)
    model = TriluGather(fn, row, col, offset, dtype).eval()
    sample_input = (torch.randn(row, col),)
    edge_model = litert_torch.convert(model, sample_input, enable_x64=False)
    expected = model(*sample_input).detach().numpy()
    actual = edge_model(*[x.numpy() for x in sample_input])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class TestTriluIndicesMlirLowering(parameterized.TestCase):
  """Verifies MLIR lowering emits a single stablehlo.constant."""

  @parameterized.named_parameters(
      (
          "triu_default_i64",
          torch.triu_indices,
          4,
          4,
          1,
          None,
          "tensor<2x6xi64>",
      ),
      (
          "triu_i32",
          torch.triu_indices,
          3,
          5,
          0,
          torch.int32,
          "tensor<2x12xi32>",
      ),
      (
          "tril_default_i64",
          torch.tril_indices,
          4,
          4,
          -1,
          None,
          "tensor<2x6xi64>",
      ),
      (
          "tril_i32",
          torch.tril_indices,
          5,
          3,
          0,
          torch.int32,
          "tensor<2x12xi32>",
      ),
      (
          "triu_empty_offset",
          torch.triu_indices,
          3,
          3,
          5,
          None,
          "tensor<2x0xi64>",
      ),
      (
          "triu_empty_zero_row",
          torch.triu_indices,
          0,
          4,
          0,
          torch.int32,
          "tensor<2x0xi32>",
      ),
      (
          "tril_empty_offset",
          torch.tril_indices,
          3,
          3,
          -5,
          None,
          "tensor<2x0xi64>",
      ),
      (
          "tril_empty_zero_col",
          torch.tril_indices,
          4,
          0,
          0,
          torch.int32,
          "tensor<2x0xi32>",
      ),
  )
  def test_lowers_to_stablehlo_constant(
      self, fn, row, col, offset, dtype, expected_type
  ):
    """Checks that lowering emits a single constant of expected shape/dtype."""

    class IndicesModule(torch.nn.Module):

      def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = {} if dtype is None else {"dtype": dtype}
        return x, fn(row, col, offset=offset, **kwargs)

    ep = torch.export.export(IndicesModule().eval(), (torch.zeros(1),))
    lowered = backend.export.exported_program_to_mlir(ep)
    ir_text = lowered.get_text()
    self.assertIn("stablehlo.constant", ir_text)
    self.assertIn(expected_type, ir_text)
    self.assertNotIn("stablehlo.iota", ir_text)
    self.assertNotIn("stablehlo.sqrt", ir_text)


if __name__ == "__main__":
  googletest.main()
