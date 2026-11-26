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
"""Tests that aten.sigmoid converts with static and dynamic shapes."""

from unittest import mock

import litert_torch
import numpy as np
import torch

from absl.testing import absltest as googletest
from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import

try:
  from ai_edge_litert.litert_wrapper.compiled_model_wrapper.hardware_accelerator import HardwareAccelerator  # pylint: disable=g-import-not-at-top
except ImportError:
  from ai_edge_litert.compiled_model import HardwareAccelerator  # pylint: disable=g-import-not-at-top


class Sigmoid(torch.nn.Module):

  def forward(self, x):
    return torch.sigmoid(x)


class SiLUMLP(torch.nn.Module):
  """A Linear followed by SiLU, which decomposes to x * sigmoid(x)."""

  def __init__(self):
    super().__init__()
    self.fc = torch.nn.Linear(8, 16, bias=False)

  def forward(self, x):
    return torch.nn.functional.silu(self.fc(x))


def _op_names(edge_model) -> list[str]:
  interpreter = tfl_interpreter.Interpreter(
      model_content=edge_model.model_content()
  )
  return [op["op_name"] for op in interpreter._get_ops_details()]  # pylint: disable=protected-access


def _dynamic_seq_len():
  return ({1: torch.export.Dim("seq_len", min=2, max=64)},)


class TestSigmoidLowering(googletest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def _run(self, edge_model, x: torch.Tensor) -> np.ndarray:
    actual = edge_model(x.numpy())
    if not isinstance(actual, np.ndarray):
      self.fail(f"Expected a single output array, got {type(actual)}.")
    return actual

  def _assert_matches_torch(self, model, edge_model, seq_lens):
    for seq_len in seq_lens:
      x = torch.randn(1, seq_len, 8)
      expected = model(x).detach().numpy()
      actual = self._run(edge_model, x)
      self.assertEqual(actual.shape, expected.shape)
      np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  def test_static_shape(self):
    model = Sigmoid().eval()
    edge_model = litert_torch.convert(model, (torch.randn(1, 5, 8),))

    self.assertEqual(_op_names(edge_model), ["LOGISTIC"])
    self._assert_matches_torch(model, edge_model, [5])

  def test_dynamic_shape(self):
    model = Sigmoid().eval()
    edge_model = litert_torch.convert(
        model, (torch.randn(1, 5, 8),), dynamic_shapes=_dynamic_seq_len()
    )

    self.assertEqual(_op_names(edge_model), ["LOGISTIC"])
    self._assert_matches_torch(model, edge_model, [5, 17])

  def test_dynamic_shape_compiled_runtime(self):
    model = Sigmoid().eval()
    edge_model = litert_torch.convert(
        model, (torch.randn(1, 5, 8),), dynamic_shapes=_dynamic_seq_len()
    )
    edge_model.use_compiled_runtime(hardware_accel=HardwareAccelerator.CPU)

    self._assert_matches_torch(model, edge_model, [5, 17])

  def test_dynamic_shape_silu_mlp_with_tfl_decomps(self):
    # The full torch_tfl decomposition table (the one export_hf applies) lowers
    # the Linear and the multiply without the JAX bridge, which fails on
    # dynamic dimensions; the sigmoid of the SiLU is left to its aten lowering.
    self.enter_context(
        mock.patch.dict("os.environ", {"LITERT_TORCH_FULL_TFL_DECOMPS": "1"})
    )
    model = SiLUMLP().eval()
    edge_model = litert_torch.convert(
        model, (torch.randn(1, 5, 8),), dynamic_shapes=_dynamic_seq_len()
    )

    self.assertEqual(_op_names(edge_model).count("LOGISTIC"), 1)
    self._assert_matches_torch(model, edge_model, [5, 17])

  def test_integer_input_is_promoted_to_float(self):
    model = Sigmoid().eval()
    x = torch.randint(-4, 5, (1, 5, 8), dtype=torch.int32)
    edge_model = litert_torch.convert(model, (x,))

    expected = model(x).numpy()
    actual = self._run(edge_model, x)
    self.assertEqual(actual.dtype, np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  googletest.main()
