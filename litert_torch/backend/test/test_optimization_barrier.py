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
from litert_torch import backend
from litert_torch.backend import optimization_barrier as optimization_barrier_lib  # Import to register the op.
import torch

from absl.testing import absltest as googletest

optimization_barrier = optimization_barrier_lib.optimization_barrier


class TestOptimizationBarrier(googletest.TestCase):
  """Test optimization barrier op implementation and lowering."""

  def test_applied_optimization_barrier_op(self):
    """Test optimization barrier op application and lowering."""

    class TestModel(torch.nn.Module):

      def forward(self, x, y):
        x, _ = optimization_barrier(x, y)
        return x

    x = torch.randn(1, 5)
    ep = torch.export.export(TestModel().eval(), (x, x))
    mlir = backend.export.exported_program_to_mlir(ep)
    mlir_text = mlir.get_text()
    self.assertEqual(
        mlir_text.count(
            "stablehlo.optimization_barrier %arg1, %arg1 : tensor<1x5xf32>,"
            " tensor<1x5xf32>"
        ),
        1,
    )

  def test_input_single_tensor(self):
    """Test optimization barrier with single tensor input."""
    x = torch.randn(1, 5)
    y = optimization_barrier(x)
    self.assertIsInstance(y, torch.Tensor)
    self.assertEqual(y.shape, (1, 5))

  def test_input_multiple_tensors(self):
    """Test optimization barrier with multiple tensors input."""
    x = torch.randn(1, 5)
    y = torch.randn(1, 6)
    z = optimization_barrier(x, y)
    self.assertIsInstance(z, tuple)
    self.assertLen(z, 2)
    self.assertIsInstance(z[0], torch.Tensor)
    self.assertIsInstance(z[1], torch.Tensor)
    self.assertEqual(z[0].shape, (1, 5))
    self.assertEqual(z[1].shape, (1, 6))

  def test_input_nested_tensors(self):
    """Test optimization barrier with nested tensor inputs."""
    x = {"foo": torch.randn(1, 5), "bar": torch.randn(1, 6)}
    z = optimization_barrier(x)
    self.assertIsInstance(z, dict)
    self.assertLen(z, 2)
    self.assertIsInstance(z["foo"], torch.Tensor)
    self.assertIsInstance(z["bar"], torch.Tensor)
    self.assertEqual(z["foo"].shape, (1, 5))
    self.assertEqual(z["bar"].shape, (1, 6))


if __name__ == "__main__":
  googletest.main()
