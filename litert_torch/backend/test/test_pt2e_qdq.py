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
import re

from litert_torch import backend
import torch

from absl.testing import absltest as googletest


class TestPT2EExport(googletest.TestCase):

  def test_per_tensor_qdq(self):

    class TestModel(torch.nn.Module):

      def forward(self, x):
        x = torch.ops.quantized_decomposed.quantize_per_tensor(
            x, 0.4, 2, -128, 127, torch.int8
        )
        x = torch.ops.quantized_decomposed.dequantize_per_tensor(
            x, 0.4, 2, -128, 127, torch.int8
        )
        return x

    x = torch.randn(2, 3, 4, 5)
    ep = torch.export.export(TestModel().eval(), (x,))
    mlir = backend.export.exported_program_to_mlir(ep)
    mlir_text = mlir.get_text()

    self.assertIn(
        "tensor<2x3x4x5x!quant.uniform<i8:f32, 4.000000e-01:2>>", mlir_text
    )
    self.assertEqual(mlir_text.count("odml_torch.uniform_quantize"), 1)
    self.assertEqual(mlir_text.count("odml_torch.uniform_dequantize"), 1)

  def test_per_channel_qdq(self):
    class TestModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.scale = torch.tensor([3.2, 5.3, 0.1, 10.1])
        self.zero_point = torch.tensor([1, 2, -1, -2], dtype=torch.int64)

      def forward(self, x):
        x = torch.ops.quantized_decomposed.quantize_per_channel(
            x, self.scale, self.zero_point, 2, -100, 100, torch.int8
        )
        x = torch.ops.quantized_decomposed.dequantize_per_channel(
            x, self.scale, self.zero_point, 2, -100, 100, torch.int8
        )
        return x

    x = torch.randn(2, 3, 4, 5)
    ep = torch.export.export(TestModel().eval(), (x,))
    mlir = backend.export.exported_program_to_mlir(ep)
    mlir_text = mlir.get_text()
    print(mlir_text)
    self.assertTrue(
        re.search(
            r"tensor<2x3x4x5x!quant.uniform<i8<-100:100>:f32:2,\s*{3.2\d*:1,5.3\d*:2,0.1\d*:-1,10.1\d*:-2}>>",
            mlir_text,
        )
    )
    self.assertEqual(mlir_text.count("odml_torch.uniform_quantize"), 1)
    self.assertEqual(mlir_text.count("odml_torch.uniform_dequantize"), 1)


if __name__ == "__main__":
  googletest.main()
