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
from litert_torch.generative.layers import scaled_dot_product_attention
import torch

from absl.testing import absltest as googletest


class ScaledDotProductAttentionTest(googletest.TestCase):

  def test_scaled_dot_product_attention(self):
    query = torch.randn(1, 16, 16, 128, dtype=torch.float32)
    key = torch.randn(1, 16, 16, 128, dtype=torch.float32)
    value = torch.randn(1, 16, 16, 128, dtype=torch.float32)
    mask = torch.ones((1, 1, 1, 16), dtype=torch.float32)
    output = scaled_dot_product_attention.scaled_dot_product_attention(
        query, key, value, head_size=128, mask=mask, scale=1.0, softcap=10.0
    )
    self.assertEqual(output.shape, (1, 16, 16, 128))

  def test_scaled_dot_product_attention_transposed(self):
    query = torch.randn(1, 16, 16, 128, dtype=torch.float32)
    key = torch.randn(1, 16, 16, 128, dtype=torch.float32)
    value = torch.randn(1, 16, 128, 16, dtype=torch.float32)
    mask = torch.ones((1, 1, 1, 16), dtype=torch.float32)
    output = (
        scaled_dot_product_attention.scaled_dot_product_attention_transposed(
            query, key, value, head_size=128, mask=mask, scale=1.0, softcap=10.0
        )
    )
    self.assertEqual(output.shape, (1, 16, 16, 128))

  def test_scaled_dot_product_attention_with_hlfb(self):
    query = torch.randn(1, 16, 16, 128, dtype=torch.float32)
    key = torch.randn(1, 16, 16, 128, dtype=torch.float32)
    value = torch.randn(1, 16, 16, 128, dtype=torch.float32)
    mask = torch.ones((1, 1, 1, 16), dtype=torch.float32)
    output = (
        scaled_dot_product_attention.scaled_dot_product_attention_with_hlfb(
            query, key, value, head_size=128, mask=mask, scale=1.0, softcap=10.0
        )
    )
    self.assertEqual(output.shape, (1, 16, 16, 128))

    def model_to_mlir(model, args):
      ep = torch.export.export(model, args)
      mlir = backend.export.exported_program_to_mlir(ep)
      return mlir.get_text()

    class SDPAModule(torch.nn.Module):

      def __init__(self):
        super().__init__()

      def forward(self, query, key, value, mask):
        return (
            scaled_dot_product_attention.scaled_dot_product_attention_with_hlfb(
                query,
                key,
                value,
                head_size=128,
                mask=mask,
                scale=1.0,
                softcap=10.0,
            )
        )

    ir_text = model_to_mlir(SDPAModule().eval(), (query, key, value, mask))
    self.assertEqual(ir_text.count("stablehlo.custom_call @mark_tensor"), 5)


if __name__ == "__main__":
  googletest.main()
