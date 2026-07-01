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

from absl.testing import absltest as googletest
from litert_torch import backend
from litert_torch.generative.layers import moe
import torch


def _model_to_mlir(model, args):
  ep = torch.export.export(model, args)
  mlir = backend.export.exported_program_to_mlir(ep)
  return mlir.get_text()


class MoeExpertsTest(googletest.TestCase):

  def test_moe_experts_fp32(self):
    torch.manual_seed(0)
    src = torch.randn(1, 3, 4)
    top_weights = torch.softmax(torch.randn(1, 3, 2), dim=-1)
    top_indices = torch.tensor([[[0, 1], [1, 0], [0, 1]]], dtype=torch.int32)
    ff_gate = moe.flatten_expert_weight(torch.randn(2, 6, 4))
    ff1 = moe.flatten_expert_weight(torch.randn(2, 6, 4))
    linear = moe.flatten_expert_weight(torch.randn(2, 4, 6))
    per_expert_scale = torch.ones(1, 1, 1, 2)

    output = moe.moe_experts(
        src,
        top_weights,
        top_indices,
        ff_gate,
        ff1,
        linear,
        per_expert_scale,
        num_experts=2,
        num_active_experts=2,
        model_dim=4,
        hidden_dim=6,
    )
    self.assertEqual(output.shape, src.shape)

  def test_moe_experts_int8(self):
    torch.manual_seed(0)
    src = torch.randn(1, 2, 4)
    top_weights = torch.softmax(torch.randn(1, 2, 2), dim=-1)
    top_indices = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int32)
    ff_gate = moe.flatten_expert_weight(
        torch.randint(-8, 8, (2, 6, 4), dtype=torch.int8)
    )
    ff1 = moe.flatten_expert_weight(
        torch.randint(-8, 8, (2, 6, 4), dtype=torch.int8)
    )
    linear = moe.flatten_expert_weight(
        torch.randint(-8, 8, (2, 4, 6), dtype=torch.int8)
    )
    ff_gate_scale = moe.flatten_expert_scale(torch.full((2, 6), 0.02))
    ff1_scale = moe.flatten_expert_scale(torch.full((2, 6), 0.03))
    linear_scale = moe.flatten_expert_scale(torch.full((2, 4), 0.04))
    per_expert_scale = torch.ones(1, 1, 1, 2)

    output = moe.moe_experts(
        src,
        top_weights,
        top_indices,
        ff_gate,
        ff1,
        linear,
        per_expert_scale,
        num_experts=2,
        num_active_experts=2,
        model_dim=4,
        hidden_dim=6,
        weight_type="int8",
        ff_gate_scale=ff_gate_scale,
        ff1_scale=ff1_scale,
        linear_scale=linear_scale,
    )
    self.assertEqual(output.shape, src.shape)

  def test_moe_experts_lowers_to_custom_op(self):

    class MoeModule(torch.nn.Module):

      def forward(
          self,
          src,
          top_weights,
          top_indices,
          ff_gate,
          ff1,
          linear,
          per_expert_scale,
      ):
        return moe.moe_experts(
            src,
            top_weights,
            top_indices,
            ff_gate,
            ff1,
            linear,
            per_expert_scale,
            num_experts=2,
            num_active_experts=2,
            model_dim=4,
            hidden_dim=6,
        )

    src = torch.randn(1, 2, 4)
    top_weights = torch.softmax(torch.randn(1, 2, 2), dim=-1)
    top_indices = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int32)
    ff_gate = moe.flatten_expert_weight(torch.randn(2, 6, 4))
    ff1 = moe.flatten_expert_weight(torch.randn(2, 6, 4))
    linear = moe.flatten_expert_weight(torch.randn(2, 4, 6))
    per_expert_scale = torch.ones(1, 1, 1, 2)

    ir_text = _model_to_mlir(
        MoeModule().eval(),
        (
            src,
            top_weights,
            top_indices,
            ff_gate,
            ff1,
            linear,
            per_expert_scale,
        ),
    )
    self.assertIn('"tfl.custom"', ir_text)
    self.assertIn('custom_code = "moe"', ir_text)
    self.assertNotIn("odml.moe_experts", ir_text)


if __name__ == "__main__":
  googletest.main()
