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

  def test_moe_experts_silu_matches_reference(self):
    torch.manual_seed(0)
    src = torch.randn(1, 2, 4)
    top_weights = torch.softmax(torch.randn(1, 2, 2), dim=-1)
    top_indices = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int32)
    gate_w = torch.randn(2, 6, 4)
    ff1_w = torch.randn(2, 6, 4)
    linear_w = torch.randn(2, 4, 6)
    output = moe.moe_experts(
        src,
        top_weights,
        top_indices,
        moe.flatten_expert_weight(gate_w),
        moe.flatten_expert_weight(ff1_w),
        moe.flatten_expert_weight(linear_w),
        torch.ones(1, 1, 1, 2),
        num_experts=2,
        num_active_experts=2,
        model_dim=4,
        hidden_dim=6,
        activation="silu",
    )
    expected = torch.zeros_like(src)
    for t in range(2):
      for k in range(2):
        e = int(top_indices[0, t, k].item())
        g = torch.nn.functional.linear(src[0, t], gate_w[e])
        u = torch.nn.functional.linear(src[0, t], ff1_w[e])
        out_e = torch.nn.functional.linear(
            torch.nn.functional.silu(g) * u, linear_w[e]
        )
        expected[0, t] += out_e * top_weights[0, t, k]
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)

  def test_litert_moe_experts_forward_swish_and_none_fallback(self):
    torch.manual_seed(1)

    class _Config:

      def __init__(self, hidden_act, hidden_activation):
        self.num_experts_per_tok = 2
        self.hidden_act = hidden_act
        self.hidden_activation = hidden_activation

    class _Experts(torch.nn.Module):

      def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = 2
        self.hidden_dim = 4
        self.intermediate_dim = 6
        self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 12, 4))
        self.down_proj = torch.nn.Parameter(torch.randn(2, 4, 6))

    x = torch.randn(1, 2, 4)
    idx = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int64)
    w = torch.softmax(torch.randn(1, 2, 2), dim=-1)

    swish_mod = _Experts(_Config(hidden_act="swish", hidden_activation=None))
    silu_mod = _Experts(_Config(hidden_act="silu", hidden_activation=None))
    silu_mod.load_state_dict(swish_mod.state_dict())
    out_swish = moe.litert_moe_experts_forward(swish_mod, x, idx, w)
    out_silu = moe.litert_moe_experts_forward(silu_mod, x, idx, w)
    torch.testing.assert_close(out_swish, out_silu, atol=1e-6, rtol=1e-6)

    none_mod = _Experts(_Config(hidden_act=None, hidden_activation=None))
    out_none = moe.litert_moe_experts_forward(none_mod, x, idx, w)
    self.assertEqual(out_none.shape, x.shape)


class BindPerExpertScaleTest(googletest.TestCase):

  class _Router(torch.nn.Module):

    def __init__(self, num_experts):
      super().__init__()
      self.per_expert_scale = torch.nn.Parameter(
          torch.arange(num_experts, dtype=torch.float32)
      )
      self.fold_per_expert_scale = True

  class _Experts(torch.nn.Module):

    def __init__(self, num_experts):
      super().__init__()
      self.num_experts = num_experts

  def _build_block(self, num_experts):
    block = torch.nn.Module()
    block.router = self._Router(num_experts)
    block.experts = self._Experts(num_experts)
    return block

  def test_unbound_experts_get_a_no_op_scale(self):
    block = self._build_block(3)

    self.assertTrue(
        torch.equal(
            moe._per_expert_scale_input(block.experts),  # pylint: disable=protected-access
            torch.ones(1, 1, 1, 3),
        )
    )

  def test_bind_per_expert_scale(self):
    block = self._build_block(3)

    moe.bind_per_expert_scale(block)

    self.assertFalse(block.router.fold_per_expert_scale)
    self.assertTrue(
        torch.equal(
            moe._per_expert_scale_input(block.experts),  # pylint: disable=protected-access
            torch.arange(3, dtype=torch.float32).reshape(1, 1, 1, 3),
        )
    )
    # The scale must not be re-registered on the experts module; that would
    # duplicate it in the state dict and emit it twice at export time.
    self.assertEmpty(dict(block.experts.named_parameters()))


if __name__ == "__main__":
  googletest.main()
