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
import transformers
from transformers.models.gemma4 import modeling_gemma4
from transformers.models.qwen3_moe import modeling_qwen3_moe


def _model_to_mlir(model, args):
  ep = torch.export.export(model, args)
  mlir = backend.export.exported_program_to_mlir(ep)
  return mlir.get_text()


def _init_experts(experts):
  experts.config._experts_implementation = "eager"  # pylint: disable=protected-access
  with torch.no_grad():
    experts.gate_up_proj.normal_()
    experts.down_proj.normal_()
  return experts


def _route(num_tokens, num_experts, num_active_experts):
  scores = torch.randn(num_tokens, num_experts).softmax(dim=-1)
  top_k_weights, top_k_index = torch.topk(scores, num_active_experts, dim=-1)
  return top_k_index, top_k_weights


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


class LitertMoeExpertsForwardTest(googletest.TestCase):

  def _assert_matches_eager(self, experts, num_active_experts):
    torch.manual_seed(0)
    hidden_states = torch.randn(6, experts.hidden_dim)
    top_k_index, top_k_weights = _route(
        6, experts.num_experts, num_active_experts
    )

    with torch.no_grad():
      expected_output = experts(hidden_states, top_k_index, top_k_weights)
      actual_output = moe.litert_moe_experts_forward(
          experts, hidden_states, top_k_index, top_k_weights
      )

    torch.testing.assert_close(
        actual_output, expected_output, rtol=1e-4, atol=1e-4
    )

  def test_reads_num_experts_per_tok(self):
    # Qwen3-MoE (like most MoE configs in transformers) has no
    # `top_k_experts`.
    config = transformers.Qwen3MoeConfig(
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="gelu_pytorch_tanh",
    )
    torch.manual_seed(0)
    experts = _init_experts(modeling_qwen3_moe.Qwen3MoeExperts(config))
    self.assertFalse(hasattr(config, "top_k_experts"))

    self._assert_matches_eager(experts, num_active_experts=2)

  def test_top_k_experts_takes_precedence(self):
    config = modeling_gemma4.Gemma4TextConfig(
        hidden_size=16, moe_intermediate_size=8, num_experts=4, top_k_experts=2
    )
    # A disagreeing `num_experts_per_tok` must not change the Gemma 4 path.
    config.num_experts_per_tok = 3
    torch.manual_seed(0)
    experts = _init_experts(modeling_gemma4.Gemma4TextExperts(config))

    self._assert_matches_eager(experts, num_active_experts=2)

  def test_missing_num_active_experts_raises(self):
    config = transformers.Qwen3MoeConfig(
        hidden_size=16,
        moe_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
    )
    experts = modeling_qwen3_moe.Qwen3MoeExperts(config)
    # A config that names the count neither way.
    experts.config = transformers.PretrainedConfig()
    top_k_index, top_k_weights = _route(2, 4, 2)

    with self.assertRaisesRegex(ValueError, "num_experts_per_tok"):
      moe.litert_moe_experts_forward(
          experts, torch.randn(2, 16), top_k_index, top_k_weights
      )


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
