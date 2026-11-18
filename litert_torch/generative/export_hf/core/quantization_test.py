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
"""Tests for quantization recipes and utilities."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.core import quantization
from litert_torch.generative.layers import moe
import torch

from litert_quantizer import qtyping
from litert_quantizer import recipe as recipe_lib
from litert_quantizer import recipe_manager

_TFLOpName = recipe_lib.TFLOperationName
_QuantGranularity = recipe_lib.QuantGranularity
# Op output names from a Qwen3.5 `export_hf` model, with the Python module
# paths before each class name dropped.
_QWEN3_5_LAYER = (
    "Qwen3_5StaticModel_model/Qwen3_5StaticDecoderLayer_{layer}/{module}/"
    "torch.nn.modules.linear.Linear_{proj};1"
)
_QWEN3_5_MLP = "FusedQwen3_5MLP_mlp"
_QWEN3_5_GDN = "Qwen3_5StaticGatedDeltaNet_linear_attn"
_QWEN3_5_ATTN = "FusedQwen3_5Attention_self_attn"


class QuantizationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("none_obj", None),
      ("empty", ""),
      ("none_str", "none"),
      ("null_str", "null"),
      ("false_str", "false"),
      ("upper_none", "NONE"),
  )
  def test_maybe_quantize_model_disabled_returns_original_path(self, recipe):
    self.assertEqual(
        quantization.maybe_quantize_model("/tmp/model.tflite", recipe),
        "/tmp/model.tflite",
    )

  @parameterized.named_parameters(
      ("dynamic_wi8_emb4_afp32", "dynamic_wi8_emb4_afp32"),
      (
          "dynamic_fc_weight_only_conv_wi8_afp32",
          "dynamic_fc_weight_only_conv_wi8_afp32",
      ),
      ("dynamic_wi4b32_octav_afp32", "dynamic_wi4b32_octav_afp32"),
      ("gemma4_mixed48_b32_octav", "gemma4_mixed48_b32_octav"),
      ("qwen3_5_mixed48_b32", "qwen3_5_mixed48_b32"),
      ("qwen3_5_mixed48_h1024", "qwen3_5_mixed48_h1024"),
      ("dynamic_wi8_moe_afp32", "dynamic_wi8_moe_afp32"),
      ("dynamic_wi4_moe_afp32", "dynamic_wi4_moe_afp32"),
      ("dynamic_wi4b32_moe_afp32", "dynamic_wi4b32_moe_afp32"),
  )
  def test_local_quantization_recipes_are_valid(self, recipe_name):
    self.assertIn(recipe_name, quantization._LOCAL_QUANTIZATION_RECIPES)
    recipe = quantization._LOCAL_QUANTIZATION_RECIPES[recipe_name]()
    self.assertNotEmpty(recipe)
    rp_manager = recipe_manager.RecipeManager()
    rp_manager.load_quantization_recipe(recipe)
    self.assertNotEmpty(rp_manager.get_quantization_recipe())

  @parameterized.named_parameters(
      (
          "wi8_moe",
          "dynamic_wi8_moe_afp32",
          8,
          qtyping.QuantGranularity.CHANNELWISE,
      ),
      (
          "wi4_moe",
          "dynamic_wi4_moe_afp32",
          4,
          qtyping.QuantGranularity.CHANNELWISE,
      ),
      (
          "wi4b32_moe",
          "dynamic_wi4b32_moe_afp32",
          4,
          qtyping.QuantGranularity.BLOCKWISE_32,
      ),
  )
  def test_moe_recipes_quantize_moe_custom_op_model(
      self, recipe_name, expected_bits, expected_granularity
  ):
    rp_manager = recipe_manager.RecipeManager()
    rp_manager.load_quantization_recipe(
        quantization._LOCAL_QUANTIZATION_RECIPES[recipe_name]()
    )
    algo_key, op_cfg = rp_manager.get_quantization_configs(
        recipe_lib.TFLOperationName.CUSTOM_OP, "moe_layer"
    )
    self.assertEqual(algo_key, recipe_lib.AlgorithmName.MIN_MAX_UNIFORM_QUANT)
    self.assertIsNotNone(op_cfg.weight_tensor_config)
    assert op_cfg.weight_tensor_config is not None
    self.assertEqual(op_cfg.weight_tensor_config.num_bits, expected_bits)
    self.assertEqual(
        op_cfg.weight_tensor_config.granularity, expected_granularity
    )
    self.assertEqual(op_cfg.weight_tensor_config.quantized_dimensions, (0, 1))

    class _TinyMoeModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(32, 32, bias=False)
        self.register_buffer(
            "gate_w", moe.flatten_expert_weight(torch.randn(2, 64, 32))
        )
        self.register_buffer(
            "ff1_w", moe.flatten_expert_weight(torch.randn(2, 64, 32))
        )
        self.register_buffer(
            "down_w", moe.flatten_expert_weight(torch.randn(2, 32, 64))
        )
        self.register_buffer("expert_scale", torch.ones(1, 1, 1, 2))

      def forward(self, x, top_w, top_idx):
        h = self.proj(x)
        return moe.moe_experts(
            h,
            top_w,
            top_idx,
            self.gate_w,
            self.ff1_w,
            self.down_w,
            self.expert_scale,
            num_experts=2,
            num_active_experts=2,
            model_dim=32,
            hidden_dim=64,
        )

    torch.manual_seed(0)
    module = _TinyMoeModule().eval()
    sample_x = torch.randn(1, 2, 32)
    sample_w = torch.full((1, 2, 2), 0.5, dtype=torch.float32)
    sample_idx = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int32)
    edge_model = litert_torch.convert(module, (sample_x, sample_w, sample_idx))
    float_path = os.path.join(
        self.create_tempdir().full_path, f"tiny_{recipe_name}.tflite"
    )
    edge_model.export(float_path)
    quantized_path = quantization.quantize_model(float_path, recipe_name)
    self.assertTrue(os.path.exists(quantized_path))
    self.assertGreater(os.path.getsize(quantized_path), 0)

  def test_octav_recipes_configure_octav_algorithm_on_fc(self):
    wi4_octav = quantization._dynamic_wi4b32_octav_afp32()
    fc_rules = [
        r
        for r in wi4_octav
        if r.get("operation") == recipe_lib.TFLOperationName.FULLY_CONNECTED
    ]
    self.assertNotEmpty(fc_rules)
    self.assertEqual(
        fc_rules[-1]["algorithm_key"], recipe_lib.AlgorithmName.OCTAV
    )

    mixed48_octav = quantization._gemma4_mixed48_b32_octav()
    per_layer_rules = [
        r for r in mixed48_octav if r.get("regex") == "per_layer"
    ]
    self.assertLen(per_layer_rules, 1)
    self.assertEqual(
        per_layer_rules[0]["algorithm_key"], recipe_lib.AlgorithmName.OCTAV
    )
    self.assertEqual(
        per_layer_rules[0]["operation"],
        recipe_lib.TFLOperationName.FULLY_CONNECTED,
    )
    self.assertEqual(
        per_layer_rules[0]["op_config"]["weight_tensor_config"]["num_bits"], 8
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="embedding",
          op_name=_TFLOpName.EMBEDDING_LOOKUP,
          scope=(
              "Qwen3_5StaticModel_model/"
              "torch.nn.modules.sparse.Embedding_embed_tokens;1"
          ),
          num_bits=4,
          granularity=_QuantGranularity.BLOCKWISE_32,
      ),
      dict(
          testcase_name="lm_head",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope="decode_logits_output",
          num_bits=4,
          granularity=_QuantGranularity.BLOCKWISE_32,
      ),
      dict(
          testcase_name="down_proj_first_layer",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=0, module=_QWEN3_5_MLP, proj="down_proj"
          ),
          num_bits=8,
          granularity=_QuantGranularity.CHANNELWISE,
      ),
      dict(
          testcase_name="down_proj_last_layer",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=23, module=_QWEN3_5_MLP, proj="down_proj"
          ),
          num_bits=8,
          granularity=_QuantGranularity.CHANNELWISE,
      ),
      dict(
          testcase_name="gated_delta_net_out_proj",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=8, module=_QWEN3_5_GDN, proj="out_proj"
          ),
          num_bits=8,
          granularity=_QuantGranularity.CHANNELWISE,
      ),
      dict(
          testcase_name="attention_o_proj",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=3, module=_QWEN3_5_ATTN, proj="o_proj"
          ),
          num_bits=8,
          granularity=_QuantGranularity.CHANNELWISE,
      ),
      dict(
          testcase_name="mlp_gate_up_proj",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=0, module=_QWEN3_5_MLP, proj="gate_up_proj"
          ),
          num_bits=4,
          granularity=_QuantGranularity.BLOCKWISE_32,
      ),
      dict(
          testcase_name="gated_delta_net_in_proj_qkv",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=0, module=_QWEN3_5_GDN, proj="in_proj_qkv"
          ),
          num_bits=4,
          granularity=_QuantGranularity.BLOCKWISE_32,
      ),
      dict(
          testcase_name="attention_qkv_proj",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=3, module=_QWEN3_5_ATTN, proj="qkv_proj"
          ),
          num_bits=4,
          granularity=_QuantGranularity.BLOCKWISE_32,
      ),
  )
  def test_qwen3_5_mixed48_b32_weight_configs(
      self, op_name, scope, num_bits, granularity
  ):
    rp_manager = recipe_manager.RecipeManager()
    rp_manager.load_quantization_recipe(quantization._qwen3_5_mixed48_b32())

    _, op_config = rp_manager.get_quantization_configs(op_name, scope)

    weight_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_config)
    self.assertEqual(weight_config.num_bits, num_bits)
    self.assertEqual(weight_config.granularity, granularity)
    self.assertTrue(weight_config.symmetric)

  @parameterized.named_parameters(
      dict(
          testcase_name="embedding",
          op_name=_TFLOpName.EMBEDDING_LOOKUP,
          scope=(
              "Qwen3_5StaticModel_model/"
              "torch.nn.modules.sparse.Embedding_embed_tokens;1"
          ),
          num_bits=4,
          algorithm_key=recipe_lib.AlgorithmName.DECOMPOSED_HADAMARD_ROTATION,
      ),
      dict(
          testcase_name="lm_head",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope="decode_logits_output",
          num_bits=4,
          algorithm_key=recipe_lib.AlgorithmName.DECOMPOSED_HADAMARD_ROTATION,
      ),
      dict(
          testcase_name="down_proj_first_layer",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=0, module=_QWEN3_5_MLP, proj="down_proj"
          ),
          num_bits=8,
          algorithm_key=recipe_lib.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      ),
      dict(
          testcase_name="down_proj_last_layer",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=23, module=_QWEN3_5_MLP, proj="down_proj"
          ),
          num_bits=8,
          algorithm_key=recipe_lib.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      ),
      dict(
          testcase_name="gated_delta_net_out_proj",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=8, module=_QWEN3_5_GDN, proj="out_proj"
          ),
          num_bits=8,
          algorithm_key=recipe_lib.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      ),
      dict(
          testcase_name="attention_o_proj",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=3, module=_QWEN3_5_ATTN, proj="o_proj"
          ),
          num_bits=8,
          algorithm_key=recipe_lib.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      ),
      dict(
          testcase_name="mlp_gate_up_proj",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=0, module=_QWEN3_5_MLP, proj="gate_up_proj"
          ),
          num_bits=4,
          algorithm_key=recipe_lib.AlgorithmName.DECOMPOSED_HADAMARD_ROTATION,
      ),
      dict(
          testcase_name="gated_delta_net_in_proj_qkv",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=0, module=_QWEN3_5_GDN, proj="in_proj_qkv"
          ),
          num_bits=4,
          algorithm_key=recipe_lib.AlgorithmName.DECOMPOSED_HADAMARD_ROTATION,
      ),
      dict(
          testcase_name="gated_delta_net_in_proj_zba",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=0, module=_QWEN3_5_GDN, proj="in_proj_zba"
          ),
          num_bits=4,
          algorithm_key=recipe_lib.AlgorithmName.DECOMPOSED_HADAMARD_ROTATION,
      ),
      dict(
          testcase_name="attention_qkv_proj",
          op_name=_TFLOpName.FULLY_CONNECTED,
          scope=_QWEN3_5_LAYER.format(
              layer=3, module=_QWEN3_5_ATTN, proj="qkv_proj"
          ),
          num_bits=4,
          algorithm_key=recipe_lib.AlgorithmName.DECOMPOSED_HADAMARD_ROTATION,
      ),
  )
  def test_qwen3_5_mixed48_h1024_weight_configs(
      self, op_name, scope, num_bits, algorithm_key
  ):
    rp_manager = recipe_manager.RecipeManager()
    rp_manager.load_quantization_recipe(quantization._qwen3_5_mixed48_h1024())

    actual_algo, op_config = rp_manager.get_quantization_configs(op_name, scope)

    self.assertEqual(actual_algo, algorithm_key)
    weight_config = op_config.weight_tensor_config
    self.assertIsNotNone(weight_config)
    assert weight_config is not None
    self.assertEqual(weight_config.num_bits, num_bits)
    self.assertEqual(weight_config.granularity, _QuantGranularity.CHANNELWISE)
    self.assertTrue(weight_config.symmetric)
    if algorithm_key == recipe_lib.AlgorithmName.DECOMPOSED_HADAMARD_ROTATION:
      self.assertEqual(
          dict(weight_config.algorithm_params),
          {
              "max_hadamard_size": 1024,
              "q_init_mode": "none",
              "randomize": True,
              "sizing_mode": "max_power_of_two",
          },
      )
    else:
      self.assertEmpty(weight_config.algorithm_params)

  def test_qwen3_5_mixed48_h1024_two_pass_quantize_model(self):
    class _TinyBlock(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, 32)
        self.gate_up_proj = torch.nn.Linear(32, 64, bias=False)
        self.down_proj = torch.nn.Linear(64, 32, bias=False)

      def forward(self, ids):
        x = self.embed_tokens(ids)
        return self.down_proj(self.gate_up_proj(x))

    torch.manual_seed(0)
    module = _TinyBlock().eval()
    sample_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    edge_model = litert_torch.convert(module, (sample_ids,))
    float_path = os.path.join(
        self.create_tempdir().full_path, "tiny_qwen3_5_h1024.tflite"
    )
    edge_model.export(float_path)

    pass1_path = quantization.quantize_model(
        float_path, "qwen3_5_mixed48_h1024"
    )
    pass2_path = quantization.quantize_model(pass1_path, "dynamic_wi8_afp32")
    self.assertTrue(os.path.exists(pass2_path))
    self.assertGreater(os.path.getsize(pass2_path), 0)


if __name__ == "__main__":
  absltest.main()
