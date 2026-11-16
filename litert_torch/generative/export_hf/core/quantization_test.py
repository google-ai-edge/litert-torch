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

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.core import quantization

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
  )
  def test_local_quantization_recipes_are_valid(self, recipe_name):
    self.assertIn(recipe_name, quantization._LOCAL_QUANTIZATION_RECIPES)
    recipe = quantization._LOCAL_QUANTIZATION_RECIPES[recipe_name]()
    self.assertNotEmpty(recipe)
    rp_manager = recipe_manager.RecipeManager()
    rp_manager.load_quantization_recipe(recipe)
    self.assertNotEmpty(rp_manager.get_quantization_recipe())

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


if __name__ == "__main__":
  absltest.main()
