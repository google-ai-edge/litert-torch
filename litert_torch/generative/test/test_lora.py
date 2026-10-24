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

"""A suite of tests to validate LoRA utilities."""

from litert_torch.generative.layers import lora as lora_utils
import litert_torch.generative.layers.model_config as cfg
import torch
from absl.testing import absltest as googletest
from tensorflow.python.platform import resource_loader  # pylint: disable=g-direct-tensorflow-import


class TestLora(googletest.TestCase):
  """Tests for LoRA utilities."""

  def test_safetensors_builder(self):
    """Converts a safetensors file to a LoRA module."""

    tensor_names = lora_utils.LoRATensorNames(
        attn_query_w_a=(
            "base_model.model.model.layers.{}.self_attn.q_proj.lora_A.weight"
        ),
        attn_query_w_b=(
            "base_model.model.model.layers.{}.self_attn.q_proj.lora_B.weight"
        ),
        attn_key_w_a=(
            "base_model.model.model.layers.{}.self_attn.k_proj.lora_A.weight"
        ),
        attn_key_w_b=(
            "base_model.model.model.layers.{}.self_attn.k_proj.lora_B.weight"
        ),
        attn_value_w_a=(
            "base_model.model.model.layers.{}.self_attn.v_proj.lora_A.weight"
        ),
        attn_value_w_b=(
            "base_model.model.model.layers.{}.self_attn.v_proj.lora_B.weight"
        ),
        attn_output_w_a=(
            "base_model.model.model.layers.{}.self_attn.o_proj.lora_A.weight"
        ),
        attn_output_w_b=(
            "base_model.model.model.layers.{}.self_attn.o_proj.lora_B.weight"
        ),
    )

    safetensors_file = resource_loader.get_path_to_datafile(
        "fixtures/test_lora_rank16.safetensors"
    )
    config = self._get_test_config(num_layers=1, head_dim=8, num_query_groups=1)
    lora = lora_utils.LoRA.from_safetensors(
        safetensors_file,
        scale=1.0,
        lora_tensor_names=tensor_names,
        config=config,
    )
    self.assertEqual(lora.get_rank(), 16)

  def test_safetensors_to_tflite_conversion(self):
    """Converts safetensors file to TFLite model and verifies tensors and metadata."""
    from tensorflow.lite.python import schema_py_generated as schema_fb  # pylint: disable=g-direct-tensorflow-import

    tensor_names = lora_utils.LoRATensorNames(
        attn_query_w_a=(
            "base_model.model.model.layers.{}.self_attn.q_proj.lora_A.weight"
        ),
        attn_query_w_b=(
            "base_model.model.model.layers.{}.self_attn.q_proj.lora_B.weight"
        ),
        attn_key_w_a=(
            "base_model.model.model.layers.{}.self_attn.k_proj.lora_A.weight"
        ),
        attn_key_w_b=(
            "base_model.model.model.layers.{}.self_attn.k_proj.lora_B.weight"
        ),
        attn_value_w_a=(
            "base_model.model.model.layers.{}.self_attn.v_proj.lora_A.weight"
        ),
        attn_value_w_b=(
            "base_model.model.model.layers.{}.self_attn.v_proj.lora_B.weight"
        ),
        attn_output_w_a=(
            "base_model.model.model.layers.{}.self_attn.o_proj.lora_A.weight"
        ),
        attn_output_w_b=(
            "base_model.model.model.layers.{}.self_attn.o_proj.lora_B.weight"
        ),
    )

    safetensors_file = resource_loader.get_path_to_datafile(
        "fixtures/test_lora_rank16.safetensors"
    )
    config = self._get_test_config(num_layers=1, head_dim=8, num_query_groups=1)
    lora = lora_utils.LoRA.from_safetensors(
        safetensors_file,
        scale=1.0,
        lora_tensor_names=tensor_names,
        config=config,
    )

    flatbuffer_model = lora.to_tflite()

    # Verify the flatbuffer outputs using TFLite's schema
    model = schema_fb.Model.GetRootAsModel(flatbuffer_model, 0)
    model = schema_fb.ModelT.InitFromObj(model)

    # 1. Verify that lora_rank metadata has been packed correctly
    self.assertLen(model.metadata, 1)
    self.assertEqual(model.metadata[0].name.decode("utf-8"), "lora_rank")
    self.assertEqual(model.metadata[0].buffer, 16)

    # 2. Verify all subgraph input tensors are present with matching names
    subgraph = model.subgraphs[0]
    tensor_names_in_model = [t.name.decode("utf-8") for t in subgraph.tensors]

    expected_tensor_names = [
        "lora_atten_q_a_prime_weight_0",
        "lora_atten_q_b_prime_weight_0",
        "lora_atten_k_a_prime_weight_0",
        "lora_atten_k_b_prime_weight_0",
        "lora_atten_v_a_prime_weight_0",
        "lora_atten_v_b_prime_weight_0",
        "lora_atten_o_a_prime_weight_0",
        "lora_atten_o_b_prime_weight_0",
    ]

    for expected_name in expected_tensor_names:
      self.assertIn(expected_name, tensor_names_in_model)

  def test_torch_export(self):
    """Tests the export of the LoRA module."""

    class TestModel(torch.nn.Module):

      def forward(self, x: torch.Tensor, lora: lora_utils.LoRA) -> torch.Tensor:
        x += lora_utils.apply_lora(x, lora.adapters[0].attention.query)
        return x

    n = 1
    head_dim = 2
    num_query_groups = 1
    config = self._get_test_config(
        num_layers=n, head_dim=head_dim, num_query_groups=num_query_groups
    )
    inputs = torch.zeros((n, 1, head_dim))
    lora = lora_utils.LoRA.zeros(rank=16, config=config)
    model = TestModel()
    exported_program = torch.export.export(model, (inputs, lora))
    input_specs = exported_program.graph_signature.input_specs
    # 9 inputs: 1 for x, 2 for query lora, 2 for key lora, 2 for value lora,
    # 2 for output lora.
    self.assertLen(input_specs, 9)
    self.assertEqual(input_specs[0].arg.name, "x")
    self.assertEqual(input_specs[1].arg.name, "lora_atten_q_a_prime_weight_0")
    self.assertEqual(input_specs[2].arg.name, "lora_atten_q_b_prime_weight_0")
    self.assertEqual(input_specs[3].arg.name, "lora_atten_k_a_prime_weight_0")
    self.assertEqual(input_specs[4].arg.name, "lora_atten_k_b_prime_weight_0")
    self.assertEqual(input_specs[5].arg.name, "lora_atten_v_a_prime_weight_0")
    self.assertEqual(input_specs[6].arg.name, "lora_atten_v_b_prime_weight_0")
    self.assertEqual(input_specs[7].arg.name, "lora_atten_o_a_prime_weight_0")
    self.assertEqual(input_specs[8].arg.name, "lora_atten_o_b_prime_weight_0")

  def test_lora_tflite_serialization(self):
    """Tests the serialization of the LoRA module."""
    config = self._get_test_config(num_layers=2, head_dim=8, num_query_groups=1)
    lora = lora_utils.LoRA.random(rank=16, config=config)
    flatbuffer_model = lora.to_tflite()
    recovered_lora = lora_utils.LoRA.from_flatbuffers(flatbuffer_model)
    self.assertEqual(lora, recovered_lora)

  def test_lora_gqa_shapes(self):
    """Tests that LoRA shapes are correct when num_query_groups > 1 (GQA)."""
    config = self._get_test_config(
        num_layers=1, head_dim=8, num_query_groups=2, num_heads=4
    )
    lora = lora_utils.LoRA.zeros(rank=16, config=config)
    attn = lora.adapters[0].attention
    self.assertEqual(attn.query.b_prime.shape, (16, 32))
    self.assertEqual(attn.key.b_prime.shape, (16, 16))
    self.assertEqual(attn.value.b_prime.shape, (16, 16))
    self.assertEqual(attn.output.a_prime.shape, (32, 16))

  def _get_test_config(
      self, num_layers, head_dim, num_query_groups, num_heads=1
  ):
    """Returns a test model config."""
    attn_config = cfg.AttentionConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        num_query_groups=num_query_groups,
    )
    block_config = cfg.TransformerBlockConfig(
        attn_config=attn_config, ff_config=None
    )
    config = cfg.ModelConfig(
        embedding_dim=head_dim,
        block_configs=block_config,
        num_layers=num_layers,
        max_seq_len=None,
        vocab_size=None,
    )
    return config


if __name__ == "__main__":
  googletest.main()
