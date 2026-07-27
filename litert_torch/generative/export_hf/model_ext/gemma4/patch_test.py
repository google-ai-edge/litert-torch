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
"""Tests for Gemma4 model export patches."""

from absl.testing import parameterized
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext.gemma4 import patch
import torch
from transformers.models.gemma4 import modeling_gemma4

from absl.testing import absltest as googletest


def _get_dummy_gemma4_text_config():
  config = modeling_gemma4.Gemma4TextConfig(
      head_dim=16,
      global_head_dim=16,
      num_attention_heads=4,
      num_key_value_heads=2,
      hidden_size=64,
      intermediate_size=128,
      attention_dropout=0.0,
      hidden_activation="gelu_pytorch_tanh",
  )
  config.layer_types = ["full_attention"]
  config.num_hidden_layers = 1
  config._attn_implementation = "eager"
  return config


class PatchTest(parameterized.TestCase):

  def test_fused_gemma4_attention_qkv(self):
    config = _get_dummy_gemma4_text_config()
    original_attn = modeling_gemma4.Gemma4TextAttention(config, layer_idx=0)
    fused_attn = patch.FusedGemma4TextAttention(original_attn, fuse_qkv=True)

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    cos = torch.randn(batch_size, seq_len, config.head_dim)
    sin = torch.randn(batch_size, seq_len, config.head_dim)
    position_embeddings = (cos, sin)
    attention_mask = torch.ones(
        (batch_size, 1, seq_len, seq_len), dtype=torch.bool
    )
    shared_kv_states = {}

    with torch.no_grad():
      expected_output, _ = original_attn(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
          attention_mask=attention_mask,
          shared_kv_states=shared_kv_states,
      )
      actual_output, _ = fused_attn(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
          attention_mask=attention_mask,
          shared_kv_states=shared_kv_states,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "Group QKV Attention Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_fused_gemma4_mlp_gate_up(self):
    config = _get_dummy_gemma4_text_config()
    original_mlp = modeling_gemma4.Gemma4TextMLP(config, layer_idx=0)
    fused_mlp = patch.FusedGemma4TextMLP(original_mlp)

    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
      expected_output = original_mlp(x)
      actual_output = fused_mlp(x)

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "Fused Gate+Up MLP Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_patch_gemma4_model(self):
    config = _get_dummy_gemma4_text_config()
    model = modeling_gemma4.Gemma4ForCausalLM(config)

    self.assertIsInstance(
        model.model.layers[0].mlp, modeling_gemma4.Gemma4TextMLP
    )
    self.assertIsInstance(
        model.model.layers[0].self_attn, modeling_gemma4.Gemma4TextAttention
    )

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir=None,
        fuse_gate_up=True,
        fuse_qkv=True,
    )

    with patch.patch_gemma4_model(model, export_config):
      self.assertIsInstance(model.model.layers[0].mlp, patch.FusedGemma4TextMLP)
      self.assertIsInstance(
          model.model.layers[0].self_attn, patch.FusedGemma4TextAttention
      )
      self.assertTrue(model.model.layers[0].self_attn.fuse_qkv)

    self.assertIsInstance(
        model.model.layers[0].mlp, modeling_gemma4.Gemma4TextMLP
    )
    self.assertIsInstance(
        model.model.layers[0].self_attn, modeling_gemma4.Gemma4TextAttention
    )


if __name__ == "__main__":
  googletest.main()
