# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Qwen3 model export patches."""

import litert_torch.generative.export_hf  # pylint: disable=unused-import
from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext.qwen3 import patch
from litert_torch.generative.layers import rotary_position_embedding as rotary_pos_emb
import torch
from transformers.models.qwen3 import modeling_qwen3

from absl.testing import absltest as googletest


def _get_dummy_qwen3_config():
  config = modeling_qwen3.Qwen3Config(
      head_dim=16,
      num_attention_heads=4,
      num_key_value_heads=2,
      hidden_size=64,
      intermediate_size=128,
      sliding_window=None,
      attention_dropout=0.0,
      hidden_act="silu",
      num_hidden_layers=2,
      rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
  )
  config._attn_implementation = "eager"
  return config


def _get_dummy_position_embeddings(batch_size, seq_len, head_dim):
  c = torch.randn(batch_size, seq_len, head_dim // 2)
  s = torch.randn(batch_size, seq_len, head_dim // 2)
  cos = torch.cat([c, c], dim=-1)
  sin = torch.cat([s, s], dim=-1)
  return (cos, sin)


class PatchTest(parameterized.TestCase):

  def test_rms_norm_on_meta_device(self):
    norm = patch.Qwen3RMSNorm(hidden_size=64).to("meta")
    x = torch.empty(1, 4, 64, device="meta")

    out = norm(x)

    self.assertEqual(out.device, torch.device("meta"))
    self.assertEqual(out.shape, x.shape)

  def test_fused_qwen3_attention_qkv(self):
    config = _get_dummy_qwen3_config()
    original_attn = modeling_qwen3.Qwen3Attention(config, layer_idx=0)
    fused_attn = patch.FusedQwen3Attention(original_attn, fuse_qkv=True)

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_embeddings = _get_dummy_position_embeddings(
        batch_size, seq_len, config.head_dim
    )

    with torch.no_grad():
      expected_output, _ = original_attn(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
          attention_mask=None,
      )
      actual_output, _ = fused_attn(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
          attention_mask=None,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "Group QKV Attention Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_fused_qwen3_mlp_gate_up(self):
    config = _get_dummy_qwen3_config()
    original_mlp = modeling_qwen3.Qwen3MLP(config)
    fused_mlp = patch.FusedQwen3MLP(original_mlp)

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

  def test_fused_qwen3_mlp_swiglu_composite(self):
    config = _get_dummy_qwen3_config()
    original_mlp = modeling_qwen3.Qwen3MLP(config)
    fused_mlp = patch.FusedQwen3MLP(original_mlp, use_swiglu_composite=True)

    batch_size = 2
    seq_len = 4
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
      expected_output = original_mlp(x)
      actual_output = fused_mlp(x)

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "Fused SwiGLU Composite MLP Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_convert_full_model_layers(self):
    config = _get_dummy_qwen3_config()
    model = modeling_qwen3.Qwen3ForCausalLM(config).eval()

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir=None,
        fuse_gate_up=True,
        use_swiglu_composite=True,
        fuse_qkv=True,
        use_qkv_norm_rope_composite=True,
    )

    with patch.patch_qwen3_model(model, export_config):
      # 1. Test MLP conversion
      mlp = model.model.layers[0].mlp
      x = torch.randn(1, 4, config.hidden_size)
      edge_mlp = litert_torch.convert(mlp, (x,))
      self.assertIsNotNone(edge_mlp)

      # 2. Test attention wrapper
      class AttnWrapper(torch.nn.Module):

        def __init__(self, attn):
          super().__init__()
          self.attn = attn

        def forward(self, h, pos):
          return self.attn(h, position_ids=pos)[0]

      attn_wrap = AttnWrapper(model.model.layers[0].self_attn).eval()
      pos = torch.arange(4, dtype=torch.int32).unsqueeze(0)
      edge_attn = litert_torch.convert(attn_wrap, (x, pos))
      self.assertIsNotNone(edge_attn)


if __name__ == "__main__":
  googletest.main()
