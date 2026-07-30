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
"""Tests for Qwen3 model export patches."""

from absl.testing import parameterized
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

  def test_fused_qwen3_attention_rope_composite(self):
    config = _get_dummy_qwen3_config()
    original_attn = modeling_qwen3.Qwen3Attention(config, layer_idx=0)
    fused_attn = patch.FusedQwen3Attention(
        original_attn, use_rope_composite=True
    )

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    rope_base = 1000000.0
    if hasattr(config, "rope_parameters") and config.rope_parameters:
      if isinstance(config.rope_parameters, dict):
        rope_base = float(config.rope_parameters.get("rope_theta", rope_base))
      elif hasattr(config.rope_parameters, "rope_theta"):
        rope_base = float(
            getattr(config.rope_parameters, "rope_theta", rope_base)
        )
    elif hasattr(config, "rope_theta"):
      rope_base = float(getattr(config, "rope_theta", rope_base))

    cos, sin = rotary_pos_emb.build_rope(
        position_ids[0], n_elem=config.head_dim, base=int(rope_base)
    )
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    position_embeddings = (cos, sin)

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
          position_ids=position_ids,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "RoPE Composite Attention Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_fused_qwen3_attention_local_rope_composite(self):
    config = _get_dummy_qwen3_config()
    config.num_local_layers_per_global = 3
    config.local_rope_theta = 10000.0
    original_attn = modeling_qwen3.Qwen3Attention(config, layer_idx=0)
    fused_attn = patch.FusedQwen3Attention(
        original_attn, use_rope_composite=True
    )

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    cos, sin = rotary_pos_emb.build_rope(
        position_ids[0], n_elem=config.head_dim, base=10000
    )
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    position_embeddings = (cos, sin)

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
          position_ids=position_ids,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "Local RoPE Composite Attention Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_patch_qwen3_model(self):
    config = _get_dummy_qwen3_config()
    model = modeling_qwen3.Qwen3ForCausalLM(config)

    # Verify originally it has standard MLP and Attention
    self.assertIsInstance(model.model.layers[0].mlp, modeling_qwen3.Qwen3MLP)
    self.assertIsInstance(
        model.model.layers[0].self_attn, modeling_qwen3.Qwen3Attention
    )

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir=None,
        fuse_gate_up=True,
        fuse_qkv=True,
        use_rope_composite=True,
    )

    with patch.patch_qwen3_model(model, export_config):
      # Verify inside context they are replaced with Fused classes
      self.assertIsInstance(model.model.layers[0].mlp, patch.FusedQwen3MLP)
      self.assertIsInstance(
          model.model.layers[0].self_attn, patch.FusedQwen3Attention
      )
      self.assertTrue(model.model.layers[0].self_attn.fuse_qkv)
      self.assertTrue(model.model.layers[0].self_attn.use_rope_composite)

    # Verify outside context they are restored
    self.assertIsInstance(model.model.layers[0].mlp, modeling_qwen3.Qwen3MLP)
    self.assertIsInstance(
        model.model.layers[0].self_attn, modeling_qwen3.Qwen3Attention
    )


if __name__ == "__main__":
  googletest.main()
