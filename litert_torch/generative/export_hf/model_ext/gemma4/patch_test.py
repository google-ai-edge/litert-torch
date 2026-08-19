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

import torch
from transformers.models.gemma4 import modeling_gemma4
from absl.testing import parameterized
import litert_torch.generative.export_hf
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext.gemma4 import patch
from litert_torch.generative.layers import moe
from litert_torch.generative.layers import rotary_position_embedding as rotary_pos_emb

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


class MockCacheLayer:

  def __init__(self, k_ts_idx=2, v_ts_idx=3):
    self.k_ts_idx = k_ts_idx
    self.v_ts_idx = v_ts_idx

  def get_k_ts_idx(self):
    return self.k_ts_idx

  def get_v_ts_idx(self):
    return self.v_ts_idx


class MockCache:

  def __init__(self, k_ts_idx=2, v_ts_idx=3):
    self.layers = [MockCacheLayer(k_ts_idx, v_ts_idx)]

  def update(self, key_states, value_states, layer_idx, **kwargs):
    if kwargs.get("apply_gpu_composites", False):
      b, n, s, h = key_states.shape
      k = key_states.reshape(1, b * n, s, h)
      v = value_states.reshape(1, b * n, s, h).transpose(-2, -1)
      return k, v
    else:
      return key_states, value_states


def _get_dummy_position_embeddings(batch_size, seq_len, head_dim):
  c = torch.randn(batch_size, seq_len, head_dim // 2)
  s = torch.randn(batch_size, seq_len, head_dim // 2)
  cos = torch.cat([c, c], dim=-1)
  sin = torch.cat([s, s], dim=-1)
  return (cos, sin)


class PatchTest(parameterized.TestCase):

  def test_fused_gemma4_attention_qkv(self):
    config = _get_dummy_gemma4_text_config()
    original_attn = modeling_gemma4.Gemma4TextAttention(config, layer_idx=0)
    fused_attn = patch.FusedGemma4TextAttention(original_attn, fuse_qkv=True)

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    position_embeddings = _get_dummy_position_embeddings(
        batch_size, seq_len, config.head_dim
    )
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

  def test_fused_gemma4_attention_rope_composite(self):
    config = _get_dummy_gemma4_text_config()
    original_attn = modeling_gemma4.Gemma4TextAttention(config, layer_idx=0)
    fused_attn = patch.FusedGemma4TextAttention(
        original_attn, use_rope_composite=True
    )

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    rope_base = float(getattr(config, "rope_theta", 500000.0))
    cos, sin = rotary_pos_emb.build_rope(
        position_ids[0], n_elem=config.head_dim, base=int(rope_base)
    )
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
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
          position_ids=position_ids,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "RoPE Composite Attention Output Mismatch.\n"
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

  def test_gemma4_experts_litert_moe_equivalence(self):
    config = _get_dummy_gemma4_text_config()
    config.num_experts = 4
    config.top_k_experts = 2
    config.moe_intermediate_size = 32

    torch.manual_seed(42)
    experts = modeling_gemma4.Gemma4TextExperts(config)
    experts.config = config
    with torch.no_grad():
      experts.gate_up_proj.normal_()
      experts.down_proj.normal_()

    batch_size = 2
    seq_len = 5
    hidden_states = torch.randn(batch_size * seq_len, config.hidden_size)
    top_k_index = torch.randint(
        0, config.num_experts, (batch_size * seq_len, config.top_k_experts)
    )
    top_k_weights = torch.softmax(
        torch.randn(batch_size * seq_len, config.top_k_experts), dim=-1
    )

    with torch.no_grad():
      expected_output = experts(hidden_states, top_k_index, top_k_weights)
      actual_output = moe.litert_moe_experts_forward(
          experts, hidden_states, top_k_index, top_k_weights
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-4, atol=1e-4),
        "Gemma4TextExperts vs litert_moe_experts_forward mismatch.\n"
        f"Max diff: {(expected_output - actual_output).abs().max().item()}\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )


if __name__ == "__main__":
  googletest.main()
