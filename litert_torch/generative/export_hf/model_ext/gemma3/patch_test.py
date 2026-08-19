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
"""Tests for Gemma3 model export patches."""

from absl.testing import parameterized
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext.gemma3 import patch
from litert_torch.generative.layers import rotary_position_embedding as rotary_pos_emb
import torch
from transformers.models.gemma3 import modeling_gemma3

from absl.testing import absltest as googletest


def _get_dummy_gemma3_text_config():
  config = modeling_gemma3.Gemma3TextConfig(
      head_dim=16,
      num_attention_heads=4,
      num_key_value_heads=2,
      hidden_size=64,
      intermediate_size=128,
      attn_logit_softcapping=None,
      query_pre_attn_scalar=16,
      sliding_window=None,
      attention_dropout=0.0,
      hidden_activation="gelu_pytorch_tanh",
  )
  config._attn_implementation = "eager"
  return config


def _get_dummy_gemma3_config():
  text_config = _get_dummy_gemma3_text_config()
  config = modeling_gemma3.Gemma3Config(text_config=text_config)
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

  def test_fused_gemma3_attention_qkv(self):
    config = _get_dummy_gemma3_text_config()
    original_attn = modeling_gemma3.Gemma3Attention(config, layer_idx=0)
    fused_attn = patch.FusedGemma3Attention(original_attn, fuse_qkv=True)

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
      )
      actual_output, _ = fused_attn(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "Group QKV Attention Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_fused_gemma3_mlp_gate_up(self):
    config = _get_dummy_gemma3_text_config()
    original_mlp = modeling_gemma3.Gemma3MLP(config)
    fused_mlp = patch.FusedGemma3MLP(original_mlp)

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

  def test_fused_gemma3_attention_rope_composite(self):
    config = _get_dummy_gemma3_text_config()
    # layer_idx=5 is a full_attention (global) layer in 6-layer pattern ((5+1)%6 == 0)
    original_attn = modeling_gemma3.Gemma3Attention(config, layer_idx=5)
    fused_attn = patch.FusedGemma3Attention(
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

    with torch.no_grad():
      expected_output, _ = original_attn(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
      )
      actual_output, _ = fused_attn(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
          position_ids=position_ids,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "Global RoPE Composite Attention Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_fused_gemma3_attention_local_rope_composite(self):
    config = _get_dummy_gemma3_text_config()
    # layer_idx=0 is a sliding_attention (local) layer
    original_attn = modeling_gemma3.Gemma3Attention(config, layer_idx=0)
    fused_attn = patch.FusedGemma3Attention(
        original_attn, use_rope_composite=True
    )

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    rope_base = float(getattr(config, "rope_local_base_freq", 10000.0))
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
      )
      actual_output, _ = fused_attn(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
          position_ids=position_ids,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "Local RoPE Composite Attention Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_patch_gemma3_model(self):
    config = _get_dummy_gemma3_text_config()
    model = modeling_gemma3.Gemma3ForCausalLM(config)

    # Verify originally it has standard MLP and Attention
    self.assertIsInstance(model.model.layers[0].mlp, modeling_gemma3.Gemma3MLP)
    self.assertIsInstance(
        model.model.layers[0].self_attn, modeling_gemma3.Gemma3Attention
    )

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir=None,
        fuse_gate_up=True,
        fuse_qkv=True,
    )

    with patch.patch_gemma3_model(model, export_config):
      # Verify inside context they are replaced with Fused classes
      self.assertIsInstance(model.model.layers[0].mlp, patch.FusedGemma3MLP)
      self.assertIsInstance(
          model.model.layers[0].self_attn, patch.FusedGemma3Attention
      )
      self.assertTrue(model.model.layers[0].self_attn.fuse_qkv)

    # Verify outside context they are restored
    self.assertIsInstance(model.model.layers[0].mlp, modeling_gemma3.Gemma3MLP)
    self.assertIsInstance(
        model.model.layers[0].self_attn, modeling_gemma3.Gemma3Attention
    )


if __name__ == "__main__":
  googletest.main()
