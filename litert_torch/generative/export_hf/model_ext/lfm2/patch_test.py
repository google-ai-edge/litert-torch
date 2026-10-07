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
"""Tests for LFM2 model export patches."""

from absl.testing import parameterized
import litert_torch
import litert_torch.generative.export_hf  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext.lfm2 import patch
import torch
from transformers.models.lfm2 import modeling_lfm2

from absl.testing import absltest as googletest


def _get_dummy_lfm2_config():
  config = modeling_lfm2.Lfm2Config(
      num_attention_heads=4,
      num_key_value_heads=2,
      hidden_size=64,
      intermediate_size=128,
      block_auto_adjust_ff_dim=False,
      num_hidden_layers=2,
      layer_types=["conv", "full_attention"],
      full_attn_idxs=[1],
      rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
  )
  config._attn_implementation = "eager"
  return config


def _get_dummy_position_embeddings(config, hidden_states, position_ids):
  rotary_emb = modeling_lfm2.Lfm2RotaryEmbedding(config)
  return rotary_emb(hidden_states, position_ids=position_ids)


class PatchTest(parameterized.TestCase):

  def test_fused_lfm2_attention_qkv(self):
    config = _get_dummy_lfm2_config()
    original_attn = modeling_lfm2.Lfm2Attention(config, layer_idx=1)
    fused_attn = patch.FusedLfm2Attention(original_attn, fuse_qkv=True)

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = _get_dummy_position_embeddings(
        config, hidden_states, position_ids
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
        "Fused QKV Attention Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_fused_lfm2_attention_qkv_norm_rope(self):
    config = _get_dummy_lfm2_config()
    original_attn = modeling_lfm2.Lfm2Attention(config, layer_idx=1)
    fused_attn = patch.FusedLfm2Attention(
        original_attn,
        fuse_qkv=True,
        use_qkv_norm_rope_composite=True,
    )

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = _get_dummy_position_embeddings(
        config, hidden_states, position_ids
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
          position_ids=position_ids,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "Fused QKV Norm RoPE Composite Attention Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_fused_lfm2_mlp_gate_up(self):
    config = _get_dummy_lfm2_config()
    original_mlp = modeling_lfm2.Lfm2MLP(config)
    fused_mlp = patch.FusedLfm2MLP(original_mlp)

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

  def test_fused_lfm2_mlp_swiglu_composite(self):
    config = _get_dummy_lfm2_config()
    original_mlp = modeling_lfm2.Lfm2MLP(config)
    fused_mlp = patch.FusedLfm2MLP(original_mlp, use_swiglu_composite=True)

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

  def test_lfm2_rms_norm(self):
    dim = 64
    original_norm = modeling_lfm2.Lfm2RMSNorm(dim)
    patched_norm = patch.Lfm2RMSNorm(dim)
    patched_norm.weight.data.copy_(original_norm.weight.data)

    x = torch.randn(2, 4, dim)
    with torch.no_grad():
      expected_output = original_norm(x)
      actual_output = patched_norm(x)

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "LFM2 RMSNorm Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_convert_full_model_layers(self):
    config = _get_dummy_lfm2_config()
    model = modeling_lfm2.Lfm2ForCausalLM(config).eval()

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir=None,
        fuse_gate_up=True,
        use_swiglu_composite=True,
        fuse_qkv=True,
        use_qkv_norm_rope_composite=True,
    )

    with patch.patch_lfm2_model(model, export_config):
      # Verify layers patched
      self.assertIsInstance(
          model.model.layers[0].feed_forward, patch.FusedLfm2MLP
      )
      self.assertIsInstance(
          model.model.layers[1].self_attn, patch.FusedLfm2Attention
      )

      # 1. Test MLP conversion
      mlp = model.model.layers[0].feed_forward
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

      attn_wrap = AttnWrapper(model.model.layers[1].self_attn).eval()
      pos = torch.arange(4, dtype=torch.int32).unsqueeze(0)
      edge_attn = litert_torch.convert(attn_wrap, (x, pos))
      self.assertIsNotNone(edge_attn)

  def test_short_conv_decode_and_prefill(self):
    config = _get_dummy_lfm2_config()
    conv = patch.short_conv_lib.Lfm2ShortConv(config, layer_idx=0)

    expected_shape = (1, config.hidden_size, config.conv_L_cache - 1)

    class DummyLayer:

      def __init__(self):
        self.conv_states = torch.zeros(expected_shape)

    class DummyCache:

      def __init__(self):
        self.layers = [DummyLayer()]

    cache = DummyCache()
    # Prefill
    x_prefill = torch.randn(1, 4, config.hidden_size)
    out_prefill = conv(x_prefill, past_key_values=cache)
    self.assertEqual(out_prefill.shape, (1, 4, config.hidden_size))
    self.assertEqual(cache.layers[0].conv_states.shape, expected_shape)

    # Decode
    x_decode = torch.randn(1, 1, config.hidden_size)
    out_decode = conv(x_decode, past_key_values=cache)
    self.assertEqual(out_decode.shape, (1, 1, config.hidden_size))
    self.assertEqual(cache.layers[0].conv_states.shape, expected_shape)

  def test_short_conv_decode_composite(self):
    config = _get_dummy_lfm2_config()
    conv_standard = patch.short_conv_lib.Lfm2ShortConv(
        config, layer_idx=0, use_short_conv_composite=False
    )
    conv_composite = patch.short_conv_lib.Lfm2ShortConv(
        config, layer_idx=0, use_short_conv_composite=True
    )
    conv_composite.load_state_dict(conv_standard.state_dict())

    expected_shape = (1, config.hidden_size, config.conv_L_cache - 1)

    class DummyLayer:

      def __init__(self, state):
        self.conv_states = state.clone()

    class DummyCache:

      def __init__(self, state):
        self.layers = [DummyLayer(state)]

    init_state = torch.randn(expected_shape)
    cache_std = DummyCache(init_state)
    cache_comp = DummyCache(init_state)

    x_decode = torch.randn(1, 1, config.hidden_size)

    with torch.no_grad():
      out_std = conv_standard(x_decode, past_key_values=cache_std)
      out_comp = conv_composite(x_decode, past_key_values=cache_comp)

    self.assertTrue(
        torch.allclose(out_std, out_comp, rtol=1e-5, atol=1e-5),
        "Output mismatch between standard and composite short conv",
    )
    self.assertTrue(
        torch.allclose(
            cache_std.layers[0].conv_states,
            cache_comp.layers[0].conv_states,
            rtol=1e-5,
            atol=1e-5,
        ),
        "Next state mismatch",
    )

  def test_patch_lfm2_model_with_single_gpu_composites_flag(self):
    torch.manual_seed(0)
    config = _get_dummy_lfm2_config()
    config.vocab_size = 256
    model = modeling_lfm2.Lfm2ForCausalLM(config).eval()

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        apply_gpu_composites=True,
    )
    export_config = extension.update_export_config(export_config, config)

    with patch.patch_lfm2_model(model, export_config):
      self.assertIsInstance(
          model.model.layers[0].feed_forward, patch.FusedLfm2MLP
      )
      self.assertTrue(model.model.layers[0].feed_forward.use_swiglu_composite)
      self.assertIsInstance(
          model.model.layers[1].self_attn, patch.FusedLfm2Attention
      )
      self.assertTrue(model.model.layers[1].self_attn.fuse_qkv)
      self.assertTrue(
          model.model.layers[1].self_attn.use_qkv_norm_rope_composite
      )


if __name__ == "__main__":

  googletest.main()
