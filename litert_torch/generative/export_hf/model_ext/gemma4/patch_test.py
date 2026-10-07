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

import copy

from absl.testing import parameterized
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
import litert_torch.generative.export_hf.model_ext as _
from litert_torch.generative.export_hf.model_ext.gemma4 import patch
from litert_torch.generative.layers import moe
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
    del layer_idx
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

    rotary_emb = modeling_gemma4.Gemma4TextRotaryEmbedding(config)
    position_embeddings = rotary_emb(
        hidden_states, position_ids, layer_type="full_attention"
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
          position_ids=position_ids,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "RoPE Composite Attention Output Mismatch.\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_fused_gemma4_attention_qkv_norm_rope_composite(self):
    config = _get_dummy_gemma4_text_config()
    original_attn = modeling_gemma4.Gemma4TextAttention(config, layer_idx=0)
    fused_attn = patch.FusedGemma4TextAttention(
        original_attn, use_qkv_norm_rope_composite=True
    )

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    rotary_emb = modeling_gemma4.Gemma4TextRotaryEmbedding(config)
    position_embeddings = rotary_emb(
        hidden_states, position_ids, layer_type="full_attention"
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
          position_ids=position_ids,
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "QKV Norm RoPE Composite Attention Output Mismatch.\n"
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

  def test_patch_gemma4_model_with_single_gpu_composites_flag(self):
    torch.manual_seed(0)
    config = _get_dummy_gemma4_text_config()
    config.vocab_size = 256
    model = modeling_gemma4.Gemma4ForCausalLM(config).eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
      expected_logits = model(input_ids=input_ids).logits

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        apply_gpu_composites=True,
    )
    export_config = _.extension.update_export_config(export_config, config)

    with patch.patch_gemma4_model(model, export_config):
      for layer in model.model.layers:
        self.assertIsInstance(layer.mlp, patch.FusedGemma4TextMLP)
        self.assertIsInstance(layer.self_attn, patch.FusedGemma4TextAttention)
        self.assertTrue(layer.self_attn.fuse_qkv)
        self.assertTrue(layer.self_attn.use_qkv_norm_rope_composite)

      with torch.no_grad():
        actual_logits = model(input_ids=input_ids).logits
      torch.testing.assert_close(
          actual_logits, expected_logits, rtol=1e-4, atol=1e-4
      )

  def test_gemma4_router_equivalence(self):
    config = _get_dummy_gemma4_text_config()
    config.num_experts = 128
    config.top_k_experts = 8
    config.hidden_size = 2816
    config.rms_norm_eps = 1e-6

    torch.manual_seed(42)
    original_router = modeling_gemma4.Gemma4TextRouter(config)
    litert_router = patch.LiteRTGemma4TextRouter(config)

    # Copy weights
    with torch.no_grad():
      litert_router.proj.weight.copy_(original_router.proj.weight)
      litert_router.scale.copy_(original_router.scale)
      litert_router.per_expert_scale.copy_(original_router.per_expert_scale)

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size * seq_len, config.hidden_size)

    with torch.no_grad():
      orig_probs, orig_weights, orig_index = original_router(hidden_states)
      lrt_probs, lrt_weights, lrt_index = litert_router(hidden_states)

    self.assertTrue(
        torch.allclose(orig_probs, lrt_probs, rtol=1e-5, atol=1e-5),
        "Router probabilities mismatch.",
    )
    self.assertTrue(
        torch.equal(orig_index, lrt_index.long()),
        "Router top-k indices mismatch.",
    )
    self.assertTrue(
        torch.allclose(orig_weights, lrt_weights, rtol=1e-5, atol=1e-5),
        "Router top-k weights mismatch.",
    )

  def test_gemma4_router_int32_indices(self):
    config = _get_dummy_gemma4_text_config()
    config.num_experts = 128
    config.top_k_experts = 8
    config.hidden_size = 2816
    config.rms_norm_eps = 1e-6

    torch.manual_seed(42)
    router = patch.LiteRTGemma4TextRouter(config)
    self.assertFalse(router.use_int32_indices)
    hidden_states = torch.randn(8, config.hidden_size)

    with torch.no_grad():
      ref_probs, ref_weights, ref_index = router(hidden_states)
      router.use_int32_indices = True
      probs, weights, index = router(hidden_states)

    self.assertEqual(ref_index.dtype, torch.int64)
    self.assertEqual(index.dtype, torch.int32)
    self.assertTrue(torch.equal(ref_index, index.long()))
    self.assertTrue(torch.allclose(ref_probs, probs))
    self.assertTrue(torch.allclose(ref_weights, weights))

    # The exported graph must route through `tfl.topk_v2` (int32 indices)
    # instead of `aten.topk`, which the converter lowers to `tfl.topk_v2`
    # followed by an int64 CAST.
    exported = torch.export.export(router, (hidden_states,))
    targets = [
        str(node.target)
        for node in exported.graph.nodes
        if node.op == "call_function"
    ]
    self.assertTrue(any("tfl.topk_v2" in t for t in targets), targets)
    self.assertFalse(any("aten.topk" in t for t in targets), targets)

  def test_gemma4_26b_experts_litert_moe_equivalence(self):
    config = _get_dummy_gemma4_text_config()
    config.num_experts = 128
    config.top_k_experts = 8
    config.hidden_size = 2816
    config.moe_intermediate_size = 704

    torch.manual_seed(42)
    experts = modeling_gemma4.Gemma4TextExperts(config)
    experts.config = config
    with torch.no_grad():
      experts.gate_up_proj.normal_(std=0.02)
      experts.down_proj.normal_(std=0.02)

    batch_size = 1
    seq_len = 4
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
        "Gemma4 26B MoE Experts vs litert_moe_experts_forward mismatch.\n"
        f"Max diff: {(expected_output - actual_output).abs().max().item()}\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_gemma4_26b_experts_pre_flattened_litert_moe_equivalence(self):
    config = _get_dummy_gemma4_text_config()
    config.num_experts = 128
    config.top_k_experts = 8
    config.hidden_size = 2816
    config.moe_intermediate_size = 704

    torch.manual_seed(42)
    experts = modeling_gemma4.Gemma4TextExperts(config)
    experts.config = config
    with torch.no_grad():
      experts.gate_up_proj.normal_(std=0.02)
      experts.down_proj.normal_(std=0.02)

    batch_size = 1
    seq_len = 4
    hidden_states = torch.randn(batch_size * seq_len, config.hidden_size)
    top_k_index = torch.randint(
        0, config.num_experts, (batch_size * seq_len, config.top_k_experts)
    )
    top_k_weights = torch.softmax(
        torch.randn(batch_size * seq_len, config.top_k_experts), dim=-1
    )

    with torch.no_grad():
      expected_output = experts(hidden_states, top_k_index, top_k_weights)

    export_lib.pre_flatten_model_experts(experts)

    self.assertTrue(hasattr(experts, "flattened_gate_weight"))
    self.assertTrue(hasattr(experts, "flattened_ff1_weight"))
    self.assertTrue(hasattr(experts, "flattened_linear_weight"))
    self.assertTrue(hasattr(experts, "per_expert_scale"))
    self.assertFalse(hasattr(experts, "gate_up_proj"))
    self.assertFalse(hasattr(experts, "down_proj"))

    with torch.no_grad():
      actual_output = moe.litert_moe_experts_forward(
          experts, hidden_states, top_k_index, top_k_weights
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-4, atol=1e-4),
        "Pre-flattened Gemma4 26B MoE Experts mismatch.\n"
        f"Max diff: {(expected_output - actual_output).abs().max().item()}\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_gemma4_decoder_layer_with_moe(self):
    config = _get_dummy_gemma4_text_config()
    config.enable_moe_block = True
    config.num_experts = 8
    config.top_k_experts = 2
    config.moe_intermediate_size = 32
    config.hidden_size = 64
    config.rms_norm_eps = 1e-6
    config.hidden_size_per_layer_input = 0

    torch.manual_seed(42)
    layer = modeling_gemma4.Gemma4TextDecoderLayer(config, layer_idx=0)
    with torch.no_grad():
      layer.experts.gate_up_proj.normal_(std=0.02)
      layer.experts.down_proj.normal_(std=0.02)
    layer.eval()

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
      expected_output = layer(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
          attention_mask=attention_mask,
          shared_kv_states=shared_kv_states,
      )

    with patch.gemma4_litert_patch():
      patched_config = _get_dummy_gemma4_text_config()
      patched_config.enable_moe_block = True
      patched_config.num_experts = 8
      patched_config.top_k_experts = 2
      patched_config.moe_intermediate_size = 32
      patched_config.hidden_size = 64
      patched_config.rms_norm_eps = 1e-6
      patched_config.hidden_size_per_layer_input = 0
      patched_config._experts_implementation = "litert_moe"

      patched_layer = modeling_gemma4.Gemma4TextDecoderLayer(
          patched_config, layer_idx=0
      )
      # Copy layer weights
      patched_layer.load_state_dict(layer.state_dict())
      patched_layer.eval()

      with torch.no_grad():
        actual_output = patched_layer(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
        )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-4, atol=1e-4),
        "Gemma4TextDecoderLayer with MoE block mismatch.\n"
        f"Max diff: {(expected_output - actual_output).abs().max().item()}\n"
        f"Expected: {expected_output}\nActual: {actual_output}",
    )

  def test_gemma4_audio_model_equivalence(self):
    audio_config = modeling_gemma4.Gemma4AudioConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        output_proj_dims=96,
        attention_chunk_size=12,
        attention_context_left=13,
        attention_context_right=0,
        conv_kernel_size=5,
        subsampling_conv_channels=[128, 8],
        dtype="float32",
    )
    audio_config._attn_implementation = "sdpa"

    torch.manual_seed(42)
    ref_model = modeling_gemma4.Gemma4AudioModel(audio_config).eval()

    with patch.gemma4_litert_patch():
      patched_model = modeling_gemma4.Gemma4AudioModel(
          copy.deepcopy(audio_config)
      ).eval()
    patched_model.load_state_dict(ref_model.state_dict())
    # Verify that even when set_attn_implementation("eager") is called on the
    # patched model (as done during export), it still matches the reference sdpa
    # boolean mask behavior.
    patched_model.set_attn_implementation("eager")

    input_features = torch.randn(1, 80, 128, dtype=torch.float32)
    attention_mask = torch.zeros(1, 80, dtype=torch.bool)
    attention_mask[:, :52] = True

    with torch.no_grad():
      ref_out = ref_model(input_features, attention_mask, return_dict=True)
      patched_out = patched_model(
          input_features, attention_mask, return_dict=True
      )

    self.assertTrue(
        torch.equal(ref_out.attention_mask, patched_out.attention_mask)
    )
    valid_mask = ref_out.attention_mask
    max_diff = (
        (
            ref_out.last_hidden_state[valid_mask]
            - patched_out.last_hidden_state[valid_mask]
        )
        .abs()
        .max()
        .item()
    )
    self.assertTrue(
        torch.allclose(
            ref_out.last_hidden_state[valid_mask],
            patched_out.last_hidden_state[valid_mask],
            rtol=1e-4,
            atol=1e-4,
        ),
        "LiteRTGemma4AudioModel vs Gemma4AudioModel mismatch.\n"
        f"Max diff: {max_diff}",
    )

  def test_bind_per_expert_scale_matches_router_folding(self):
    config = _get_dummy_gemma4_text_config()
    config.num_experts = 4
    config.top_k_experts = 2
    config.moe_intermediate_size = 32

    torch.manual_seed(42)
    block = torch.nn.Module()
    block.router = patch.LiteRTGemma4TextRouter(config)
    block.experts = modeling_gemma4.Gemma4TextExperts(config)
    block.experts.config = config
    with torch.no_grad():
      block.experts.gate_up_proj.normal_()
      block.experts.down_proj.normal_()
      block.router.per_expert_scale.uniform_(0.5, 1.5)

    hidden_states = torch.randn(6, config.hidden_size)

    with torch.no_grad():
      # Baseline: the router folds `per_expert_scale` into `top_k_weights` and
      # the custom op receives an all-ones scale.
      _, folded_weights, top_k_index = block.router(hidden_states)
      expected_output = moe.litert_moe_experts_forward(
          block.experts, hidden_states, top_k_index, folded_weights
      )

      moe.bind_per_expert_scale(block)
      self.assertFalse(block.router.fold_per_expert_scale)
      self.assertTrue(block.router.use_int32_indices)

      _, unfolded_weights, top_k_index = block.router(hidden_states)
      self.assertEqual(top_k_index.dtype, torch.int32)
      actual_output = moe.litert_moe_experts_forward(
          block.experts, hidden_states, top_k_index, unfolded_weights
      )

    self.assertFalse(torch.allclose(folded_weights, unfolded_weights))
    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-5, atol=1e-5),
        "Per-expert scale binding changed the MoE output.\n"
        f"Max diff: {(expected_output - actual_output).abs().max().item()}",
    )


if __name__ == "__main__":
  googletest.main()
