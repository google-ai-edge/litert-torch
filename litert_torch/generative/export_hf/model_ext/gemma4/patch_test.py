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
import types

from absl.testing import parameterized
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import weights_loader
import litert_torch.generative.export_hf.model_ext as _
from litert_torch.generative.export_hf.model_ext.gemma4 import patch
from litert_torch.generative.layers import moe
import torch
import transformers
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

  @parameterized.named_parameters(
      ("first_sliding_layer", 0, None),
      ("first_full_layer", 4, None),
      ("sliding_donor", 13, True),
      ("full_donor", 14, True),
      ("first_shared_sliding_layer", 15, None),
      ("first_shared_full_layer", 19, None),
      ("last_sliding_reader", 33, False),
      ("last_full_reader", 34, False),
  )
  def test_get_prefill_skip_cache_update_override_gemma4_e2b(
      self, layer_idx, expected
  ):
    # Gemma4-E2B: 35 layers, the last 20 share KV, every 5th is full attention.
    config = types.SimpleNamespace(
        num_hidden_layers=35,
        num_kv_shared_layers=20,
        layer_types=[
            "full_attention" if i % 5 == 4 else "sliding_attention"
            for i in range(35)
        ],
    )
    self.assertEqual(
        patch.get_prefill_skip_cache_update_override(config, layer_idx),
        expected,
    )

  def test_get_prefill_skip_cache_update_override_without_kv_sharing(self):
    config = types.SimpleNamespace(
        num_hidden_layers=2,
        num_kv_shared_layers=0,
        layer_types=["sliding_attention", "full_attention"],
    )
    for layer_idx in range(2):
      self.assertIsNone(
          patch.get_prefill_skip_cache_update_override(config, layer_idx)
      )

  @parameterized.named_parameters(
      # (is_kv_shared_layer, override, sliding_prefill, readers_live, skip)
      ("owner", False, None, True, True, False),
      ("shared_reader", True, None, True, True, True),
      ("donor_defers", False, True, True, True, True),
      ("last_reader_writes", True, False, True, True, False),
      ("donor_decode", False, True, False, True, False),
      ("last_reader_decode", True, False, False, True, True),
      ("donor_dead_readers", False, True, True, False, False),
      ("last_reader_dead_readers", True, False, True, False, True),
  )
  def test_resolve_skip_cache_update(
      self, is_kv_shared_layer, override, sliding_prefill, readers_live, skip
  ):
    # The override applies only to sliding-window prefill with live readers;
    # otherwise the donor writes first and KV-shared layers skip.
    self.assertEqual(
        patch.resolve_skip_cache_update(
            is_kv_shared_layer,
            override,
            is_sliding_prefill=sliding_prefill,
            shared_kv_readers_live=readers_live,
        ),
        skip,
    )

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

  def test_patch_gemma4_model_enables_rope_composite_by_default(self):
    torch.manual_seed(0)
    config = _get_dummy_gemma4_text_config()
    config.vocab_size = 256
    model = modeling_gemma4.Gemma4ForCausalLM(config).eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
      expected_logits = model(input_ids=input_ids).logits

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
    )

    with patch.patch_gemma4_model(model, export_config):
      for layer in model.model.layers:
        self.assertIsInstance(layer.mlp, modeling_gemma4.Gemma4TextMLP)
        self.assertIsInstance(layer.self_attn, patch.FusedGemma4TextAttention)
        self.assertFalse(layer.self_attn.fuse_qkv)
        self.assertFalse(layer.self_attn.use_qkv_norm_rope_composite)
        self.assertTrue(layer.self_attn.use_rope_composite)

      with torch.no_grad():
        actual_logits = model(input_ids=input_ids).logits
      torch.testing.assert_close(
          actual_logits, expected_logits, rtol=1e-4, atol=1e-4
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

  def test_patch_gemma4_model_on_meta_device(self):
    config = _get_dummy_gemma4_text_config()
    config.vocab_size = 256
    config.attention_k_eq_v = True
    with weights_loader.init_params_on_meta():
      model = modeling_gemma4.Gemma4ForCausalLM(config).eval()

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        apply_gpu_composites=True,
    )
    with patch.patch_gemma4_model(model, export_config):
      layer = model.model.layers[0]
      self.assertIsInstance(layer.mlp, patch.FusedGemma4TextMLP)
      self.assertTrue(layer.mlp.gate_up_proj.weight.is_meta)
      self.assertIsInstance(layer.self_attn, patch.FusedGemma4TextAttention)
      self.assertTrue(layer.self_attn.qkv_proj.weight.is_meta)

  def _build_container_with_gemma4_decoder_layer(self, config):
    """Builds a minimal container holding a patched `Gemma4TextDecoderLayer`.

    `patch_gemma4_model` only needs `model.modules()`/`named_children` to
    find a `Gemma4TextDecoderLayer`; it doesn't require a full
    `Gemma4ForCausalLM`, so this sidesteps constructing the full causal LM
    (which isn't exercised under `gemma4_litert_patch` elsewhere and hits an
    unrelated MoE-dispatch incompatibility in this transformers version).
    """
    with patch.gemma4_litert_patch():
      layer = modeling_gemma4.Gemma4TextDecoderLayer(config, layer_idx=0)
    container = torch.nn.Module()
    container.layers = torch.nn.ModuleList([layer])
    return container, layer

  def test_patch_gemma4_model_elides_rmsnorm_final_scale(self):
    config = _get_dummy_gemma4_text_config()
    model, layer = self._build_container_with_gemma4_decoder_layer(config)

    self.assertIsInstance(layer.self_attn.q_norm, patch.Gemma4RMSNorm)
    self.assertFalse(layer.self_attn.q_norm.elide_final_scale)
    self.assertFalse(layer.input_layernorm.elide_final_scale)

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir=None,
        fuse_qkv=True,
        extra_kwargs={"elide_rmsnorm_final_scale": True},
    )

    with patch.patch_gemma4_model(model, export_config):
      fused_attn = model.layers[0].self_attn
      self.assertIsInstance(fused_attn, patch.FusedGemma4TextAttention)
      # `q_norm`/`k_norm` are reused by reference (not copied) from the
      # original attention module when it's fused; this regresses a bug
      # where flagging `Gemma4RMSNorm` instances inside the same
      # `named_children` walk used to fuse MLP/Attention meant the fusion
      # replaced the parent before the walk ever reached these norms.
      self.assertTrue(fused_attn.q_norm.elide_final_scale)
      self.assertTrue(fused_attn.k_norm.elide_final_scale)
      self.assertTrue(model.layers[0].input_layernorm.elide_final_scale)

    # Restored once the trace-time patch context exits.
    self.assertFalse(layer.self_attn.q_norm.elide_final_scale)
    self.assertFalse(layer.input_layernorm.elide_final_scale)

  def test_patch_gemma4_model_keeps_final_scale_by_default(self):
    config = _get_dummy_gemma4_text_config()
    model, _ = self._build_container_with_gemma4_decoder_layer(config)

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir=None,
        fuse_qkv=True,
    )

    with patch.patch_gemma4_model(model, export_config):
      fused_attn = model.layers[0].self_attn
      self.assertFalse(fused_attn.q_norm.elide_final_scale)
      self.assertFalse(model.layers[0].input_layernorm.elide_final_scale)

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
    # The reference audio model only produces a correct boolean attention mask
    # under "sdpa" in older transformers releases, while newer releases
    # (>= 5.20) implement the chunked local attention in eager mode only and
    # reject "sdpa" at construction time. Pick whichever the installed
    # transformers version supports.
    if getattr(modeling_gemma4.Gemma4AudioModel, "_supports_sdpa", False):
      audio_config._attn_implementation = "sdpa"
    else:
      audio_config._attn_implementation = "eager"

    torch.manual_seed(42)
    ref_model = modeling_gemma4.Gemma4AudioModel(audio_config).eval()

    with patch.gemma4_litert_patch():
      patched_model = modeling_gemma4.Gemma4AudioModel(
          copy.deepcopy(audio_config)
      ).eval()
    for layer in (
        patched_model.subsample_conv_projection.layer0,
        patched_model.subsample_conv_projection.layer1,
    ):
      self.assertIsInstance(layer.norm, patch.normalization.LayerNorm)
      self.assertTrue(layer.norm.enable_hlfb)
      self.assertIsNone(layer.norm.bias)
    patched_model.load_state_dict(ref_model.state_dict())
    # Verify that even when set_attn_implementation("eager") is called on the
    # patched model (as done during export), it still matches the reference
    # mask behavior.
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

  def test_gemma4_litert_patch_syncs_lazy_module_for_auto_model(self):
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
    audio_config._attn_implementation = "eager"
    text_config = _get_dummy_gemma4_text_config()

    # Populate `_LazyModule` caches on `transformers.models.gemma4` and
    # `transformers` before entering `gemma4_litert_patch`.
    unpatched_audio = transformers.AutoModel.from_config(audio_config)
    unpatched_text = transformers.AutoModel.from_config(text_config)
    _ = getattr(transformers, "Gemma4AudioModel", None)
    _ = getattr(transformers, "Gemma4TextModel", None)
    self.assertNotIsInstance(unpatched_audio, patch.LiteRTGemma4AudioModel)
    self.assertNotIsInstance(unpatched_text, patch.LiteRTGemma4TextModel)

    with patch.gemma4_litert_patch():
      patched_audio = transformers.AutoModel.from_config(
          copy.deepcopy(audio_config)
      ).eval()
      patched_text = transformers.AutoModel.from_config(
          copy.deepcopy(text_config)
      ).eval()
      self.assertIsInstance(patched_audio, patch.LiteRTGemma4AudioModel)
      self.assertIsInstance(patched_text, patch.LiteRTGemma4TextModel)

      input_features = torch.randn(1, 80, 128, dtype=torch.float32)
      attention_mask = torch.ones(1, 80, dtype=torch.bool)
      with torch.no_grad():
        out = patched_audio(input_features, attention_mask, return_dict=True)
      self.assertEqual(out.last_hidden_state.shape[0], 1)

    restored_audio = transformers.AutoModel.from_config(audio_config)
    restored_text = transformers.AutoModel.from_config(text_config)
    self.assertNotIsInstance(restored_audio, patch.LiteRTGemma4AudioModel)
    self.assertNotIsInstance(restored_text, patch.LiteRTGemma4TextModel)

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

  def test_vision_patch_embedder_precomputes_tables(self):
    vision_config = modeling_gemma4.Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        patch_size=14,
        position_embedding_size=64,
        pooling_kernel_size=2,
    )
    torch.manual_seed(0)
    ref_embedder = modeling_gemma4.Gemma4VisionPatchEmbedder(
        vision_config
    ).eval()
    with torch.no_grad():
      ref_embedder.position_embedding_table.normal_()

    with patch.gemma4_litert_patch():
      patched_embedder = modeling_gemma4.Gemma4VisionPatchEmbedder(
          vision_config
      ).eval()
    patched_embedder.load_state_dict(ref_embedder.state_dict())

    self.assertIn("table_x", patched_embedder._buffers)
    self.assertIn("table_y", patched_embedder._buffers)
    torch.testing.assert_close(
        patched_embedder.table_x, ref_embedder.position_embedding_table[0]
    )
    torch.testing.assert_close(
        patched_embedder.table_y, ref_embedder.position_embedding_table[1]
    )

    pixel_values = torch.randn(1, 16, 3 * 14 * 14)
    pixel_position_ids = torch.randint(0, 8, (1, 16, 2))
    padding_positions = torch.zeros((1, 16), dtype=torch.bool)
    with torch.no_grad():
      ref_out = ref_embedder(
          pixel_values, pixel_position_ids, padding_positions
      )
      patched_out = patched_embedder(
          pixel_values, pixel_position_ids, padding_positions
      )
    torch.testing.assert_close(patched_out, ref_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  googletest.main()
