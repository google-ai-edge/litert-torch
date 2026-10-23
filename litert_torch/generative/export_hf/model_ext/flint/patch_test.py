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
"""Tests for Flint model export patches."""

import copy
import os
from unittest import mock

from absl.testing import parameterized
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
import litert_torch.generative.export_hf.model_ext as _
from litert_torch.generative.export_hf.model_ext.flint import configuration_flint
from litert_torch.generative.export_hf.model_ext.flint import modeling_flint
from litert_torch.generative.export_hf.model_ext.flint import patch
from litert_torch.generative.layers import moe
import safetensors.torch
import torch

from absl.testing import absltest as googletest


def _get_dummy_flint_config():
  config = configuration_flint.FlintConfig(
      vocab_size=256,
      hidden_size=64,
      intermediate_size=128,
      num_hidden_layers=3,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=16,
      moe_intermediate_size=32,
      shared_expert_intermediate_size=64,
      num_experts=8,
      num_experts_per_tok=2,
      mlp_only_layers=[0],
      layer_types=["full_attention", "sliding_attention", "full_attention"],
      sliding_window=8,
  )
  config._attn_implementation = "eager"
  return config


class PatchTest(parameterized.TestCase):

  def test_flint_router_equivalence_and_int32_indices(self):
    config = _get_dummy_flint_config()
    torch.manual_seed(42)
    orig_router = modeling_flint.FlintTopKRouter(config)
    lrt_router = patch.LiteRTFlintTopKRouter(config)
    with torch.no_grad():
      orig_router.weight.normal_(std=0.02)
      lrt_router.weight.copy_(orig_router.weight)

    hidden_states = torch.randn(6, config.hidden_size)
    with torch.no_grad():
      orig_logits, orig_weights, orig_indices = orig_router(hidden_states)
      lrt_logits, lrt_weights, lrt_indices = lrt_router(hidden_states)
      lrt_router.use_int32_indices = True
      _, lrt_weights_i32, lrt_indices_i32 = lrt_router(hidden_states)

    self.assertTrue(
        torch.allclose(orig_logits, lrt_logits, rtol=1e-5, atol=1e-5)
    )
    self.assertTrue(
        torch.allclose(orig_weights, lrt_weights, rtol=1e-5, atol=1e-5)
    )
    self.assertTrue(torch.equal(orig_indices, lrt_indices))
    self.assertEqual(lrt_indices_i32.dtype, torch.int32)
    self.assertTrue(torch.equal(orig_indices, lrt_indices_i32.long()))
    self.assertTrue(
        torch.allclose(orig_weights, lrt_weights_i32, rtol=1e-5, atol=1e-5)
    )

    exported = torch.export.export(lrt_router, (hidden_states,))
    targets = [
        str(node.target)
        for node in exported.graph.nodes
        if node.op == "call_function"
    ]
    self.assertTrue(any("tfl.topk_v2" in t for t in targets), targets)
    self.assertFalse(any("aten.topk" in t for t in targets), targets)

  def test_flint_experts_litert_moe_equivalence(self):
    config = _get_dummy_flint_config()
    torch.manual_seed(42)
    experts = modeling_flint.FlintExperts(config)
    with torch.no_grad():
      experts.gate_up_proj.normal_(std=0.02)
      experts.down_proj.normal_(std=0.02)

    hidden_states = torch.randn(4, config.hidden_size)
    top_k_index = torch.randint(
        0, config.num_experts, (4, config.num_experts_per_tok)
    )
    top_k_weights = torch.softmax(
        torch.randn(4, config.num_experts_per_tok), dim=-1
    )

    with torch.no_grad():
      expected_output = experts(hidden_states, top_k_index, top_k_weights)
      actual_output = moe.litert_moe_experts_forward(
          experts, hidden_states, top_k_index, top_k_weights
      )

    self.assertTrue(
        torch.allclose(expected_output, actual_output, rtol=1e-4, atol=1e-4)
    )

    export_lib.pre_flatten_model_experts(experts)
    self.assertTrue(hasattr(experts, "flattened_gate_weight"))
    self.assertTrue(hasattr(experts, "flattened_ff1_weight"))
    self.assertTrue(hasattr(experts, "flattened_linear_weight"))

    with torch.no_grad():
      flattened_output = moe.litert_moe_experts_forward(
          experts, hidden_states, top_k_index, top_k_weights
      )

    self.assertTrue(
        torch.allclose(expected_output, flattened_output, rtol=1e-4, atol=1e-4)
    )

  def test_fused_flint_mlp(self):
    config = _get_dummy_flint_config()
    torch.manual_seed(42)
    orig_mlp = modeling_flint.FlintMLP(config)
    fused_mlp = patch.FusedFlintMLP(orig_mlp, use_swiglu_composite=True)

    x = torch.randn(2, 4, config.hidden_size)
    with torch.no_grad():
      expected = orig_mlp(x)
      actual = fused_mlp(x)

    self.assertTrue(torch.allclose(expected, actual, rtol=1e-5, atol=1e-5))

  @parameterized.named_parameters(
      ("full_attention_no_rope", 0),
      ("sliding_attention_with_rope", 1),
  )
  def test_fused_flint_attention(self, layer_idx):
    config = _get_dummy_flint_config()
    torch.manual_seed(42)
    orig_attn = modeling_flint.FlintAttention(config, layer_idx=layer_idx)
    with torch.no_grad():
      orig_attn.sinks.normal_(std=0.02)
    fused_attn = patch.FusedFlintAttention(
        orig_attn,
        fuse_qkv=True,
        use_rope_composite=True,
        use_qkv_norm_rope_composite=True,
    )

    batch_size = 2
    seq_len = 4
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    rotary_emb = modeling_flint.FlintRotaryEmbedding(config)
    position_embeddings = rotary_emb(hidden_states, position_ids)
    attention_mask = torch.zeros(
        (batch_size, 1, seq_len, seq_len), dtype=torch.float32
    )

    with torch.no_grad():
      expected, _ = orig_attn(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
          attention_mask=attention_mask,
          position_ids=position_ids,
      )
      actual, _ = fused_attn(
          hidden_states=hidden_states,
          position_embeddings=position_embeddings,
          attention_mask=attention_mask,
          position_ids=position_ids,
      )

    self.assertTrue(torch.allclose(expected, actual, rtol=1e-4, atol=1e-4))

  def test_flint_for_causal_lm_end_to_end_patch_equivalence(self):
    torch.manual_seed(42)
    config = _get_dummy_flint_config()
    ref_model = modeling_flint.FlintForCausalLM(config).eval()

    with patch.flint_litert_patch():
      patched_config = copy.deepcopy(config)
      patched_config._experts_implementation = "litert_moe"
      patched_model = modeling_flint.FlintForCausalLM(patched_config).eval()
    patched_model.load_state_dict(ref_model.state_dict())
    moe.bind_per_expert_scale(patched_model)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
      expected_logits = ref_model(input_ids=input_ids).logits

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        fuse_gate_up=True,
        fuse_qkv=True,
        use_rope_composite=True,
        use_qkv_norm_rope_composite=True,
        use_swiglu_composite=True,
    )
    with patch.patch_flint_model(patched_model, export_config):
      with torch.no_grad():
        actual_logits = patched_model(input_ids=input_ids).logits

    torch.testing.assert_close(
        actual_logits, expected_logits, rtol=1e-4, atol=1e-4
    )

  def test_flint_converter_v2_export_and_4bit_quantize(self):
    torch.manual_seed(42)
    config = _get_dummy_flint_config()
    ref_model = modeling_flint.FlintForCausalLM(config).eval()
    with torch.no_grad():
      for p in ref_model.parameters():
        p.normal_(std=0.02)

    ckpt_dir = self.create_tempdir("flint_ckpt").full_path
    output_dir = self.create_tempdir("flint_out").full_path
    config.save_pretrained(ckpt_dir)
    raw_sd = {}
    for k, v in ref_model.state_dict().items():
      if k.endswith(".self_attn.sinks"):
        raw_sd[k.removesuffix(".sinks") + ".softmax_offset"] = v
      elif k.endswith(".mlp.experts.gate_up_proj"):
        prefix = k.removesuffix("gate_up_proj")
        for idx in range(v.shape[0]):
          gate_w, up_w = v[idx].chunk(2, dim=0)
          raw_sd[f"{prefix}{idx}.gate_proj.weight"] = gate_w.contiguous()
          raw_sd[f"{prefix}{idx}.up_proj.weight"] = up_w.contiguous()
      elif k.endswith(".mlp.experts.down_proj"):
        prefix = k.removesuffix("down_proj")
        for idx in range(v.shape[0]):
          raw_sd[f"{prefix}{idx}.down_proj.weight"] = v[idx].contiguous()
      else:
        raw_sd[k] = v
    safetensors.torch.save_file(
        raw_sd, os.path.join(ckpt_dir, "model.safetensors")
    )

    with mock.patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
      mock_tok.return_value = mock.MagicMock()
      source_artifacts = export_lib.load_model(
          ckpt_dir,
          exportable_module_config.ExportableModuleConfig(
              model=ckpt_dir,
              output_dir=output_dir,
              work_dir=output_dir,
              use_v2=True,
              moe_exports_implementation="litert_moe",
              quantization_recipe="dynamic_wi4b32_afp32",
              prefill_lengths=[16],
              cache_length=32,
              externalize_embedder=False,
              split_cache=False,
          ),
      )
    export_config = export_lib.update_export_config(
        exportable_module_config.ExportableModuleConfig(
            model=ckpt_dir,
            output_dir=output_dir,
            work_dir=output_dir,
            use_v2=True,
            delete_in_memory_params=True,
            moe_exports_implementation="litert_moe",
            quantization_recipe="dynamic_wi4b32_afp32",
            prefill_lengths=[16],
            cache_length=32,
            externalize_embedder=False,
            split_cache=False,
        ),
        source_artifacts,
    )
    exported = export_lib.export_text_prefill_decode_model(
        source_artifacts,
        export_config,
        export_lib.ExportedModelArtifacts(),
    )
    self.assertIsNotNone(exported.prefill_decode_model_path)
    self.assertTrue(os.path.exists(exported.prefill_decode_model_path))
    self.assertGreater(os.path.getsize(exported.prefill_decode_model_path), 0)


if __name__ == "__main__":
  googletest.main()
