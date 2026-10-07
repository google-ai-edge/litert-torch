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
"""Tests for Qwen3.5 model patch."""

import importlib.metadata
from typing import Any

orig_version = importlib.metadata.version


def patched_version(distribution_name: str) -> str:
  if distribution_name == "torchao":
    return "0.4.0"
  return orig_version(distribution_name)


importlib.metadata.version = patched_version

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.qwen3_5 import modeling_qwen3_5_static
from litert_torch.generative.export_hf.model_ext.qwen3_5 import patch
import torch
from transformers.models.qwen3_5 import configuration_qwen3_5
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5 import modular_qwen3_5


class Qwen3_5PatchTest(parameterized.TestCase):

  def _make_config(self, partial_rotary_factor: float = 0.25) -> Any:
    cfg = configuration_qwen3_5.Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": partial_rotary_factor,
        },
        rms_norm_eps=1e-6,
    )
    cfg._attn_implementation = "eager"
    return cfg

  def test_qwen3_5_rms_norm_equivalence(self):
    torch.manual_seed(0)
    orig_norm = modeling_qwen3_5.Qwen3_5RMSNorm(64, eps=1e-6)
    orig_norm.weight.data.uniform_(-0.2, 0.2)

    fused_norm = patch.Qwen3_5RMSNorm(64, eps=1e-6)
    fused_norm.weight = torch.nn.Parameter(
        orig_norm.weight.to(torch.float32) + 1.0
    )

    self.assertIn("eps=1e-06", fused_norm.extra_repr())
    x = torch.randn(2, 5, 64)
    expected = orig_norm(x)
    actual = fused_norm(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name="fused_gate_up_only", use_swiglu_composite=False),
      dict(testcase_name="swiglu_composite", use_swiglu_composite=True),
  )
  def test_fused_qwen3_5_mlp_equivalence(self, use_swiglu_composite: bool):
    torch.manual_seed(0)
    cfg = self._make_config()
    orig_mlp = modeling_qwen3_5.Qwen3_5MLP(cfg, cfg.intermediate_size).eval()
    fused_mlp = patch.FusedQwen3_5MLP(
        orig_mlp, use_swiglu_composite=use_swiglu_composite
    ).eval()

    x = torch.randn(1, 4, cfg.hidden_size)
    with torch.no_grad():
      expected = orig_mlp(x)
      actual = fused_mlp(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  def test_fused_qwen3_5_mlp_rejects_non_silu_activation(self):
    cfg = self._make_config()
    orig_mlp = modeling_qwen3_5.Qwen3_5MLP(cfg, cfg.intermediate_size).eval()
    orig_mlp.act_fn = torch.nn.GELU()
    with self.assertRaisesRegex(ValueError, "Expected SiLU activation"):
      patch.FusedQwen3_5MLP(orig_mlp, use_swiglu_composite=True)

  @parameterized.named_parameters(
      dict(
          testcase_name="fuse_qkv_only",
          fuse_qkv=True,
          use_rope_composite=False,
          partial_rotary_factor=0.25,
          with_bias=False,
      ),
      dict(
          testcase_name="fuse_qkv_with_bias",
          fuse_qkv=True,
          use_rope_composite=False,
          partial_rotary_factor=0.25,
          with_bias=True,
      ),
      dict(
          testcase_name="rope_composite_partial",
          fuse_qkv=False,
          use_rope_composite=True,
          partial_rotary_factor=0.25,
          with_bias=False,
      ),
      dict(
          testcase_name="rope_composite_full",
          fuse_qkv=False,
          use_rope_composite=True,
          partial_rotary_factor=1.0,
          with_bias=False,
      ),
      dict(
          testcase_name="fuse_qkv_and_rope_composite",
          fuse_qkv=True,
          use_rope_composite=True,
          partial_rotary_factor=0.25,
          with_bias=False,
      ),
  )
  def test_fused_qwen3_5_attention_equivalence(
      self,
      fuse_qkv: bool,
      use_rope_composite: bool,
      partial_rotary_factor: float,
      with_bias: bool,
  ):
    torch.manual_seed(0)
    cfg = self._make_config(partial_rotary_factor=partial_rotary_factor)
    orig_attn = modeling_qwen3_5.Qwen3_5Attention(cfg, layer_idx=1).eval()
    if with_bias:
      orig_attn.q_proj.bias = torch.nn.Parameter(
          torch.randn(orig_attn.q_proj.out_features)
      )
      orig_attn.k_proj.bias = torch.nn.Parameter(
          torch.randn(orig_attn.k_proj.out_features)
      )
      orig_attn.v_proj.bias = torch.nn.Parameter(
          torch.randn(orig_attn.v_proj.out_features)
      )

    fused_attn = patch.FusedQwen3_5Attention(
        orig_attn,
        fuse_qkv=fuse_qkv,
        use_rope_composite=use_rope_composite,
    ).eval()

    x = torch.randn(1, 4, cfg.hidden_size)
    rotary_emb = modeling_qwen3_5_static.Qwen3_5StaticRotaryEmbedding(cfg)
    pos_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
    pos_emb = rotary_emb(x, pos_ids)

    with torch.no_grad():
      expected, _ = orig_attn(x, pos_emb, attention_mask=None)
      actual, _ = fused_attn(
          x, pos_emb, attention_mask=None, position_ids=pos_ids
      )
      torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

      if use_rope_composite:
        actual_default_pos, _ = fused_attn(x, pos_emb, attention_mask=None)
        torch.testing.assert_close(
            actual_default_pos, expected, rtol=1e-5, atol=1e-5
        )
        actual_1d_pos, _ = fused_attn(
            x, pos_emb, attention_mask=None, position_ids=pos_ids.squeeze(0)
        )
        torch.testing.assert_close(
            actual_1d_pos, expected, rtol=1e-5, atol=1e-5
        )

  @parameterized.named_parameters(
      dict(
          testcase_name="rmsnorm_only",
          fuse_gate_up=False,
          fuse_qkv=False,
          use_swiglu_composite=False,
          use_rope_composite=False,
      ),
      dict(
          testcase_name="fuse_gate_up_only",
          fuse_gate_up=True,
          fuse_qkv=False,
          use_swiglu_composite=False,
          use_rope_composite=False,
      ),
      dict(
          testcase_name="swiglu_composite_only",
          fuse_gate_up=False,
          fuse_qkv=False,
          use_swiglu_composite=True,
          use_rope_composite=False,
      ),
      dict(
          testcase_name="fuse_qkv_only",
          fuse_gate_up=False,
          fuse_qkv=True,
          use_swiglu_composite=False,
          use_rope_composite=False,
      ),
      dict(
          testcase_name="rope_composite_only",
          fuse_gate_up=False,
          fuse_qkv=False,
          use_swiglu_composite=False,
          use_rope_composite=True,
      ),
      dict(
          testcase_name="all_combined",
          fuse_gate_up=True,
          fuse_qkv=True,
          use_swiglu_composite=True,
          use_rope_composite=True,
      ),
  )
  def test_patch_qwen3_5_model_combinations(
      self,
      fuse_gate_up: bool,
      fuse_qkv: bool,
      use_swiglu_composite: bool,
      use_rope_composite: bool,
  ):
    torch.manual_seed(0)
    cfg = self._make_config()
    model = modeling_qwen3_5_static.Qwen3_5StaticForCausalLM(cfg).eval()
    for module in model.modules():
      if isinstance(
          module,
          (modeling_qwen3_5.Qwen3_5RMSNorm, modular_qwen3_5.Qwen3_5RMSNorm),
      ):
        module.weight.data.uniform_(-0.1, 0.1)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    positions = torch.arange(4, dtype=torch.int32).unsqueeze(0)

    with torch.no_grad():
      expected_logits, _ = model(input_ids=input_ids, positions=positions)

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        fuse_gate_up=fuse_gate_up,
        fuse_qkv=fuse_qkv,
        use_swiglu_composite=use_swiglu_composite,
        use_rope_composite=use_rope_composite,
    )

    with patches_lib.patch_model(model, "qwen3_5", export_config):
      for layer in model.model.layers:
        self.assertIsInstance(layer.input_layernorm, patch.Qwen3_5RMSNorm)
        self.assertIsInstance(
            layer.post_attention_layernorm, patch.Qwen3_5RMSNorm
        )
        if fuse_gate_up or use_swiglu_composite:
          self.assertIsInstance(layer.mlp, patch.FusedQwen3_5MLP)
          self.assertEqual(
              layer.mlp.use_swiglu_composite, use_swiglu_composite
          )
        else:
          self.assertIsInstance(
              layer.mlp,
              (modeling_qwen3_5.Qwen3_5MLP, modular_qwen3_5.Qwen3_5MLP),
          )

        if hasattr(layer, "self_attn"):
          if fuse_qkv or use_rope_composite:
            self.assertIsInstance(
                layer.self_attn, patch.FusedQwen3_5Attention
            )
            self.assertEqual(layer.self_attn.fuse_qkv, fuse_qkv)
            self.assertEqual(
                layer.self_attn.use_rope_composite, use_rope_composite
            )
          else:
            self.assertIsInstance(
                layer.self_attn, modeling_qwen3_5.Qwen3_5Attention
            )

      with torch.no_grad():
        actual_logits, _ = model(input_ids=input_ids, positions=positions)
      torch.testing.assert_close(
          actual_logits, expected_logits, rtol=1e-4, atol=1e-4
      )

    # Verify modules are restored on exit
    for layer in model.model.layers:
      self.assertIsInstance(
          layer.input_layernorm,
          (modeling_qwen3_5.Qwen3_5RMSNorm, modular_qwen3_5.Qwen3_5RMSNorm),
      )
      self.assertIsInstance(
          layer.mlp,
          (modeling_qwen3_5.Qwen3_5MLP, modular_qwen3_5.Qwen3_5MLP),
      )
      if hasattr(layer, "self_attn"):
        self.assertIsInstance(
            layer.self_attn, modeling_qwen3_5.Qwen3_5Attention
        )


if __name__ == "__main__":
  absltest.main()
