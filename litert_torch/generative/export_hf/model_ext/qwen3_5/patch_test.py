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
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.qwen3_5 import modeling_qwen3_5_static
from litert_torch.generative.export_hf.model_ext.qwen3_5 import patch
import torch
from transformers.models.qwen3_5 import configuration_qwen3_5
from transformers.models.qwen3_5 import modeling_qwen3_5


class Qwen3_5PatchTest(absltest.TestCase):

  def _make_config(self) -> Any:
    return configuration_qwen3_5.Qwen3_5TextConfig(
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
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        rms_norm_eps=1e-6,
    )

  def test_qwen3_5_rms_norm_equivalence(self):
    torch.manual_seed(0)
    orig_norm = modeling_qwen3_5.Qwen3_5RMSNorm(64, eps=1e-6)
    orig_norm.weight.data.uniform_(-0.2, 0.2)

    fused_norm = patch.Qwen3_5RMSNorm(64, eps=1e-6)
    fused_norm.weight = torch.nn.Parameter(
        orig_norm.weight.to(torch.float32) + 1.0
    )

    x = torch.randn(2, 5, 64)
    expected = orig_norm(x)
    actual = fused_norm(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  def test_fused_qwen3_5_mlp_equivalence(self):
    torch.manual_seed(0)
    cfg = self._make_config()
    orig_mlp = modeling_qwen3_5.Qwen3_5MLP(cfg, cfg.intermediate_size).eval()

    fused_mlp = patch.FusedQwen3_5MLP(orig_mlp, use_swiglu_composite=False).eval()
    fused_mlp_composite = patch.FusedQwen3_5MLP(
        orig_mlp, use_swiglu_composite=True
    ).eval()

    x = torch.randn(1, 4, cfg.hidden_size)
    expected = orig_mlp(x)
    actual = fused_mlp(x)
    actual_composite = fused_mlp_composite(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        actual_composite, expected, rtol=1e-5, atol=1e-5
    )

  def test_fused_qwen3_5_attention_equivalence(self):
    torch.manual_seed(0)
    cfg = self._make_config()
    cfg._attn_implementation = "eager"
    orig_attn = modeling_qwen3_5.Qwen3_5Attention(cfg, layer_idx=1).eval()

    fused_attn = patch.FusedQwen3_5Attention(
        orig_attn, fuse_qkv=True, use_rope_composite=False
    ).eval()

    x = torch.randn(1, 4, cfg.hidden_size)
    rotary_emb = modeling_qwen3_5_static.Qwen3_5StaticRotaryEmbedding(cfg)
    pos_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
    pos_emb = rotary_emb(x, pos_ids)

    expected, _ = orig_attn(x, pos_emb, attention_mask=None)
    actual, _ = fused_attn(x, pos_emb, attention_mask=None)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  def test_patch_qwen3_5_model_context_manager(self):
    torch.manual_seed(0)
    cfg = self._make_config()
    model = modeling_qwen3_5_static.Qwen3_5StaticForCausalLM(cfg).eval()
    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        fuse_gate_up=True,
        fuse_qkv=True,
        use_swiglu_composite=True,
        use_rope_composite=True,
    )

    with patches_lib.patch_model(model, "qwen3_5", export_config):
      for layer in model.model.layers:
        self.assertIsInstance(layer.mlp, patch.FusedQwen3_5MLP)
        self.assertIsInstance(layer.input_layernorm, patch.Qwen3_5RMSNorm)
        if hasattr(layer, "self_attn"):
          self.assertIsInstance(layer.self_attn, patch.FusedQwen3_5Attention)


if __name__ == "__main__":
  absltest.main()
