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
"""Tests for the Qwen3.5 `ModelExtension` checkpoint weight streaming."""

import os
import types

from absl.testing import absltest
from litert_torch.generative.export_hf.core import weights_loader as weights_loader_lib
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.qwen3_5 import extension as qwen3_5_extension
import safetensors.torch
import torch


class Qwen35ExtensionWeightsLoaderTest(absltest.TestCase):
  """Tests Qwen3.5 RMSNorm and qkv_norm_rope transforms in Qwen3_5Extension."""

  def setUp(self):
    super().setUp()
    self.ext = qwen3_5_extension.Qwen3_5Extension()
    self.ckpt_dir = self.create_tempdir().full_path
    self.num_heads = 2
    self.num_kv_heads = 1
    self.head_dim = 8
    self.hidden_size = 16
    # rotary_dim = 8 * 0.25 = 2 -> half_rot = 1, half_dim = 4
    # perm = [0, 2, 3, 4, 1, 5, 6, 7]
    self.expected_perm = torch.tensor(
        [0, 2, 3, 4, 1, 5, 6, 7], dtype=torch.long
    )

    self.q_only = torch.arange(
        self.num_heads * self.head_dim * self.hidden_size, dtype=torch.float32
    ).view(self.num_heads, self.head_dim, self.hidden_size)
    self.gate = (
        torch.arange(
            self.num_heads * self.head_dim * self.hidden_size,
            dtype=torch.float32,
        ).view(self.num_heads, self.head_dim, self.hidden_size)
        + 1000.0
    )
    q_proj_w = torch.stack([self.q_only, self.gate], dim=1).reshape(
        self.num_heads * 2 * self.head_dim, self.hidden_size
    )
    self.k_proj_w = (
        torch.arange(
            self.num_kv_heads * self.head_dim * self.hidden_size,
            dtype=torch.float32,
        ).view(self.num_kv_heads * self.head_dim, self.hidden_size)
        + 2000.0
    )
    self.v_proj_w = (
        torch.arange(
            self.num_kv_heads * self.head_dim * self.hidden_size,
            dtype=torch.float32,
        ).view(self.num_kv_heads * self.head_dim, self.hidden_size)
        + 3000.0
    )

    self.q_only_b = torch.arange(
        self.num_heads * self.head_dim, dtype=torch.float32
    ).view(self.num_heads, self.head_dim)
    self.gate_b = (
        torch.arange(self.num_heads * self.head_dim, dtype=torch.float32).view(
            self.num_heads, self.head_dim
        )
        + 100.0
    )
    q_proj_b = torch.stack([self.q_only_b, self.gate_b], dim=1).reshape(-1)
    self.k_proj_b = (
        torch.arange(self.num_kv_heads * self.head_dim, dtype=torch.float32)
        + 200.0
    )
    self.v_proj_b = (
        torch.arange(self.num_kv_heads * self.head_dim, dtype=torch.float32)
        + 300.0
    )

    self.q_norm_w = torch.arange(self.head_dim, dtype=torch.float32) + 10.0
    self.k_norm_w = torch.arange(self.head_dim, dtype=torch.float32) + 20.0
    self.final_norm_w = (
        torch.arange(self.hidden_size, dtype=torch.float32) + 0.5
    )
    self.linear_attn_norm_w = (
        torch.arange(self.head_dim, dtype=torch.float32) + 5.0
    )

    safetensors.torch.save_file(
        {
            "model.norm.weight": self.final_norm_w,
            "model.layers.0.linear_attn.norm.weight": self.linear_attn_norm_w,
            "model.layers.0.self_attn.q_proj.weight": q_proj_w,
            "model.layers.0.self_attn.k_proj.weight": self.k_proj_w,
            "model.layers.0.self_attn.v_proj.weight": self.v_proj_w,
            "model.layers.0.self_attn.q_proj.bias": q_proj_b,
            "model.layers.0.self_attn.k_proj.bias": self.k_proj_b,
            "model.layers.0.self_attn.v_proj.bias": self.v_proj_b,
            "model.layers.0.self_attn.q_norm.weight": self.q_norm_w,
            "model.layers.0.self_attn.k_norm.weight": self.k_norm_w,
        },
        os.path.join(self.ckpt_dir, "model.safetensors"),
    )
    self.config = types.SimpleNamespace(
        model_type="qwen3_5_text",
        num_attention_heads=self.num_heads,
        num_key_value_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        hidden_size=self.hidden_size,
        rope_parameters={"partial_rotary_factor": 0.25},
    )

  def _make_loader(self, use_qkv_norm_rope: bool):
    export_config = types.SimpleNamespace(
        use_qkv_norm_rope_composite=use_qkv_norm_rope
    )
    return weights_loader_lib.HFCheckpointWeightsLoader(
        self.ckpt_dir,
        extra_fused_specs=self.ext.fused_weight_specs,
        tensor_resolver=self.ext.resolve_checkpoint_tensor,
        config=self.config,
        export_config=export_config,
    )

  def test_resolves_in_proj_zba_fused_weight_spec(self):
    self.assertIn(
        extension_lib.FusedWeightSpec(
            "in_proj_zba", ("in_proj_z", "in_proj_b", "in_proj_a"), dim=0
        ),
        self.ext.fused_weight_specs,
    )

  def test_deinterleaves_q_gate_and_permutes_partial_rope(self):
    loader = self._make_loader(use_qkv_norm_rope=True)

    qkv_w = loader("static_model.model.layers.0.self_attn.qkv_proj.weight")
    expected_q_w = self.q_only[:, self.expected_perm, :].reshape(
        self.num_heads * self.head_dim, self.hidden_size
    )
    expected_k_w = self.k_proj_w.view(
        self.num_kv_heads, self.head_dim, self.hidden_size
    )[:, self.expected_perm, :].reshape(
        self.num_kv_heads * self.head_dim, self.hidden_size
    )
    expected_gate_w = self.gate.reshape(
        self.num_heads * self.head_dim, self.hidden_size
    )
    torch.testing.assert_close(
        qkv_w,
        torch.cat(
            [expected_q_w, expected_k_w, self.v_proj_w, expected_gate_w], dim=0
        ),
    )

    qkv_b = loader("static_model.model.layers.0.self_attn.qkv_proj.bias")
    expected_q_b = self.q_only_b[:, self.expected_perm].reshape(-1)
    expected_k_b = self.k_proj_b.view(self.num_kv_heads, self.head_dim)[
        :, self.expected_perm
    ].reshape(-1)
    expected_gate_b = self.gate_b.reshape(-1)
    torch.testing.assert_close(
        qkv_b,
        torch.cat(
            [expected_q_b, expected_k_b, self.v_proj_b, expected_gate_b], dim=0
        ),
    )

    torch.testing.assert_close(
        loader("static_model.model.layers.0.self_attn.q_norm.weight"),
        (self.q_norm_w.to(torch.float32) + 1.0)[self.expected_perm],
    )
    torch.testing.assert_close(
        loader("static_model.model.layers.0.self_attn.k_norm.weight"),
        (self.k_norm_w.to(torch.float32) + 1.0)[self.expected_perm],
    )
    torch.testing.assert_close(
        loader("static_model.model.norm.weight"),
        self.final_norm_w.to(torch.float32) + 1.0,
    )
    # Gated RMSNorm on GatedDeltaNet uses standard weights without +1.0 shift.
    torch.testing.assert_close(
        loader("static_model.model.layers.0.linear_attn.norm.weight"),
        self.linear_attn_norm_w,
    )

  def test_keeps_raw_layout_when_composite_disabled(self):
    loader = self._make_loader(use_qkv_norm_rope=False)
    torch.testing.assert_close(
        loader("model.layers.0.self_attn.q_norm.weight"),
        self.q_norm_w.to(torch.float32) + 1.0,
    )
    qkv_w = loader("model.layers.0.self_attn.qkv_proj.weight")
    raw_q_w = loader("model.layers.0.self_attn.q_proj.weight")
    torch.testing.assert_close(
        qkv_w, torch.cat([raw_q_w, self.k_proj_w, self.v_proj_w], dim=0)
    )


if __name__ == "__main__":
  absltest.main()
