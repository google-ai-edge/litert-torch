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
"""Unit tests for MiniCPM-o 4.5 weight loading and quantization utilities."""

import os
import tempfile

from absl.testing import absltest
import litert_torch
from litert_torch.generative.export_hf.model_ext.minicpm_o import weights_utils
import safetensors.torch
import torch
from torch import nn


class WeightsUtilsTest(absltest.TestCase):
  """Tests for weights_utils module."""

  def test_make_linear_and_make_embedding_thresholds(self):
    """Tests make_linear and make_embedding around the large-tensor limit."""
    at_thresh_linear = weights_utils.make_linear(1000, 1000, bias=True)
    self.assertEqual(at_thresh_linear.weight.shape, (1000, 1000))
    self.assertIsNotNone(at_thresh_linear.bias)
    self.assertGreater(float(at_thresh_linear.weight.abs().sum().item()), 0.0)

    above_thresh_no_bias = weights_utils.make_linear(1001, 1000, bias=False)
    self.assertEqual(above_thresh_no_bias.weight.shape, (1000, 1001))
    self.assertIsNone(above_thresh_no_bias.bias)
    self.assertEqual(float(above_thresh_no_bias.weight.abs().sum().item()), 0.0)

    above_thresh_with_bias = weights_utils.make_linear(1001, 1000, bias=True)
    self.assertIsNotNone(above_thresh_with_bias.bias)
    self.assertEqual(float(above_thresh_with_bias.bias.abs().sum().item()), 0.0)

    at_thresh_emb = weights_utils.make_embedding(1000, 1000)
    self.assertEqual(at_thresh_emb.weight.shape, (1000, 1000))
    self.assertGreater(float(at_thresh_emb.weight.abs().sum().item()), 0.0)

    above_thresh_emb = weights_utils.make_embedding(1001, 1000)
    self.assertEqual(above_thresh_emb.weight.shape, (1001, 1000))
    self.assertEqual(float(above_thresh_emb.weight.abs().sum().item()), 0.0)

  def test_get_weight_from_state_dict_and_safetensors_shard(self):
    """Tests get_weight lookup from state_dict, safetensors, or missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      shard_path = os.path.join(tmp_dir, "shard.safetensors")
      shard_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
      safetensors.torch.save_file({"from_shard": shard_tensor}, shard_path)
      mem_tensor = torch.ones(2, 2, dtype=torch.float32)
      state_dict = {"from_mem": mem_tensor}
      weight_map = {"from_shard": shard_path}

      got_mem = weights_utils.get_weight(state_dict, weight_map, "from_mem")
      self.assertIsNotNone(got_mem)
      torch.testing.assert_close(got_mem, mem_tensor)

      got_shard = weights_utils.get_weight(state_dict, weight_map, "from_shard")
      self.assertIsNotNone(got_shard)
      torch.testing.assert_close(got_shard, shard_tensor)

      got_missing = weights_utils.get_weight(
          state_dict, weight_map, "nonexistent"
      )
      self.assertIsNone(got_missing)

  def test_copy_weight_primary_fallback_missing_and_mismatch(self):
    """Tests copy_weight with primary, fallback, missing, or mismatched keys."""
    state_dict = {
        "primary": torch.full((2, 3), 2.0),
        "fallback": torch.full((2, 3), 5.0),
        "wrong_shape": torch.zeros(4, 4),
    }
    weight_map: dict[str, str] = {}
    populated: set[int] = set()

    target = nn.Parameter(torch.zeros(2, 3))
    weights_utils.copy_weight(
        target,
        state_dict,
        weight_map,
        "missing_key",
        "fallback",
        strict=True,
        populated=populated,
    )
    self.assertIn(target.data_ptr(), populated)
    torch.testing.assert_close(target.data, state_dict["fallback"])

    # Missing key in non-strict mode leaves target intact without populating.
    pop_missing: set[int] = set()
    weights_utils.copy_weight(
        target,
        state_dict,
        weight_map,
        "missing_key",
        strict=False,
        populated=pop_missing,
    )
    self.assertNotIn(target.data_ptr(), pop_missing)

    # Missing key in strict mode raises KeyError.
    with self.assertRaises(KeyError):
      weights_utils.copy_weight(
          target, state_dict, weight_map, "missing_key", strict=True
      )

    # Shape mismatch in non-strict mode leaves target intact.
    pop_mismatch: set[int] = set()
    weights_utils.copy_weight(
        target,
        state_dict,
        weight_map,
        "wrong_shape",
        strict=False,
        populated=pop_mismatch,
    )
    self.assertNotIn(target.data_ptr(), pop_mismatch)
    torch.testing.assert_close(target.data, state_dict["fallback"])

    # Shape mismatch in strict mode raises ValueError.
    with self.assertRaises(ValueError):
      weights_utils.copy_weight(
          target, state_dict, weight_map, "wrong_shape", strict=True
      )

  def test_copy_qkv_in_proj(self):
    """Tests copy_qkv_in_proj for valid, missing, and mismatched Q/K/V keys."""
    state_dict = {
        "in_proj.weight": torch.arange(48, dtype=torch.float32).reshape(12, 4),
        "in_proj.bias": torch.arange(12, dtype=torch.float32),
        "bad.weight": torch.zeros(2, 4),
        "bad.bias": torch.zeros(2),
    }
    q_proj = nn.Linear(4, 4, bias=True)
    k_proj = nn.Linear(4, 4, bias=True)
    v_proj = nn.Linear(4, 4, bias=True)
    populated: set[int] = set()

    weights_utils.copy_qkv_in_proj(
        q_proj,
        k_proj,
        v_proj,
        state_dict,
        {},
        "in_proj.weight",
        "in_proj.bias",
        strict=True,
        populated=populated,
    )
    self.assertIn(q_proj.weight.data_ptr(), populated)
    self.assertIn(k_proj.weight.data_ptr(), populated)
    self.assertIn(v_proj.weight.data_ptr(), populated)
    self.assertIn(q_proj.bias.data_ptr(), populated)
    torch.testing.assert_close(
        q_proj.weight.data, state_dict["in_proj.weight"][:4]
    )
    torch.testing.assert_close(
        k_proj.weight.data, state_dict["in_proj.weight"][4:8]
    )
    torch.testing.assert_close(
        v_proj.weight.data, state_dict["in_proj.weight"][8:]
    )
    torch.testing.assert_close(q_proj.bias.data, state_dict["in_proj.bias"][:4])

    # Missing weight key in non-strict vs strict mode.
    weights_utils.copy_qkv_in_proj(
        q_proj,
        k_proj,
        v_proj,
        state_dict,
        {},
        "missing.weight",
        "in_proj.bias",
        strict=False,
        populated=set(),
    )
    with self.assertRaises(KeyError):
      weights_utils.copy_qkv_in_proj(
          q_proj,
          k_proj,
          v_proj,
          state_dict,
          {},
          "missing.weight",
          "in_proj.bias",
          strict=True,
          populated=set(),
      )

    # Mismatched weight shape in non-strict vs strict mode.
    weights_utils.copy_qkv_in_proj(
        q_proj,
        k_proj,
        v_proj,
        state_dict,
        {},
        "bad.weight",
        "in_proj.bias",
        strict=False,
        populated=set(),
    )
    with self.assertRaises(ValueError):
      weights_utils.copy_qkv_in_proj(
          q_proj,
          k_proj,
          v_proj,
          state_dict,
          {},
          "bad.weight",
          "in_proj.bias",
          strict=True,
          populated=set(),
      )

    # Missing bias key in non-strict vs strict mode.
    weights_utils.copy_qkv_in_proj(
        q_proj,
        k_proj,
        v_proj,
        state_dict,
        {},
        "in_proj.weight",
        "missing.bias",
        strict=False,
        populated=set(),
    )
    with self.assertRaises(KeyError):
      weights_utils.copy_qkv_in_proj(
          q_proj,
          k_proj,
          v_proj,
          state_dict,
          {},
          "in_proj.weight",
          "missing.bias",
          strict=True,
          populated=set(),
      )

    # Mismatched bias shape in non-strict vs strict mode.
    weights_utils.copy_qkv_in_proj(
        q_proj,
        k_proj,
        v_proj,
        state_dict,
        {},
        "in_proj.weight",
        "bad.bias",
        strict=False,
        populated=set(),
    )
    with self.assertRaises(ValueError):
      weights_utils.copy_qkv_in_proj(
          q_proj,
          k_proj,
          v_proj,
          state_dict,
          {},
          "in_proj.weight",
          "bad.bias",
          strict=True,
          populated=set(),
      )

  def test_copy_weight_norm_direct_and_parametrized(self):
    """Tests copy_weight_norm for direct weight_key and parametrized weights."""
    direct_w = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    orig0 = torch.tensor([[2.0], [5.0]])
    orig1 = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    state_dict = {
        "direct.weight": direct_w,
        "bad_direct.weight": torch.zeros(1, 2),
        "wn.orig0": orig0,
        "wn.orig1": orig1,
        "bad_wn.orig0": torch.ones(1, 1),
        "bad_wn.orig1": torch.ones(1, 2),
    }

    # 1. Direct weight_key branch (matching shape).
    target_direct = nn.Parameter(torch.zeros(2, 2))
    populated_direct: set[int] = set()
    weights_utils.copy_weight_norm(
        target_direct,
        state_dict,
        {},
        "direct.weight",
        "missing.orig0",
        "missing.orig1",
        strict=True,
        populated=populated_direct,
    )
    self.assertIn(target_direct.data_ptr(), populated_direct)
    torch.testing.assert_close(target_direct.data, direct_w)

    # Direct weight_key branch with shape mismatch (non-strict and strict).
    weights_utils.copy_weight_norm(
        target_direct,
        state_dict,
        {},
        "bad_direct.weight",
        "missing.orig0",
        "missing.orig1",
        strict=False,
        populated=set(),
    )
    torch.testing.assert_close(target_direct.data, direct_w)
    with self.assertRaises(ValueError):
      weights_utils.copy_weight_norm(
          target_direct,
          state_dict,
          {},
          "bad_direct.weight",
          "missing.orig0",
          "missing.orig1",
          strict=True,
          populated=set(),
      )

    # 2. Parametrized (orig0, orig1) branch:
    # Row 0: 2.0 * [3, 4] / 5 = [1.2, 1.6]; Row 1: 5.0 * [0, 2] / 2 = [0.0, 5.0]
    target_wn = nn.Parameter(torch.zeros(2, 2))
    populated_wn: set[int] = set()
    weights_utils.copy_weight_norm(
        target_wn,
        state_dict,
        {},
        "missing.weight",
        "wn.orig0",
        "wn.orig1",
        strict=True,
        populated=populated_wn,
    )
    self.assertIn(target_wn.data_ptr(), populated_wn)
    expected_wn = torch.tensor([[1.2, 1.6], [0.0, 5.0]])
    torch.testing.assert_close(
        target_wn.data, expected_wn, atol=1e-6, rtol=1e-6
    )

    # Parametrized branch with shape mismatch (non-strict and strict).
    weights_utils.copy_weight_norm(
        target_wn,
        state_dict,
        {},
        "missing.weight",
        "bad_wn.orig0",
        "bad_wn.orig1",
        strict=False,
        populated=set(),
    )
    with self.assertRaises(ValueError):
      weights_utils.copy_weight_norm(
          target_wn,
          state_dict,
          {},
          "missing.weight",
          "bad_wn.orig0",
          "bad_wn.orig1",
          strict=True,
          populated=set(),
      )

    # Missing both direct and parametrized keys (non-strict and strict).
    weights_utils.copy_weight_norm(
        target_wn,
        state_dict,
        {},
        "missing.weight",
        "missing.orig0",
        "missing.orig1",
        strict=False,
        populated=set(),
    )
    with self.assertRaises(KeyError):
      weights_utils.copy_weight_norm(
          target_wn,
          state_dict,
          {},
          "missing.weight",
          "missing.orig0",
          "missing.orig1",
          strict=True,
          populated=set(),
      )

  def test_copy_from_dict_linear_wn_and_verify_populated(self):
    """Tests copy_from_dict, copy_linear_from_dict, and copy_wn_from_dict."""
    sd = {
        "lin.weight": torch.full((2, 3), 3.0),
        "lin.bias": torch.full((2,), 1.0),
        "mismatch.weight": torch.ones(5, 5),
        "conv_direct.weight": torch.full((2, 3, 1), 4.0),
        "conv_direct.bias": torch.full((2,), 0.5),
        "conv_bad.weight": torch.ones(9, 9, 9),
        "conv_wn.parametrizations.weight.original0": torch.full((2, 1, 1), 2.0),
        "conv_wn.parametrizations.weight.original1": torch.full((2, 1, 1), 1.0),
        "conv_wn.bias": torch.full((2,), -1.0),
        "conv_wn_bad.parametrizations.weight.original0": torch.ones(5, 1, 1),
        "conv_wn_bad.parametrizations.weight.original1": torch.ones(5, 1, 1),
    }
    lin = nn.Linear(3, 2, bias=True)
    populated: set[int] = set()
    weights_utils.copy_linear_from_dict(
        lin, sd, "lin", strict=True, populated=populated
    )
    weights_utils.verify_all_parameters_populated(lin, populated, "lin")
    torch.testing.assert_close(lin.weight.data, sd["lin.weight"])
    torch.testing.assert_close(lin.bias.data, sd["lin.bias"])

    # In non-strict mode, shape mismatch leaves target unchanged and unpopulated
    # (no silent slicing/padding).
    sentinel = torch.full((2, 3), -7.0)
    target = nn.Parameter(sentinel.clone())
    pop_mismatch: set[int] = set()
    weights_utils.copy_from_dict(
        target,
        sd,
        "mismatch.weight",
        strict=False,
        populated=pop_mismatch,
    )
    self.assertNotIn(target.data_ptr(), pop_mismatch)
    torch.testing.assert_close(target.data, sentinel)

    with self.assertRaises(ValueError):
      weights_utils.copy_from_dict(
          target,
          sd,
          "mismatch.weight",
          strict=True,
          populated=pop_mismatch,
      )

    # Missing key in non-strict vs strict mode.
    weights_utils.copy_from_dict(
        target, sd, "missing.key", strict=False, populated=pop_mismatch
    )
    self.assertNotIn(target.data_ptr(), pop_mismatch)
    with self.assertRaises(KeyError):
      weights_utils.copy_from_dict(
          target, sd, "missing.key", strict=True, populated=pop_mismatch
      )

    # copy_wn_from_dict with direct weight key and with parametrized keys.
    conv_direct = nn.Conv1d(3, 2, kernel_size=1)
    pop_conv: set[int] = set()
    weights_utils.copy_wn_from_dict(
        conv_direct,
        sd,
        "conv_direct",
        strict=True,
        populated=pop_conv,
    )
    self.assertIn(conv_direct.weight.data_ptr(), pop_conv)
    assert conv_direct.bias is not None
    self.assertIn(conv_direct.bias.data_ptr(), pop_conv)
    torch.testing.assert_close(
        conv_direct.weight.data, sd["conv_direct.weight"]
    )

    # Direct weight shape mismatch in non-strict and strict mode.
    weights_utils.copy_wn_from_dict(
        conv_direct, sd, "conv_bad", strict=False, populated=set()
    )
    with self.assertRaises(ValueError):
      weights_utils.copy_wn_from_dict(
          conv_direct, sd, "conv_bad", strict=True, populated=set()
      )

    conv_wn = nn.Conv1d(1, 2, kernel_size=1)
    pop_wn: set[int] = set()
    weights_utils.copy_wn_from_dict(
        conv_wn, sd, "conv_wn", strict=True, populated=pop_wn
    )
    self.assertIn(conv_wn.weight.data_ptr(), pop_wn)
    assert conv_wn.bias is not None
    self.assertIn(conv_wn.bias.data_ptr(), pop_wn)
    torch.testing.assert_close(conv_wn.weight.data, torch.full((2, 1, 1), 2.0))

    # Parametrized weight shape mismatch in non-strict and strict mode.
    weights_utils.copy_wn_from_dict(
        conv_wn, sd, "conv_wn_bad", strict=False, populated=set()
    )
    with self.assertRaises(ValueError):
      weights_utils.copy_wn_from_dict(
          conv_wn, sd, "conv_wn_bad", strict=True, populated=set()
      )

    # Missing weight norm keys in non-strict and strict mode.
    weights_utils.copy_wn_from_dict(
        conv_wn, sd, "nonexistent", strict=False, populated=set()
    )
    with self.assertRaises(KeyError):
      weights_utils.copy_wn_from_dict(
          conv_wn, sd, "nonexistent", strict=True, populated=set()
      )

    # verify_all_parameters_populated raises ValueError when a param is missing.
    with self.assertRaises(ValueError):
      weights_utils.verify_all_parameters_populated(lin, set(), "lin")

  def test_maybe_quantize(self):
    """Tests maybe_quantize with disabled and enabled quantization recipes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      emb = nn.Embedding(16, 8).eval()
      sample_ids = torch.tensor([[1, 2]], dtype=torch.int32)
      raw_path = os.path.join(tmp_dir, "tiny_fp32.tflite")
      litert_torch.convert(emb, sample_kwargs={"input": sample_ids}).export(
          raw_path
      )

      for noop_recipe in (None, "", "none", "None"):
        out = weights_utils.maybe_quantize(raw_path, noop_recipe)
        self.assertEqual(out, raw_path)
        self.assertTrue(os.path.exists(raw_path))

      out_q = weights_utils.maybe_quantize(raw_path, "dynamic_wi8_afp32")
      self.assertNotEqual(out_q, raw_path)
      self.assertTrue(os.path.exists(out_q))
      self.assertFalse(os.path.exists(raw_path))

  def test_copy_module_parameters_and_parameter_dict(self):
    """Tests copy_module_parameters and copy_parameter_dict."""
    mod = nn.Sequential(nn.Linear(4, 2, bias=True), nn.LayerNorm(2))
    sd = {
        "blk.0.weight": torch.ones(2, 4),
        "blk.0.bias": torch.full((2,), 0.5),
        "blk.1.weight": torch.full((2,), 2.0),
        "blk.1.bias": torch.full((2,), -1.0),
    }
    pop: set[int] = set()
    weights_utils.copy_module_parameters(
        mod, sd, {}, "blk", strict=True, populated=pop
    )
    self.assertLen(pop, 4)
    torch.testing.assert_close(mod[0].weight, sd["blk.0.weight"])
    torch.testing.assert_close(mod[1].bias, sd["blk.1.bias"])

    # Exercise param_map (including identity fallback for unmapped params) with
    # an empty prefix.
    sd_mapped = {
        "linear.w": torch.full((2, 4), -2.0),
        "linear.b": torch.full((2,), 3.0),
        "1.weight": torch.full((2,), 4.0),
        "1.bias": torch.full((2,), -0.5),
    }
    pop_mapped: set[int] = set()
    weights_utils.copy_module_parameters(
        mod,
        sd_mapped,
        {},
        "",
        param_map={"0.weight": "linear.w", "0.bias": "linear.b"},
        strict=True,
        populated=pop_mapped,
    )
    self.assertLen(pop_mapped, 4)
    torch.testing.assert_close(mod[0].weight, sd_mapped["linear.w"])
    torch.testing.assert_close(mod[0].bias, sd_mapped["linear.b"])
    torch.testing.assert_close(mod[1].weight, sd_mapped["1.weight"])
    torch.testing.assert_close(mod[1].bias, sd_mapped["1.bias"])

    lin = nn.Linear(3, 3, bias=True)
    sd2 = {
        "primary.weight": torch.full((3, 3), 3.0),
        "primary.bias": torch.full((3,), 1.5),
    }
    pop2: set[int] = set()
    weights_utils.copy_parameter_dict(
        {
            lin.weight: ("fallback.weight", "primary.weight"),
            lin.bias: "primary.bias",
        },
        sd2,
        {},
        strict=True,
        populated=pop2,
    )
    self.assertLen(pop2, 2)
    torch.testing.assert_close(lin.weight, sd2["primary.weight"])
    torch.testing.assert_close(lin.bias, sd2["primary.bias"])


if __name__ == "__main__":
  absltest.main()
