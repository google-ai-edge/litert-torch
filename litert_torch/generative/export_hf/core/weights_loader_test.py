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
"""Tests for weights_loader."""

import json
import os
import types
from unittest import mock

from absl.testing import absltest
from litert_torch.generative.export_hf.core import weights_loader as weights_loader_lib
import safetensors.torch
import torch
from torch import nn


class _Rotary(nn.Module):

  def __init__(self):
    super().__init__()
    self.register_buffer(
        "inv_freq", torch.arange(4, dtype=torch.float32), persistent=False
    )


class _Tiny(nn.Module):

  def __init__(self):
    super().__init__()
    self.embed_tokens = nn.Embedding(8, 4)
    self.proj = nn.Linear(4, 4)
    self.rotary = _Rotary()
    self.lm_head = nn.Linear(4, 8, bias=False)
    self.lm_head.weight = self.embed_tokens.weight


class _Wrapper(nn.Module):

  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model


class _Layer(nn.Module):
  """Decoder layer with Gemma-4 style buffers."""

  def __init__(self):
    super().__init__()
    self.proj = nn.Linear(4, 4)
    self.rotary = _Rotary()
    self.register_buffer("layer_scalar", torch.ones(1))
    self.register_buffer("not_in_checkpoint", torch.zeros(2))


class _LanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.layers = nn.ModuleList([_Layer()])


class _Multimodal(nn.Module):
  """Holds the text model at `model.language_model`, as Gemma-4 does."""

  def __init__(self):
    super().__init__()
    self.model = nn.Module()
    self.model.language_model = _LanguageModel()


class InitAndCastTest(absltest.TestCase):

  def test_init_params_on_meta_keeps_buffers_real(self):
    with weights_loader_lib.init_params_on_meta():
      model = _Tiny()
    self.assertTrue(all(p.is_meta for p in model.parameters()))
    self.assertFalse(model.rotary.inv_freq.is_meta)
    torch.testing.assert_close(
        model.rotary.inv_freq, torch.arange(4, dtype=torch.float32)
    )
    # Patch is removed on exit.
    self.assertFalse(nn.Linear(2, 2).weight.is_meta)

  def test_cast_params_keeps_buffers_and_ties(self):
    model = weights_loader_lib.cast_params_(_Tiny(), torch.bfloat16)
    self.assertEqual(model.proj.weight.dtype, torch.bfloat16)
    self.assertEqual(model.rotary.inv_freq.dtype, torch.float32)
    self.assertIs(model.lm_head.weight, model.embed_tokens.weight)

  def test_submodule_weights_loader(self):
    root = _Wrapper(_Tiny())
    requested = []

    def loader(name):
      requested.append(name)
      return torch.zeros(1)

    sub_loader = weights_loader_lib.submodule_weights_loader(
        loader, root, root.model.embed_tokens
    )
    sub_loader("model.weight")
    self.assertEqual(requested, ["model.embed_tokens.weight"])
    with self.assertRaises(KeyError):
      sub_loader("other.weight")

  def test_buffers_on_meta_serves_values_and_restores(self):
    tiny = _Tiny()
    real = tiny.rotary.inv_freq
    roots = (_Wrapper(tiny), _Wrapper(tiny))
    with weights_loader_lib.buffers_on_meta(*roots) as values:
      self.assertTrue(tiny.rotary.inv_freq.is_meta)
      self.assertEqual(list(values), ["model.rotary.inv_freq"])
      self.assertIs(values["model.rotary.inv_freq"], real)

      def base_loader(name):
        raise KeyError(name)

      loader = weights_loader_lib.with_tensors(base_loader, values)
      self.assertIs(loader("model.rotary.inv_freq"), real)
      with self.assertRaises(KeyError):
        loader("model.proj.weight")
    self.assertIs(tiny.rotary.inv_freq, real)


class HFCheckpointWeightsLoaderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = self.create_tempdir().full_path
    self.tensors = {
        "model.language_model.embed_tokens.weight": torch.randn(8, 4),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(
            4, 4
        ),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(
            2, 4
        ),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(
            2, 4
        ),
        "model.language_model.layers.1.self_attn.q_proj.weight": torch.randn(
            4, 4
        ),
        "model.language_model.layers.1.self_attn.k_proj.weight": torch.randn(
            2, 4
        ),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(6, 4),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(6, 4),
        "model.language_model.layers.0.layer_scalar": torch.tensor([0.5]),
    }
    safetensors.torch.save_file(
        self.tensors, os.path.join(self.ckpt_dir, "model.safetensors")
    )
    self.loader = weights_loader_lib.HFCheckpointWeightsLoader(self.ckpt_dir)

  def test_prefix_stripping(self):
    torch.testing.assert_close(
        self.loader("model.model.language_model.embed_tokens.weight"),
        self.tensors["model.language_model.embed_tokens.weight"],
    )
    torch.testing.assert_close(
        self.loader("static_model.model.layers.0.self_attn.q_proj.weight"),
        self.tensors["model.language_model.layers.0.self_attn.q_proj.weight"],
    )

  def test_fused_projections(self):
    prefix = "model.language_model.layers.0."
    qkv = self.loader("model.model.layers.0.self_attn.qkv_proj.weight")
    torch.testing.assert_close(
        qkv,
        torch.cat([
            self.tensors[f"{prefix}self_attn.{p}_proj.weight"]
            for p in ("q", "k", "v")
        ]),
    )
    gate_up = self.loader("model.model.layers.0.mlp.gate_up_proj.weight")
    torch.testing.assert_close(
        gate_up,
        torch.cat([
            self.tensors[f"{prefix}mlp.gate_proj.weight"],
            self.tensors[f"{prefix}mlp.up_proj.weight"],
        ]),
    )
    # Gemma-4 global attention with `attention_k_eq_v=True` has no `v_proj`.
    l1_prefix = "model.language_model.layers.1."
    qk = self.loader("model.model.layers.1.self_attn.qkv_proj.weight")
    torch.testing.assert_close(
        qk,
        torch.cat([
            self.tensors[f"{l1_prefix}self_attn.q_proj.weight"],
            self.tensors[f"{l1_prefix}self_attn.k_proj.weight"],
        ]),
    )

  def test_embed_tokens_on_cpu_meta_and_real(self):
    class _ScaledEmbedding(nn.Embedding):

      def __init__(self):
        super().__init__(8, 4)
        self.register_buffer(
            "embed_scale",
            torch.tensor(2.0, dtype=torch.float32),
            persistent=False,
        )

      def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale

    class _ModelWithEmb(nn.Module):

      def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = _ScaledEmbedding()

      def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    real_model = _ModelWithEmb()
    with torch.no_grad():
      real_model.get_input_embeddings().weight.copy_(
          self.tensors["model.language_model.embed_tokens.weight"]
      )
    expected = weights_loader_lib.embed_tokens_on_cpu(real_model, [3])

    with weights_loader_lib.init_params_on_meta():
      meta_model = _ModelWithEmb()
    weights_loader_lib.load_persistent_buffers_(meta_model, self.loader)
    actual = weights_loader_lib.embed_tokens_on_cpu(meta_model, [3])
    self.assertFalse(actual.is_meta)
    torch.testing.assert_close(actual, expected)

  def test_tied_lm_head(self):
    loader = weights_loader_lib.HFCheckpointWeightsLoader(
        self.ckpt_dir, tie_word_embeddings=True
    )
    torch.testing.assert_close(
        loader("model.lm_head.weight"),
        self.tensors["model.language_model.embed_tokens.weight"],
    )

  def test_untied_missing_lm_head_raises(self):
    with self.assertRaises(KeyError):
      self.loader("model.lm_head.weight")

  def test_missing_key_raises(self):
    with self.assertRaises(KeyError):
      self.loader("model.layers.0.self_attn.o_proj.weight")

  def test_close_reopens_lazily(self):
    self.loader.close()
    self.assertIsNotNone(
        self.loader("model.language_model.embed_tokens.weight")
    )

  def test_load_persistent_buffers(self):
    with weights_loader_lib.init_params_on_meta():
      model = _Multimodal()
    layer = model.model.language_model.layers[0]
    with self.assertLogs(level="WARNING") as logs:
      weights_loader_lib.load_persistent_buffers_(model, self.loader)
    # Resolved by the same lookup as `model.language_model.layers.0.*` params.
    torch.testing.assert_close(
        layer.layer_scalar,
        self.tensors["model.language_model.layers.0.layer_scalar"],
    )
    # Non-persistent and missing buffers keep their init values.
    torch.testing.assert_close(
        layer.rotary.inv_freq, torch.arange(4, dtype=torch.float32)
    )
    torch.testing.assert_close(layer.not_in_checkpoint, torch.zeros(2))
    self.assertIn(
        "model.language_model.layers.0.not_in_checkpoint", logs.output[0]
    )
    self.assertTrue(layer.proj.weight.is_meta)

  def test_load_persistent_buffers_shape_mismatch_raises(self):
    model = _Multimodal()
    layer = model.model.language_model.layers[0]
    layer.layer_scalar = torch.ones(2)
    with self.assertRaisesRegex(ValueError, "layer_scalar"):
      weights_loader_lib.load_persistent_buffers_(model, self.loader)

  def test_empty_checkpoint_raises(self):
    with self.assertRaises(ValueError):
      weights_loader_lib.HFCheckpointWeightsLoader(
          self.create_tempdir().full_path
      )

  def test_extension_extra_prefixes_and_fused_specs_and_resolver(self):
    ckpt_dir = self.create_tempdir().full_path
    tensors = {
        "backbone.embed_tokens.weight": torch.randn(8, 4),
        "backbone.layers.0.mlp.w1.weight": torch.randn(6, 4),
        "backbone.layers.0.mlp.w3.weight": torch.randn(6, 4),
        "backbone.layers.0.attn.col_a.weight": torch.randn(4, 2),
        "backbone.layers.0.attn.col_b.weight": torch.randn(4, 2),
        "raw_packed.proj_t": torch.randn(3, 5),
    }
    safetensors.torch.save_file(
        tensors, os.path.join(ckpt_dir, "model.safetensors")
    )

    def _custom_resolver(param_name, try_load):
      if param_name.endswith("transposed_proj.weight"):
        raw = try_load("raw_packed.proj_t")
        return raw.t().contiguous() if raw is not None else None
      return None

    loader = weights_loader_lib.HFCheckpointWeightsLoader(
        ckpt_dir,
        tie_word_embeddings=True,
        extra_prefixes=("backbone",),
        extra_fused_specs=(
            ("w13", ("w1", "w3")),
            ("col_ab", ("col_a", "col_b"), 1),
        ),
        tensor_resolver=_custom_resolver,
    )
    torch.testing.assert_close(
        loader("model.lm_head.weight"),
        tensors["backbone.embed_tokens.weight"],
    )
    torch.testing.assert_close(
        loader("model.layers.0.mlp.w13.weight"),
        torch.cat(
            [
                tensors["backbone.layers.0.mlp.w1.weight"],
                tensors["backbone.layers.0.mlp.w3.weight"],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        loader("model.layers.0.attn.col_ab.weight"),
        torch.cat(
            [
                tensors["backbone.layers.0.attn.col_a.weight"],
                tensors["backbone.layers.0.attn.col_b.weight"],
            ],
            dim=1,
        ),
    )
    torch.testing.assert_close(
        loader("model.layers.0.transposed_proj.weight"),
        tensors["raw_packed.proj_t"].t(),
    )


class ResolveCheckpointDirTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cached_dir = self.create_tempdir().full_path
    self.downloaded_dir = self.create_tempdir().full_path
    safetensors.torch.save_file(
        {"w": torch.zeros(1)},
        os.path.join(self.downloaded_dir, "model.safetensors"),
    )

  def _fake_snapshot_download(self, repo_id, allow_patterns, **kwargs):
    del repo_id, allow_patterns  # Unused.
    if kwargs.get("local_files_only"):
      return self.cached_dir
    return self.downloaded_dir

  def _resolve(self) -> tuple[str, mock.MagicMock]:
    with mock.patch.object(
        weights_loader_lib.huggingface_hub,
        "snapshot_download",
        autospec=True,
        side_effect=self._fake_snapshot_download,
    ) as snapshot_download:
      resolved = weights_loader_lib.HFCheckpointWeightsLoader(
          "org/model"
      )._checkpoint_dir  # pylint: disable=protected-access
    return resolved, snapshot_download

  def test_uses_complete_cached_snapshot(self):
    safetensors.torch.save_file(
        {"w": torch.zeros(1)},
        os.path.join(self.cached_dir, "model.safetensors"),
    )
    resolved, snapshot_download = self._resolve()
    self.assertEqual(resolved, self.cached_dir)
    snapshot_download.assert_called_once()

  def test_downloads_if_cached_snapshot_has_no_weights(self):
    # E.g. only `config.json` was cached by `from_config`.
    with open(os.path.join(self.cached_dir, "config.json"), "w") as f:
      f.write("{}")
    resolved, snapshot_download = self._resolve()
    self.assertEqual(resolved, self.downloaded_dir)
    self.assertEqual(snapshot_download.call_count, 2)

  def test_downloads_if_cached_snapshot_misses_shards(self):
    with open(
        os.path.join(self.cached_dir, "model.safetensors.index.json"), "w"
    ) as f:
      json.dump(
          {
              "weight_map": {
                  "a": "model-00001-of-00002.safetensors",
                  "b": "model-00002-of-00002.safetensors",
              }
          },
          f,
      )
    safetensors.torch.save_file(
        {"a": torch.zeros(1)},
        os.path.join(self.cached_dir, "model-00001-of-00002.safetensors"),
    )
    resolved, _ = self._resolve()
    self.assertEqual(resolved, self.downloaded_dir)


class Qwen35QkvNormRopeWeightsLoaderTest(absltest.TestCase):
  """Tests Qwen3.5 qkv_norm_rope composite transforms in HFCheckpointWeightsLoader."""

  def setUp(self):
    super().setUp()
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

    safetensors.torch.save_file(
        {
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

  def test_deinterleaves_q_gate_and_permutes_partial_rope(self):
    """Verifies [Q, K, V, Gate] ordering and partial-rotary permutation."""
    export_config = types.SimpleNamespace(use_qkv_norm_rope_composite=True)
    loader = weights_loader_lib.HFCheckpointWeightsLoader(
        self.ckpt_dir, config=self.config, export_config=export_config
    )

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
        self.q_norm_w[self.expected_perm],
    )
    torch.testing.assert_close(
        loader("static_model.model.layers.0.self_attn.k_norm.weight"),
        self.k_norm_w[self.expected_perm],
    )

  def test_keeps_raw_layout_when_composite_disabled(self):
    """Verifies raw QKV concatenation when qkv_norm_rope composite is off."""
    export_config = types.SimpleNamespace(use_qkv_norm_rope_composite=False)
    loader = weights_loader_lib.HFCheckpointWeightsLoader(
        self.ckpt_dir, config=self.config, export_config=export_config
    )
    torch.testing.assert_close(
        loader("model.layers.0.self_attn.q_norm.weight"), self.q_norm_w
    )
    qkv_w = loader("model.layers.0.self_attn.qkv_proj.weight")
    raw_q_w = loader("model.layers.0.self_attn.q_proj.weight")
    torch.testing.assert_close(
        qkv_w, torch.cat([raw_q_w, self.k_proj_w, self.v_proj_w], dim=0)
    )


if __name__ == "__main__":
  absltest.main()
