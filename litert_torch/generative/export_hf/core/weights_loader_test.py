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

import os

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
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(6, 4),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(6, 4),
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

  def test_empty_checkpoint_raises(self):
    with self.assertRaises(ValueError):
      weights_loader_lib.HFCheckpointWeightsLoader(
          self.create_tempdir().full_path
      )


if __name__ == "__main__":
  absltest.main()
