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
"""Tests for the LFM2 `ModelExtension`."""

import litert_torch.generative.export_hf  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import weights_loader as weights_loader_lib
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext import patches
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
from litert_torch.generative.export_hf.model_ext.lfm2 import extension as lfm2_extension
import transformers
from transformers.models.lfm2 import modeling_lfm2

from absl.testing import absltest as googletest


def _get_dummy_lfm2_config() -> modeling_lfm2.Lfm2Config:
  return modeling_lfm2.Lfm2Config(
      vocab_size=32,
      num_attention_heads=4,
      num_key_value_heads=2,
      hidden_size=16,
      intermediate_size=32,
      block_auto_adjust_ff_dim=False,
      num_hidden_layers=2,
      layer_types=["conv", "full_attention"],
      full_attn_idxs=[1],
  )


class Lfm2ExtensionTest(googletest.TestCase):

  def test_resolves_gate_up_fused_weight_spec(self):
    ext = extension_lib.resolve_model_extension("lfm2")
    self.assertIsInstance(ext, lfm2_extension.Lfm2Extension)
    self.assertIn(
        extension_lib.FusedWeightSpec("gate_up_proj", ("w1", "w3"), dim=0),
        ext.fused_weight_specs,
    )

  def test_lfm2_vl_text_backbone_inherits_fused_weight_spec(self):
    config = transformers.Lfm2VlConfig(text_config={"model_type": "lfm2"})
    ext = extension_lib.resolve_model_extension(config)
    self.assertIsNotNone(ext)
    self.assertIn(
        ("gate_up_proj", ("w1", "w3"), 0),
        [tuple(spec) for spec in ext.fused_weight_specs],
    )

  def test_v2_weights_resolvable_with_all_fusions(self):
    ext = lfm2_extension.Lfm2Extension()
    with patches.get_patch_context("lfm2"):
      model = modeling_lfm2.Lfm2ForCausalLM(_get_dummy_lfm2_config()).eval()
    resolved = ext_testing.assert_v2_weights_resolvable(ext, model)
    self.assertIn(
        "model.model.layers.0.feed_forward.gate_up_proj.weight", resolved
    )
    self.assertIn("model.model.layers.1.self_attn.qkv_proj.weight", resolved)

  def test_weights_loader_without_extension_spec_raises(self):
    model = modeling_lfm2.Lfm2ForCausalLM(_get_dummy_lfm2_config())
    ckpt_dir = self.create_tempdir().full_path
    model.save_pretrained(ckpt_dir, safe_serialization=True)

    loader = weights_loader_lib.HFCheckpointWeightsLoader(ckpt_dir)
    with self.assertRaises(KeyError):
      loader("model.model.layers.0.feed_forward.gate_up_proj.weight")


if __name__ == "__main__":
  googletest.main()
