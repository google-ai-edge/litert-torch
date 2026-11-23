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
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import weights_loader as weights_loader_lib
from litert_torch.generative.export_hf.core.split_cache import attention_mask as split_attention_mask
from litert_torch.generative.export_hf.core.split_cache import cache as split_cache_lib
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext import patches
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
from litert_torch.generative.export_hf.model_ext.lfm2 import extension as lfm2_extension
import torch
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


def _export_split_cache(k_ts_idx=2, v_ts_idx=3):
  """Traces a tiny LFM2 in the split-cache layout (8-token prefill, decode)."""
  export_config = exportable_module_config.ExportableModuleConfig(
      model="dummy",
      prefill_lengths=[8],
      cache_length=16,
      cache_lengths=[16],
      split_cache=True,
      k_ts_idx=k_ts_idx,
      v_ts_idx=v_ts_idx,
  )
  config = _get_dummy_lfm2_config()
  with patches.get_patch_context("lfm2"):
    model = modeling_lfm2.Lfm2ForCausalLM(config).eval()
  programs = ext_testing.assert_tiny_prefill_decode_torch_export(
      lfm2_extension.Lfm2Extension(), model, export_config=export_config
  )

  def make_inputs(embeddings, valid_mask):
    """Inputs for a call on an empty cache, with the mask the runtime gets."""
    seq_len = embeddings.shape[1]
    head_dim = config.hidden_size // config.num_attention_heads
    return {
        "embeddings": embeddings,
        "pos_emb": {
            "cos": torch.ones(1, seq_len, 1, head_dim),
            "sin": torch.zeros(1, seq_len, 1, head_dim),
        },
        "mask": {
            "global": split_attention_mask.build_full_mask(
                torch.zeros((1, seq_len), dtype=torch.int32),
                None,
                export_config.cache_length,
                torch.tensor(0, dtype=torch.int32),
                valid_mask,
            )
        },
        "kv_cache": split_cache_lib.LiteRTLMSplitCache.create_from_config(
            config, export_config
        ),
    }

  return programs, make_inputs


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

  def test_split_cache_export_names_slices_by_layer_index(self):
    for k_ts_idx, v_ts_idx in ((2, 3), (3, 2)):
      with self.subTest(k_ts_idx=k_ts_idx, v_ts_idx=v_ts_idx):
        programs, make_inputs = _export_split_cache(k_ts_idx, v_ts_idx)
        outputs = programs["decode"].module()(
            **make_inputs(torch.randn(1, 1, 16), torch.ones(1, 1).bool())
        )
        # Layer 0 is conv and layer 1 attention: no `kv_slice_k_0`.
        self.assertEqual(
            set(outputs),
            {"kv_slice_c_0", "kv_slice_k_1", "kv_slice_v_1", "logits"},
        )
        self.assertEqual(outputs["kv_slice_c_0"].shape, (1, 2, 16))
        self.assertEqual(outputs["kv_slice_k_1"].shape[k_ts_idx], 1)
        self.assertEqual(outputs["kv_slice_v_1"].shape[v_ts_idx], 1)

  def test_split_cache_padded_prefill_keeps_conv_state(self):
    programs, make_inputs = _export_split_cache()
    num_real = 5
    embeddings = torch.randn(1, 8, 16)
    valid_mask = (torch.arange(8) < num_real).unsqueeze(0)

    # The runtime pads a short prefill chunk: 5 real tokens and 3 padding rows.
    state = programs["prefill_8"].module()(
        **make_inputs(embeddings, valid_mask)
    )["kv_slice_c_0"]

    # The same 5 tokens, one at a time through decode.
    decode = programs["decode"].module()
    expected = None
    for i in range(num_real):
      inputs = make_inputs(embeddings[:, i : i + 1], torch.ones(1, 1).bool())
      if expected is not None:
        inputs["kv_cache"].layers[0].conv_states = expected
      expected = decode(**inputs)["kv_slice_c_0"]
    torch.testing.assert_close(state, expected)


if __name__ == "__main__":
  googletest.main()
