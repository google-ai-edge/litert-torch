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
"""Tests for Gemma3n exportable modules."""

import copy

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext.gemma3n import exportable_module
from litert_torch.generative.export_hf.model_ext.gemma3n import patch
import torch
from transformers.models.gemma3n import modeling_gemma3n

from absl.testing import absltest as googletest

_CACHE_LENGTH = 16
_TOKEN_ID = 5


class _Gemma3nTextOnlyModel(torch.nn.Module):
  """Text-only stand-in for `Gemma3nForConditionalGeneration`.

  The real class builds a timm vision tower, and timm is not a test dependency.
  The exportable modules only use `model.language_model`, `lm_head` and
  `config`. As in the real class, `config` is a `Gemma3nConfig`, which keeps
  `final_logit_softcapping` in its text config.
  """

  def __init__(self, causal_lm):
    super().__init__()
    self.model = torch.nn.Module()
    self.model.language_model = causal_lm.model
    self.lm_head = causal_lm.lm_head
    self.config = modeling_gemma3n.Gemma3nConfig(text_config=causal_lm.config)


class Gemma3nExportableModuleTest(googletest.TestCase):

  def test_decode_logits_match_hf(self):
    config = modeling_gemma3n.Gemma3nTextConfig(
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=2,
        sliding_window=8,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        activation_sparsity_pattern=[0.0] * 4,
        laurel_rank=4,
        final_logit_softcapping=30.0,
    )
    torch.manual_seed(0)
    reference_model = modeling_gemma3n.Gemma3nForCausalLM(config).eval()
    with torch.no_grad():
      # Amplify lm_head so raw pre-cap logits exceed +-30.0. It is tied to
      # `embed_tokens`, so the input embeddings are scaled too.
      reference_model.lm_head.weight.mul_(200.0)
    with patch.gemma3n_litert_patch():
      causal_lm = modeling_gemma3n.Gemma3nForCausalLM(
          copy.deepcopy(config)
      ).eval()
    causal_lm.load_state_dict(reference_model.state_dict())
    model = _Gemma3nTextOnlyModel(causal_lm)
    # Set after building `Gemma3nConfig`, which resets the attention
    # implementation of its text config.
    causal_lm.set_attn_implementation("lrt_transposed_attention")

    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        output_dir=None,
        cache_length=_CACHE_LENGTH,
        externalize_embedder=True,
    )
    decode_module = exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder(
        model, export_config
    )
    inputs, _ = decode_module.get_sample_inputs(config)["decode"]

    tokens = torch.tensor([[_TOKEN_ID]])
    # The first token is decoded into an empty cache, so only cache position 0
    # is visible.
    mask = torch.full((1, 1, 1, _CACHE_LENGTH), float("-inf"))
    mask[..., 0] = 0
    with torch.no_grad():
      language_model = model.model.language_model
      inputs["embeddings"] = language_model.embed_tokens(tokens)
      inputs["per_layer_embeddings"] = language_model.get_per_layer_inputs(
          tokens
      )
      inputs["input_pos"] = torch.tensor([0], dtype=torch.int32)
      inputs["mask"] = mask
      actual_logits = decode_module(**inputs)["logits"]
      expected_logits = reference_model(input_ids=tokens).logits
      raw_logits = reference_model.lm_head(
          reference_model.model(input_ids=tokens).last_hidden_state
      )

    self.assertGreater(raw_logits.abs().max().item(), 40.0)
    torch.testing.assert_close(actual_logits, expected_logits)


if __name__ == "__main__":
  googletest.main()
