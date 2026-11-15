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

"""Unit tests for Whisper model extension and stateful decoder."""

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.speech import asr_model
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext.whisper import extension as _  # Registers WhisperExtension
from litert_torch.generative.export_hf.model_ext.whisper import whisper
import torch
import transformers

from absl.testing import absltest as googletest


class WhisperStatefulDecoderTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    # Create a small synthetic Whisper config for fast testing.
    self.config = transformers.WhisperConfig(
        vocab_size=1000,
        pad_token_id=0,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        decoder_ffn_dim=128,
        encoder_ffn_dim=128,
        num_mel_bins=80,
        max_source_positions=1500,
        max_target_positions=128,
    )
    self.hf_model = transformers.WhisperForConditionalGeneration(self.config).eval()

  def test_stateful_decoder_matches_stateless_logits(self):
    max_decode_steps = 32
    stateful_decoder = whisper.WhisperDecoder(
        self.hf_model, stateful=True, max_decode_steps=max_decode_steps
    ).eval()
    stateless_decoder = whisper.WhisperDecoder(
        self.hf_model, stateful=False
    ).eval()

    # Synthetic encoder hidden states [1, 100, d_model]
    enc_states = torch.randn(1, 100, self.config.d_model)

    # Sequence of 4 token IDs
    test_tokens = [50, 100, 150, 200]
    unrolled_tokens = torch.tensor([test_tokens], dtype=torch.int32)
    causal_mask = asr_model.get_causal_mask(len(test_tokens))

    # 1. Stateless forward on full sequence
    with torch.no_grad():
      stateless_out = stateless_decoder(enc_states, unrolled_tokens, causal_mask)
    stateless_logits = stateless_out[0]  # [1, 4, vocab_size]

    # 2. Stateful step-by-step forward
    num_heads = self.config.decoder_attention_heads
    head_dim = self.config.d_model // num_heads
    kv_caches = []
    for _ in range(self.config.decoder_layers):
      kv_caches.append(torch.zeros(1, num_heads, max_decode_steps, head_dim))
      kv_caches.append(torch.zeros(1, num_heads, max_decode_steps, head_dim))

    step = torch.zeros(1, dtype=torch.int32)
    for i, tok in enumerate(test_tokens):
      tok_tensor = torch.tensor([[tok]], dtype=torch.int32)
      with torch.no_grad():
        step_out = stateful_decoder(enc_states, tok_tensor, step, *kv_caches)
      step_logits = step_out[0]  # [1, 1, vocab_size]
      next_step = step_out[1]
      kv_caches = step_out[2:]

      self.assertEqual(next_step.item(), i + 1)
      step = next_step

      # Compare logits at position i
      torch.testing.assert_close(
          step_logits[0, 0],
          stateless_logits[0, i],
          atol=1e-4,
          rtol=1e-4,
          msg=f"Logits mismatch at decode step {i}",
      )

    # Verify stateful_decoder transparently handles stateless fallback when called
    # with sequence tokens and causal mask (e.g. stateful_after <= 0 export).
    tokens_tensor = torch.tensor([test_tokens], dtype=torch.int32)
    mask = asr_model.get_causal_mask(len(test_tokens))
    stateless_fallback_out = stateful_decoder(enc_states, tokens_tensor, mask)
    torch.testing.assert_close(
        stateless_fallback_out[0], stateless_logits, atol=1e-4, rtol=1e-4
    )

  def test_get_decoder_sample_input_stateful_and_stateless(self):
    speech_model = whisper.Whisper.__new__(whisper.Whisper)
    speech_model._model = self.hf_model
    speech_model._stateful = True
    speech_model._max_decode_steps = 64

    enc_output = (torch.zeros(1, 100, self.config.d_model),)

    # Stateful sample inputs (num_tokens = 1)
    stateful_inputs = speech_model.get_decoder_sample_input(enc_output, 1)
    expected_states = 1 + 2 * self.config.decoder_layers  # step + 2*layers
    self.assertLen(stateful_inputs, 2 + expected_states)
    self.assertEqual(stateful_inputs[0].shape, (1, 100, self.config.d_model))
    self.assertEqual(stateful_inputs[1].shape, (1, 1))  # tokens
    self.assertEqual(stateful_inputs[2].shape, (1,))  # step
    for k in range(3, len(stateful_inputs)):
      self.assertEqual(
          stateful_inputs[k].shape,
          (1, self.config.decoder_attention_heads, 64, self.config.d_model // self.config.decoder_attention_heads),
      )

    # Stateless fallback when num_tokens > 1
    stateless_inputs = speech_model.get_decoder_sample_input(enc_output, 16)
    self.assertLen(stateless_inputs, 3)
    self.assertEqual(stateless_inputs[0].shape, (1, 100, self.config.d_model))
    self.assertEqual(stateless_inputs[1].shape, (1, 16))
    self.assertEqual(stateless_inputs[2].shape, (1, 1, 16, 16))

  def test_extension_export_defaults_and_update_config(self):
    ext = extension.get_model_extension("whisper")
    self.assertIsNotNone(ext)
    defaults = ext.get_export_defaults({})
    self.assertEqual(defaults.export_kwargs.get("input_sec"), 30.0)

    cfg = exportable_module_config.ExportableModuleConfig(model="whisper")
    updated = ext.update_export_config(cfg, None)
    self.assertEqual(updated.stateful_after, 1)

  def test_speech2text_metadata_sets_decode_statefully_after(self):
    speech_model = whisper.Whisper.__new__(whisper.Whisper)
    speech_model._model = self.hf_model
    speech_model._processor = None
    speech_model._stateful = True

    # When stateful is True and stateful_after > 0, decode_statefully_after is 1.
    cfg = exportable_module_config.ExportableModuleConfig(
        model="whisper", stateful_after=4
    )
    meta = speech_model.get_speech2text_metadata(cfg)
    self.assertEqual(meta.decode_statefully_after, 1)

    # When export_config specifies stateful_after <= 0, stateful decode is disabled.
    cfg_stateless = exportable_module_config.ExportableModuleConfig(
        model="whisper", stateful_after=-1
    )
    meta_stateless_cfg = speech_model.get_speech2text_metadata(cfg_stateless)
    self.assertFalse(meta_stateless_cfg.HasField("decode_statefully_after"))

    # When stateful is False, decode_statefully_after is cleared.
    speech_model._stateful = False
    meta_stateless = speech_model.get_speech2text_metadata(cfg)
    self.assertFalse(meta_stateless.HasField("decode_statefully_after"))


if __name__ == "__main__":
  googletest.main()
