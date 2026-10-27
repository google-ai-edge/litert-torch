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
"""Tests for the Qwen3-ASR export wrapper."""

import types

from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.model_ext.qwen3 import qwen3_asr
import numpy as np
import torch
from transformers.models.qwen3_asr import configuration_qwen3_asr
from transformers.models.qwen3_asr import modeling_qwen3_asr

from absl.testing import absltest as googletest


def _get_dummy_encoder_config(n_window_infer: int = 800):
  config = configuration_qwen3_asr.Qwen3ASREncoderConfig(
      d_model=32,
      encoder_layers=2,
      encoder_attention_heads=2,
      encoder_ffn_dim=64,
      downsample_hidden_size=8,
      output_dim=32,
      n_window=50,
      n_window_infer=n_window_infer,
  )
  config._attn_implementation = "eager"
  return config


def _init_weights(module: torch.nn.Module) -> torch.nn.Module:
  # With the default init (std 0.02), the output barely depends on the
  # attention windows.
  for m in module.modules():
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
      torch.nn.init.normal_(m.weight, std=0.2)
  return module.eval()


def _patch_audio_attention(module: torch.nn.Module):
  for m in module.modules():
    if isinstance(m, modeling_qwen3_asr.Qwen3ASRAudioAttention):
      m.forward = types.MethodType(qwen3_asr._audio_attention_forward, m)


def _max_abs_diff(a, b) -> float:
  return float(np.abs(np.asarray(a) - np.asarray(b)).max())


class Qwen3AsrTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # 65 encoder tokens, one window.
      ("5s", 500, 800),
      # 156 encoder tokens, windows of 104 and 52.
      ("12s", 1200, 800),
      # 390 encoder tokens, windows of 104, 104, 104 and 78.
      ("30s", 3000, 800),
      # 156 encoder tokens, windows of 52.
      ("12s_n_window_infer_400", 1200, 400),
  )
  def test_audio_attention_matches_transformers(
      self, num_frames, n_window_infer
  ):
    torch.manual_seed(0)
    config = _get_dummy_encoder_config(n_window_infer)
    encoder = _init_weights(modeling_qwen3_asr.Qwen3ASREncoder(config))
    input_features = torch.randn(1, config.num_mel_bins, num_frames)
    input_features_mask = torch.ones(1, num_frames, dtype=torch.long)

    with torch.no_grad():
      expected = encoder(input_features, input_features_mask)
      _patch_audio_attention(encoder)
      actual = encoder(input_features, input_features_mask)

    self.assertLess(
        _max_abs_diff(expected.last_hidden_state, actual.last_hidden_state),
        1e-5,
    )

  def test_13_token_chunks_do_not_match_transformers(self):
    torch.manual_seed(0)
    config = _get_dummy_encoder_config()
    encoder = _init_weights(modeling_qwen3_asr.Qwen3ASREncoder(config))
    input_features = torch.randn(1, config.num_mel_bins, 1200)
    input_features_mask = torch.ones(1, 1200, dtype=torch.long)
    # Attention within each 13-token chunk of the 156 encoder tokens.
    cu_seqlens = torch.arange(0, 156 + 1, 13, dtype=torch.int32)

    with torch.no_grad():
      expected = encoder(input_features, input_features_mask)
      chunked = encoder(
          input_features, input_features_mask, cu_seqlens=cu_seqlens
      )

    self.assertGreater(
        _max_abs_diff(expected.last_hidden_state, chunked.last_hidden_state),
        1e-2,
    )

  def test_convert_audio_attention(self):
    torch.manual_seed(0)
    config = _get_dummy_encoder_config()
    attention = _init_weights(modeling_qwen3_asr.Qwen3ASRAudioAttention(config))
    hidden_states = torch.randn(156, config.d_model)
    cu_seqlens = torch.tensor([0, 104, 156], dtype=torch.int32)

    with torch.no_grad():
      expected = attention(hidden_states, cu_seqlens)
    _patch_audio_attention(attention)
    edge_model = litert_torch.convert(attention, (hidden_states,))

    self.assertLess(_max_abs_diff(expected, edge_model(hidden_states)), 1e-5)


if __name__ == "__main__":
  googletest.main()
