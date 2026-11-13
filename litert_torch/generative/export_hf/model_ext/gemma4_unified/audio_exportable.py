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
"""Exportable module for the Gemma4 unified (encoder-free) audio embedder."""

from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
import torch

# Audio soft tokens per encoder window: `audio_seq_length` in the Gemma 4
# Unified processor config (30 s of 16 kHz audio at 40 ms per token).
DEFAULT_AUDIO_TOKENS = 750


class LiteRTExportableModuleForGemma4UnifiedAudioEncoder(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for the Gemma4 unified audio embedder.

  Gemma 4 Unified has no audio tower: each token is a frame of
  `audio_samples_per_token` raw waveform samples, projected straight into the
  language model space by `model.model.embed_audio` (RMSNorm -> Linear).

  The graph has a single `src_inputs` input and a single `features` output and
  is packaged without an audio adapter. LiteRT-LM then treats it as a
  stateless streaming encoder (with `skip_mel_spectrogram_extraction` in
  `LlmMetadata` feeding raw frames), so audio of any length is encoded in
  windows of `DEFAULT_AUDIO_TOKENS` frames and the valid token count is taken
  from the input mask. Tokens are independent, so padding frames need no mask.
  """

  def __init__(self, model: torch.nn.Module, export_config):
    super().__init__(export_config)
    self.model = model

  def forward(self, src_inputs):
    return {'features': self.model.model.embed_audio(inputs_embeds=src_inputs)}

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    del kwargs
    samples_per_token = model_config.audio_config.audio_samples_per_token
    inputs = {
        'src_inputs': torch.zeros(
            (1, DEFAULT_AUDIO_TOKENS, samples_per_token), dtype=torch.float32
        ),
    }
    return {f'audio_{DEFAULT_AUDIO_TOKENS}': (inputs, {})}
