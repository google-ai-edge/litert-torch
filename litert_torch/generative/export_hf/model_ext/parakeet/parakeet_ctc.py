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

"""Wrapper for ParakeetCTC to compute logits."""

from litert_torch.generative.export_hf.core.speech import asr_model
import torch
import transformers


class ParakeetCTCEncoder(torch.nn.Module):
  """Wrapper for ParakeetCTC for encoder outputs."""

  def __init__(self, model: torch.nn.Module):
    super().__init__()
    self._encoder = model.encoder
    self._ctc_head = model.ctc_head

  def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Simplified forward() to remove unnecessary BROADCAST_TO."""
    hidden_states = self._encoder.subsampling(input_features)
    hidden_states = hidden_states * self._encoder.input_scale
    position_embeddings = self._encoder.encode_positions(hidden_states)

    for encoder_layer in self._encoder.layers:
      hidden_states = encoder_layer(
          hidden_states, position_embeddings=position_embeddings
      )

    logits = self._ctc_head(hidden_states.transpose(1, 2)).transpose(1, 2)
    return (logits,)


class ParakeetCTC(asr_model.AsrModel):
  """Wrapper for ParakeetCTC to return only sequences."""

  HF_MODEL_ID = 'nvidia/parakeet-ctc-0.6b'

  def __init__(
      self, model_id: str = HF_MODEL_ID, override_transformers: bool = False
  ):
    super().__init__(override_transformers)
    factory = transformers.AutoModelForCTC
    self._model = factory.from_pretrained(model_id).float().eval()
    self._replace_normalizations(self._model)
    self._encoder = ParakeetCTCEncoder(self._model).eval()
    self._processor = asr_model.TransformersProcessor(model_id)

  def get_encoder(self) -> torch.nn.Module:
    return self._encoder

  def get_processor(self) -> asr_model.AsrProcessor:
    return self._processor

  def get_encoder_sample_input(
      self, processed_audio: dict[str, torch.Tensor]
  ) -> tuple[torch.Tensor, ...]:
    return (processed_audio['input_features'],)

  def run_original_model(
      self, processed_audio: dict[str, torch.Tensor]
  ) -> asr_model.AsrModel.OriginalModelOutput:
    out = self._model.generate(**processed_audio, return_dict_in_generate=True)
    return asr_model.AsrModel.OriginalModelOutput(
        logits=out.logits, tokens=out.sequences
    )
