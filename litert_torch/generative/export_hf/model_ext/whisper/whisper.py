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

"""Wrapper for WhisperForConditionalGeneration to compute logits."""

from litert_torch.generative.export_hf.core.speech import asr_model
import numpy as np
import torch
import transformers


class WhisperDecoder(torch.nn.Module):
  """Wrapper for WhisperForConditionalGeneration for decoder outputs."""

  def __init__(self, model: torch.nn.Module):
    super().__init__()
    self._model = model

  def forward(
      self,
      encoder_hidden_states: torch.Tensor,
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    """Returns the decoder's logits for each token including the next token."""
    decoder_outputs = self._model.get_decoder()(
        encoder_hidden_states=encoder_hidden_states,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = self._model.proj_out(decoder_outputs.last_hidden_state)
    return (logits,)


class WhisperEncoder(torch.nn.Module):
  """Wrapper for WhisperForConditionalGeneration for encoder outputs."""

  def __init__(self, model: torch.nn.Module):
    super().__init__()
    self._model = model

  def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Returns the encoder's last hidden state."""
    encoder_outputs = self._model.get_encoder()(input_features=input_features)
    return (encoder_outputs.last_hidden_state,)


class Whisper(asr_model.AsrModel):
  """Wrapper for WhisperForConditionalGeneration for encoder outputs."""

  HF_MODEL_ID = 'openai/whisper-tiny'

  def __init__(
      self, model_id: str = HF_MODEL_ID, override_transformers: bool = False
  ):
    super().__init__(override_transformers)
    factory = transformers.WhisperForConditionalGeneration
    self._model = factory.from_pretrained(model_id).float().eval()
    self._replace_normalizations(self._model)
    self._encoder = WhisperEncoder(self._model).eval()
    self._decoder = WhisperDecoder(self._model).eval()
    self._processor = asr_model.TransformersProcessor(model_id)

  def get_encoder(self) -> torch.nn.Module:
    return self._encoder

  def get_decoder(self) -> torch.nn.Module:
    return self._decoder

  def get_processor(self) -> asr_model.AsrProcessor:
    return self._processor

  def get_encoder_sample_input(
      self, processed_audio: tuple[str, torch.Tensor]
  ) -> tuple[torch.Tensor, ...]:
    return (processed_audio['input_features'],)

  def get_decoder_sample_input(
      self, encoder_output: tuple[torch.Tensor, ...], num_tokens: int
  ) -> tuple[torch.Tensor, ...]:
    tokens = torch.arange(num_tokens, dtype=torch.int32).unsqueeze(0)
    return encoder_output + (tokens, asr_model.get_causal_mask(num_tokens))

  def get_decode_start_token_id(self) -> int:
    return 50258  # <|startoftranscript|>

  def get_decode_stop_token_id(self) -> int:
    return 50257  # <|endoftext|>

  def run_original_model(
      self, processed_audio: dict[str, torch.Tensor]
  ) -> asr_model.AsrModel.OriginalModelOutput:
    out = self._model.generate(
        **processed_audio,
        generation_config=transformers.GenerationConfig(
            return_dict_in_generate=True,  # Required for logits.
            output_logits=True,
        ),
    )
    return asr_model.AsrModel.OriginalModelOutput(
        logits=torch.from_numpy(np.array(out['logits']).transpose(1, 0, 2)),
        tokens=out['sequences'],
    )
