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

"""Wrapper for ParakeetTDT to compute logits."""

from litert_torch.generative.export_hf.core.speech import asr_model
import numpy as np
import torch


class ParakeetTDTDecoder(torch.nn.Module):
  """Wrapper of ParakeetTDT for decoder outputs."""

  def __init__(self, model: torch.nn.Module):
    super().__init__()
    self._decoder = model.decoder.eval()
    self._joint = model.joint.eval()
    # It is needed for correct inference though it's not clear why.
    self._joint.set_fuse_loss_wer(False)

  def forward(
      self,
      encoder_hidden_states: torch.Tensor,
      input_ids: torch.Tensor,
      hidden_states: torch.Tensor,
      cell_states: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    """Returns the joint network's logits and new states."""
    decoder_outputs, (hidden_states, cell_states) = self._decoder.predict(
        input_ids, (hidden_states, cell_states), add_sos=False
    )
    logits = self._joint(
        encoder_outputs=encoder_hidden_states,
        decoder_outputs=decoder_outputs.transpose(1, 2),
    )
    return (logits, hidden_states, cell_states)


class ParakeetTDTEncoder(torch.nn.Module):
  """Wrapper of ParakeetTDT for encoder outputs."""

  def __init__(self, model: torch.nn.Module):
    super().__init__()
    self._encoder = model.encoder.eval()

  def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Simplified forward() to remove unnecessary operations."""
    hidden_states, _ = self._encoder(audio_signal=input_features, length=None)
    return (hidden_states,)


class ParakeetAsrProcessor(asr_model.AsrProcessor):
  """Wrapper of NeMo's processor to behave like a HF processor."""

  def __init__(self, model: torch.nn.Module):
    self._preprocessor = model.preprocessor
    self._tokenizer = model.tokenizer

  def get_sampling_rate(self) -> int:
    return self._preprocessor._sample_rate  # pylint: disable=protected-access

  def process(self, audio: np.ndarray) -> dict[str, torch.Tensor]:
    processed_signal, _ = self._preprocessor(
        input_signal=torch.FloatTensor(audio).unsqueeze(0),
        length=torch.IntTensor([len(audio)]),
    )
    # returns input audio for original model inference during verification.
    return {'input_features': processed_signal, 'audio': audio}

  def decode(self, tokens: np.ndarray | torch.Tensor) -> list[str]:
    return [self._tokenizer.ids_to_text(id) for id in tokens]


class _MaskedConvSequential(torch.nn.Sequential):
  """Simplified MaskedConvSequential for conversion and GPU inference."""

  def forward(self, x, lengths):
    x = x.unsqueeze(1)  # (batch, 1, time, features)
    mask = self._create_mask(x, lengths)
    for _, layer in enumerate(self):
      x = x * mask
      x = layer(x)
      if hasattr(layer, 'stride') and layer.stride != (1, 1):
        lengths = lengths + layer.padding[0] + layer.padding[1]
        lengths = (lengths - layer.kernel_size[0]) // layer.stride[0] + 1
        mask = self._create_mask(x, lengths)
    x = x * mask
    return x, lengths

  def _create_mask(self, x, lengths):
    """Create mask matching tensor dimensions."""
    b, _, t, f = x.shape
    time_arange = torch.cat([torch.arange(t).unsqueeze(0)] * b, dim=0)
    time_mask = (time_arange < lengths.unsqueeze(1)).float()
    return torch.cat([time_mask.unsqueeze(-1)] * f, dim=-1).unsqueeze(1)


class ParakeetTDT(asr_model.AsrModel):
  """Wrapper for ParakeetTDT to return only sequences."""

  HF_MODEL_ID = 'nvidia/parakeet-tdt-0.6b-v3'
  NUM_DURATIONS = 5  # number of durations at the end of logits
  BLANK_TOKEN_ID = 8192  # token ID for <blank>

  # LSTM with 2 layers and 640 hidden dimensions.
  _NUM_LAYERS = 2
  _HIDDEN_DIM = 640

  def __init__(
      self, model_id: str = HF_MODEL_ID, override_transformers: bool = False
  ):
    super().__init__(override_transformers)
    import nemo.collections.asr as nemo_asr  # pylint: disable=g-import-not-at-top
    # Replace subsampling.MaskedConvSequential to conversion and GPU inference.
    nemo_asr.parts.submodules.subsampling.MaskedConvSequential = (
        _MaskedConvSequential
    )
    model = nemo_asr.models.asr_model.ASRModel.from_pretrained(model_id)
    self._model = model.float().eval()
    self._replace_normalizations(self._model)
    self._encoder = ParakeetTDTEncoder(self._model).eval()
    self._decoder = ParakeetTDTDecoder(self._model).eval()
    self._processor = ParakeetAsrProcessor(self._model)

  def get_encoder(self) -> torch.nn.Module:
    return self._encoder

  def get_decoder(self) -> torch.nn.Module:
    return self._decoder

  def get_processor(self) -> asr_model.AsrProcessor:
    return self._processor

  def get_encoder_sample_input(
      self, processed_audio: dict[str, torch.Tensor]
  ) -> tuple[torch.Tensor, ...]:
    return (processed_audio['input_features'],)

  def get_decoder_sample_input(
      self, encoder_output: tuple[torch.Tensor, ...], num_tokens: int
  ) -> tuple[torch.Tensor, ...]:
    tokens = torch.arange(num_tokens, dtype=torch.int32).unsqueeze(0)
    # NeMo's RNNTDecoder LSTM expects a list of 2 tensors: (hidden, cell).
    # Each state has shape [layers, batch, hidden_dim].
    # See AbstractRNNTDecoder.predict() for more details.
    hidden_state = torch.zeros((self._NUM_LAYERS, 1, self._HIDDEN_DIM)).float()
    # Must be a separate tensor to avoid being omitted from conversion.
    cell_state = hidden_state.clone()
    return encoder_output + (tokens, hidden_state, cell_state)

  def get_decode_start_token_id(self) -> int:
    return self.BLANK_TOKEN_ID

  def run_original_model(
      self, processed_audio: dict[str, torch.Tensor]
  ) -> asr_model.AsrModel.OriginalModelOutput:
    out = self._model.transcribe(
        # transcribe expects a list of audios
        audio=[processed_audio['audio']], return_hypotheses=True
    )
    tokens = out[0].y_sequence.unsqueeze(0).long()
    return asr_model.AsrModel.OriginalModelOutput(
        # Return fake logits as it's hard to get the raw logits from the model.
        logits=torch.zeros((1, len(tokens), 8192), dtype=torch.float32),
        tokens=out[0].y_sequence.unsqueeze(0).long(),
        # timestamps=out[0].timestamp.unsqueeze(0),
    )
