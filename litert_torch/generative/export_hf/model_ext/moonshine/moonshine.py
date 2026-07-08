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

"""Wrapper for MoonshineForConditionalGeneration to compute logits."""

from litert_torch.generative.export_hf.core.speech import asr_model
import numpy as np
import torch
import transformers
from transformers.models.moonshine import modeling_moonshine as moonshine


def _interleave_half(x: torch.Tensor) -> torch.Tensor:
  """h.repeat_interleave(2, dim=-1) without BROADCAST_TO for GPU inference."""
  h = x[..., : x.shape[-1] // 2]
  h = h.unsqueeze(-1)
  h = torch.cat([h, h], dim=-1)
  return h.reshape(x.shape)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
  """Applies rotary embeddings on the first half or full tensor."""
  assert x.shape[0] == 1
  x = x.squeeze(0)
  x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
  x_embed = (x_rot * cos) + (moonshine.rotate_half(x_rot) * sin)
  return torch.cat([x_embed, x_pass], dim=-1).unsqueeze(0)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Moonshine.apply_rotary_pos_emb() with less dimensions for GPU inference."""
  cos = _interleave_half(cos)
  sin = _interleave_half(sin)
  rotary_dim = cos.shape[-1]
  q_embed = _apply_rotary(q, cos, sin, rotary_dim)
  k_embed = _apply_rotary(k, cos, sin, rotary_dim)
  return q_embed, k_embed


class MoonshineDecoder(torch.nn.Module):
  """Wrapper for MoonshineForConditionalGeneration for decoder outputs."""

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


class MoonshineEncoder(torch.nn.Module):
  """Wrapper for MoonshineForConditionalGeneration for encoder outputs."""

  def __init__(self, model: torch.nn.Module):
    super().__init__()
    self._model = model

  def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Returns the encoder's last hidden state."""
    encoder_outputs = self._model.get_encoder()(input_values)
    return (encoder_outputs.last_hidden_state,)


class Moonshine(asr_model.AsrModel):
  """Wrapper for MoonshineForConditionalGeneration for encoder outputs."""

  HF_MODEL_ID = 'UsefulSensors/moonshine-tiny'

  def __init__(
      self, model_id: str = HF_MODEL_ID, override_transformers: bool = False
  ):
    super().__init__(override_transformers)
    if override_transformers:
      moonshine.apply_rotary_pos_emb = _apply_rotary_pos_emb
    factory = transformers.MoonshineForConditionalGeneration
    self._model = factory.from_pretrained(model_id).float().eval()
    self._replace_normalizations(self._model)
    self._encoder = MoonshineEncoder(self._model).eval()
    self._decoder = MoonshineDecoder(self._model).eval()
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
    return (processed_audio['input_values'],)

  def get_decoder_sample_input(
      self, encoder_output: tuple[torch.Tensor, ...], num_tokens: int
  ) -> tuple[torch.Tensor, ...]:
    tokens = torch.arange(num_tokens, dtype=torch.int32).unsqueeze(0)
    return encoder_output + (tokens, asr_model.get_causal_mask(num_tokens))

  def get_decode_start_token_id(self) -> int:
    return 1  # <s>

  def get_decode_stop_token_id(self) -> int:
    return 2  # </s>

  def run_original_model(
      self, processed_audio: dict[str, torch.Tensor]
  ) -> asr_model.AsrModel.OriginalModelOutput:
    out = self._model.generate(
        **processed_audio,
        return_dict_in_generate=True,  # Required for logits.
        generation_config=transformers.GenerationConfig(output_logits=True),
    )
    return asr_model.AsrModel.OriginalModelOutput(
        logits=torch.from_numpy(np.array(out['logits']).transpose(1, 0, 2)),
        tokens=out['sequences'],
    )
