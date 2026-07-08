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

"""Wrapper for Qwen3ASRForConditionalGeneration to compute logits."""

import types

from litert_torch.generative.export_hf.core.speech import asr_model
import numpy as np
import torch
from torch.nn import functional as F
import transformers


def _apply_interleaved_mrope(
    freqs: torch.Tensor, mrope_section: list[int]
) -> torch.Tensor:
  """Interleave MRoPE without in-place slice assignment for GPU inference."""
  assert mrope_section == [24, 20, 20]
  # freqs shape: [3, c, s, 64]
  _, c, s, _ = freqs.shape
  freqs_60 = freqs[:, :, :, :60]  # [3, c, s, 60]
  freqs_t = freqs_60[0].reshape(c, s, 20, 3)
  freqs_h = freqs_60[1].reshape(c, s, 20, 3)
  freqs_w = freqs_60[2].reshape(c, s, 20, 3)
  mfreqs_t = freqs_t[..., 0]  # [c, s, 20]
  mfreqs_h = freqs_h[..., 1]  # [c, s, 20]
  mfreqs_w = freqs_w[..., 2]  # [c, s, 20]
  mfreqs_60 = torch.stack([mfreqs_t, mfreqs_h, mfreqs_w], dim=-1)  # [c,s,20,3]
  mfreqs_60 = mfreqs_60.reshape(c, s, 60)  # [c, s, 60]
  freqs_tail = freqs[0, :, :, 60:]  # [c, s, 4]
  return torch.cat([mfreqs_60, freqs_tail], dim=-1)  # [c, s, 64]


def _rotary_forward(
    self, x: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Qwen3ASRThinkerTextRotaryEmbedding.forward() for GPU inference."""
  assert position_ids.ndim == 3
  assert position_ids.shape[0] == 3
  assert position_ids.shape[1] == 1
  position_ids = position_ids.unsqueeze(2).float()  # [3, 1, 1, pos]
  # Replacing expand with torch.cat() ends up to make this tensor as INT
  # probably because converter treats it as weights. So, leave as is for now.
  inv_freq = self.inv_freq.reshape(1, 1, -1, 1).expand(3, -1, -1, -1)
  freqs = (inv_freq @ position_ids).transpose(2, 3)
  freqs = _apply_interleaved_mrope(freqs, self.mrope_section)
  emb = torch.cat([freqs, freqs], dim=-1)
  cos = emb.cos() * self.attention_scaling
  sin = emb.sin() * self.attention_scaling
  return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3AsrDecoder(torch.nn.Module):
  """Wrapper for Qwen3ASRForConditionalGeneration for decoder outputs."""

  def __init__(self, model: torch.nn.Module, override_transformers: bool):
    super().__init__()
    self._model = model.thinker.model
    self._lm_head = model.thinker.lm_head
    if override_transformers:
      # Replace rotary embedding with GPU-compatible version.
      rotary_emb = self._model.rotary_emb
      rotary_emb.forward = types.MethodType(_rotary_forward, rotary_emb)

  def forward(
      self,
      prompt_embeds: torch.Tensor,
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    """Returns the decoder's logits for each token including the next token."""
    inputs_embeds = self._model.get_input_embeddings()(input_ids).float()
    # Concatenate prompt_embeds (audio) and inputs_embeds (text) along the
    # sequence dimension (dim=1).
    full_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
    # Prepare position ids here to remove BROADCAST_TO ops.
    position_ids = torch.arange(0, full_embeds.shape[1]).reshape(1, 1, -1).int()
    decoder_outputs = self._model(
        inputs_embeds=full_embeds,
        attention_mask=attention_mask,
        # The hard coded `3` is for temporal, height and width.
        position_ids=torch.cat([position_ids] * 3, dim=0),
    )
    prompt_len = prompt_embeds.shape[1]
    logits = self._lm_head(decoder_outputs.last_hidden_state[:, prompt_len:, :])
    return (logits,)


class Qwen3AsrEncoder(torch.nn.Module):
  """Wrapper for Qwen3ASRForConditionalGeneration for encoder outputs."""

  # Corresponding to Qwen3AsrProcessor.PROMPT.
  # <|im_start|>user<|audio_start|>
  _INPUT_IDS_PREFIX = [151644, 872, 151669]
  # <|audio_end|><|im_end|><|im_start|>assistant
  _INPUT_IDS_POSTFIX = [151670, 151645, 151644, 77091]

  def __init__(self, model: torch.nn.Module):
    super().__init__()
    self._encoder = model.thinker.audio_tower
    prefix_ids = torch.LongTensor(self._INPUT_IDS_PREFIX).unsqueeze(0)
    self._prefix_embeds = model.thinker.model.embed_tokens(prefix_ids)
    postfix_ids = torch.LongTensor(self._INPUT_IDS_POSTFIX).unsqueeze(0)
    self._postfix_embeds = model.thinker.model.embed_tokens(postfix_ids)

  def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Simplifed version of Qwen3ASRAudioEncoder.forward()."""
    padded_embeds = F.gelu(self._encoder.conv2d1(input_features.unsqueeze(1)))
    padded_embeds = F.gelu(self._encoder.conv2d2(padded_embeds))
    padded_embeds = F.gelu(self._encoder.conv2d3(padded_embeds))
    b, c, f, t = padded_embeds.size()
    padded_embeds = self._encoder.conv_out(
        padded_embeds.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    )
    positional_embeds = self._encoder.positional_embedding.positional_embedding[
        : padded_embeds.shape[1], :
    ].unsqueeze(0)

    hidden_states = (padded_embeds + positional_embeds).view(b * t, -1)
    cu_seqlens = torch.arange(0, b + 1).int() * t
    for layer in self._encoder.layers:
      hidden_states = layer(hidden_states, cu_seqlens)[0]

    hidden_states = self._encoder.ln_post(hidden_states)
    hidden_states = self._encoder.proj1(hidden_states)
    hidden_states = self._encoder.act(hidden_states)
    hidden_states = self._encoder.proj2(hidden_states).unsqueeze(0)

    prompt_embeds = torch.cat(
        [self._prefix_embeds, hidden_states, self._postfix_embeds], dim=1
    )
    return (prompt_embeds,)


class Qwen3AsrProcessor(asr_model.TransformersProcessor):
  """Wrapper for Qwen3AsrProcessor to pass a default text."""

  _PROMPT = (
      '<|im_start|>user<|audio_start|><|audio_pad|><|audio_end|><|im_end|>'
      '<|im_start|>assistant\n'
  )

  def process(self, audio: np.ndarray) -> dict[str, torch.Tensor]:
    return self._processor(text=self._PROMPT, audio=audio, return_tensors='pt')


class Qwen3Asr(asr_model.AsrModel):
  """Wrapper for Qwen3ASRForConditionalGeneration for encoder outputs."""

  HF_MODEL_ID = 'Qwen/Qwen3-ASR-0.6B'

  def __init__(
      self, model_id: str = HF_MODEL_ID, override_transformers: bool = False
  ):
    super().__init__(override_transformers)
    import qwen_asr  # pylint: disable=g-import-not-at-top,unused-import
    factory = transformers.AutoModel
    self._model = factory.from_pretrained(model_id).float().eval()
    self._replace_normalizations(self._model)
    self._replace_rmsnorms(self._model)
    self._encoder = Qwen3AsrEncoder(self._model).eval()
    self._decoder = Qwen3AsrDecoder(self._model, override_transformers).eval()
    self._processor = Qwen3AsrProcessor(model_id)

  def _replace_rmsnorms(self, module: torch.nn.Module, parent: str = ''):
    """Replaces Qwen3ASRTextRMSNorm with composite ops for GPU inference."""
    import qwen_asr  # pylint: disable=g-import-not-at-top
    qwen3_asr = qwen_asr.core.transformers_backend.modeling_qwen3_asr
    for name, child in list(module.named_children()):
      full_name = f'{parent}.{name}'
      if isinstance(child, qwen3_asr.Qwen3ASRTextRMSNorm):
        print(f'Replacing {full_name} with AsrRMSNorm')
        setattr(module, name, asr_model.AsrRMSNorm(
            child.variance_epsilon, child.weight
        ))
      else:
        self._replace_rmsnorms(child, full_name)

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
    num_masks = encoder_output[0].shape[1] + num_tokens
    return encoder_output + (tokens, asr_model.get_causal_mask(num_masks))

  def get_decode_start_token_id(self) -> int:
    return 198  # \n following 'assistant'

  def get_decode_stop_token_id(self) -> int:
    return 151645  # <|im_end|>

  def run_original_model(
      self, processed_audio: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    out = self._model.generate(
        **processed_audio,
        generation_config=transformers.GenerationConfig(output_logits=True),
    )
    return asr_model.AsrModel.OriginalModelOutput(
        logits=torch.from_numpy(np.array(out.logits).transpose(1, 0, 2)),
        tokens=out.sequences,
    )
