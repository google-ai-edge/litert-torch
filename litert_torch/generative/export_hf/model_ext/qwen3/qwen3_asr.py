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

# Fixed chunk length for Qwen3ASRAudioAttention.
_CHUNK_LEN = 13


def _audio_attention_forward(
    self, hidden_states: torch.Tensor, **kwargs
) -> torch.Tensor:
  """Patched Qwen3ASRAudioAttention.forward() to avoid splits and loops."""
  seqlen, _ = hidden_states.size()
  num_chunks = seqlen // _CHUNK_LEN

  q = self.q_proj(hidden_states).reshape(seqlen, self.num_heads, -1)
  k = self.k_proj(hidden_states).reshape(seqlen, self.num_heads, -1)
  v = self.v_proj(hidden_states).reshape(seqlen, self.num_heads, -1)

  q = q.view(num_chunks, _CHUNK_LEN, self.num_heads, self.head_dim)
  k = k.view(num_chunks, _CHUNK_LEN, self.num_heads, self.head_dim)
  v = v.view(num_chunks, _CHUNK_LEN, self.num_heads, self.head_dim)

  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)

  attn_output, _ = asr_model._sdpa(
      self, q, k, v, attention_mask=None, scaling=self.scaling, **kwargs
  )
  attn_output = attn_output.view(seqlen, -1)
  attn_output = self.out_proj(attn_output)
  return attn_output


class Qwen3AsrDecoder(torch.nn.Module):
  """Wrapper for Qwen3ASRForConditionalGeneration for decoder outputs."""

  def __init__(self, model: torch.nn.Module, override_transformers: bool):
    super().__init__()
    self._model = model.model.language_model
    self._lm_head = model.lm_head

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
    position_ids = torch.arange(0, full_embeds.shape[1]).reshape(1, -1).int()
    decoder_outputs = self._model(
        inputs_embeds=full_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
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
    self._encoder = model.model.audio_tower
    self._projector = model.model.multi_modal_projector
    prefix_ids = torch.LongTensor(self._INPUT_IDS_PREFIX).unsqueeze(0)
    self._prefix_embeds = model.model.language_model.embed_tokens(prefix_ids)
    postfix_ids = torch.LongTensor(self._INPUT_IDS_POSTFIX).unsqueeze(0)
    self._postfix_embeds = model.model.language_model.embed_tokens(postfix_ids)

  def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Simplifed version of Qwen3ASRAudioEncoder.forward()."""
    batch_size, num_mel_bins, padded_feature_length = input_features.shape
    chunk_len = self._encoder.n_window * 2
    num_chunks = padded_feature_length // chunk_len
    chunked = (
        input_features.view(batch_size, num_mel_bins, num_chunks, chunk_len)
        .permute(0, 2, 1, 3)
        .reshape(batch_size * num_chunks, 1, num_mel_bins, chunk_len)
    )
    conv_out = F.gelu(self._encoder.conv2d1(chunked))
    conv_out = F.gelu(self._encoder.conv2d2(conv_out))
    conv_out = F.gelu(self._encoder.conv2d3(conv_out))
    b, c, f, t = conv_out.size()
    conv_out = self._encoder.conv_out(
        conv_out.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
    )
    conv_out += self._encoder.positional_embedding.positional_embedding[:t]

    hidden_states = conv_out.view(b * t, -1)
    cu_seqlens = torch.arange(0, b + 1).int() * t
    for layer in self._encoder.layers:
      hidden_states = layer(hidden_states, cu_seqlens)[0]

    hidden_states = self._encoder.ln_post(hidden_states)
    hidden_states = self._projector.linear_1(hidden_states)
    hidden_states = self._projector.act(hidden_states)
    hidden_states = self._projector.linear_2(hidden_states)
    hidden_states = hidden_states.view(batch_size, num_chunks * t, -1)

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

  HF_MODEL_ID = 'Qwen/Qwen3-ASR-0.6B-hf'

  def __init__(
      self, model_id: str = HF_MODEL_ID, override_transformers: bool = False
  ):
    super().__init__(override_transformers)
    factory = transformers.Qwen3ASRForConditionalGeneration
    self._model = factory.from_pretrained(model_id).float().eval()
    if override_transformers:
      modeling_qwen3_asr = transformers.models.qwen3_asr.modeling_qwen3_asr
      for module in self._model.modules():
        if isinstance(module, modeling_qwen3_asr.Qwen3ASRAudioAttention):
          module.forward = types.MethodType(_audio_attention_forward, module)
    self._replace_normalizations(self._model)
    self._replace_rmsnorms(self._model)
    self._encoder = Qwen3AsrEncoder(self._model).eval()
    self._decoder = Qwen3AsrDecoder(self._model, override_transformers).eval()
    self._processor = Qwen3AsrProcessor(model_id)

  def _replace_rmsnorms(self, module: torch.nn.Module):
    """Replaces Qwen3RMSNorm with composite ops for GPU inference."""
    modeling_qwen3 = transformers.models.qwen3.modeling_qwen3
    for name, child in list(module.named_children()):
      if isinstance(child, modeling_qwen3.Qwen3RMSNorm):
        setattr(module, name, asr_model.AsrRMSNorm(
            child.variance_epsilon, child.weight
        ))
      else:
        self._replace_rmsnorms(child)

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
    num_masks = encoder_output[0].shape[1] + num_tokens
    return encoder_output + (tokens, asr_model.get_causal_mask(num_masks))

  def get_decode_start_token_id(self) -> int:
    return 198  # \n following 'assistant'

  def get_decode_stop_token_id(self) -> int:
    return 151645  # <|im_end|>

  def run_original_model(
      self, processed_audio: dict[str, torch.Tensor]
  ) -> asr_model.AsrModel.OriginalModelOutput:
    out = self._model.generate(
        **processed_audio,
        generation_config=transformers.GenerationConfig(
            return_dict_in_generate=True, output_logits=True
        ),
    )
    prompt_len = processed_audio['input_ids'].shape[1]
    return asr_model.AsrModel.OriginalModelOutput(
        logits=torch.stack(out.logits, dim=0).transpose(0, 1),
        tokens=out.sequences[:, prompt_len:],
    )
