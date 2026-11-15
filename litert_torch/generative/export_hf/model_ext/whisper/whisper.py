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

import litert_torch.generative.custom_ops.dynamic_update_slice as tfl_dus
from litert_torch.generative.export_hf.core.speech import asr_model
import numpy as np
import torch
import transformers


class WhisperDecoder(torch.nn.Module):
  """Wrapper for WhisperForConditionalGeneration for decoder outputs.

  Supports both stateful autoregressive decoding with recurrent KV cache states
  (O(1) step latency) and legacy stateless unrolled sequence execution.
  """

  def __init__(
      self,
      model: torch.nn.Module,
      stateful: bool = True,
      max_decode_steps: int = 256,
  ):
    super().__init__()
    self._model = model
    self._stateful = stateful
    self._max_decode_steps = max_decode_steps
    self.register_buffer(
        '_pos',
        torch.arange(max_decode_steps, dtype=torch.int32).reshape(
            1, 1, 1, max_decode_steps
        ),
    )

  def forward(
      self,
      encoder_hidden_states: torch.Tensor,
      input_ids: torch.Tensor,
      *states: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    """Returns decoder logits and updated recurrent KV cache states.

    Args:
      encoder_hidden_states: Speech encoder representation [1, 1500, d_model].
      input_ids: Next token id [1, 1] (or sequence [1, seq_len] if stateless).
      *states: For stateful decoding, step [1] followed by 2 * num_layers KV caches
        [1, num_heads, max_decode_steps, head_dim]. For stateless decoding, optional
        attention_mask [1, 1, seq_len, seq_len].

    Returns:
      logits: Output logits [1, 1, vocab_size] (or [1, seq_len, vocab_size]).
      next_step: Updated step [1] (only when stateful).
      *new_kv_caches: Updated KV caches (only when stateful).
    """
    if not self._stateful or len(states) <= 1 or input_ids.shape[-1] > 1:
      attention_mask = states[0] if states else None
      decoder_outputs = self._model.get_decoder()(
          encoder_hidden_states=encoder_hidden_states,
          input_ids=input_ids,
          attention_mask=attention_mask,
      )
      logits = self._model.proj_out(decoder_outputs.last_hidden_state)
      return (logits,)

    step = states[0]
    kv_caches = states[1:]

    decoder = self._model.get_decoder()
    config = self._model.config
    num_heads = config.decoder_attention_heads
    head_dim = config.d_model // num_heads

    input_ids = input_ids.reshape(1, -1)
    step = step.reshape(1)

    # 1. Embed current token and add positional embedding.
    inputs_embeds = decoder.embed_tokens(input_ids)
    position_ids = step.reshape(1, 1).to(torch.long)
    positions = torch.nn.functional.embedding(
        position_ids, decoder.embed_positions.weight
    )
    hidden_states = inputs_embeds + positions

    zeros = torch.zeros((), dtype=torch.int32, device=step.device)
    start_indices = [zeros, zeros, step.reshape(()).to(torch.int32), zeros]

    # Causal mask: attend only to positions <= step; future positions are masked.
    causal_mask = torch.where(
        self._pos <= step.reshape(1, 1, 1, 1),
        torch.zeros(1, dtype=hidden_states.dtype, device=hidden_states.device),
        torch.tensor(-10000.0, dtype=hidden_states.dtype, device=hidden_states.device),
    )

    new_kv_caches = []
    num_layers = len(decoder.layers)
    for l in range(num_layers):
      layer = decoder.layers[l]
      k_cache = kv_caches[2 * l]
      v_cache = kv_caches[2 * l + 1]

      # --- Self-Attention ---
      residual = hidden_states
      hidden_states = layer.self_attn_layer_norm(hidden_states)

      scaling = layer.self_attn.scaling
      q = layer.self_attn.q_proj(hidden_states) * scaling
      q = q.view(1, 1, num_heads, head_dim).transpose(1, 2)

      k = layer.self_attn.k_proj(hidden_states)
      k = k.view(1, 1, num_heads, head_dim).transpose(1, 2)

      v = layer.self_attn.v_proj(hidden_states)
      v = v.view(1, 1, num_heads, head_dim).transpose(1, 2)

      # Update recurrent KV cache at slice `step`.
      k_cache = tfl_dus.dynamic_update_slice(k_cache, k, start_indices)
      v_cache = tfl_dus.dynamic_update_slice(v_cache, v, start_indices)
      new_kv_caches.append(k_cache)
      new_kv_caches.append(v_cache)

      attn_weights = torch.matmul(q, k_cache.transpose(2, 3))
      attn_weights = attn_weights + causal_mask
      attn_weights = torch.softmax(attn_weights, dim=-1)

      attn_output = torch.matmul(attn_weights, v_cache)
      attn_output = attn_output.transpose(1, 2).reshape(1, 1, -1)
      attn_output = layer.self_attn.out_proj(attn_output)
      hidden_states = residual + attn_output

      # --- Cross-Attention ---
      residual = hidden_states
      hidden_states = layer.encoder_attn_layer_norm(hidden_states)
      cross_out, _ = layer.encoder_attn(
          hidden_states,
          key_value_states=encoder_hidden_states,
      )
      hidden_states = residual + cross_out

      # --- MLP ---
      residual = hidden_states
      hidden_states = layer.final_layer_norm(hidden_states)
      hidden_states = layer.activation_fn(layer.fc1(hidden_states))
      hidden_states = layer.fc2(hidden_states)
      hidden_states = residual + hidden_states

    hidden_states = decoder.layer_norm(hidden_states)
    logits = self._model.proj_out(hidden_states)
    next_step = (step + 1).to(torch.int32)
    return (logits, next_step, *new_kv_caches)


class WhisperEncoder(torch.nn.Module):
  """Wrapper for WhisperForConditionalGeneration for encoder outputs."""

  def __init__(self, model: torch.nn.Module):
    super().__init__()
    self._model = model

  def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Returns the encoder's last hidden state."""
    encoder_outputs = self._model.get_encoder()(input_features=input_features)
    return (encoder_outputs.last_hidden_state,)


_WHISPER_START_OF_TRANSCRIPT_TOKEN_ID = 50258
_WHISPER_DEFAULT_LANGUAGE_TOKEN_ID = 50259  # <|en|>
_WHISPER_TRANSCRIBE_TOKEN_ID = 50359
_WHISPER_NO_TIMESTAMPS_TOKEN_ID = 50363
_WHISPER_END_OF_TEXT_TOKEN_ID = 50257

# Ordered 99 shared Whisper language codes corresponding to token IDs
# 50259..50357 across whisper-tiny/base/small/medium.
_WHISPER_LANGUAGE_CODES: tuple[str, ...] = (
    'en',
    'zh',
    'de',
    'es',
    'ru',
    'ko',
    'fr',
    'ja',
    'pt',
    'tr',
    'pl',
    'ca',
    'nl',
    'ar',
    'sv',
    'it',
    'id',
    'hi',
    'fi',
    'vi',
    'he',
    'uk',
    'el',
    'ms',
    'cs',
    'ro',
    'da',
    'hu',
    'ta',
    'no',
    'th',
    'ur',
    'hr',
    'bg',
    'lt',
    'la',
    'mi',
    'ml',
    'cy',
    'sk',
    'te',
    'fa',
    'lv',
    'bn',
    'sr',
    'az',
    'sl',
    'kn',
    'et',
    'mk',
    'br',
    'eu',
    'is',
    'hy',
    'ne',
    'mn',
    'bs',
    'kk',
    'sq',
    'sw',
    'gl',
    'mr',
    'pa',
    'si',
    'km',
    'sn',
    'yo',
    'so',
    'af',
    'oc',
    'ka',
    'be',
    'tg',
    'sd',
    'gu',
    'am',
    'yi',
    'lo',
    'uz',
    'fo',
    'ht',
    'ps',
    'tk',
    'nn',
    'mt',
    'sa',
    'lb',
    'my',
    'bo',
    'tl',
    'mg',
    'as',
    'tt',
    'haw',
    'ln',
    'ha',
    'ba',
    'jw',
    'su',
)


_WHISPER_LANGUAGE_ALIASES = {
    'cmn': 'zh',
    'cn': 'zh',
    'jp': 'ja',
    'kr': 'ko',
    'iw': 'he',
    'in': 'id',
    'ji': 'yi',
    'nb': 'no',
    'jv': 'jw',
    'fil': 'tl',
}


def _normalize_whisper_language(language: str) -> str:
  """Normalizes language tags and common aliases (`cmn`/`cn`/`jp`/`kr`)."""
  normalized = language.strip()
  if normalized.startswith('<|') and normalized.endswith('|>'):
    normalized = normalized[2:-2].strip()
  normalized = normalized.lower().replace('_', '-').split('-', 1)[0]
  return _WHISPER_LANGUAGE_ALIASES.get(normalized, normalized)


def resolve_whisper_start_token_ids(language: str, tokenizer=None) -> list[int]:
  """Resolves Whisper `[sot, lang, transcribe, notimestamps]` token IDs."""
  normalized = _normalize_whisper_language(language)
  if not normalized:
    raise ValueError(f'Unsupported whisper_language {language!r}.')

  if tokenizer is not None and hasattr(tokenizer, 'convert_tokens_to_ids'):
    sot_id = tokenizer.convert_tokens_to_ids('<|startoftranscript|>')
    translate_id = tokenizer.convert_tokens_to_ids('<|translate|>')
    transcribe_id = tokenizer.convert_tokens_to_ids('<|transcribe|>')
    notimestamps_id = tokenizer.convert_tokens_to_ids('<|notimestamps|>')
    if (
        isinstance(sot_id, int)
        and isinstance(translate_id, int)
        and isinstance(transcribe_id, int)
        and isinstance(notimestamps_id, int)
        and sot_id < translate_id < transcribe_id < notimestamps_id
    ):
      lang_id = tokenizer.convert_tokens_to_ids(f'<|{normalized}|>')
      if not isinstance(lang_id, int) or not (sot_id < lang_id < translate_id):
        raise ValueError(f'Unsupported whisper_language {language!r}.')
      return [sot_id, lang_id, transcribe_id, notimestamps_id]

  try:
    idx = _WHISPER_LANGUAGE_CODES.index(normalized)
  except ValueError as exc:
    raise ValueError(
        f'Unsupported whisper_language {language!r}; expected one of'
        f' {_WHISPER_LANGUAGE_CODES}.'
    ) from exc
  return [
      _WHISPER_START_OF_TRANSCRIPT_TOKEN_ID,
      _WHISPER_DEFAULT_LANGUAGE_TOKEN_ID + idx,
      _WHISPER_TRANSCRIBE_TOKEN_ID,
      _WHISPER_NO_TIMESTAMPS_TOKEN_ID,
  ]


class Whisper(asr_model.AsrModel):
  """Wrapper for WhisperForConditionalGeneration for encoder outputs."""

  HF_MODEL_ID = 'openai/whisper-tiny'

  def __init__(
      self,
      model_id: str = HF_MODEL_ID,
      override_transformers: bool = False,
      stateful: bool = True,
      max_decode_steps: int = 256,
  ):
    super().__init__(override_transformers)
    factory = transformers.WhisperForConditionalGeneration
    self._model = factory.from_pretrained(model_id).float().eval()
    self._replace_normalizations(self._model)
    self._encoder = WhisperEncoder(self._model).eval()
    self._stateful = stateful
    self._max_decode_steps = max_decode_steps
    self._decoder = WhisperDecoder(
        self._model, stateful=stateful, max_decode_steps=max_decode_steps
    ).eval()
    self._processor = asr_model.TransformersProcessor(model_id)

  def get_speech2text_metadata(self, export_config):
    # WhisperFeatureExtractor and WhisperEncoder always use a fixed 30 s
    # (3000-frame) encoder window regardless of `export_config.input_sec`,
    # matching the runtime's 30 s Whisper defaults.
    metadata = self.build_speech2text_metadata('whisper')
    is_stateful = getattr(self, '_stateful', False)
    if export_config is not None and getattr(export_config, 'stateful_after', None) is not None:
      is_stateful = is_stateful and (export_config.stateful_after > 0)
    if is_stateful:
      # Whisper only supports single-token stateful step decoding (stateful_after=1).
      metadata.decode_statefully_after = 1
    elif metadata.HasField('decode_statefully_after'):
      metadata.ClearField('decode_statefully_after')

    model = getattr(self, '_model', None)
    model_config = getattr(model, 'config', None)
    num_mel_bins = getattr(model_config, 'num_mel_bins', 80)
    if num_mel_bins and int(num_mel_bins) != 80:
      metadata.log_mel_spectrogram_config.n_mels = int(num_mel_bins)

    extra_kwargs = (
        getattr(export_config, 'extra_kwargs', None)
        if export_config is not None
        else None
    )
    whisper_language = (
        extra_kwargs.get('whisper_language') if extra_kwargs else None
    )
    processor = getattr(self, '_processor', None)
    hf_proc = getattr(processor, '_processor', None)
    tokenizer = getattr(hf_proc, 'tokenizer', None)
    if whisper_language or tokenizer is not None:
      metadata.decode_start_token_ids.extend(
          resolve_whisper_start_token_ids(
              str(whisper_language) if whisper_language else 'en',
              tokenizer,
          )
      )
    return metadata

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
    if not self._stateful or num_tokens > 1:
      tokens = torch.arange(num_tokens, dtype=torch.int32).unsqueeze(0)
      return encoder_output + (tokens, asr_model.get_causal_mask(num_tokens))

    tokens = torch.zeros((1, 1), dtype=torch.int32)
    step = torch.zeros((1,), dtype=torch.int32)
    config = self._model.config
    num_heads = config.decoder_attention_heads
    head_dim = config.d_model // num_heads
    kv_caches = []
    for _ in range(config.decoder_layers):
      k_cache = torch.zeros(
          (1, num_heads, self._max_decode_steps, head_dim), dtype=torch.float32
      )
      v_cache = torch.zeros(
          (1, num_heads, self._max_decode_steps, head_dim), dtype=torch.float32
      )
      kv_caches.append(k_cache)
      kv_caches.append(v_cache)
    return encoder_output + (tokens, step, *kv_caches)

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
