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
      self, model_id: str = HF_MODEL_ID, override_transformers: bool = False
  ):
    super().__init__(override_transformers)
    factory = transformers.WhisperForConditionalGeneration
    self._model = factory.from_pretrained(model_id).float().eval()
    self._replace_normalizations(self._model)
    self._encoder = WhisperEncoder(self._model).eval()
    self._decoder = WhisperDecoder(self._model).eval()
    self._processor = asr_model.TransformersProcessor(model_id)

  def get_speech2text_metadata(self, export_config):
    # WhisperFeatureExtractor and WhisperEncoder always use a fixed 30 s
    # (3000-frame) encoder window regardless of `export_config.input_sec`,
    # matching the runtime's 30 s Whisper defaults.
    metadata = self.build_speech2text_metadata('whisper')
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
