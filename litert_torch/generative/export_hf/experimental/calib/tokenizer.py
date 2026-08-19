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
"""Tokenizer."""

import dataclasses
import io

import numpy as np
from PIL import Image
import transformers

import sentencepiece as spm

IMAGE_PREPREOCESSORS = {}


def _register_image_preprocessor(model_type: str):
  """Decorator to register an image preprocessor."""

  def decorator(image_preprocessor_cls):
    IMAGE_PREPREOCESSORS[model_type] = image_preprocessor_cls
    return image_preprocessor_cls

  return decorator


@dataclasses.dataclass
class DataItem:
  """Data item."""

  image_bytes: bytes | None = None
  text: str | None = None


@dataclasses.dataclass
class Request:
  """Request."""

  contents: list[DataItem]


@dataclasses.dataclass
class TokenizerConfig:
  vocab_path: str | None = None
  transformers_model_path: str | None = None
  tokenizer_json_path: str | None = None
  chat_template: str | None = None

  def make(self) -> 'Tokenizer':
    return Tokenizer(
        spm_model_path=self.vocab_path,
        transformers_model_path=self.transformers_model_path,
        tokenizer_json_path=self.tokenizer_json_path,
        chat_template=self.chat_template,
    )


class Tokenizer:
  """Tokenizes the input string."""

  def __init__(
      self,
      spm_model_path: str | None = None,
      transformers_model_path: str | None = None,
      tokenizer_json_path: str | None = None,
      chat_template: str | None = None,
  ):
    self.spm = None
    self.tx_tokenizer = None
    self._image_preprocessor = None

    if tokenizer_json_path:
      self.tx_tokenizer = transformers.PreTrainedTokenizerFast(
          tokenizer_file=tokenizer_json_path,
          chat_template=chat_template,
      )
    elif transformers_model_path:
      self.tx_tokenizer = transformers.AutoTokenizer.from_pretrained(
          transformers_model_path
      )

    if spm_model_path:
      self.spm = spm.SentencePieceProcessor()
      self.spm.Load(spm_model_path)

    if not self.spm and not self.tx_tokenizer:
      raise ValueError(
          'Must specify either spm_model_path, tokenizer_json_path, or'
          ' transformers_model_path.'
      )

    if transformers_model_path:
      try:
        config = transformers.AutoConfig.from_pretrained(
            transformers_model_path
        )
        image_preprocessor_cls = IMAGE_PREPREOCESSORS.get(
            config.model_type, None
        )
      except (ValueError, OSError):
        image_preprocessor_cls = None
      if image_preprocessor_cls:
        self._image_preprocessor = image_preprocessor_cls(
            transformers_model_path, self
        )

  def tokenize_internal(self, input_string: str):
    """Tokenizes the input string."""
    if self.spm:
      input_ids = self.spm.EncodeAsIds(input_string)
    elif self.tx_tokenizer:
      input_ids = self.tx_tokenizer.encode(  # pytype: disable=attribute-error
          input_string, add_special_tokens=False
      )
    else:
      raise ValueError('No tokenizer available.')
    return np.array(input_ids, dtype=np.int32)

  @property
  def eos_id(self) -> int:
    """Returns the EOS id."""
    if self.spm:
      return self.spm.eos_id()
    elif self.tx_tokenizer:
      eos = getattr(self.tx_tokenizer, 'eos_token_id', -1)
      if eos is None:
        return -1
      if isinstance(eos, (list, tuple, set)):
        return int(list(eos)[0]) if eos else -1
      return int(eos)
    else:
      raise ValueError('No tokenizer available.')

  @property
  def bos_id(self) -> int | None:
    """Returns the BOS id or None if no BOS token is configured."""
    if self.spm:
      return self.spm.bos_id()
    elif self.tx_tokenizer:
      bos = getattr(self.tx_tokenizer, 'bos_token_id', None)
      if isinstance(bos, int) and bos >= 0:
        return bos
      for token_str in ('<bos>', '<s>', '<|begin_of_text|>'):
        conv = getattr(self.tx_tokenizer, 'convert_tokens_to_ids', None)
        if callable(conv):
          tid = conv(token_str)
          if (
              isinstance(tid, int)
              and tid >= 0
              and tid != getattr(self.tx_tokenizer, 'unk_token_id', -1)
          ):
            return tid
      return None
    else:
      raise ValueError('No tokenizer available.')

  @property
  def stop_token_ids(self) -> set[int]:
    """Returns set of stop/EOS token IDs automatically discovered from tokenizer."""
    ids = set()
    if self.spm:
      ids.add(self.spm.eos_id())
    elif self.tx_tokenizer:
      eos = getattr(self.tx_tokenizer, 'eos_token_id', None)
      if isinstance(eos, (list, tuple, set)):
        ids.update(int(x) for x in eos if x is not None)
      elif isinstance(eos, int) and eos >= 0:
        ids.add(eos)
      for token_str in ('<end_of_turn>', '<eos>', '</s>', '<|endoftext|>'):
        conv = getattr(self.tx_tokenizer, 'convert_tokens_to_ids', None)
        if callable(conv):
          tid = conv(token_str)
          if (
              isinstance(tid, int)
              and tid >= 0
              and tid != getattr(self.tx_tokenizer, 'unk_token_id', -1)
          ):
            ids.add(tid)
    return ids

  def detokenize_internal(self, input_ids) -> str:
    """Detokenizes the input string."""
    if self.spm:
      return self.spm.DecodeIds(input_ids)
    elif self.tx_tokenizer:
      return self.tx_tokenizer.decode(  # pytype: disable=attribute-error
          input_ids,
          skip_special_tokens=False,
      )
    else:
      raise ValueError('No tokenizer available.')

  def tokenize(self, input_string: str, prepend_bos: bool = True):
    """Tokenizes the input string."""
    input_ids = self.tokenize_internal(input_string)
    if prepend_bos:
      bos_id = self.bos_id
      if bos_id is not None:
        input_ids = np.array(input_ids).tolist()
        if len(input_ids) > 0 and input_ids[0] == bos_id:
          pass
        else:
          input_ids = [bos_id] + input_ids
    input_ids = np.array(
        np.array(input_ids).tolist(),
        dtype=np.int32,
    )
    return np.array(input_ids).astype(np.int32)

  def process_request(self, request: Request, prepend_bos: bool = True):
    """Processes the request."""
    ids_with_indices = []
    image_idx: set[int] = set()
    for item_idx, item in enumerate(request.contents):
      if item.text is not None:
        text_ids = self.tokenize_internal(item.text)
        text_ids = np.array(text_ids).astype(np.int32).tolist()
        ids_with_indices.append((item_idx, text_ids))
      elif item.image_bytes is not None:
        ids_with_indices.append(
            (item_idx, self._preprocess_image(item.image_bytes))
        )
        image_idx.add(item_idx)

    ids_with_indices = sorted(ids_with_indices, key=lambda x: x[0])

    bos_id = self.bos_id

    ids = []
    index_media = []
    index_feat_in_media = []
    images = []

    for item_idx, item in ids_with_indices:
      if item_idx in image_idx:
        assert (
            self._image_preprocessor
            and self._image_preprocessor.num_tokens_per_image
        ), 'num_tokens_per_image must be specified for image requests.'
        ids.append(self._image_preprocessor.special_tokens.get('soi', None))
        index_media.append(-1)
        index_feat_in_media.append(-1)

        ids += [-2] * self._image_preprocessor.num_tokens_per_image
        index_media += [len(images)] * (
            self._image_preprocessor.num_tokens_per_image
        )
        index_feat_in_media += list(
            range(self._image_preprocessor.num_tokens_per_image)
        )

        ids.append(self._image_preprocessor.special_tokens.get('eoi', None))
        index_media.append(-1)
        index_feat_in_media.append(-1)

        images.append(item)
      else:
        ids += item
        index_media += [-1] * len(item)
        index_feat_in_media += [-1] * len(item)

    if prepend_bos and bos_id is not None:
      if len(ids) == 0 or ids[0] != bos_id:
        ids = [bos_id] + ids
        index_media = [-1] + index_media
        index_feat_in_media = [-1] + index_feat_in_media

    return (
        np.asarray(ids).astype(np.int32)[None, :],
        images,
        np.asarray(index_media).astype(np.int32)[None, :],
        np.asarray(index_feat_in_media).astype(np.int32)[None, :],
    )

  def _preprocess_image(self, image_bytes: bytes) -> dict[str, np.ndarray]:
    """Preprocesses the image."""
    if self._image_preprocessor:
      return self._image_preprocessor(image_bytes)
    else:
      raise NotImplementedError()


def interleave_media_features_in_text(
    inputs,
    index_media,
    index_feat_in_media,
    media_features,
):
  """Interleaves media features into a text sequence."""
  output = np.copy(inputs)
  input_len = inputs.shape[1]

  pad_len = input_len - index_media.shape[1]
  assert pad_len >= 0, 'Inputs sequence length smaller than index arrays.'

  if pad_len > 0:
    padding_config = ((0, 0), (0, pad_len))
    index_media = np.pad(
        index_media, padding_config, mode='constant', constant_values=-1
    )
    index_feat_in_media = np.pad(
        index_feat_in_media, padding_config, mode='constant', constant_values=-1
    )

  mask = index_media >= 0
  batch_indices, seq_indices = np.where(mask)

  if batch_indices.size == 0:
    return output

  media_indices_to_use = index_media[batch_indices, seq_indices]
  feat_indices_to_use = index_feat_in_media[batch_indices, seq_indices]

  features_to_insert = media_features[
      batch_indices, media_indices_to_use, feat_indices_to_use
  ]

  output[batch_indices, seq_indices] = features_to_insert

  return output


@_register_image_preprocessor('lfm2_vl')
class LFM2VLImagePreprocessor:
  """LFM2VL image preprocessor."""

  def __init__(
      self,
      transformers_model_path: str,
      tokenizer: Tokenizer,
  ):
    self.parent = tokenizer
    self.transformers_model_path = transformers_model_path
    # Hardcoded here
    self.num_tokens_per_image = 256
    self.pre_preprocess_image_res = 512

    self.processor = transformers.AutoImageProcessor.from_pretrained(
        transformers_model_path
    )
    tx_tokenizer = transformers.AutoTokenizer.from_pretrained(
        transformers_model_path
    )
    self.special_tokens = {
        'soi': tx_tokenizer.convert_tokens_to_ids('<|image_start|>'),  # pyrefly: ignore[missing-attribute]
        'eoi': tx_tokenizer.convert_tokens_to_ids('<|image_end|>'),  # pyrefly: ignore[missing-attribute]
    }

  def __call__(self, image_bytes: bytes) -> dict[str, np.ndarray]:
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)

    image = image.convert('RGB')
    target_shape = (
        self.pre_preprocess_image_res,
        self.pre_preprocess_image_res,
    )
    image = image.resize(target_shape, resample=Image.Resampling.LANCZOS)

    processed_image = self.processor(image, return_tensors='np').pixel_values

    return {'images': processed_image}
