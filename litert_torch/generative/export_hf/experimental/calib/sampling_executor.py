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
"""Sampling executor for TFLite models."""

import copy
import dataclasses
import os
from typing import Sequence

from litert_torch.generative.export_hf.experimental.calib import tokenizer as tokenizer_lib
import numpy as np
import torch

from ai_edge_quantizer import calibrator
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor
from ai_edge_quantizer.utils import calibration_utils as cu

TokenizerConfig = tokenizer_lib.TokenizerConfig
Request = tokenizer_lib.Request
DataItem = tokenizer_lib.DataItem

RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[94m'
RESET = '\033[0m'


@dataclasses.dataclass
class TFLModelEntry:
  """An TFLite model entry."""

  path: str
  signature_name: str | None = None


@dataclasses.dataclass(kw_only=True)
class TflSamplingExecutorConfig:
  """Sampling executor for TFLite models."""

  prefill_model_entries: dict[int, TFLModelEntry]
  decode_model_entry: TFLModelEntry
  max_kv_cache_size: int

  prefill_mask_model_entries: dict[int, TFLModelEntry]
  decode_mask_model_entry: TFLModelEntry

  prefill_rope_model_entries: dict[int, TFLModelEntry]
  decode_rope_model_entry: TFLModelEntry

  prefill_embedder_model_entries: dict[int, TFLModelEntry]
  decode_embedder_model_entry: TFLModelEntry

  tokenizer_config: tokenizer_lib.TokenizerConfig

  prefill_cache_update_model_entries: dict[int, TFLModelEntry]
  decode_cache_update_model_entry: TFLModelEntry

  prefill_per_layer_embedder_model_entries: dict[int, TFLModelEntry] | None = (
      None
  )
  decode_per_layer_embedder_model_entry: TFLModelEntry | None = None

  mm_encoder_model_entry: TFLModelEntry | None = None
  mm_adapter_model_entry: TFLModelEntry | None = None

  early_terminate_suffix: str | None = None
  stop_token: int | None = None
  enable_calibration: bool = False
  enable_min_max_calibration_update: bool = True


def load_model(
    path: str, enable_calibration: bool = False
) -> calibrator.CalibrationInterpreter:
  """Loads the model."""
  print(f'Loading model from {path}...')
  return calibrator.CalibrationInterpreter(
      path,
      mode=calibrator.CalibrationMode.CALIBRATION
      if enable_calibration
      else calibrator.CalibrationMode.INFERENCE,
  )


@dataclasses.dataclass
class DecodeState:
  """Decode state."""

  kv_cache: dict[str, np.ndarray]
  num_input_tokens: int
  token_ids: np.ndarray
  sampled_tokens: np.ndarray
  logits: np.ndarray | None
  time_step: np.ndarray
  generate: bool
  done: bool

  next_decode_token: np.ndarray | None = None
  processed_embeds: np.ndarray | None = None
  images: list[dict[str, np.ndarray]] | None = None
  index_media: np.ndarray | None = None
  index_feat_in_media: np.ndarray | None = None


class Executor:
  """Executor for TFLite models."""

  def __init__(
      self,
      config: TflSamplingExecutorConfig,
      stream_output: bool = False,
  ):
    self.config = config
    self.stream_output = stream_output
    self.interpreters: dict[str, calibrator.CalibrationInterpreter] = {}

    self.prefill_runners = {}
    for input_length, model_entry in config.prefill_model_entries.items():
      self.prefill_runners[input_length] = self._get_interpreter(
          model_entry.path, config.enable_calibration
      ).get_signature_runner(model_entry.signature_name)

    self.cache_length = config.max_kv_cache_size
    self.decode_runner = self._get_interpreter(
        config.decode_model_entry.path, config.enable_calibration
    ).get_signature_runner(config.decode_model_entry.signature_name)

    self.prefill_mask_runners = {}
    for input_length, model_entry in config.prefill_mask_model_entries.items():
      self.prefill_mask_runners[input_length] = self._get_interpreter(
          model_entry.path, config.enable_calibration
      ).get_signature_runner(model_entry.signature_name)

    self.decode_mask_runner = self._get_interpreter(
        config.decode_mask_model_entry.path, config.enable_calibration
    ).get_signature_runner(config.decode_mask_model_entry.signature_name)

    self.prefill_rope_runners = {}
    for input_length, model_entry in config.prefill_rope_model_entries.items():
      self.prefill_rope_runners[input_length] = self._get_interpreter(
          model_entry.path, config.enable_calibration
      ).get_signature_runner(model_entry.signature_name)

    self.decode_rope_runner = self._get_interpreter(
        config.decode_rope_model_entry.path, config.enable_calibration
    ).get_signature_runner(config.decode_rope_model_entry.signature_name)

    self.prefill_embedder_runners = {}
    for (
        input_length,
        model_entry,
    ) in config.prefill_embedder_model_entries.items():
      self.prefill_embedder_runners[input_length] = self._get_interpreter(
          model_entry.path, config.enable_calibration
      ).get_signature_runner(model_entry.signature_name)

    self.decode_embedder_runner = self._get_interpreter(
        config.decode_embedder_model_entry.path, config.enable_calibration
    ).get_signature_runner(config.decode_embedder_model_entry.signature_name)

    self.prefill_cache_update_runners = {}
    for seq_len in self.prefill_runners:
      self.prefill_cache_update_runners[seq_len] = self._get_interpreter(
          config.prefill_cache_update_model_entries[seq_len].path,
          config.enable_calibration,
      ).get_signature_runner(f'prefill_cache_update_{seq_len}')

    self.decode_cache_update_runner = self._get_interpreter(
        config.decode_cache_update_model_entry.path, config.enable_calibration
    ).get_signature_runner('decode_cache_update')

    self.prefill_per_layer_embedder_runners = None
    if config.prefill_per_layer_embedder_model_entries:
      self.prefill_per_layer_embedder_runners = {}
      for (
          input_length,
          model_entry,
      ) in config.prefill_per_layer_embedder_model_entries.items():
        self.prefill_per_layer_embedder_runners[input_length] = (
            self._get_interpreter(
                model_entry.path, config.enable_calibration
            ).get_signature_runner(model_entry.signature_name)
        )
    if config.decode_per_layer_embedder_model_entry:
      self.decode_per_layer_embedder_runner = self._get_interpreter(
          config.decode_per_layer_embedder_model_entry.path,
          config.enable_calibration,
      ).get_signature_runner(
          config.decode_per_layer_embedder_model_entry.signature_name
      )
    else:
      self.decode_per_layer_embedder_runner = None

    self.mm_encoder_runner = None
    if config.mm_encoder_model_entry is not None:
      self.mm_encoder_runner = self._get_interpreter(
          config.mm_encoder_model_entry.path, config.enable_calibration
      ).get_signature_runner(config.mm_encoder_model_entry.signature_name)

    self.mm_adapter_runner = None
    if config.mm_adapter_model_entry is not None:
      self.mm_adapter_runner = self._get_interpreter(
          config.mm_adapter_model_entry.path, config.enable_calibration
      ).get_signature_runner(config.mm_adapter_model_entry.signature_name)

    self.tokenizer = config.tokenizer_config.make()

  def _get_interpreter(
      self, path: str, enable_calibration: bool
  ) -> calibrator.CalibrationInterpreter:
    """Returns the interpreter for the given path."""
    if path not in self.interpreters:
      if self.config.enable_min_max_calibration_update:
        qsv_update_func = cu.min_max_update
      else:
        qsv_update_func = cu.moving_average_update
      self.interpreters[path] = calibrator.CalibrationInterpreter(
          path,
          mode=calibrator.CalibrationMode.CALIBRATION
          if enable_calibration
          else calibrator.CalibrationMode.INFERENCE,
          qsv_update_func=qsv_update_func,
      )
    return self.interpreters[path]

  def get_calibration_results(self):
    """Returns the calibration results."""
    results = {}
    for path, interp in self.interpreters.items():
      results[path] = interp.get_calibration_results()
    return results

  def save_calibration_results(
      self, output_dir: str, extra_metadata: dict[str, str] | None = None
  ):
    """Saves the calibration results."""
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    for path, interp in self.interpreters.items():
      model_name = path.split('/')[-1]
      output_path = f'{output_dir}/{model_name}.json'
      temp_output_path = output_path + '.tmp'
      interp.save_calibration_result(temp_output_path, extra_metadata)
      os.replace(temp_output_path, output_path)
      print(f'--- Saved calibration results for {model_name} to {output_path}')

  def load_calibration_results(self, output_dir: str):
    """Loads the calibration results."""
    for path, interp in self.interpreters.items():
      model_name = path.split('/')[-1]
      output_path = f'{output_dir}/{model_name}.json'
      if os.path.exists(output_path):
        # Accessing protected member _calibrator to load QSVs.
        # pylint: disable=protected-access
        interp._calibrator.load_model_qsvs(output_path)
        print(
            f'--- Loaded calibration results for {model_name} from'
            f' {output_path}'
        )

  def init_cache(self) -> dict[str, np.ndarray]:
    """Init cache to zeros."""
    input_details = self.decode_runner.get_input_details()
    kv_cache = {
        k: np.zeros(input_details[k]['shape'], dtype=np.float32)
        for k in input_details
        if 'cache' in k
    }
    return kv_cache

  def init_decode_state(self, request: str | tokenizer_lib.Request):
    """Initializes the decode state."""
    if isinstance(request, str):
      text = request
      tokens = self.tokenizer.tokenize(text)[None, :]

      cache = self.init_cache()
      decode_state = DecodeState(
          kv_cache=cache,
          num_input_tokens=len(tokens[0]),
          token_ids=tokens,
          sampled_tokens=np.asarray([[]], dtype=np.int32),
          logits=None,
          time_step=np.asarray(0, dtype=np.int32),
          generate=False,
          done=False,
          next_decode_token=np.asarray([[]], dtype=np.int32),
      )
      return decode_state
    else:
      tokens, images, index_media, index_feat_in_media = (
          self.tokenizer.process_request(request)
      )

      cache = self.init_cache()
      decode_state = DecodeState(
          kv_cache=cache,
          num_input_tokens=len(tokens[0]),
          token_ids=tokens,
          sampled_tokens=np.asarray([[]], dtype=np.int32),
          logits=None,
          time_step=np.asarray(0, dtype=np.int32),
          generate=False,
          done=False,
          images=images,
          index_media=index_media,
          index_feat_in_media=index_feat_in_media,
          next_decode_token=np.asarray([[]], dtype=np.int32),
      )
      return decode_state

  def tokenize(self, text: str) -> Sequence[int]:
    """Tokenizes the text."""
    return self.tokenizer.tokenize_internal(text).tolist()

  def prefill_embeds(self, decode_state: DecodeState):
    """Prefills the embeddings."""
    time_step = decode_state.time_step.tolist()
    num_input_tokens = decode_state.num_input_tokens
    remaining_input_size = num_input_tokens - time_step
    embeds = []

    while remaining_input_size:
      # Find the smallest input size that is larger than the current input size.
      available_input_size = [
          x for x in self.prefill_runners.keys() if x >= remaining_input_size
      ]
      if available_input_size:
        input_size = min(available_input_size)
        padded_tokens = np.pad(
            decode_state.token_ids[:, time_step:],
            ((0, 0), (0, input_size - remaining_input_size)),
        )
      else:
        input_size = max(x for x in self.prefill_runners.keys())
        padded_tokens = decode_state.token_ids[
            :, time_step : time_step + input_size
        ]

      if time_step + input_size > self.cache_length:
        raise ValueError('Prefill chunk exceeds the cache length.')

      input_embeds = try_run_signature_with_quant_dequant(
          {'token_ids': padded_tokens},
          self.prefill_embedder_runners[input_size],
      )['embeddings']

      embeds.append(input_embeds)
      processed_token_length = min(remaining_input_size, input_size)
      time_step += processed_token_length
      remaining_input_size = num_input_tokens - time_step
    embeds = np.concatenate(embeds, axis=1)
    if decode_state.processed_embeds is not None:
      embeds = np.concatenate([decode_state.processed_embeds, embeds], axis=1)
    decode_state = dataclasses.replace(
        decode_state,
        processed_embeds=embeds,
    )
    decode_state = self.encode_images(decode_state)
    return decode_state

  def encode_images(self, decode_state: DecodeState):
    """Encodes the images."""
    if decode_state.images is None or len(decode_state.images) == 0:
      return decode_state

    num_images = len(decode_state.images)
    mm_embs = []
    mm_embedding = None
    for i in range(num_images):
      img = decode_state.images[i]
      img_features = self.mm_encoder_runner(
          **img,
      )
      img_features = img_features['features']
      mm_embedding = self.mm_adapter_runner(
          soft_tokens=img_features,
      )['mm_embedding']
      mm_embs.append(mm_embedding)
    if mm_embs:
      mm_embedding = np.concatenate(mm_embs, axis=0)
    input_embeddings = decode_state.processed_embeds

    if mm_embs:
      interleaved_embeddings = tokenizer_lib.interleave_media_features_in_text(
          input_embeddings,
          decode_state.index_media,
          decode_state.index_feat_in_media,
          mm_embedding[None, ...],
      )
    else:
      interleaved_embeddings = input_embeddings

    decode_state = dataclasses.replace(
        decode_state,
        processed_embeds=interleaved_embeddings,
    )
    return decode_state

  def prefill_chunk(self, decode_state: DecodeState):
    """Prefills the chunk."""
    assert not decode_state.generate, 'Generate should be false.'

    time_step = decode_state.time_step
    num_input_tokens = decode_state.num_input_tokens

    remaining_input_size = num_input_tokens - time_step

    # Find the smallest input size that is larger than the current input size.
    available_input_size = [
        x for x in self.prefill_runners.keys() if x >= remaining_input_size
    ]
    if available_input_size:
      input_size = min(available_input_size)
      padded_tokens = np.pad(
          decode_state.token_ids[:, time_step:],
          ((0, 0), (0, input_size - remaining_input_size)),
      )
    else:
      input_size = max(x for x in self.prefill_runners.keys())
      padded_tokens = decode_state.token_ids[
          :, time_step : time_step + input_size
      ]

    if time_step + input_size > self.cache_length:
      raise ValueError('Prefill chunk exceeds the cache length.')

    positions = np.arange(time_step, time_step + input_size, dtype=np.int32)

    prefill_masks = self.prefill_mask_runners[input_size](
        time_step=np.asarray(time_step, dtype=np.int32),
        input_tokens=padded_tokens,
    )

    input_embeds = decode_state.processed_embeds[
        :, time_step : time_step + input_size
    ]
    rope_runner = self.prefill_rope_runners[input_size]
    rope = rope_runner(input_pos=positions)

    ple_input_embeds = {}
    if self.prefill_per_layer_embedder_runners:
      ple_input_embeds = {
          'per_layer_embeddings': try_run_signature_with_quant_dequant(
              {'token_ids': padded_tokens},
              self.prefill_per_layer_embedder_runners[input_size],
          )['embeddings']
      }

    # Only pass the KV cache entries that are used in the prefill runner.
    # For split cache implementation, prefill will not require all layers' KV
    # cache.
    kv_cache_input = {
        x: decode_state.kv_cache[x]
        for x in decode_state.kv_cache
        if x in self.prefill_runners[input_size].get_input_details()
    }
    kv_slice: dict[str, np.ndarray] = try_run_signature_with_quant_dequant(
        {
            'embeddings': input_embeds,
            **rope,
            **prefill_masks,
            **kv_cache_input,
            **ple_input_embeds,
        },
        self.prefill_runners[input_size],
    )

    new_kv_cache = try_run_signature_with_quant_dequant(
        {
            **kv_slice,
            **decode_state.kv_cache,
            'input_pos': positions,
        },
        self.prefill_cache_update_runners[input_size],
    )

    processed_token_length = min(remaining_input_size, input_size)
    time_step += processed_token_length
    if time_step == num_input_tokens:
      generate = True
      time_step -= 1
    elif time_step < num_input_tokens:
      generate = False
    else:
      # time_step > num_input_tokens
      raise ValueError(f'Unknown time_step: {time_step}')

    decode_state = dataclasses.replace(
        decode_state,
        kv_cache=new_kv_cache,
        time_step=time_step,
        generate=generate,
        next_decode_token=decode_state.token_ids[:, time_step : time_step + 1],
    )
    return decode_state

  def sample_logits(self, logits: np.ndarray) -> np.ndarray:
    # Greedy sampling here...
    return np.argsort(logits, axis=-1).astype(np.int32)[..., -1]

  def decode_step(
      self,
      decode_state: DecodeState,
      token_ids_override: np.ndarray | None = None,
  ):
    """Decodes the next token."""

    time_step = decode_state.time_step
    input_tokens = decode_state.next_decode_token
    positions = np.asarray([time_step], dtype=np.int32)

    decode_masks = self.decode_mask_runner(
        time_step=np.asarray(time_step, dtype=np.int32),
        input_tokens=input_tokens,
    )

    input_embeds = try_run_signature_with_quant_dequant(
        {'token_ids': input_tokens},
        self.decode_embedder_runner,
    )['embeddings']

    if time_step < len(decode_state.token_ids[0]):
      processed_embeds = decode_state.processed_embeds
    else:
      processed_embeds = np.concatenate(
          [decode_state.processed_embeds, input_embeds], axis=1
      )

    ple_input_embeds = {}
    if self.decode_per_layer_embedder_runner:
      ple_input_embeds = {
          'per_layer_embeddings': try_run_signature_with_quant_dequant(
              {'embedding': input_tokens},
              self.decode_per_layer_embedder_runner,
          )['embeddings']
      }

    rope = self.decode_rope_runner(input_pos=positions)

    kv_slice: dict[str, np.ndarray] = try_run_signature_with_quant_dequant(
        {
            'embeddings': input_embeds,
            **rope,
            **decode_masks,
            **decode_state.kv_cache,
            **ple_input_embeds,
        },
        self.decode_runner,
    )

    logits = kv_slice.pop('logits')

    new_kv_cache = try_run_signature_with_quant_dequant(
        {
            **kv_slice,
            **decode_state.kv_cache,
            'input_pos': positions,
        },
        self.decode_cache_update_runner,
    )

    # Sample
    if token_ids_override is not None:
      next_token_ids = token_ids_override
    else:
      next_token_ids = self.sample_logits(logits)

    sampled_tokens = np.concatenate(
        [decode_state.sampled_tokens, next_token_ids], axis=-1
    )
    done = self.tokenizer.eos_id in next_token_ids

    current_tokens = sampled_tokens.tolist()[0]
    current_output = self.tokenizer.detokenize_internal(current_tokens)
    if (
        self.config.early_terminate_suffix is not None
        and current_output.endswith(self.config.early_terminate_suffix)
    ) or (
        self.config.stop_token is not None
        and current_tokens[-1] == self.config.stop_token
    ):
      done = True
    if self.stream_output:
      print(current_output)

    new_logits = (
        np.concatenate([decode_state.logits, logits], axis=1)
        if decode_state.logits is not None
        else logits
    )
    decode_state = dataclasses.replace(
        decode_state,
        kv_cache=new_kv_cache,
        time_step=time_step + 1,
        done=done,
        sampled_tokens=sampled_tokens,
        logits=new_logits,
        processed_embeds=processed_embeds,
        next_decode_token=next_token_ids,
    )
    return decode_state

  def sample_text(
      self,
      request: str | tokenizer_lib.Request,
      max_sample_step: int | None = None,
  ):
    """Samples text."""
    print('--- sample_text, initializing decode state (prefill)...')
    decode_state = self.init_decode_state(request)

    if decode_state.num_input_tokens >= self.cache_length:
      print('num_input_tokens >= cache_length')
      return ['']
    print('--- Starting prefill...')
    decode_state = self.prefill_embeds(decode_state)
    while not decode_state.generate:
      decode_state = self.prefill_chunk(decode_state)
    print('--- Prefill done. Starting decode...')

    if max_sample_step is None:
      max_sample_step = self.cache_length - decode_state.time_step
    else:
      max_sample_step = min(
          self.cache_length, max_sample_step + decode_state.time_step
      )

    decode_state = self.decode_step(decode_state)

    while decode_state.time_step < max_sample_step and not decode_state.done:
      decode_state = self.decode_step(decode_state)

    if self.stream_output:
      print()

    return [
        self.tokenizer.detokenize_internal(x)
        for x in decode_state.sampled_tokens.tolist()
    ]

  def score_suffix(self, suffix: str, decode_state: DecodeState):
    """Scores the suffix."""
    suffix_tokens = self.tokenizer.tokenize_internal(suffix)[
        np.newaxis, :
    ]  # [1, S]
    ret = np.asarray([]).astype(np.float32)
    # Note the last token is not sampled as the token_ids_override provided will
    # override the NEXT token to be encoded.
    for i in range(suffix_tokens.shape[1]):
      decode_state = self.decode_step(decode_state, suffix_tokens[:, i : i + 1])
      # Collects the prob of i-th token in the suffix.
      logits = decode_state.logits[0, -1, :]  # [V]
      token = suffix_tokens[0, i : i + 1]  # [1]
      probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
      ret = np.concatenate([ret, probs[token]])  # [S]
    return decode_state, ret  # [S]

  def score(self, text, suffix_list):
    """Samples text."""
    decode_state = self.init_decode_state(text)

    decode_state = self.prefill_embeds(decode_state)
    while not decode_state.generate:
      decode_state = self.prefill_chunk(decode_state)

    # decode_state_collection is a dict of {suffix: decode_state}, not a single
    # decode_state.
    decode_state_collection, scores = {}, {}
    for suffix in suffix_list:
      # score_suffix mutates the decode_state, so we need to make a copy.
      orig_decode_state = copy.deepcopy(decode_state)
      decode_state_collection[suffix], scores[suffix] = self.score_suffix(
          suffix, orig_decode_state
      )
    return decode_state_collection, scores


class ConversationExecutor(Executor):
  """Conversation executor for TFLite models."""

  decode_state: DecodeState | None = None

  def __init__(
      self,
      config: TflSamplingExecutorConfig,
      stream_output: bool = False,
  ):
    super().__init__(config, stream_output)
    self.decode_state: DecodeState | None = None

  def clear(self):
    """Clears the cache."""
    self.decode_state = None

  def extend_decode_state(self, request: str | tokenizer_lib.Request):
    """Initializes the decode state."""
    assert self.decode_state is not None, 'Decode state is not initialized.'
    if isinstance(request, str):
      text = request
      tokens = self.tokenizer.tokenize_internal(text)[None, :]
      tokens = np.concatenate(
          [self.decode_state.sampled_tokens[:, -1:], tokens], axis=1
      )

      decode_state = DecodeState(
          kv_cache=self.decode_state.kv_cache,
          num_input_tokens=self.decode_state.num_input_tokens + len(tokens[0]),
          token_ids=np.concatenate(
              [self.decode_state.token_ids, tokens], axis=1
          ),
          sampled_tokens=np.asarray([[]], dtype=np.int32),
          logits=None,
          time_step=self.decode_state.time_step,
          generate=False,
          done=False,
          processed_embeds=self.decode_state.processed_embeds[
              :, : self.decode_state.time_step, :
          ],
          images=self.decode_state.images,
          index_media=self.decode_state.index_media,
          index_feat_in_media=self.decode_state.index_feat_in_media,
      )
      return decode_state
    else:
      raise NotImplementedError('Not implemented for multimodal.')

  def sample_text(
      self,
      request: str | tokenizer_lib.Request,
      max_sample_step: int | None = None,
  ):
    """Samples text."""
    if self.decode_state is None:
      self.decode_state = self.init_decode_state(request)
    else:
      self.decode_state = self.extend_decode_state(request)

    if self.decode_state.num_input_tokens >= self.cache_length:
      print('num_input_tokens >= cache_length')
      return ['']

    self.decode_state = self.prefill_embeds(self.decode_state)
    while not self.decode_state.generate:
      self.decode_state = self.prefill_chunk(self.decode_state)

    if max_sample_step is None:
      max_sample_step = self.cache_length - self.decode_state.time_step
    else:
      max_sample_step = min(
          self.cache_length, max_sample_step + self.decode_state.time_step
      )

    while (
        self.decode_state.time_step < max_sample_step
        and not self.decode_state.done
    ):
      self.decode_state = self.decode_step(self.decode_state)

    if self.stream_output:
      print()

    print()
    print(
        self.tokenizer.detokenize_internal(
            self.decode_state.token_ids.tolist()[0]
        )
    )

    return [
        self.tokenizer.detokenize_internal(x)
        for x in self.decode_state.sampled_tokens.tolist()
    ]


def try_run_signature_with_quant_dequant(signature_input, signature_runner):
  """Runs signature with quantization and dequantization if applicable.

  Args:
    signature_input: key,value for inputs to the model. Key is the SignatureDef
      input Value is numpy array with the value.
    signature_runner: the signature runner for the model.

  Returns:
    dictionary of the `float` (either dequantized or original) results from the
    model invoke.
  """
  in_kwargs = try_get_quantized_input(signature_input, signature_runner)
  out_kwargs = signature_runner(**in_kwargs)
  return try_get_dequantized_output(out_kwargs, signature_runner)


def is_quantized(tensor_detail):
  """Returns whether the tensor is quantized."""
  quant_params = tensor_detail['quantization_parameters']
  return len(quant_params['scales']) > 0


def try_get_quantized_input(signature_input, signature_runner):
  """Returns quantized input if applicable."""
  input_details = signature_runner.get_input_details()
  for k, detail in input_details.items():
    input_data = signature_input[k]
    if input_data.dtype == np.float32 and is_quantized(detail):
      quant_params = qtyping.UniformQuantParams.from_tfl_tensor_details(detail)
      signature_input[k] = uniform_quantize_tensor.uniform_quantize(
          input_data, quant_params
      )
  return signature_input


def try_get_dequantized_output(
    signature_output,
    signature_runner,
):
  """Returns dequantized output if applicable."""
  output_details = signature_runner.get_output_details()
  for k, detail in output_details.items():
    if is_quantized(detail):
      quant_params = qtyping.UniformQuantParams.from_tfl_tensor_details(detail)
      signature_output[k] = uniform_quantize_tensor.uniform_dequantize(
          signature_output[k], quant_params
      ).astype(np.float32)
  return signature_output
