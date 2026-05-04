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
"""Loader for models."""

from typing import Tuple
from litert_torch.generative.export_hf.experimental.calib import sampling_executor as tfl_sampling_executor
from litert_torch.generative.export_hf.experimental.calib import tokenizer as tokenizer_lib


def _infer_model_specs(
    prefill_model_path: str,
) -> list[int]:
  """Infers the model specs."""
  prefill_model = tfl_sampling_executor.load_model(prefill_model_path)

  prefill_signatures = [
      x for x in prefill_model.get_signature_list() if 'prefill_' in x
  ]
  prefill_lengths = [int(x.split('_')[-1]) for x in prefill_signatures]
  del prefill_model
  return prefill_lengths


def _infer_drafter_step(decode_model_path: str):
  """Infers the drafter step."""
  decode_model = tfl_sampling_executor.load_model(decode_model_path)
  runner = decode_model.get_signature_runner('verify')
  drafter_step = runner.get_input_details()['embeddings']['shape'][1] - 1
  return drafter_step


def load_models(
    model_path: str | Tuple[str, str],
    embedder_model_path: str,
    spm_path: str | None,
    transformers_model_path: str | None,
    max_kv_cache_size: int | None,
    auxiliary_model_path: str | None = None,
    mask_model_path: str | None = None,
    rope_model_path: str | None = None,
    cache_update_model_path: str | None = None,
    per_layer_embedder_model_path: str | None = None,
    mm_encoder_model_path: str | None = None,
    mm_adapter_model_path: str | None = None,
    enable_calibration: bool = False,
    enable_min_max_calibration_update: bool = True,
) -> tfl_sampling_executor.TflSamplingExecutorConfig:
  """Loads the models."""
  if isinstance(model_path, tuple):
    prefill_model_path, decode_model_path = model_path
    decode_model_path = decode_model_path or prefill_model_path
  else:
    prefill_model_path = model_path
    decode_model_path = model_path
  prefill_lengths = _infer_model_specs(prefill_model_path)
  prefill_model_entries = {
      i: tfl_sampling_executor.TFLModelEntry(
          path=prefill_model_path,
          signature_name=f'prefill_{i}',
      )
      for i in prefill_lengths
  }
  decode_model_entry = tfl_sampling_executor.TFLModelEntry(
      path=decode_model_path,
      signature_name='decode',
  )
  prefill_embedder_model_entries = {
      i: tfl_sampling_executor.TFLModelEntry(
          path=embedder_model_path,
          signature_name=f'prefill_embedder_{i}',
      )
      for i in prefill_lengths
  }
  decode_embedder_model_entry = tfl_sampling_executor.TFLModelEntry(
      path=embedder_model_path,
      signature_name='decode_embedder',
  )
  if per_layer_embedder_model_path:
    prefill_per_layer_embedder_model_entries = {
        i: tfl_sampling_executor.TFLModelEntry(
            path=per_layer_embedder_model_path,
            signature_name=f'prefill_per_layer_embedder_{i}',
        )
        for i in prefill_lengths
    }
    decode_per_layer_embedder_model_entry = tfl_sampling_executor.TFLModelEntry(
        path=per_layer_embedder_model_path,
        signature_name='decode_per_layer_embedder',
    )
  else:
    prefill_per_layer_embedder_model_entries = None
    decode_per_layer_embedder_model_entry = None

  if mm_encoder_model_path and mm_adapter_model_path:
    mm_encoder_model_entry = tfl_sampling_executor.TFLModelEntry(
        path=mm_encoder_model_path,
        signature_name=None,
    )
    mm_adapter_model_entry = tfl_sampling_executor.TFLModelEntry(
        path=mm_adapter_model_path,
        signature_name=None,
    )
  else:
    mm_encoder_model_entry = None
    mm_adapter_model_entry = None

  mask_model_path = mask_model_path or auxiliary_model_path
  prefill_mask_model_entries = {
      i: tfl_sampling_executor.TFLModelEntry(
          path=mask_model_path,
          signature_name=f'prefill_mask_{i}',
      )
      for i in prefill_lengths
  }
  decode_mask_model_entry = tfl_sampling_executor.TFLModelEntry(
      path=mask_model_path,
      signature_name='decode_mask',
  )

  rope_model_path = rope_model_path or auxiliary_model_path
  prefill_rope_model_entries = {
      i: tfl_sampling_executor.TFLModelEntry(
          path=rope_model_path,
          signature_name=f'prefill_rope_{i}',
      )
      for i in prefill_lengths
  }
  decode_rope_model_entry = tfl_sampling_executor.TFLModelEntry(
      path=rope_model_path,
      signature_name='decode_rope',
  )

  cache_update_model_path = cache_update_model_path or auxiliary_model_path
  prefill_cache_update_model_entries = {
      i: tfl_sampling_executor.TFLModelEntry(
          path=cache_update_model_path,
          signature_name=f'prefill_cache_update_{i}',
      )
      for i in prefill_lengths
  }
  decode_cache_update_model_entry = tfl_sampling_executor.TFLModelEntry(
      path=cache_update_model_path,
      signature_name='decode_cache_update',
  )
  tokenizer_config = tokenizer_lib.TokenizerConfig(
      vocab_path=spm_path,
      transformers_model_path=transformers_model_path
  )
  return tfl_sampling_executor.TflSamplingExecutorConfig(
      prefill_model_entries=prefill_model_entries,
      decode_model_entry=decode_model_entry,
      max_kv_cache_size=max_kv_cache_size,
      prefill_mask_model_entries=prefill_mask_model_entries,
      decode_mask_model_entry=decode_mask_model_entry,
      prefill_rope_model_entries=prefill_rope_model_entries,
      decode_rope_model_entry=decode_rope_model_entry,
      prefill_embedder_model_entries=prefill_embedder_model_entries,
      decode_embedder_model_entry=decode_embedder_model_entry,
      tokenizer_config=tokenizer_config,
      prefill_cache_update_model_entries=prefill_cache_update_model_entries,
      decode_cache_update_model_entry=decode_cache_update_model_entry,
      prefill_per_layer_embedder_model_entries=prefill_per_layer_embedder_model_entries,
      decode_per_layer_embedder_model_entry=decode_per_layer_embedder_model_entry,
      mm_encoder_model_entry=mm_encoder_model_entry,
      mm_adapter_model_entry=mm_adapter_model_entry,
      enable_calibration=enable_calibration,
      enable_min_max_calibration_update=enable_min_max_calibration_update,
  )
