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
"""Sampling executor main."""

from collections.abc import Sequence
import json

from absl import app
from absl import flags
from litert_torch.generative.export_hf.experimental.calib import loader
from litert_torch.generative.export_hf.experimental.calib import sampling_executor as tfl_sampling_executor

_KV_CACHE_MAX_LEN = flags.DEFINE_integer(
    'kv_cache_max_len',
    1280,
    'The maximum size of KV cache buffer, including both prefill and decode.',
)

_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    None,
    'Path to the model.',
    required=True,
)
_DECODE_MODEL_PATH = flags.DEFINE_string(
    'decode_model_path',
    None,
    'Optional. Path to the decode model.',
)
_EMBEDDER_MODEL_PATH = flags.DEFINE_string(
    'embedder_model_path',
    None,
    'Path to the embedder model.',
    required=True,
)
_AUXILIARY_MODEL_PATH = flags.DEFINE_string(
    'auxiliary_model_path',
    None,
    'Path to the auxiliary model.',
    required=True,
)
_PLE_MODEL_PATH = flags.DEFINE_string(
    'ple_model_path',
    None,
    'Path to the per layer embedder model.',
)
_MM_ENCODER_MODEL_PATH = flags.DEFINE_string(
    'mm_encoder_model_path',
    None,
    'Path to the MM encoder model.',
)
_MM_ADAPTER_MODEL_PATH = flags.DEFINE_string(
    'mm_adapter_model_path',
    None,
    'Path to the MM adapter model.',
)
_SPM_PATH = flags.DEFINE_string(
    'spm_path',
    None,
    'Path to the SPM.',
)
_TRANSFORMERS_MODEL_PATH = flags.DEFINE_string(
    'transformers_model_path',
    None,
    'Path to the transformers model.',
)
_PROMPT = flags.DEFINE_string(
    'prompt',
    None,
    'Input prompt.',
)
_SECOND_PROMPT = flags.DEFINE_string(
    'second_prompt',
    None,
    'Second input prompt.',
)
_PROMPT_FILE = flags.DEFINE_string(
    'prompt_file',
    None,
    'Input prompt file.',
)
_EARLY_TERMINATE_SUFFIX = flags.DEFINE_string(
    'early_terminate_suffix',
    '',
    'Early terminate suffix.',
)
_STOP_TOKEN = flags.DEFINE_integer(
    'stop_token',
    None,
    'Stop token.',
)
_MAX_DECODE_STEPS = flags.DEFINE_integer(
    'max_decode_steps',
    None,
    'Maximum number of decode steps.',
)
_IMAGE_FILES = flags.DEFINE_multi_string(
    'image_files',
    None,
    'Input image file.',
)
_ENABLE_CALIBRATION = flags.DEFINE_bool(
    'enable_calibration',
    False,
    'Enable calibration.',
)
_CALIBRATION_RESULT_SAVE_DIR = flags.DEFINE_string(
    'calibration_result_save_dir',
    None,
    'Path to the output calibration result directory.',
)
_ENABLE_MIN_MAX_CALIBRATION_UPDATE = flags.DEFINE_bool(
    'enable_min_max_calibration_update',
    True,
    'Enable min max update for calibration. This is useful for NanoV4 as the'
    ' model is already heavily quantized. Using min/max will help us to find'
    ' the true range for k/v cache and rope signals.',
)

_ENABLE_FORMATTING = flags.DEFINE_bool(
    'enable_formatting',
    True,
    'Whether to enable formatting for the input prompts.',
)

_VIT_MM_ENCODER = flags.DEFINE_bool(
    'vit_mm_encoder',
    False,
    'Whether to use ViT mm encoder.',
)
_STREAM_OUTPUT = flags.DEFINE_bool(
    'stream_output',
    False,
    'Whether to stream the output.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _PROMPT_FILE.value is not None:
    if not _PROMPT_FILE.value.endswith('.riegeli'):
      with open(_PROMPT_FILE.value, 'r') as f:
        if _PROMPT_FILE.value.endswith('.json'):
          prompt = json.load(f)
        else:
          prompt = f.read()
    else:
      prompt = None  # Will be processed later
  else:
    prompt = _PROMPT.value

  print('--- Configuring executor...')
  config = loader.load_models(
      max_kv_cache_size=_KV_CACHE_MAX_LEN.value,
      model_path=(_MODEL_PATH.value, _DECODE_MODEL_PATH.value),
      embedder_model_path=_EMBEDDER_MODEL_PATH.value,
      spm_path=_SPM_PATH.value,
      transformers_model_path=_TRANSFORMERS_MODEL_PATH.value,
      auxiliary_model_path=_AUXILIARY_MODEL_PATH.value,
      per_layer_embedder_model_path=_PLE_MODEL_PATH.value,
      mm_encoder_model_path=_MM_ENCODER_MODEL_PATH.value,
      mm_adapter_model_path=_MM_ADAPTER_MODEL_PATH.value,
      enable_calibration=_ENABLE_CALIBRATION.value,
      enable_min_max_calibration_update=_ENABLE_MIN_MAX_CALIBRATION_UPDATE.value,
  )

  if _ENABLE_FORMATTING.value:
    assert (
        _TRANSFORMERS_MODEL_PATH.value is not None
    ), 'Transformers model path is required for formatting.'
    messages = [{'role': 'user', 'content': prompt}]
    tokenizer = config.tokenizer_config.make().tx_tokenizer
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f'--- Formatted prompt:\n\n{prompt}')

  if _EARLY_TERMINATE_SUFFIX.value:
    config.early_terminate_suffix = _EARLY_TERMINATE_SUFFIX.value
  if _STOP_TOKEN.value is not None:
    config.stop_token = _STOP_TOKEN.value
  print('--- Initializing executor. Loading models...')
  executor_cls = (
      tfl_sampling_executor.ConversationExecutor
      if _SECOND_PROMPT.value
      else tfl_sampling_executor.Executor
  )
  executor = executor_cls(config, stream_output=_STREAM_OUTPUT.value)
  print('--- Models loaded. Starting sampling...')

  images = []
  if _IMAGE_FILES.value is not None:
    for image_file in _IMAGE_FILES.value:
      with open(image_file, 'rb') as f:
        images.append(f.read())
    assert prompt is not None, 'Prompt is required.'
    prompt_parts = prompt.split('<img>')
    contents = []
    if prompt_parts[0]:
      contents.append(tfl_sampling_executor.DataItem(text=prompt_parts[0]))
    for i in range(1, len(prompt_parts)):
      contents.append(tfl_sampling_executor.DataItem(image_bytes=images[i - 1]))
      if prompt_parts[i]:
        contents.append(tfl_sampling_executor.DataItem(text=prompt_parts[i]))
    print('\n--- Processing prompt with images ---')
    print(f'Prompt:\n{prompt}')
    response = executor.sample_text(
        tfl_sampling_executor.Request(contents=contents),
        max_sample_step=_MAX_DECODE_STEPS.value,
    )
    print(f'Response:\n{response}')

  if isinstance(prompt, list):
    for i, p in enumerate(prompt):
      print(f'\n--- Processing prompt {i} ---')
      print(f'Prompt:\n{p}')
      response = executor.sample_text(
          p, max_sample_step=_MAX_DECODE_STEPS.value
      )
      print(f'Response:\n{response}')
  else:
    print('\n--- Processing prompt ---')
    print(f'Prompt:\n{prompt}')
    response = executor.sample_text(
        prompt, max_sample_step=_MAX_DECODE_STEPS.value
    )
    print(f'Response:\n{response}')

  if _SECOND_PROMPT.value:
    second_prompt = _SECOND_PROMPT.value
    if _ENABLE_FORMATTING.value:
      assert (
          _TRANSFORMERS_MODEL_PATH.value is not None
      ), 'Transformers model path is required for formatting.'
      messages = [{'role': 'user', 'content': second_prompt}]
      tokenizer = config.tokenizer_config.make().tx_tokenizer
      second_prompt = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True,
      )
      if tokenizer.special_tokens_map.get('bos_token', None):
        bos_token = tokenizer.special_tokens_map['bos_token']
        second_prompt = '\n' + second_prompt.removeprefix(bos_token)
      print(f'--- Formatted second prompt:\n\n{second_prompt}')
    print('\n--- Processing second prompt ---')
    print(f'Prompt:\n{second_prompt}')
    response = executor.sample_text(
        second_prompt, max_sample_step=_MAX_DECODE_STEPS.value
    )
    print(f'Response:\n{response}')

  if _ENABLE_CALIBRATION.value and _CALIBRATION_RESULT_SAVE_DIR.value:
    executor.save_calibration_results(_CALIBRATION_RESULT_SAVE_DIR.value)


if __name__ == '__main__':
  app.run(main)
