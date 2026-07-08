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
"""Calibration script for TFLite models."""

from typing import Sequence

from absl import app
from absl import flags
from litert_torch.generative.export_hf.experimental.calib import loader
from litert_torch.generative.export_hf.experimental.calib import quant_utils
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
    'Path to the transformers model directory (containing tokenizer.json).',
)

_EVAL_TASK_NAMES = flags.DEFINE_list(
    'eval_task_names',
    '',
    'Comma-separated list of eval task name(s) to run. If ALL is included,'
    ' run all tasks in eval_task_utils.EVAL_TASK_RIEGELI_PATH.',
)

_CALIBRATION_RESULT_SAVE_DIR = flags.DEFINE_string(
    'calibration_result_save_dir',
    None,
    'Path to the output calibration result directory.',
    required=True,
)

_ENABLE_FORMATTING = flags.DEFINE_bool(
    'enable_formatting',
    True,
    'Whether to enable formatting for the input prompts.',
)
_ENABLE_MIN_MAX_CALIBRATION_UPDATE = flags.DEFINE_bool(
    'enable_min_max_calibration_update',
    True,
    'Whether to use min-max tracking instead of EMA for calibration updates.',
)
_USE_PROFILER_BASED_CALIBRATION = flags.DEFINE_bool(
    'use_profiler_based_calibration',
    False,
    'Use profiler-based calibration.',
)

_EMA_SMOOTHING_FACTOR = flags.DEFINE_float(
    'ema_smoothing_factor',
    0.95,
    'Smoothing factor (alpha) for Exponential Moving Average calibration'
    ' updates.',
)

_MAX_DECODE_STEPS = flags.DEFINE_integer(
    'max_decode_steps',
    None,
    'Maximum number of decode steps.',
)

_MAX_EXAMPLES = flags.DEFINE_integer(
    'max_examples',
    None,
    'Maximum number of examples to process per task.',
)

_DATASET_DIR = flags.DEFINE_string(
    'dataset_dir',
    None,
    'Path to the dataset directory. If not specified, uses the default CNS'
    ' path.',
)

_DEFAULT_DATASET_FORMAT = 'json'

_DATASET_FORMAT = flags.DEFINE_enum(
    'dataset_format',
    _DEFAULT_DATASET_FORMAT,
    ['riegeli', 'json', 'jsonl'],
    'Format of the dataset.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not _SPM_PATH.value and not _TRANSFORMERS_MODEL_PATH.value:
    raise app.UsageError(
        'Must specify either --spm_path or --transformers_model_path.'
    )

  print('--- Configuring executor...')
  config = loader.load_models(
      max_kv_cache_size=_KV_CACHE_MAX_LEN.value,
      model_path=(_MODEL_PATH.value, _DECODE_MODEL_PATH.value),  # pyrefly: ignore[bad-argument-type]
      embedder_model_path=_EMBEDDER_MODEL_PATH.value,
      spm_path=_SPM_PATH.value,
      transformers_model_path=_TRANSFORMERS_MODEL_PATH.value,
      auxiliary_model_path=_AUXILIARY_MODEL_PATH.value,
      per_layer_embedder_model_path=_PLE_MODEL_PATH.value,
      mm_encoder_model_path=_MM_ENCODER_MODEL_PATH.value,
      mm_adapter_model_path=_MM_ADAPTER_MODEL_PATH.value,
      enable_calibration=True,
      enable_min_max_calibration_update=_ENABLE_MIN_MAX_CALIBRATION_UPDATE.value,
      ema_smoothing_factor=_EMA_SMOOTHING_FACTOR.value,
      use_profiler_based_calibration=_USE_PROFILER_BASED_CALIBRATION.value,
  )

  print('--- Initializing executor. Loading models...')
  executor = tfl_sampling_executor.Executor(config, stream_output=False)
  print('--- Models loaded. Starting calibration...')

  # Load calibration state from previous runs, if any.
  state = quant_utils.CalibrationState.load(_CALIBRATION_RESULT_SAVE_DIR.value)

  if state.task_idx > 0 or state.example_idx > 0:
    print(
        f'--- Resuming from task index {state.task_idx}, example index'
        f' {state.example_idx}'
    )
    executor.load_calibration_results(_CALIBRATION_RESULT_SAVE_DIR.value)

  ext = _DATASET_FORMAT.value
  tasks = quant_utils.get_calibration_tasks(
      _EVAL_TASK_NAMES.value, dataset_dir=_DATASET_DIR.value, ext=ext
  )
  task_items = list(tasks.items())

  for task_idx, (task_name, dataset_path) in enumerate(task_items):
    # Skip tasks that have already been completed in a previous run.
    if task_idx < state.task_idx:
      print(f'--- Skipping task: {task_name} (already completed) ---')
      continue

    print(f'\n--- Running calibration for task: {task_name} ---')
    if _DATASET_FORMAT.value == 'json':
      examples = quant_utils.read_from_json(dataset_path)
    elif _DATASET_FORMAT.value == 'jsonl':
      examples = quant_utils.read_from_jsonl(dataset_path)
    else:
      raise ValueError("Only json and jsonl formats are supported in OSS.")

    current_start_example_idx = (
        state.example_idx if task_idx == state.task_idx else 0
    )

    for i, example in enumerate(examples):
      if _MAX_EXAMPLES.value is not None and i >= _MAX_EXAMPLES.value:
        print(
            f'--- Reached max examples limit ({_MAX_EXAMPLES.value}).'
            ' Stopping task. ---'
        )
        break

      # Skip examples that have already been completed in a previous run.
      if i < current_start_example_idx:
        print(
            f'--- Skipping example {i+1}/{len(examples)} (already'
            ' completed) ---'
        )
        continue

      prompt = quant_utils.get_example_prompt(
          example,
          enable_formatting=_ENABLE_FORMATTING.value,
          tokenizer=executor.tokenizer,
      )
      print(f'Processing example {i+1}/{len(examples)}')
      result = executor.sample_text(
          prompt, max_sample_step=_MAX_DECODE_STEPS.value
      )
      print(f'Result: {result}')

      # Save state and results
      state.update(len(examples))

      if _CALIBRATION_RESULT_SAVE_DIR.value:
        executor.save_calibration_results(
            _CALIBRATION_RESULT_SAVE_DIR.value,
            extra_metadata={'task_name': task_name},
        )
        state.save(_CALIBRATION_RESULT_SAVE_DIR.value)


if __name__ == '__main__':
  app.run(main)
