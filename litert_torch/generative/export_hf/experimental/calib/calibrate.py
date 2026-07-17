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
from litert_torch.generative.export_hf.experimental.litertlm_bundle import litertlm_bundle as litertlm_utils


def _define_flag(define_fn, name, *args, **kwargs):
  try:
    return define_fn(name, *args, **kwargs)
  except flags.DuplicateFlagError:
    return flags.FLAGS[name]


_INPUT_LITERTLM = _define_flag(
    flags.DEFINE_string,
    'input_litertlm',
    None,
    'Optional path to input .litertlm file.',
)

_KV_CACHE_MAX_LEN = _define_flag(
    flags.DEFINE_integer,
    'kv_cache_max_len',
    1280,
    'The maximum size of KV cache buffer, including both prefill and decode.',
)

_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'model_path',
    None,
    'Path to the model.',
)
_DECODE_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'decode_model_path',
    None,
    'Optional. Path to the decode model.',
)
_EMBEDDER_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'embedder_model_path',
    None,
    'Path to the embedder model.',
)
_AUXILIARY_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'auxiliary_model_path',
    None,
    'Path to the auxiliary model.',
)
_PLE_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'ple_model_path',
    None,
    'Path to the per layer embedder model.',
)
_MM_ENCODER_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'mm_encoder_model_path',
    None,
    'Path to the MM encoder model.',
)
_MM_ADAPTER_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'mm_adapter_model_path',
    None,
    'Path to the MM adapter model.',
)
_SPM_PATH = _define_flag(
    flags.DEFINE_string,
    'spm_path',
    None,
    'Path to the SPM.',
)
_TRANSFORMERS_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'transformers_model_path',
    None,
    'Path to the transformers model directory (containing tokenizer.json).',
)

_EVAL_TASK_NAMES = _define_flag(
    flags.DEFINE_list,
    'eval_task_names',
    '',
    'Comma-separated list of eval task name(s) to run. If ALL is included,'
    ' run all tasks in eval_task_utils.EVAL_TASK_RIEGELI_PATH.',
)

_CALIBRATION_RESULT_SAVE_DIR = _define_flag(
    flags.DEFINE_string,
    'calibration_result_save_dir',
    None,
    'Path to the output calibration result directory.',
    required=True,
)

_ENABLE_FORMATTING = _define_flag(
    flags.DEFINE_bool,
    'enable_formatting',
    True,
    'Whether to enable formatting for the input prompts.',
)
_ENABLE_MIN_MAX_CALIBRATION_UPDATE = _define_flag(
    flags.DEFINE_bool,
    'enable_min_max_calibration_update',
    True,
    'Whether to use min-max tracking instead of EMA for calibration updates.',
)
_USE_PROFILER_BASED_CALIBRATION = _define_flag(
    flags.DEFINE_bool,
    'use_profiler_based_calibration',
    False,
    'Use profiler-based calibration.',
)

_EMA_SMOOTHING_FACTOR = _define_flag(
    flags.DEFINE_float,
    'ema_smoothing_factor',
    0.95,
    'Smoothing factor (alpha) for Exponential Moving Average calibration'
    ' updates.',
)

_MAX_DECODE_STEPS = _define_flag(
    flags.DEFINE_integer,
    'max_decode_steps',
    None,
    'Maximum number of decode steps.',
)

_MAX_EXAMPLES = _define_flag(
    flags.DEFINE_integer,
    'max_examples',
    None,
    'Maximum number of examples to process per task.',
)

_DATASET_DIR = _define_flag(
    flags.DEFINE_string,
    'dataset_dir',
    None,
    'Path to the dataset directory. If not specified, uses the default CNS'
    ' path.',
)

_DEFAULT_DATASET_FORMAT = 'json'

_DATASET_FORMAT = _define_flag(
    flags.DEFINE_enum,
    'dataset_format',
    _DEFAULT_DATASET_FORMAT,
    ['riegeli', 'json', 'jsonl'],
    'Format of the dataset.',
)


def calibrate(
    input_litertlm: str | None = None,
    dataset_dir: str | None = None,
    calibration_result_save_dir: str | None = None,
    dataset_format: str = 'riegeli',
    eval_task_names: list[str] | str = 'ALL',
    max_decode_steps: int = 32,
    use_profiler_based_calibration: bool = True,
    enable_min_max_calibration_update: bool = True,
    max_examples: int | None = None,
    spm_path: str | None = None,
    transformers_model_path: str | None = None,
    model_path: str | None = None,
    decode_model_path: str | None = None,
    embedder_model_path: str | None = None,
    auxiliary_model_path: str | None = None,
    ple_model_path: str | None = None,
    mm_encoder_model_path: str | None = None,
    mm_adapter_model_path: str | None = None,
    enable_formatting: bool = True,
    ema_smoothing_factor: float = 0.1,
    kv_cache_max_len: int = 1280,
) -> None:
  """Runs profiler-based calibration over prompt datasets."""
  if not spm_path and not transformers_model_path and not input_litertlm:
    raise ValueError(
        'Must specify either spm_path, transformers_model_path, or'
        ' input_litertlm.'
    )

  print('--- Configuring executor...')
  if input_litertlm:
    print(f'--- Inspecting input .litertlm bundle: {input_litertlm}')
    print(litertlm_utils.peek_litertlm(input_litertlm))

  config = loader.load_models(
      max_kv_cache_size=kv_cache_max_len,
      model_path=(model_path, decode_model_path),  # pyrefly: ignore[bad-argument-type]
      embedder_model_path=embedder_model_path,
      spm_path=spm_path,
      transformers_model_path=transformers_model_path,
      auxiliary_model_path=auxiliary_model_path,
      per_layer_embedder_model_path=ple_model_path,
      mm_encoder_model_path=mm_encoder_model_path,
      mm_adapter_model_path=mm_adapter_model_path,
      enable_calibration=True,
      enable_min_max_calibration_update=enable_min_max_calibration_update,
      ema_smoothing_factor=ema_smoothing_factor,
      use_profiler_based_calibration=use_profiler_based_calibration,
      input_litertlm=input_litertlm,
  )

  print('--- Initializing executor. Loading models...')
  executor = tfl_sampling_executor.Executor(config, stream_output=False)
  print('--- Models loaded. Starting calibration...')

  # Load calibration state from previous runs, if any.
  state = quant_utils.CalibrationState.load(calibration_result_save_dir)

  if state.task_idx > 0 or state.example_idx > 0:
    print(
        f'--- Resuming from task index {state.task_idx}, example index'
        f' {state.example_idx}'
    )
    executor.load_calibration_results(calibration_result_save_dir)

  ext = dataset_format
  task_names_list = (
      [eval_task_names] if isinstance(eval_task_names, str) else eval_task_names
  )
  tasks = quant_utils.get_calibration_tasks(
      task_names_list, dataset_dir=dataset_dir, ext=ext
  )
  task_items = list(tasks.items())

  for task_idx, (task_name, dataset_path) in enumerate(task_items):
    if task_idx < state.task_idx:
      print(f'--- Skipping task: {task_name} (already completed) ---')
      continue

    print(f'\n--- Running calibration for task: {task_name} ---')
    if dataset_format == 'json':
      examples = quant_utils.read_from_json(dataset_path)
    elif dataset_format == 'jsonl':
      examples = quant_utils.read_from_jsonl(dataset_path)
    else:
      raise ValueError("Only json and jsonl formats are supported in OSS.")

    current_start_example_idx = (
        state.example_idx if task_idx == state.task_idx else 0
    )

    for i, example in enumerate(examples):
      if max_examples is not None and i >= max_examples:
        print(
            f'--- Reached max examples limit ({max_examples}).'
            ' Stopping task. ---'
        )
        break

      if i < current_start_example_idx:
        print(
            f'--- Skipping example {i+1}/{len(examples)} (already'
            ' completed) ---'
        )
        continue

      prompt = quant_utils.get_example_prompt(
          example,
          enable_formatting=enable_formatting,
          tokenizer=executor.tokenizer,
      )
      print(f'Processing example {i+1}/{len(examples)}')
      result = executor.sample_text(prompt, max_sample_step=max_decode_steps)
      print(f'Result: {result}')

      state.update(len(examples))

      if calibration_result_save_dir:
        executor.save_calibration_results(
            calibration_result_save_dir,
            extra_metadata={'task_name': task_name},
        )
        state.save(calibration_result_save_dir)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if (
      not _SPM_PATH.value
      and not _TRANSFORMERS_MODEL_PATH.value
      and not _INPUT_LITERTLM.value
  ):
    raise app.UsageError(
        'Must specify either --spm_path, --transformers_model_path, or'
        ' --input_litertlm.'
    )

  calibrate(
      input_litertlm=_INPUT_LITERTLM.value,
      dataset_dir=_DATASET_DIR.value,
      calibration_result_save_dir=_CALIBRATION_RESULT_SAVE_DIR.value,
      dataset_format=_DATASET_FORMAT.value,
      eval_task_names=_EVAL_TASK_NAMES.value,
      max_decode_steps=_MAX_DECODE_STEPS.value,
      use_profiler_based_calibration=_USE_PROFILER_BASED_CALIBRATION.value,
      enable_min_max_calibration_update=_ENABLE_MIN_MAX_CALIBRATION_UPDATE.value,
      max_examples=_MAX_EXAMPLES.value,
      spm_path=_SPM_PATH.value,
      transformers_model_path=_TRANSFORMERS_MODEL_PATH.value,
      model_path=_MODEL_PATH.value,
      decode_model_path=_DECODE_MODEL_PATH.value,
      embedder_model_path=_EMBEDDER_MODEL_PATH.value,
      auxiliary_model_path=_AUXILIARY_MODEL_PATH.value,
      ple_model_path=_PLE_MODEL_PATH.value,
      mm_encoder_model_path=_MM_ENCODER_MODEL_PATH.value,
      mm_adapter_model_path=_MM_ADAPTER_MODEL_PATH.value,
      enable_formatting=_ENABLE_FORMATTING.value,
      ema_smoothing_factor=_EMA_SMOOTHING_FACTOR.value,
      kv_cache_max_len=_KV_CACHE_MAX_LEN.value,
  )


if __name__ == '__main__':
  app.run(main)
