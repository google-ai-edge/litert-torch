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
"""Utility functions for quantization and calibration."""

import dataclasses
import json
import logging
import os
import re
from typing import Any

from litert_torch.generative.export_hf.experimental.calib import tokenizer as tokenizer_lib

from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from ai_edge_quantizer.algorithms.utils import common_utils
from ai_edge_quantizer.utils import calibration_utils as cu
from ai_edge_quantizer.utils import qsv_utils as qsu
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
class gfile:
  @staticmethod
  def Exists(path): return os.path.exists(path)
  @staticmethod
  def MakeDirs(path): os.makedirs(path, exist_ok=True)
  @staticmethod
  def Open(path, mode='r'): return open(path, mode)
  @staticmethod
  def Rename(src, dst, overwrite=False):
    if overwrite and os.path.exists(dst): os.remove(dst)
    os.rename(src, dst)
  @staticmethod
  def ListDir(path): return os.listdir(path)
  @staticmethod
  def IsDirectory(path): return os.path.isdir(path)

# Following BARD250 format.
PROMPT_TEMPLATE_PREFIX = '<start_of_turn>user\n'
PROMPT_TEMPLATE_SUFFIX = '<end_of_turn>\n<start_of_turn>model\n'

BASE_SAVE_DIR = ''


def get_calibration_tasks(
    task_names: list[str], dataset_dir: str | None = None, ext: str = 'json'
) -> dict[str, str]:
  """Returns a dictionary of calibration tasks."""
  base_dir = dataset_dir or BASE_SAVE_DIR

  if dataset_dir is not None:
    return {
        task_name: f'{base_dir}/{task_name}.{ext}' for task_name in task_names
    }

  if dataset_dir is None:
    raise ValueError("dataset_dir must be specified in open-source environments.")
  return {
      task_name: os.path.join(base_dir, f'{task_name}.{ext}')
      for task_name in task_names
  }


def read_from_json(input_file: str) -> list[dict[str, Any]]:
  """Reads examples from a JSON file."""
  with open(input_file, 'r') as f:
    return json.load(f)


def read_from_jsonl(input_file: str) -> list[dict[str, Any]]:
  """Reads examples from a JSONL file."""
  examples = []
  with gfile.Open(input_file, 'r') as f:
    for line in f:
      if line.strip():
        examples.append(json.loads(line))
  return examples


def format_prompt(
    prompt: str | list[str], enable_formatting: bool
) -> str | list[str]:
  """Formats a single prompt or a list of prompts by optionally wrapping them in control tokens."""
  if not enable_formatting:
    return prompt

  if isinstance(prompt, list):
    return [PROMPT_TEMPLATE_PREFIX + p + PROMPT_TEMPLATE_SUFFIX for p in prompt]
  else:
    return PROMPT_TEMPLATE_PREFIX + prompt + PROMPT_TEMPLATE_SUFFIX


def get_example_prompt(
    example: Any,
    enable_formatting: bool = True,
    tokenizer: tokenizer_lib.Tokenizer | None = None,
) -> str | tokenizer_lib.Request:
  """Gets the prompt from the example."""
  if isinstance(example, str):
    prompt = example
    if enable_formatting:
      prompt = PROMPT_TEMPLATE_PREFIX + prompt + PROMPT_TEMPLATE_SUFFIX
    return prompt

  if isinstance(example, dict):
    if 'text' in example and isinstance(example['text'], str):
      prompt = example['text']
      if enable_formatting:
        prompt = PROMPT_TEMPLATE_PREFIX + prompt + PROMPT_TEMPLATE_SUFFIX
      return prompt

    if 'prompt' in example and isinstance(example['prompt'], str):
      prompt = example['prompt']
      if enable_formatting:
        prompt = PROMPT_TEMPLATE_PREFIX + prompt + PROMPT_TEMPLATE_SUFFIX
      return prompt

    if 'messages' in example:
      user_messages = [
          msg for msg in example['messages'] if msg.get('role') == 'user'
      ]
      if tokenizer and tokenizer.tx_tokenizer and enable_formatting:
        prompt = tokenizer.tx_tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f'\n--- Formatted prompt using chat template:\n{prompt}\n---')
        return prompt
      else:
        prompt = '\n'.join([msg.get('content', '') for msg in user_messages])
        if enable_formatting:
          prompt = PROMPT_TEMPLATE_PREFIX + prompt + PROMPT_TEMPLATE_SUFFIX
          print(f'\n--- Formatted prompt (fallback): {prompt} ---')
        return prompt

    if 'inputs' in example and example['inputs']:
      prompt = example['inputs']
      if enable_formatting:
        prompt = PROMPT_TEMPLATE_PREFIX + prompt + PROMPT_TEMPLATE_SUFFIX
        print(f'\n--- Formatted prompt: {prompt} ---')
      return prompt

    contents = []
    if enable_formatting:
      contents.append(tokenizer_lib.DataItem(text=PROMPT_TEMPLATE_PREFIX))

    for data_item in example.get('data_items', []):
      if 'text' in data_item:
        contents.append(tokenizer_lib.DataItem(text=data_item['text']))
      elif 'image_bytes' in data_item:
        contents.append(
            tokenizer_lib.DataItem(image_bytes=data_item['image_bytes'])
        )

    if enable_formatting:
      contents.append(tokenizer_lib.DataItem(text=PROMPT_TEMPLATE_SUFFIX))

    return tokenizer_lib.Request(contents=contents)

  if isinstance(example, str):
    prompt = example
    if enable_formatting:
      prompt = PROMPT_TEMPLATE_PREFIX + prompt + PROMPT_TEMPLATE_SUFFIX
    return prompt
  raise ValueError("Only dict or str examples are supported in OSS.")


@dataclasses.dataclass
class CalibrationState:
  """Calibration state."""

  task_idx: int = 0
  example_idx: int = 0
  total_calibrated_samples: int = 0

  def save(self, save_dir: str):
    """Saves the calibration state."""
    if not gfile.Exists(save_dir):
      gfile.MakeDirs(save_dir)
    state_file_path = os.path.join(save_dir, 'calibration_state.json')
    temp_state_file_path = state_file_path + '.tmp'
    with gfile.Open(temp_state_file_path, 'w') as f:
      json.dump(dataclasses.asdict(self), f)
    gfile.Rename(temp_state_file_path, state_file_path, overwrite=True)

  @classmethod
  def load(cls, save_dir: str) -> 'CalibrationState':
    """Loads the calibration state."""
    state_file_path = os.path.join(save_dir, 'calibration_state.json')
    if gfile.Exists(state_file_path):
      with gfile.Open(state_file_path, 'r') as f:
        data = json.load(f)
        return cls(**data)
    return cls()

  def update(self, num_examples_in_task: int):
    """Updates the calibration state."""
    self.example_idx += 1
    self.total_calibrated_samples += 1
    if self.example_idx >= num_examples_in_task:
      self.example_idx = 0
      self.task_idx += 1


class CalibrationResultsMerger:
  """Merges calibration results from multiple tasks."""

  def __init__(self, input_dir: str, output_dir: str):
    self._input_dir = input_dir
    self._output_dir = output_dir
    self._model_qsvs = {}
    self._metadata = {}
    self._task_files = {}
    self._merged_tasks = []

  def load_all(self):
    """Discovers all calibration task files."""
    logging.info('Discovering calibration results in %s', self._input_dir)
    task_dirs = gfile.ListDir(self._input_dir)
    logging.info(
        'Found %d task directories in %s', len(task_dirs), self._input_dir
    )

    for task_dir_name in task_dirs:
      task_path = os.path.join(self._input_dir, task_dir_name)
      if not gfile.IsDirectory(task_path):
        continue
      if os.path.normpath(task_path) == os.path.normpath(self._output_dir):
        logging.info('Skipping output directory: %s', task_path)
        continue
      self._process_task_dir(task_dir_name, task_path)

  def _process_task_dir(self, task_name: str, task_path: str):
    """Processes a task directory to find valid files."""
    logging.info('Processing task directory: %s', task_path)
    files = gfile.ListDir(task_path)
    valid_files = []
    for file_name in files:
      if (
          not file_name.endswith('.json')
          or file_name == 'calibration_state.json'
      ):
        continue
      valid_files.append(os.path.join(task_path, file_name))

    if valid_files:
      self._task_files[task_name] = valid_files

  def get_loaded_tasks(self) -> list[str]:
    """Returns the list of loaded tasks."""
    return list(self._task_files.keys())

  def merge(self):
    """Merges the calibration results."""
    for task_name, files in self._task_files.items():
      logging.info('Merging task: %s', task_name)
      if all(
          self._load_and_merge_file(file_path, os.path.basename(file_path))
          for file_path in files
      ):
        self._merged_tasks.append(task_name)

  def _load_and_merge_file(self, file_path: str, file_name: str) -> bool:
    """Loads and merges a calibration file."""
    logging.info('Loading calibration results from %s', file_path)
    try:
      new_qsvs, new_metadata = cu.load_calibration_results(file_path)
    except Exception as e:  # pylint: disable=broad-except
      print(f'--- Failed to load calibration results from {file_path}: {e}')
      return False
    if file_name not in self._model_qsvs:
      self._model_qsvs[file_name] = new_qsvs
      self._metadata[file_name] = new_metadata
    else:
      self._merge_qsvs(file_name, new_qsvs)
      self._merge_metadata(file_name, new_metadata)
    return True

  def _merge_metadata(self, file_name: str, new_metadata: dict[str, Any]):
    """Merges metadata."""
    current_metadata = self._metadata[file_name]
    # Sum num_samples_calibrated
    if 'num_samples_calibrated' in new_metadata:
      current_metadata['num_samples_calibrated'] = current_metadata.get(
          'num_samples_calibrated', 0
      ) + new_metadata.get('num_samples_calibrated', 0)

  def _merge_qsvs(self, file_name: str, new_qsvs: dict[str, Any]):
    """Merges QSVs."""
    current_qsvs = self._model_qsvs[file_name]
    for tensor_name, qsv in new_qsvs.items():
      if tensor_name not in current_qsvs:
        current_qsvs[tensor_name] = qsv
      else:
        current_qsvs[tensor_name] = qsu.min_max_update(
            current_qsvs[tensor_name], qsv
        )

  def save(self):
    """Saves the merged results."""
    if not gfile.Exists(self._output_dir):
      gfile.MakeDirs(self._output_dir)

    for file_name, qsvs in self._model_qsvs.items():
      output_path = os.path.join(self._output_dir, file_name)
      print(f'--- Saving merged calibration results to {output_path} ---')
      metadata = self._metadata.get(file_name, {})
      metadata['merged_tasks'] = self._merged_tasks
      output = {
          'model_qsvs': qsvs,
          'metadata': metadata,
      }
      with gfile.Open(output_path, 'w') as f:
        json.dump(output, f, cls=cu.NumpyEncoder)


def get_tensor_name_from_signature(
    model: qtyping.ModelT,
    signature_tensor_name: str,
    signature_key: str | None = None,
) -> list[str]:
  """Gets the actual tensor names from a signature's tensor name."""
  if not model.signatureDefs:
    return []

  tensor_names = []
  for signature_def in model.signatureDefs:
    if (
        signature_key is not None
        and signature_def.signatureKey.decode('utf-8') != signature_key
    ):
      continue

    subgraph_idx = signature_def.subgraphIndex
    subgraph = model.subgraphs[subgraph_idx]

    # Check inputs
    for signature_item in signature_def.inputs:
      if signature_item.name.decode('utf-8') == signature_tensor_name:
        tensor = subgraph.tensors[signature_item.tensorIndex]
        tensor_names.append(tfl_flatbuffer_utils.get_tensor_name(tensor))

    # Check outputs
    for signature_item in signature_def.outputs:
      if signature_item.name.decode('utf-8') == signature_tensor_name:
        tensor = subgraph.tensors[signature_item.tensorIndex]
        tensor_names.append(tfl_flatbuffer_utils.get_tensor_name(tensor))

  return tensor_names


def get_output_tensors_to_skip(
    q: quantizer.Quantizer, float_output_signature_regex: str
) -> set[str]:
  """Gets the set of output tensors to skip quantization."""
  tensors_to_skip = set()
  if not float_output_signature_regex:
    return tensors_to_skip

  model = q._float_model
  if not model.signatureDefs:
    return tensors_to_skip

  for signature_def in model.signatureDefs:
    for output in signature_def.outputs:
      sig_name = output.name.decode('utf-8')
      if re.match(float_output_signature_regex, sig_name):
        actual_names = get_tensor_name_from_signature(model, sig_name)
        tensors_to_skip.update(actual_names)

  return tensors_to_skip


def get_custom_materialize_output(tensors_to_skip: set[str]):
  """Returns a custom materialization function for the output op."""

  def _custom_materialize_output(
      op_info: qtyping.OpInfo,
      graph_info: qtyping.GraphInfo,
      tensor_name_to_qsv: dict[str, Any],
      tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
      get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature = naive_min_max_quantize.get_tensor_quant_params,
  ) -> list[qtyping.TensorTransformationParams]:
    inputs_to_ignore = []
    for opr_idx, tensor_idx in enumerate(op_info.op.inputs):
      tensor_name = tfl_flatbuffer_utils.get_tensor_name(
          graph_info.subgraph_tensors[tensor_idx]
      )
      if tensor_name in tensors_to_skip:
        inputs_to_ignore.append(opr_idx)
        print(f'--- Skipping quantization of output tensor: {tensor_name}')

    return common_utils.materialize_standard_op(
        op_info,
        graph_info,
        tensor_name_to_qsv,
        get_tensor_quant_params_fn,
        inputs_to_ignore=inputs_to_ignore,
        tensor_quant_params_cache=tensor_quant_params_cache,
    )

  return _custom_materialize_output


def get_custom_materialize_input(float_input_tensor_regex: str):
  """Returns a custom materialization function for the input op."""

  def _custom_materialize_input(
      op_info: qtyping.OpInfo,
      graph_info: qtyping.GraphInfo,
      tensor_name_to_qsv: dict[str, Any],
      tensor_quant_params_cache: common_utils.TensorQuantParamsCache,
      get_tensor_quant_params_fn: qtyping.GetTensorQuantParamsFuncSignature = naive_min_max_quantize.get_tensor_quant_params,
  ) -> list[qtyping.TensorTransformationParams]:
    outputs_to_ignore = []
    for opr_idx, tensor_idx in enumerate(op_info.op.outputs):
      tensor_name = tfl_flatbuffer_utils.get_tensor_name(
          graph_info.subgraph_tensors[tensor_idx]
      )
      if float_input_tensor_regex and re.match(
          float_input_tensor_regex, tensor_name
      ):
        outputs_to_ignore.append(opr_idx)
        print(f'--- Skipping quantization of tensor: {tensor_name}')

    return common_utils.materialize_standard_op(
        op_info,
        graph_info,
        tensor_name_to_qsv,
        get_tensor_quant_params_fn,
        outputs_to_ignore=outputs_to_ignore,
        tensor_quant_params_cache=tensor_quant_params_cache,
    )

  return _custom_materialize_input


def _add_stablehlo_composite_graph_recipe(
    q: quantizer.Quantizer,
    norm_regex: str = '^(?!.*(rotated|rotate|reshaped)).*[rR][mM][sS]_?[nN][oO][rR][mM].*',
    activation_num_bits: int = 8,
) -> quantizer.Quantizer:
  """Adds a quantization recipe for StableHLO composite graphs."""
  q.update_quantization_recipe(
      regex=norm_regex,
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      algorithm_key=quantizer.AlgorithmName.NO_QUANTIZE,
  )
  for op in [
      qtyping.TFLOperationName.INPUT,
      qtyping.TFLOperationName.OUTPUT,
      qtyping.TFLOperationName.STABLEHLO_COMPOSITE,
  ]:
    q.add_static_config(
        regex=norm_regex,
        operation_name=op,
        activation_num_bits=activation_num_bits,
        weight_num_bits=8,
    )
  return q


def add_main_model_quant_recipe(
    q: quantizer.Quantizer,
    allow_float_operations: bool = True,
    a16w8: bool = False,
    attn_bits: int = 0,
) -> quantizer.Quantizer:
  """Adds the quantization recipe for the main model."""
  activation_num_bits = 16 if a16w8 else 8
  print(f'--- Using {activation_num_bits}-bit activation for main model.')
  q.add_static_config(
      regex='.*',
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      activation_num_bits=activation_num_bits,
      weight_num_bits=8,
  )
  print('--- Forcing a16w4 per-tensor for Hadamard rotation FC ops.')
  q.add_static_config(
      regex='.*rotated.*',
      operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
      activation_num_bits=16,
      weight_num_bits=4,
      weight_granularity=qtyping.QuantGranularity.TENSORWISE,
  )
  if attn_bits in [8, 16]:
    print(f'--- Forcing {attn_bits}-bit activation for attention ops.')
    q.add_static_config(
        regex='.*dot_product_attention.*',
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=attn_bits,
        weight_num_bits=8,
    )
  if allow_float_operations:
    q.update_quantization_recipe(
        regex='.*',
        operation_name=qtyping.TFLOperationName.STABLEHLO_COMPOSITE,
        algorithm_key=quantizer.AlgorithmName.NO_QUANTIZE,
    )
    q.update_quantization_recipe(
        regex='.*post_qkv.*',
        operation_name=qtyping.TFLOperationName.ADD,
        algorithm_key=quantizer.AlgorithmName.NO_QUANTIZE,
    )
    q.update_quantization_recipe(
        regex='.*apply_skip_scale.*',
        operation_name=qtyping.TFLOperationName.MUL,
        algorithm_key=quantizer.AlgorithmName.NO_QUANTIZE,
    )
  else:
    q.add_static_config(
        regex='.*post_qkv.*',
        operation_name=qtyping.TFLOperationName.ADD,
        activation_num_bits=16,
        weight_num_bits=8,
    )
    q.add_static_config(
        regex='.*apply_skip_scale.*',
        operation_name=qtyping.TFLOperationName.MUL,
        activation_num_bits=16,
        weight_num_bits=8,
    )
    norm_activation_num_bits = 16
    q.add_static_config(
        regex='.*',
        operation_name=qtyping.TFLOperationName.STABLEHLO_COMPOSITE,
        activation_num_bits=norm_activation_num_bits,
        weight_num_bits=8,
    )
    q = _add_stablehlo_composite_graph_recipe(
        q, activation_num_bits=norm_activation_num_bits
    )

  return q


def add_vision_encoder_quant_recipe(
    q: quantizer.Quantizer,
    allow_float_operations: bool = True,
    a16w8: bool = False,
    attn_bits: int = 0,
) -> quantizer.Quantizer:
  """Adds the quantization recipe for the vision encoder model."""
  print('--- Adding vision encoder quant recipe.')
  q = add_main_model_quant_recipe(
      q,
      allow_float_operations=allow_float_operations,
      a16w8=a16w8,
      attn_bits=attn_bits,
  )
  print('--- Forcing 16-bit activation for entry and exit ops.')
  q.add_static_config(
      regex='.*(entry|exit).*',
      operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      activation_num_bits=16,
      weight_num_bits=8,
  )
  for op in [
      qtyping.TFLOperationName.SELECT,
      qtyping.TFLOperationName.SELECT_V2,
  ]:
    q.update_quantization_recipe(
        regex='.*',
        operation_name=op,
        algorithm_key=quantizer.AlgorithmName.NO_QUANTIZE,
    )

  return q


def _get_signature_data(
    alignment_utils: cu.CalibrationQsvAlignmentUtils,
    signature_tensor_names: list[str],
) -> dict[str, list[str]]:
  """Dynamically constructs signature_data by searching all signatures."""
  signature_data = {}
  for sig_key, runner in alignment_utils._signature_runners.items():
    for signature_tensor_name in signature_tensor_names:
      if (
          signature_tensor_name in runner.get_input_details()
          or signature_tensor_name in runner.get_output_details()
      ):
        if sig_key not in signature_data:
          signature_data[sig_key] = []
        signature_data[sig_key].append(signature_tensor_name)
  print(f'Aligning kv cache for {signature_data}')
  return signature_data


def _align_kv_tensors(
    signature_tensor_names: list[str],
    alignment_utils: cu.CalibrationQsvAlignmentUtils,
    calibration_results: dict[str, qtyping.QSV],
    aux_alignment_utils: cu.CalibrationQsvAlignmentUtils | None,
    aux_calibration_results: dict[str, qtyping.QSV] | None,
):
  """Aligns quantization parameters for a list of KV tensors."""
  signature_data_main = _get_signature_data(
      alignment_utils, signature_tensor_names
  )
  if not signature_data_main:
    return None, None
  min_val, max_val = alignment_utils.align_quant_stats(
      calibration_results, signature_data_main
  )

  if aux_alignment_utils and aux_calibration_results is not None:
    signature_data_aux = _get_signature_data(
        aux_alignment_utils, signature_tensor_names
    )
    if signature_data_aux:
      aux_alignment_utils.update_quant_stats(
          aux_calibration_results, signature_data_aux, min_val, max_val
      )

  return min_val, max_val


def _get_num_layers(
    alignment_utils: cu.CalibrationQsvAlignmentUtils,
    kv_cache_patterns: list[str],
) -> int:
  """Gets the number of layers by inspecting signatures."""
  max_idx = -1
  for pattern in kv_cache_patterns:
    prefix = pattern.split('{}')[0]
    regex = re.compile(f'{prefix}(\\d+)')

    for _, runner in alignment_utils._signature_runners.items():
      for name in runner.get_input_details().keys():
        match = regex.match(name)
        if match:
          max_idx = max(max_idx, int(match.group(1)))
      for name in runner.get_output_details().keys():
        match = regex.match(name)
        if match:
          max_idx = max(max_idx, int(match.group(1)))

  return max_idx + 1


def align_kv_cache_params(
    calibration_results: dict[str, qtyping.QSV],
    model_path: str,
    kv_cache_k_patterns: list[str],
    kv_cache_v_patterns: list[str],
    aux_calibration_results: dict[str, qtyping.QSV] | None = None,
    aux_model_path: str | None = None,
) -> None:
  """Aligns KV cache quantization parameters."""
  main_model_utils = cu.CalibrationQsvAlignmentUtils(model_path)
  aux_model_utils = None
  if aux_model_path and aux_calibration_results is not None:
    aux_model_utils = cu.CalibrationQsvAlignmentUtils(aux_model_path)

  num_layers = _get_num_layers(main_model_utils, kv_cache_k_patterns)
  for i in range(num_layers):
    print(f'--- Aligning KV cache for layer {i} ...')

    k_names = [pattern.format(i) for pattern in kv_cache_k_patterns]
    _align_kv_tensors(
        k_names,
        main_model_utils,
        calibration_results,
        aux_model_utils,
        aux_calibration_results,
    )

    v_names = [pattern.format(i) for pattern in kv_cache_v_patterns]
    _align_kv_tensors(
        v_names,
        main_model_utils,
        calibration_results,
        aux_model_utils,
        aux_calibration_results,
    )

  print(f'--- Aligned {num_layers} layers of KV cache.')
