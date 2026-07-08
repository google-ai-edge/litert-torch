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
"""Quantization script for TFLite models."""

import dataclasses
import os
from typing import Any, Sequence

from absl import app
from absl import flags
from litert_torch.generative.export_hf.experimental.calib import fuse_q
from litert_torch.generative.export_hf.experimental.calib import quant_utils
import numpy as np

from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from ai_edge_quantizer.utils import calibration_utils
from litert_converter.tools import model_utils as mu

_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    None,
    'Path to the unquantized TFLite model.',
    required=True,
)
_CALIBRATION_PATH = flags.DEFINE_string(
    'calibration_path',
    None,
    'Path to the calibration results JSON file.',
    required=True,
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    'Path to save the quantized TFLite model.',
    required=True,
)
_A16W8 = flags.DEFINE_bool(
    'a16w8',
    False,
    'Whether to use 16-bit activation quantization.',
)
_ALIGN_KV_CACHE = flags.DEFINE_bool(
    'align_kv_cache',
    True,
    'Whether to align KV cache quantization parameters.',
)
_ALLOW_FLOAT_OPERATIONS = flags.DEFINE_bool(
    'allow_float_operations',
    True,
    'Whether to allow float operations (e.g. RMS Norm, residual ADD) in the'
    ' quantized model.',
)
_SKIP_MLIR_PASSES = flags.DEFINE_bool(
    'skip_mlir_passes',
    False,
    'Whether to skip post-quantization MLIR graph surgery passes.',
)
_KV_CACHE_K_NAME_PATTERN = flags.DEFINE_list(
    'kv_cache_k_name_pattern',
    ['kv_cache_k_{}', 'kv_slice_k_{}'],
    'List of patterns for KV cache K tensor names.',
)
_KV_CACHE_V_NAME_PATTERN = flags.DEFINE_list(
    'kv_cache_v_name_pattern',
    ['kv_cache_v_{}', 'kv_slice_v_{}'],
    'List of patterns for KV cache V tensor names.',
)
_AUX_MODEL_PATH = flags.DEFINE_string(
    'aux_model_path',
    None,
    'Optional. Path to the unquantized auxiliary TFLite model.',
)
_AUX_CALIBRATION_PATH = flags.DEFINE_string(
    'aux_calibration_path',
    None,
    'Optional. Path to the auxiliary calibration results JSON file.',
)
_AUX_OUTPUT_PATH = flags.DEFINE_string(
    'aux_output_path',
    None,
    'Optional. Path to save the quantized auxiliary TFLite model.',
)
_CALIBRATION_RANGE_SCALE = flags.DEFINE_float(
    'calibration_range_scale',
    1.0,
    'Scale factor to apply to the calibration min/max ranges before'
    ' quantization.',
)


def _scale_calibration_results(
    calibration_result: dict[str, Any], scale_factor: float
) -> dict[str, Any]:
  """Scales the calibration min/max ranges by a scale factor."""
  if scale_factor == 1.0:
    return calibration_result

  print(f'--- Scaling calibration ranges by factor {scale_factor} ...')
  scaled_result = {}
  for tensor_name, qsv in calibration_result.items():
    scaled_qsv = qsv.copy()
    if 'min' in qsv:
      min_val = qsv['min']
      scaled_min = np.where(
          min_val < 0, min_val * scale_factor, min_val * (2.0 - scale_factor)
      )
      scaled_qsv['min'] = scaled_min
    if 'max' in qsv:
      max_val = qsv['max']
      scaled_max = np.where(
          max_val > 0, max_val * scale_factor, max_val * (2.0 - scale_factor)
      )
      scaled_qsv['max'] = scaled_max
    scaled_result[tensor_name] = scaled_qsv
  return scaled_result


def _apply_mlir_passes(quantization_result: Any) -> Any:
  """Applies MLIR QDQ and quantized BMM fusion passes."""
  if _SKIP_MLIR_PASSES.value:
    return quantization_result

  print('--- Starting MLIR post-quantization passes...')
  module, ctx = mu.read_flatbuffer(
      content=bytes(quantization_result.quantized_model)
  )
  with ctx:
    mu.passes.MlirPass('builtin.module(tfl-fuse-qdq)')(module)
    fuse_q.FuseQuantizedBmmPass()(module)
    module.cleanup()
    quantization_result = dataclasses.replace(  # pytype: disable=wrong-arg-types
        quantization_result, quantized_model=mu.write_flatbuffer(module)
    )
  return quantization_result


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('--- Registering custom INPUT override for embeddings...')
  algorithm_manager.register_quantized_op(
      algorithm_key=algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      tfl_op_name=qtyping.TFLOperationName.INPUT,
      init_qsv_func=naive_min_max_quantize.init_qsvs,
      calibration_func=naive_min_max_quantize.min_max_calibrate,
      materialize_func=quant_utils.get_custom_materialize_input('.*embeddings'),
  )

  print(f'--- Loading calibration results from: {_CALIBRATION_PATH.value} ...')
  calibration_result, _ = calibration_utils.load_calibration_results(
      _CALIBRATION_PATH.value
  )
  print(f'Loaded calibration results for {len(calibration_result)} tensors.')

  if _CALIBRATION_RANGE_SCALE.value != 1.0:
    calibration_result = _scale_calibration_results(
        calibration_result, _CALIBRATION_RANGE_SCALE.value
    )

  aux_calibration_result = None
  if _AUX_CALIBRATION_PATH.value:
    print(
        '--- Loading aux calibration results from:'
        f' {_AUX_CALIBRATION_PATH.value} ...'
    )
    aux_calibration_result, _ = calibration_utils.load_calibration_results(
        _AUX_CALIBRATION_PATH.value
    )
    if _CALIBRATION_RANGE_SCALE.value != 1.0:
      aux_calibration_result = _scale_calibration_results(
          aux_calibration_result, _CALIBRATION_RANGE_SCALE.value
      )

  if _ALIGN_KV_CACHE.value:
    print('--- Aligning KV cache parameters across models...')
    quant_utils.align_kv_cache_params(
        calibration_results=calibration_result,
        model_path=_MODEL_PATH.value,
        kv_cache_k_patterns=_KV_CACHE_K_NAME_PATTERN.value,
        kv_cache_v_patterns=_KV_CACHE_V_NAME_PATTERN.value,
        aux_calibration_results=aux_calibration_result,
        aux_model_path=_AUX_MODEL_PATH.value,
    )

  # Quantize Main Model
  print(f'--- Initializing Quantizer for model: {_MODEL_PATH.value} ...')
  q = quantizer.Quantizer(_MODEL_PATH.value)

  print(
      f'--- Adding main model quantization recipe (a16w8={_A16W8.value},'
      f' allow_float={_ALLOW_FLOAT_OPERATIONS.value}) ...'
  )
  q = quant_utils.add_main_model_quant_recipe(
      q,
      allow_float_operations=_ALLOW_FLOAT_OPERATIONS.value,
      a16w8=_A16W8.value,
  )

  print('--- Running main model quantization...')
  quantization_result = q.quantize(calibration_result)

  # Run MLIR post-quantization passes
  quantization_result = _apply_mlir_passes(quantization_result)

  output_dir = os.path.dirname(_OUTPUT_PATH.value)
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

  print(f'--- Exporting quantized model to: {_OUTPUT_PATH.value} ...')
  quantization_result.export_model(_OUTPUT_PATH.value, overwrite=True)

  # Quantize Aux Model if provided
  if (
      _AUX_MODEL_PATH.value
      and _AUX_OUTPUT_PATH.value
      and aux_calibration_result is not None
  ):
    print(
        f'--- Initializing Quantizer for aux model: {_AUX_MODEL_PATH.value} ...'
    )
    q_aux = quantizer.Quantizer(_AUX_MODEL_PATH.value)

    print(
        f'--- Adding aux model quantization recipe (a16w8={_A16W8.value},'
        f' allow_float={_ALLOW_FLOAT_OPERATIONS.value}) ...'
    )
    q_aux = quant_utils.add_main_model_quant_recipe(
        q_aux,
        allow_float_operations=_ALLOW_FLOAT_OPERATIONS.value,
        a16w8=_A16W8.value,
    )

    print('--- Running aux model quantization...')
    aux_quantization_result = q_aux.quantize(aux_calibration_result)

    # Run MLIR post-quantization passes for aux model
    aux_quantization_result = _apply_mlir_passes(aux_quantization_result)

    aux_output_dir = os.path.dirname(_AUX_OUTPUT_PATH.value)
    if aux_output_dir and not os.path.exists(aux_output_dir):
      os.makedirs(aux_output_dir, exist_ok=True)

    print(f'--- Exporting quantized aux model to: {_AUX_OUTPUT_PATH.value} ...')
    aux_quantization_result.export_model(_AUX_OUTPUT_PATH.value, overwrite=True)

  print('--- Quantization completed successfully!')


if __name__ == '__main__':
  app.run(main)
