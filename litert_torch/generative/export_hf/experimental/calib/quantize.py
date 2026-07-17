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
import tempfile
from typing import Any, Sequence

from absl import app
from absl import flags
from litert_torch.generative.export_hf.experimental.calib import fuse_q
from litert_torch.generative.export_hf.experimental.calib import quant_utils
from litert_torch.generative.export_hf.experimental.litertlm_bundle import litertlm_bundle as litertlm_utils
import numpy as np

from ai_edge_quantizer import algorithm_manager
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer
from ai_edge_quantizer.algorithms.uniform_quantize import naive_min_max_quantize
from ai_edge_quantizer.utils import calibration_utils
from litert_converter.tools import model_utils as mu


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
_OUTPUT_LITERTLM = _define_flag(
    flags.DEFINE_string,
    'output_litertlm',
    None,
    'Optional path to output .litertlm file.',
)
_EMBEDDER_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'embedder_model_path',
    None,
    'Optional path to unquantized embedder model for bundling.',
)
_PLE_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'ple_model_path',
    None,
    'Optional path to unquantized per-layer embedder model for bundling.',
)
_SPM_PATH = _define_flag(
    flags.DEFINE_string,
    'spm_path',
    None,
    'Optional path to SentencePiece tokenizer (.model) for bundling.',
)
_TRANSFORMERS_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'transformers_model_path',
    None,
    'Optional path to HuggingFace tokenizer directory or file for bundling.',
)

_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'model_path',
    None,
    'Path to the unquantized TFLite model.',
)
_CALIBRATION_PATH = _define_flag(
    flags.DEFINE_string,
    'calibration_path',
    None,
    'Path to the calibration results JSON file.',
)
_OUTPUT_PATH = _define_flag(
    flags.DEFINE_string,
    'output_path',
    None,
    'Path to save the quantized TFLite model.',
)
_A16W8 = _define_flag(
    flags.DEFINE_bool,
    'a16w8',
    False,
    'Whether to use 16-bit activation quantization.',
)
_ALIGN_KV_CACHE = _define_flag(
    flags.DEFINE_bool,
    'align_kv_cache',
    True,
    'Whether to align KV cache quantization parameters.',
)
_ALLOW_FLOAT_OPERATIONS = _define_flag(
    flags.DEFINE_bool,
    'allow_float_operations',
    True,
    'Whether to allow float operations (e.g. RMS Norm, residual ADD) in the'
    ' quantized model.',
)
_SKIP_MLIR_PASSES = _define_flag(
    flags.DEFINE_bool,
    'skip_mlir_passes',
    True,
    'Whether to skip post-quantization MLIR graph surgery passes.',
)
_KV_CACHE_K_NAME_PATTERN = _define_flag(
    flags.DEFINE_list,
    'kv_cache_k_name_pattern',
    ['kv_cache_k_{}', 'kv_slice_k_{}'],
    'List of patterns for KV cache K tensor names.',
)
_KV_CACHE_V_NAME_PATTERN = _define_flag(
    flags.DEFINE_list,
    'kv_cache_v_name_pattern',
    ['kv_cache_v_{}', 'kv_slice_v_{}'],
    'List of patterns for KV cache V tensor names.',
)
_AUX_MODEL_PATH = _define_flag(
    flags.DEFINE_string,
    'aux_model_path',
    None,
    'Optional. Path to the unquantized auxiliary TFLite model.',
)
_AUX_CALIBRATION_PATH = _define_flag(
    flags.DEFINE_string,
    'aux_calibration_path',
    None,
    'Optional. Path to the auxiliary calibration results JSON file.',
)
_AUX_OUTPUT_PATH = _define_flag(
    flags.DEFINE_string,
    'aux_output_path',
    None,
    'Optional. Path to save the quantized auxiliary TFLite model.',
)
_CALIBRATION_RANGE_SCALE = _define_flag(
    flags.DEFINE_float,
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


def _apply_mlir_passes(
    quantization_result: Any, skip_mlir_passes: bool = False
) -> Any:
  """Applies MLIR QDQ and quantized BMM fusion passes."""
  if skip_mlir_passes or _SKIP_MLIR_PASSES.value:
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


def quantize(
    input_litertlm: str | None = None,
    output_litertlm: str | None = None,
    calibration_path: str | None = None,
    aux_calibration_path: str | None = None,
    model_path: str | None = None,
    output_path: str | None = None,
    embedder_model_path: str | None = None,
    aux_model_path: str | None = None,
    ple_model_path: str | None = None,
    spm_path: str | None = None,
    transformers_model_path: str | None = None,
    a16w8: bool = True,
    allow_float_operations: bool = False,
    skip_mlir_passes: bool = True,
    align_kv_cache: bool = True,
    calibration_range_scale: float = 1.0,
    calibration_dir: str | None = None,
    quantization_recipe: str | None = None,
    kv_cache_k_name_pattern: list[str] | None = None,
    kv_cache_v_name_pattern: list[str] | None = None,
    overwrite: bool = True,
) -> None:
  """Applies static range quantization to TFLite / LiteRT-LM models."""
  if overwrite and output_litertlm and gfile.Exists(output_litertlm):
    gfile.Remove(output_litertlm)
  unpacked = {}
  if input_litertlm:
    unpack_dir = tempfile.mkdtemp(prefix='litertlm_quant_unpacked_')
    unpacked = litertlm_utils.unpack_litertlm(input_litertlm, unpack_dir)

  model_path = model_path or unpacked.get('tf_lite_prefill_decode')
  aux_model_path = aux_model_path or unpacked.get('tf_lite_aux')
  embedder_path = embedder_model_path or unpacked.get('tf_lite_embedder')
  ple_path = ple_model_path or unpacked.get('tf_lite_per_layer_embedder')
  spm_path = spm_path or unpacked.get('SP_Tokenizer')
  hf_path = transformers_model_path or unpacked.get('transformers_model_path')
  llm_metadata_path = unpacked.get('LlmMetadataProto') or unpacked.get(
      'LlmMetadata'
  )

  if not model_path:
    raise ValueError('Must specify model_path or input_litertlm.')

  if calibration_dir and os.path.isdir(calibration_dir):
    for fname in os.listdir(calibration_dir):
      if fname.endswith('.json'):
        full_p = os.path.join(calibration_dir, fname)
        if 'prefill_decode' in fname and not calibration_path:
          calibration_path = full_p
        elif 'aux' in fname and not aux_calibration_path:
          aux_calibration_path = full_p

  if not calibration_path:
    raise ValueError('Must provide calibration_path or valid calibration_dir.')

  print('--- Registering custom INPUT override for embeddings...')
  algorithm_manager.register_quantized_op(
      algorithm_key=algorithm_manager.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
      tfl_op_name=qtyping.TFLOperationName.INPUT,
      init_qsv_func=naive_min_max_quantize.init_qsvs,
      calibration_func=naive_min_max_quantize.min_max_calibrate,
      materialize_func=quant_utils.get_custom_materialize_input('.*embeddings'),
  )

  print(f'--- Loading calibration results from: {calibration_path} ...')
  calibration_result, _ = calibration_utils.load_calibration_results(
      calibration_path
  )
  print(f'Loaded calibration results for {len(calibration_result)} tensors.')

  if calibration_range_scale != 1.0:
    calibration_result = _scale_calibration_results(
        calibration_result, calibration_range_scale
    )

  aux_calibration_result = None
  if aux_calibration_path:
    print(
        f'--- Loading aux calibration results from: {aux_calibration_path} ...'
    )
    aux_calibration_result, _ = calibration_utils.load_calibration_results(
        aux_calibration_path
    )
    if calibration_range_scale != 1.0:
      aux_calibration_result = _scale_calibration_results(
          aux_calibration_result, calibration_range_scale
      )

  if align_kv_cache:
    print('--- Aligning KV cache parameters across models...')
    quant_utils.align_kv_cache_params(
        calibration_results=calibration_result,
        model_path=model_path,
        kv_cache_k_patterns=kv_cache_k_name_pattern
        or _KV_CACHE_K_NAME_PATTERN.value,
        kv_cache_v_patterns=kv_cache_v_name_pattern
        or _KV_CACHE_V_NAME_PATTERN.value,
        aux_calibration_results=aux_calibration_result,
        aux_model_path=aux_model_path,
    )

  print(f'--- Initializing Quantizer for model: {model_path} ...')
  q_main = quantizer.Quantizer(model_path)
  if quantization_recipe:
    q_main.load_quantization_recipe(quantization_recipe)
  else:
    q_main = quant_utils.add_main_model_quant_recipe(
        q_main,
        allow_float_operations=allow_float_operations,
        a16w8=a16w8,
    )

  print('--- Running main model quantization...')
  quantization_result = q_main.quantize(calibration_result)

  if not skip_mlir_passes:
    quantization_result = _apply_mlir_passes(
        quantization_result, skip_mlir_passes=skip_mlir_passes
    )

  output_path = output_path or (
      output_litertlm + '.intermediate.tflite'
      if output_litertlm
      else model_path + '.quantized.tflite'
  )
  output_dir = os.path.dirname(output_path)
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

  print(f'--- Exporting quantized model to: {output_path} ...')
  quantization_result.export_model(output_path, overwrite=True)

  aux_output_path = None
  if aux_model_path and aux_calibration_result:
    print(f'--- Initializing Quantizer for aux model: {aux_model_path} ...')
    q_aux = quantizer.Quantizer(aux_model_path)
    if quantization_recipe:
      q_aux.load_quantization_recipe(quantization_recipe)
    else:
      q_aux = quant_utils.add_main_model_quant_recipe(
          q_aux,
          allow_float_operations=allow_float_operations,
          a16w8=a16w8,
      )

    print('--- Running aux model quantization...')
    aux_quantization_result = q_aux.quantize(aux_calibration_result)
    aux_quantization_result = _apply_mlir_passes(
        aux_quantization_result, skip_mlir_passes=skip_mlir_passes
    )

    aux_output_path = (
        output_litertlm + '.aux_intermediate.tflite'
        if output_litertlm
        else aux_model_path + '.quantized.tflite'
    )
    aux_output_dir = os.path.dirname(aux_output_path)
    if aux_output_dir and not os.path.exists(aux_output_dir):
      os.makedirs(aux_output_dir, exist_ok=True)

    print(f'--- Exporting quantized aux model to: {aux_output_path} ...')
    aux_quantization_result.export_model(aux_output_path, overwrite=True)

  if output_litertlm:
    print(f'\n--- Packaging model components into {output_litertlm} ---')
    litertlm_utils.pack_litertlm(
        output_litertlm=output_litertlm,
        model_path=output_path,
        embedder_model_path=embedder_path,
        auxiliary_model_path=aux_output_path,
        ple_model_path=ple_path,
        spm_path=spm_path,
        transformers_model_path=hf_path,
        llm_metadata_path=llm_metadata_path,
    )
    print(f'Done packaging {output_litertlm}')

  print('--- Quantization completed successfully!')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not _MODEL_PATH.value and not _INPUT_LITERTLM.value:
    raise app.UsageError('Must specify --model_path or --input_litertlm.')

  quantize(
      input_litertlm=_INPUT_LITERTLM.value,
      output_litertlm=_OUTPUT_LITERTLM.value,
      calibration_path=_CALIBRATION_PATH.value,
      aux_calibration_path=_AUX_CALIBRATION_PATH.value,
      model_path=_MODEL_PATH.value,
      output_path=_OUTPUT_PATH.value,
      embedder_model_path=_EMBEDDER_MODEL_PATH.value,
      aux_model_path=_AUX_MODEL_PATH.value,
      ple_model_path=_PLE_MODEL_PATH.value,
      spm_path=_SPM_PATH.value,
      transformers_model_path=_TRANSFORMERS_MODEL_PATH.value,
      a16w8=_A16W8.value,
      allow_float_operations=_ALLOW_FLOAT_OPERATIONS.value,
      skip_mlir_passes=_SKIP_MLIR_PASSES.value,
      align_kv_cache=_ALIGN_KV_CACHE.value,
      calibration_range_scale=_CALIBRATION_RANGE_SCALE.value,
      kv_cache_k_name_pattern=_KV_CACHE_K_NAME_PATTERN.value,
      kv_cache_v_name_pattern=_KV_CACHE_V_NAME_PATTERN.value,
  )


if __name__ == '__main__':
  app.run(main)
