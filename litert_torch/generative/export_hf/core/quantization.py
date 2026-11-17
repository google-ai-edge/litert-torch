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
"""Shared post-conversion quantization for export_hf models.

Kept separate from `export_lib` so model extensions (which `export_lib` imports
through the `model_ext` registry) can quantize their own artifacts without an
import cycle.
"""

from litert_torch import progress

from litert_quantizer import qtyping
from litert_quantizer import quantizer as quantizer_lib
from litert_quantizer import recipe as recipe_lib
from litert_quantizer import recipe_manager


def _dynamic_wi8_emb4_afp32():
  """Local recipe with 4-bit embedding tables and 8-bit fully connected weights."""
  return recipe_lib.dynamic_wi4c_afp32(
      operation_name=recipe_lib.TFLOperationName.EMBEDDING_LOOKUP,
  ) + recipe_lib.dynamic_wi8c_afp32(
      operation_name=recipe_lib.TFLOperationName.FULLY_CONNECTED,
  )


def _dynamic_fc_weight_only_conv_wi8_afp32():
  """Local recipe with 8-bit dynamic FC and 8-bit weight-only Conv2D."""
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_dynamic_config(
      regex='.*',
      operation_name=recipe_lib.TFLOperationName.FULLY_CONNECTED,
      num_bits=8,
  )
  rp_manager.add_weight_only_config(
      regex='.*',
      operation_name=recipe_lib.TFLOperationName.CONV_2D,
      num_bits=8,
  )
  return rp_manager.get_quantization_recipe()


def _dynamic_wi4b32_octav_afp32():
  """Local recipe with 4-bit blockwise-32 OCTAV FC weights and float32 activations."""
  return recipe_lib.dynamic_wi4b32_afp32() + recipe_lib.dynamic_wi4b32_afp32(
      algorithm_key=recipe_lib.AlgorithmName.OCTAV,
      operation_name=recipe_lib.TFLOperationName.FULLY_CONNECTED,
  )


def _gemma4_mixed48_b32_octav():
  """Local recipe with 4-bit blockwise-32 OCTAV FC weights and 8-bit OCTAV per-layer FC."""
  return (
      recipe_lib.dynamic_wi4b32_afp32()
      + recipe_lib.dynamic_wi4b32_afp32(
          algorithm_key=recipe_lib.AlgorithmName.OCTAV,
          operation_name=recipe_lib.TFLOperationName.FULLY_CONNECTED,
      )
      + recipe_lib.dynamic_wi8c_afp32(
          algorithm_key=recipe_lib.AlgorithmName.OCTAV,
          regex='per_layer',
          operation_name=recipe_lib.TFLOperationName.FULLY_CONNECTED,
      )
  )


def _qwen3_5_mixed48_b32():
  """Local Qwen3.5 recipe with int4 weights and int8 output projections.

  Weights are int4 blockwise-32, except the projections that write back to the
  residual stream in every decoder layer (MLP `down_proj`, GatedDeltaNet
  `out_proj` and attention `o_proj`), which are int8 channelwise. On
  Qwen3.5-0.8B that recovers part of the accuracy lost with all-int4 weights.
  The tied embedding / LM head stays int4: it is the largest weight, and int8
  there costs the most decode speed. The regex matches the op output names that
  `export_hf` emits for Qwen3.5.
  """
  return recipe_lib.dynamic_wi4b32_afp32() + recipe_lib.dynamic_wi8c_afp32(
      regex=r'Linear_(down|out|o)_proj',
      operation_name=recipe_lib.TFLOperationName.FULLY_CONNECTED,
  )


def _dynamic_moe_afp32(
    num_bits: int,
    granularity: qtyping.QuantGranularity = (
        qtyping.QuantGranularity.CHANNELWISE
    ),
):
  """Local recipe with explicit multi-axis (0, 1) MoE CUSTOM_OP quantization."""
  rp_manager = recipe_manager.RecipeManager()
  rp_manager.add_dynamic_config(
      regex='.*',
      operation_name=recipe_lib.TFLOperationName.ALL_SUPPORTED,
      num_bits=num_bits,
      granularity=granularity,
  )
  rp_manager.add_quantization_config(
      regex='.*',
      operation_name=recipe_lib.TFLOperationName.CUSTOM_OP,
      op_config=qtyping.OpQuantizationConfig(
          weight_tensor_config=qtyping.TensorQuantizationConfig(
              num_bits=num_bits,
              symmetric=True,
              granularity=granularity,
              quantized_dimensions=(0, 1),
          ),
          compute_precision=qtyping.ComputePrecision.INTEGER,
          explicit_dequantize=False,
      ),
  )
  return rp_manager.get_quantization_recipe()


def _dynamic_wi8_moe_afp32():
  """Local recipe with 8-bit channelwise dynamic quantization including MoE."""
  return _dynamic_moe_afp32(num_bits=8)


def _dynamic_wi4_moe_afp32():
  """Local recipe with 4-bit channelwise dynamic quantization including MoE."""
  return _dynamic_moe_afp32(num_bits=4)


def _dynamic_wi4b32_moe_afp32():
  """Local recipe with 4-bit blockwise-32 dynamic quantization including MoE."""
  return _dynamic_moe_afp32(
      num_bits=4,
      granularity=qtyping.QuantGranularity.BLOCKWISE_32,
  )


_LOCAL_QUANTIZATION_RECIPES = {
    'dynamic_wi8_emb4_afp32': _dynamic_wi8_emb4_afp32,
    'dynamic_fc_weight_only_conv_wi8_afp32': (
        _dynamic_fc_weight_only_conv_wi8_afp32
    ),
    'dynamic_wi4b32_octav_afp32': _dynamic_wi4b32_octav_afp32,
    'gemma4_mixed48_b32_octav': _gemma4_mixed48_b32_octav,
    'qwen3_5_mixed48_b32': _qwen3_5_mixed48_b32,
    'dynamic_wi8_moe_afp32': _dynamic_wi8_moe_afp32,
    'dynamic_wi4_moe_afp32': _dynamic_wi4_moe_afp32,
    'dynamic_wi4b32_moe_afp32': _dynamic_wi4b32_moe_afp32,
}


def maybe_quantize_model(
    model_path: str,
    quantization_recipe: str | None = None,
) -> str:
  """Quantizes a .tflite model if a recipe is provided.

  Args:
    model_path: Path of the float .tflite model.
    quantization_recipe: A `litert_quantizer` recipe name, a local recipe name
      or a path to a `.json` recipe. `None`, `''`, `'none'`, `'null'` and
      `'false'` disable quantization.

  Returns:
    The path of the quantized model, or `model_path` if quantization is
    disabled.

  Raises:
    ValueError: If the recipe is unknown or cannot be loaded.
  """
  if not quantization_recipe or str(quantization_recipe).strip().lower() in (
      'none',
      'null',
      'false',
      '',
  ):
    return model_path
  return quantize_model(model_path, quantization_recipe)


@progress.task('Quantize model')
def quantize_model(
    model_path: str,
    quantization_recipe: str,
) -> str:
  """Quantizes a .tflite model with the given recipe.

  Args:
    model_path: Path of the float .tflite model.
    quantization_recipe: A `litert_quantizer` recipe name, a local recipe name
      or a path to a `.json` recipe.

  Returns:
    The path of the quantized model, `<model>_quantized.tflite`.

  Raises:
    ValueError: If the recipe is unknown or cannot be loaded.
  """
  quantized_model_path = (
      model_path.removesuffix('.tflite').removesuffix('_quantized')
      + '_quantized.tflite'
  )
  qt = quantizer_lib.Quantizer(model_path)
  try:
    if quantization_recipe.endswith('.json'):
      recipe = quantization_recipe
    elif quantization_recipe in _LOCAL_QUANTIZATION_RECIPES:
      recipe = _LOCAL_QUANTIZATION_RECIPES[quantization_recipe]()
    else:
      recipe = recipe_lib.__dict__[quantization_recipe]()
    qt.load_quantization_recipe(recipe)
  except Exception as e:
    raise ValueError(
        f'Invalid quantization recipe: {quantization_recipe}. Please check'
        ' the recipe name.'
    ) from e
  qt.quantize().export_model(quantized_model_path, overwrite=True)
  return quantized_model_path
