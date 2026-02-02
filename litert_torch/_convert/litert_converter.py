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
"""LiteRT converter integrations: MLIR to flatbuffer conversions."""

from litert_torch import backend
from litert_torch._convert import signature
from litert_torch.quantize import quant_config as qcfg
from litert_torch.quantize import translate_recipe
from ai_edge_litert.mlir import passmanager
import torch

from ai_edge_litert.mlir._mlir_libs import converter_api_ext


def _get_output_names(
    exported_program: torch.export.ExportedProgram,
    lowered: backend.export.MlirLowered,
) -> list[str]:
  """Get the output names from the exported program."""
  spec = exported_program.call_spec.out_spec

  # Default output names if the out_spec is not available.
  if not spec or not spec.context:
    flat_names = []
    for i in range(len(lowered.output_signature)):
      flat_names.append(f"output_{i}")
    return flat_names

  flat_names = backend.export_utils.flat_dict_names(
      spec.children_specs, spec.context
  )
  return flat_names


def exported_programs_to_flatbuffer(
    exported_programs: list[torch.export.ExportedProgram],
    signatures: list[signature.Signature],
    *,
    quant_config: qcfg.QuantConfig | None = None,
):
  """Converts a ExportedPrograms to a LiteRT model.

  Args:
    exported_programs: A list of ExportedProgram.
    signatures: A list of Signature.
    quant_config: A QuantConfig.

  Returns:
    A LiteRT model.
  """
  if not exported_programs:
    raise ValueError("The number of exported programs must be greater than 0.")
  if len(exported_programs) != len(signatures):
    raise ValueError(
        "The number of exported programs must match the number of signatures."
    )

  ir_context = backend.export_utils.create_ir_context()

  lowered_programs = []
  for exported_program, sig in zip(exported_programs, signatures):
    # Convert ExportedProgram to Mlir Module.
    lowered = backend.export.exported_program_to_mlir(
        exported_program,
        ir_context=ir_context,
    )

    # Set signature.
    sig_name = sig.name
    input_names = sig.flat_arg_names
    output_names = _get_output_names(exported_program, lowered)
    converter_api_ext.set_signature(
        lowered.module.operation,
        signature_name=sig_name,
        input_names=input_names,
        output_names=output_names,
    )
    lowered_programs.append(lowered)

  # Merge all lowered modules into one module.
  merged_module = converter_api_ext.merge_modules(
      [lowered.module for lowered in lowered_programs]
  )

  # Prepare ai-edge-quantizer recipe.
  translated_recipe = None
  if (
      quant_config is not None
      and quant_config._quantizer_mode
      == qcfg.QuantConfig._QuantizerMode.AI_EDGE_QUANTIZER
  ):
    translated_recipe = translate_recipe.translate_to_ai_edge_recipe(
        quant_config.generative_recipe
    )

  # Prepare LiteRT converter config.
  config = converter_api_ext.ConvertToTFLConfig()
  config.model_origin_framework = "PYTORCH"
  if quant_config is not None:
    quantizer_mode = quant_config._quantizer_mode
    if quantizer_mode == qcfg.QuantConfig._QuantizerMode.PT2E_DYNAMIC:
      config.qdq_conversion_mode = "DYNAMIC"
    elif quantizer_mode == qcfg.QuantConfig._QuantizerMode.PT2E_STATIC:
      config.qdq_conversion_mode = "STATIC"
  config.unsafe_fuse_dynamic_shaped_broadcast = True
  # litert-torch handles inf clamping before the MLIR conversion.
  config.canonicalizing_inf_as_min_max_float = False

  # Run LiteRT converter passes.
  with ir_context:
    pass_manager = passmanager.PassManager()
    converter_api_ext.run_convert_to_tfl_passes(
        merged_module, pass_manager, config
    )

  # Convert module to flatbuffer.
  tflite_model = converter_api_ext.export_flatbuffer_to_bytes(merged_module)

  # Quantize the model if needed.
  if translated_recipe:
    tflite_model = translate_recipe.quantize_model(
        tflite_model, translated_recipe
    )

  return tflite_model
