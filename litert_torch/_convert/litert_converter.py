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

import dataclasses

from litert_torch import backend
from litert_torch import model as model_lib
from litert_torch import progress
from litert_torch._convert import signature
from litert_torch.backend import inline_consts as inline_consts_lib
from litert_torch.quantize import quant_config as qcfg
from litert_torch.quantize import translate_recipe
from ai_edge_litert.mlir import ir
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


@dataclasses.dataclass
class LazyModelExporter(model_lib.ModelExporter):
  """A model exporter that exports the module lazily.

  The lazy exporter reduces the export time and memory usage when user does not
  mean to load the model in memory but just to save it to a file.
  """

  module: ir.Module | ir.Operation | None = None
  content: bytes | None = None

  @property
  def _module_op(self) -> ir.Operation | None:
    if isinstance(self.module, ir.Module):
      return self.module.operation
    return self.module

  def to_file(self, path: str):
    """Exports the module to a flatbuffer file."""
    path = str(path)

    if self.content is not None:
      with open(path, "wb") as f:
        f.write(self.content)
      return

    with progress.task(f"Write Model to {path}"):
      try:
        # TODO b/478909085 - Remove the try-except once converter_api_ext is
        # stable in OSS.
        converter_api_ext.export_flatbuffer_to_file(self._module_op, path)
      except TypeError:
        converter_api_ext.export_flatbuffer_to_file(self.module, path)

  def to_bytes(self) -> bytes:
    """Returns the flatbuffer bytes of the module."""
    if self.content is not None:
      return self.content

    with progress.task("Write Model to Bytes"):
      try:
        # TODO b/478909085 - Remove the try-except once converter_api_ext is
        # stable in OSS.
        self.content = converter_api_ext.export_flatbuffer_to_bytes(
            self._module_op
        )
      except TypeError:
        self.content = converter_api_ext.export_flatbuffer_to_bytes(self.module)

    self.module = None
    return self.content


def exported_programs_to_flatbuffer(
    exported_programs: list[torch.export.ExportedProgram],
    signatures: list[signature.Signature],
    *,
    quant_config: qcfg.QuantConfig | None = None,
    lightweight_conversion: bool = False,
) -> LazyModelExporter:
  """Convert ExportedPrograms to a LiteRT model."""
  if not exported_programs:
    raise ValueError("The number of exported programs must be greater than 0.")
  if len(exported_programs) != len(signatures):
    raise ValueError(
        "The number of exported programs must match the number of signatures."
    )

  ir_context = backend.export_utils.create_ir_context()
  cross_program_inline_consts_ctx = inline_consts_lib.InlineConstsContext(
      enable_resource_constants=lightweight_conversion,
  )

  lowered_programs = []
  for exported_program, sig in zip(exported_programs, signatures):
    # Convert ExportedProgram to Mlir Module.
    with progress.task(f"Lower to MLIR: {sig.name}"):
      lowered = backend.export.exported_program_to_mlir(
          exported_program,
          ir_context=ir_context,
          lowering_context_plugins=[cross_program_inline_consts_ctx],
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
  with progress.task("Merge MLIR Modules"):
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
  with ir_context, progress.task("Run LiteRT Converter Passes"):
    pass_manager = passmanager.PassManager()
    converter_api_ext.run_convert_to_tfl_passes(
        merged_module, pass_manager, config
    )

  # Creates the lazy flatbuffer exporter.
  exporter = LazyModelExporter(module=merged_module)

  # Quantize the model if needed.
  if translated_recipe:
    with progress.task("Quantize Model"):
      model_bytes = translate_recipe.quantize_model(
          exporter.to_bytes(), translated_recipe
      )
      exporter = LazyModelExporter(content=model_bytes)

  return exporter
