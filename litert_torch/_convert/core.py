# Copyright 2024 The LiteRT Torch Authors.
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
"""LiteRT-Torch conversion core functions."""

from __future__ import annotations

import logging
from typing import Any, Literal

from litert_torch import fx_infra
from litert_torch import model
from litert_torch import progress
from litert_torch._convert import fx_passes
from litert_torch._convert import litert_converter
from litert_torch._convert import signature
from litert_torch.generative import fx_passes as generative_fx_passes
from litert_torch.quantize import quant_config as qcfg
import torch

from ai_edge_litert.aot import aot_compile as aot_compile_lib
try:
  from ai_edge_litert.aot.core import aot_types as litert_types
except ImportError:
  from ai_edge_litert.aot.core import types as litert_types


def _run_convert_passes(
    exported_program: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
  exported_program = generative_fx_passes.run_generative_passes(
      exported_program
  )

  passes = [
      fx_passes.EliminateDeadCodePass(),
      fx_passes.OptimizeLayoutTransposesPass(),
      fx_passes.CanonicalizePass(),
      fx_passes.ReduceViewRankPass(),
      fx_passes.BuildAtenCompositePass(),
      fx_passes.RemoveNonUserOutputsPass(),
      fx_passes.CastInputsBf16ToF32Pass(),
  ]
  exported_program = fx_infra.run_passes(exported_program, passes)
  return exported_program


def export_torch_signature(
    sig: signature.Signature,
    strict_export: Literal["auto"] | bool = False,
) -> torch.export.ExportedProgram:
  """Exports a single signature into an ExportedProgram."""
  if strict_export == "auto":
    try:
      exported_program = torch.export.export(
          sig.module,
          args=sig.args,
          kwargs=sig.kwargs,
          dynamic_shapes=sig.dynamic_shapes,
          strict=False,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logging.warning(
          "torch.export.export(..., strict=False) failed. Retrying with"
          " strict=True"
      )
      exported_program = torch.export.export(
          sig.module,
          args=sig.args,
          kwargs=sig.kwargs,
          dynamic_shapes=sig.dynamic_shapes,
          strict=True,
      )
  elif not strict_export:
    exported_program = torch.export.export(
        sig.module,
        args=sig.args,
        kwargs=sig.kwargs,
        dynamic_shapes=sig.dynamic_shapes,
        strict=False,
    )
  else:
    exported_program = torch.export.export(
        sig.module,
        args=sig.args,
        kwargs=sig.kwargs,
        dynamic_shapes=sig.dynamic_shapes,
        strict=True,
    )

  exported_program = fx_infra.graph_utils.reset_from_node_meta(exported_program)
  exported_program = fx_infra.safe_run_decompositions(
      exported_program,
      fx_infra.decomp.pre_convert_decomp(),
      can_skip=False,
  )
  return exported_program


def _warn_training_modules(signatures: list[signature.Signature]):
  """Warns the user if the module is in training mode (.eval not called)."""
  for sig in signatures:
    if not sig.module.training:
      continue

    message = (
        "Your model {sig_name}is converted in training mode. Please set the"
        " module in evaluation mode with `module.eval()` for better on-device"
        " performance and compatibility."
    )
    if len(signatures) == 1 and sig.name == model.DEFAULT_SIGNATURE_NAME:
      # User does not specify any signature names explicitly.
      message = message.format(sig_name="")
    else:
      message = message.format(sig_name=f'"{sig.name}" ')

    logging.warning(message)


@progress.task("LiteRT-Torch Convert")
def convert_signatures(
    signatures: list[signature.Signature],
    *,
    strict_export: Literal["auto"] | bool = False,
    quant_config: qcfg.QuantConfig | None = None,
    lightweight_conversion: bool = False,
    enable_x64: bool = True,
    runtime_constant_folding: bool | None = None,
    use_v2: bool = False,
    export_dir: str | None = None,
    output_file_path: str | None = None,
    delete_in_memory_params: bool = False,
    fold_fp16_resource_casts: bool = True,
    allow_reuse_intermediates: bool = False,
    _litert_converter_flags: dict[str, Any] | None = None,
) -> model.LiteRTModel:
  """Converts a list of `signature.Signature`s and embeds them into one `model.LiteRTModel`.

  Args:
      signatures: The list of 'signature.Signature' objects containing PyTorch
        modules to be converted.
      strict_export: Experimental `strict` arg for torch.export.export. When
        enabled, the export function will trace the program through TorchDynamo
        and ensure the soundness of the exported graph. When
        strict_export="auto", the function will try to export module in both
        modes and use the first one succeeds for downstream conversion.
      quant_config: User-defined quantization method and scheme of the model.
      lightweight_conversion: (Experimental) If True, prioritizes a faster
        conversion process and a reduced memory footprint. This is achieved by
        handling constants lazily during the conversion phase, making it ideal
        for large models that might otherwise hit memory limits. Note that
        enabling this mode may bypass certain graph optimizations, such as
        constant folding, in the resulting model.
      enable_x64: If False, downcast x64 tensors and inputs to x32.
      runtime_constant_folding: If True, uses the LiteRT runtime to fold
        constants beyond what the standard converter can resolve. If None
        (default), this is enabled automatically when `lightweight_conversion`
        is True to maintain model quality.
      use_v2: If True, uses the LiteRT Converter V2 export bridge and C++
        pipeline.
      export_dir: Optional directory to persist intermediate MLIR bytecode and
        weight files.
      output_file_path: Optional destination path for the converted .tflite
        model.
      delete_in_memory_params: If True, deletes in-memory parameter tensors to
        reduce RAM usage during conversion.
      fold_fp16_resource_casts: If True, folds fp16 resource casts during V2
        conversion.
      allow_reuse_intermediates: If True, reuses existing intermediate artifacts
        in export_dir without re-exporting from PyTorch.
      _litert_converter_flags: Optional flags to configure the LiteRT converter.

  Returns:
    The converted `model.LiteRTModel` object.
  """
  if use_v2:
    if quant_config is not None:
      raise ValueError(
          "quant_config is not currently supported with use_v2=True."
      )
    if lightweight_conversion:
      logging.warning("lightweight_conversion is ignored when use_v2=True.")
    if not enable_x64:
      logging.warning(
          "enable_x64=False is not currently handled in use_v2 mode."
      )

    from litert_torch._convert import converter_v2  # pylint: disable=g-import-not-at-top

    return converter_v2.convert_signatures_v2(
        signatures,
        strict_export=strict_export,
        export_dir=export_dir,
        output_file_path=output_file_path,
        delete_in_memory_params=delete_in_memory_params,
        fold_fp16_resource_casts=fold_fp16_resource_casts,
        allow_reuse_intermediates=allow_reuse_intermediates,
        _litert_converter_flags=_litert_converter_flags,
    )

  _warn_training_modules(signatures)

  exported_programs = []
  for sig in signatures:
    with progress.task(f"Torch Export: {sig.name}"):
      exported_program = export_torch_signature(
          sig, strict_export=strict_export
      )
    exported_programs.append(exported_program)

  # Apply default fx passes
  with progress.task("Run FX Passes"):
    exported_programs = list(map(_run_convert_passes, exported_programs))

  exporter = litert_converter.exported_programs_to_flatbuffer(
      exported_programs,
      signatures,
      enable_x64=enable_x64,
      quant_config=quant_config,
      lightweight_conversion=lightweight_conversion,
      runtime_constant_folding=runtime_constant_folding,
  )

  return model.LiteRTModel(exporter)


def aot_compile(
    compilation_configs: list[litert_types.CompilationConfig],
    cpu_model: model.LiteRTModel,
) -> litert_types.CompilationResult:
  """Compiles the given CPU model.

  Args:
    compilation_configs: The list of compilation configs to use.
    cpu_model: The CPU model to compile.

  Returns:
    The compilation result.
  """
  litert_model = litert_types.Model.create_from_bytes(cpu_model.model_content())
  return aot_compile_lib.aot_compile(
      litert_model,
      config=compilation_configs,
  )
