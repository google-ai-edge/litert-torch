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
"""Converter V2 for LiteRT Torch.

Exports PyTorch models to MLIR bytecode and decoupled binary weights
(params.bin and weights_metadata.json) for the LiteRT Converter V2 C++ pipeline.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import tempfile
from typing import Any, Literal, Optional
import uuid

from litert_torch import backend
from litert_torch import model
from litert_torch import progress
from litert_torch._convert import core
from litert_torch._convert import litert_converter
from litert_torch._convert import signature as signature_module
import tensorflow as tf
import torch


def _apply_tfl_converter_flags(
    converter: tf.lite.TFLiteConverter, tfl_converter_flags: dict[str, Any]
) -> None:
  """Applies TFLite converter flags to the converter."""

  def _set_converter_flag(path: list[Any]):
    if len(path) < 2:
      raise ValueError("Expecting at least two values in the path.")

    target_obj = converter
    for idx in range(len(path) - 2):
      target_obj = getattr(target_obj, path[idx])

    setattr(target_obj, path[-2], path[-1])

  def _iterate_dict_tree(flags_dict: dict[str, Any], path: list[Any]):
    for key, value in flags_dict.items():
      path.append(key)
      if isinstance(value, dict):
        _iterate_dict_tree(value, path)
      else:
        path.append(value)
        _set_converter_flag(path)
        path.pop()
      path.pop()

  _iterate_dict_tree(tfl_converter_flags, [])


def _get_param_id(p: torch.Tensor) -> Any:
  """Returns a unique fingerprint for a parameter tensor to enable deduplication."""
  if hasattr(p, "untyped_storage") and callable(p.untyped_storage):
    try:
      storage = p.untyped_storage()
      return (
          storage.data_ptr(),
          p.storage_offset(),
          tuple(p.shape),
          tuple(p.stride()),
          p.dtype,
      )
    except Exception:
      return id(p)
  return id(p)


def _tensor_to_raw_bytes(tensor: torch.Tensor) -> bytes:
  """Converts a torch tensor to raw bytes with C-contiguous layout."""
  t = tensor.contiguous().detach().cpu()
  if t.dtype == torch.bfloat16:
    return t.view(torch.uint8).numpy().tobytes()
  return t.numpy().tobytes()


def _estimate_total_param_bytes(
    signatures: list[signature_module.Signature],
) -> int:
  """Estimates exact weight payload byte count across all signatures."""
  total_bytes = 0
  seen_ids = set()
  for sig in signatures:
    if sig.module is None:
      continue
    for p in sig.module.parameters():
      pid = _get_param_id(p)
      if pid not in seen_ids:
        seen_ids.add(pid)
        total_bytes += p.numel() * p.element_size()
    for b in sig.module.buffers():
      pid = _get_param_id(b)
      if pid not in seen_ids:
        seen_ids.add(pid)
        total_bytes += b.numel() * b.element_size()
  return total_bytes


def _has_complete_intermediates(
    export_dir: str, signatures: list[signature_module.Signature]
) -> bool:
  """Checks if params.bin, weights_metadata.json, and all per-signature .mlirbc exist."""
  if not os.path.exists(os.path.join(export_dir, "params.bin")):
    return False
  if not os.path.exists(os.path.join(export_dir, "weights_metadata.json")):
    return False
  if signatures:
    for sig in signatures:
      mlirbc_path = os.path.join(export_dir, f"{sig.name}.mlirbc")
      if not os.path.exists(mlirbc_path):
        return False
    return True
  else:
    mlirbc_files = [f for f in os.listdir(export_dir) if f.endswith(".mlirbc")]
    return len(mlirbc_files) > 0


def _has_any_intermediates(
    export_dir: str, signatures: list[signature_module.Signature]
) -> bool:
  if os.path.exists(os.path.join(export_dir, "params.bin")) or os.path.exists(
      os.path.join(export_dir, "weights_metadata.json")
  ):
    return True
  for sig in signatures:
    if os.path.exists(os.path.join(export_dir, f"{sig.name}.mlirbc")):
      return True
  mlirbc_files = [f for f in os.listdir(export_dir) if f.endswith(".mlirbc")]
  return len(mlirbc_files) > 0


class ParameterRegistry:
  """Tracks and deduplicates parameters, generating metadata for export."""

  def __init__(self):
    self.unique_params: list[tuple[Any, torch.Tensor]] = []
    self.id_to_offset: dict[Any, int] = {}
    self.current_offset: int = 0
    self.metadata: dict[str, Any] = {
        "signatures": {},
        "signature_inputs": {},
        "signature_outputs": {},
    }

  def register_signature(
      self,
      name: str,
      exported_program: torch.export.ExportedProgram,  # pylint: disable=unused-argument
      lowered: backend.export.MlirLowered,
      input_names: list[str] | None = None,
      output_names: list[str] | None = None,
  ):
    """Registers a signature and its parameter mapping from lowered MLIR signature."""
    sig_metadata = []

    for i, var_sig in enumerate(lowered.input_signature):
      if not var_sig.input_spec.is_user_input:
        # Parameter or buffer constant
        param_name = var_sig.input_spec.name
        tensor = lowered.state_dict.get(param_name)
        if tensor is None:
          continue

        param_id = _get_param_id(tensor)
        if param_id not in self.id_to_offset:
          # Align offset to 64-byte boundary for SIMD / mmap alignment
          self.current_offset = (self.current_offset + 63) // 64 * 64
          offset = self.current_offset
          self.id_to_offset[param_id] = offset
          self.unique_params.append((param_id, tensor))
          nbytes = tensor.numel() * tensor.element_size()
          self.current_offset += nbytes

        sig_metadata.append({
            "arg_index": i,
            "offset": self.id_to_offset[param_id],
        })

    self.metadata["signatures"][name] = sig_metadata

    if input_names is not None:
      self.metadata["signature_inputs"][name] = input_names
    else:
      num_user_inputs = sum(
          1 for s in lowered.input_signature if s.input_spec.is_user_input
      )
      self.metadata["signature_inputs"][name] = [
          f"args_{i}" for i in range(num_user_inputs)
      ]

    if output_names is not None:
      self.metadata["signature_outputs"][name] = output_names
    else:
      self.metadata["signature_outputs"][name] = [
          f"output_{i}" for i in range(len(lowered.output_signature))
      ]

  def save(self, bin_path: str, json_path: str):
    """Saves the parameter buffer and metadata JSON with 64-byte padding."""
    current_pos = 0
    with open(bin_path, "wb") as f:
      for _, p in self.unique_params:
        aligned_pos = (current_pos + 63) // 64 * 64
        if aligned_pos > current_pos:
          f.write(b"\x00" * (aligned_pos - current_pos))
          current_pos = aligned_pos
        raw_bytes = _tensor_to_raw_bytes(p)
        f.write(raw_bytes)
        current_pos += len(raw_bytes)

    with open(json_path, "w") as f:
      json.dump(self.metadata, f, indent=2)

  def delete_in_memory_params(self):
    """Releases physical memory of tracked PyTorch tensors and clears references."""
    with torch.no_grad():
      for _, p in self.unique_params:
        if hasattr(p, "untyped_storage"):
          try:
            p.detach().untyped_storage().resize_(0)
          except Exception:  # pylint: disable=broad-exception-caught
            pass
        if hasattr(p, "data"):
          try:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
          except Exception:  # pylint: disable=broad-exception-caught
            pass
    self.unique_params.clear()
    self.id_to_offset.clear()


def export_to_dir(
    signatures: list[signature_module.Signature],
    export_dir: str,
    *,
    strict_export: Literal["auto"] | bool = False,
    delete_in_memory_params: bool = False,
) -> ParameterRegistry:
  """Exports PyTorch signatures to MLIR bytecode, params.bin, and weights_metadata.json."""
  os.makedirs(export_dir, exist_ok=True)
  registry = ParameterRegistry()

  exported_programs = []
  for sig in signatures:
    with progress.task(f"Torch Export: {sig.name}"):
      exported_program = core.export_torch_signature(
          sig, strict_export=strict_export
      )
    exported_programs.append(exported_program)

  # Apply default fx passes
  with progress.task("Run FX Passes"):
    exported_programs = list(map(core._run_convert_passes, exported_programs))

  # Lower each exported program to MLIR without inlining constants
  for sig, exported_program in zip(signatures, exported_programs):
    ir_context = backend.export_utils.create_ir_context()
    with progress.task(f"Lower to MLIR (V2): {sig.name}"):
      lowered = backend.export.exported_program_to_mlir(
          exported_program,
          ir_context=ir_context,
          inline_constants=False,
      )

    # Save MLIR bytecode
    mlirbc_path = os.path.join(export_dir, f"{sig.name}.mlirbc")
    with open(mlirbc_path, "wb") as f:
      lowered.module.operation.write_bytecode(file=f)

    # Register parameters and metadata
    output_names = litert_converter._get_output_names(exported_program, lowered)
    registry.register_signature(
        sig.name,
        exported_program,
        lowered,
        input_names=sig.flat_arg_names,
        output_names=output_names,
    )

  # Save params.bin and metadata JSON
  with progress.task("Save Params Bin & Metadata"):
    registry.save(
        os.path.join(export_dir, "params.bin"),
        os.path.join(export_dir, "weights_metadata.json"),
    )

  if delete_in_memory_params:
    with torch.no_grad():
      for sig in signatures:
        if sig.module is not None:
          for p in sig.module.parameters():
            if hasattr(p, "untyped_storage"):
              try:
                p.detach().untyped_storage().resize_(0)
              except Exception:  # pylint: disable=broad-exception-caught
                pass
            if hasattr(p, "data"):
              try:
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
              except Exception:  # pylint: disable=broad-exception-caught
                pass
          for b in sig.module.buffers():
            if hasattr(b, "untyped_storage"):
              try:
                b.detach().untyped_storage().resize_(0)
              except Exception:  # pylint: disable=broad-exception-caught
                pass
            if hasattr(b, "data"):
              try:
                b.data = torch.empty(0, dtype=b.dtype, device=b.device)
              except Exception:  # pylint: disable=broad-exception-caught
                pass
    registry.delete_in_memory_params()
    gc.collect()

  return registry


def convert_signatures_v2(
    signatures: list[signature_module.Signature],
    *,
    strict_export: Literal["auto"] | bool = False,
    export_dir: Optional[str] = None,
    output_file_path: Optional[str] = None,
    delete_in_memory_params: bool = False,
    fold_fp16_resource_casts: bool = True,
    allow_reuse_intermediates: bool = False,
    _litert_converter_flags: Optional[dict[str, Any]] = None,
) -> model.LiteRTModel:
  """Converts a list of signatures to a LiteRT model using the V2 bridge."""
  if not signatures and export_dir is None:
    raise ValueError("No signatures added to the converter.")

  core._warn_training_modules(signatures)

  if export_dir is not None:
    os.makedirs(export_dir, exist_ok=True)
    has_complete = _has_complete_intermediates(export_dir, signatures)
    has_any = _has_any_intermediates(export_dir, signatures)

    if has_any or has_complete:
      if not allow_reuse_intermediates:
        raise ValueError(
            f"Intermediate conversion files found in export_dir '{export_dir}'."
            " To overwrite them, remove existing files or use a clean"
            " directory. To reuse existing per-signature intermediates without"
            " re-tracing, set allow_reuse_intermediates=True."
        )
      if not has_complete:
        raise ValueError(
            "Incomplete per-signature intermediate files found in"
            f" '{export_dir}' even though allow_reuse_intermediates=True. Some"
            " signature .mlirbc, params.bin, or weights_metadata.json files"
            " are missing."
        )
      logging.info(
          "[CONVERTER V2] Found complete intermediate files in %s and"
          " allow_reuse_intermediates=True; skipping PyTorch export.",
          export_dir,
      )
    else:
      if not signatures:
        raise ValueError("No signatures added to the converter.")
      export_to_dir(
          signatures,
          export_dir,
          strict_export=strict_export,
          delete_in_memory_params=delete_in_memory_params,
      )

    target_path = output_file_path or os.path.join(export_dir, "model.tflite")
    tfl_converter = tf.lite.TFLiteConverter._from_mlir_bytecode(export_dir)  # pyrefly: ignore[missing-attribute] # pylint: disable=protected-access
    tfl_converter._experimental_enable_composite_direct_lowering = True
    tfl_converter._experimental_fold_fp16_resource_casts = (
        fold_fp16_resource_casts
    )
    if _litert_converter_flags:
      _apply_tfl_converter_flags(tfl_converter, dict(_litert_converter_flags))
    tfl_converter.convert(target_path)
    return model.LiteRTModel.load(target_path)
  else:
    param_bytes = _estimate_total_param_bytes(signatures)
    required_bytes = param_bytes * 2 + 50_000_000

    tmp_dir = tempfile.gettempdir()
    try:
      free_bytes = shutil.disk_usage(tmp_dir).free
    except OSError:
      free_bytes = 0

    if free_bytes >= required_bytes:
      root_dir = tmp_dir
    else:
      root_dir = os.path.expanduser("~/tmp")
      os.makedirs(root_dir, exist_ok=True)
      logging.warning(
          "[CONVERTER V2] System temp dir '%s' has insufficient free space (%d"
          " MB free vs %d MB required). Falling back to '%s' for conversion"
          " intermediates.",
          tmp_dir,
          free_bytes // (1024 * 1024),
          required_bytes // (1024 * 1024),
          root_dir,
      )

    uuid_str = str(uuid.uuid4())
    temp_dir = os.path.join(root_dir, "litert-torch-conversion", uuid_str)
    os.makedirs(temp_dir, exist_ok=True)

    target_path = output_file_path or os.path.join(temp_dir, "model.tflite")
    try:
      export_to_dir(
          signatures,
          temp_dir,
          strict_export=strict_export,
          delete_in_memory_params=delete_in_memory_params,
      )

      tfl_converter = tf.lite.TFLiteConverter._from_mlir_bytecode(temp_dir)  # pyrefly: ignore[missing-attribute] # pylint: disable=protected-access
      tfl_converter._experimental_enable_composite_direct_lowering = True
      tfl_converter._experimental_fold_fp16_resource_casts = (
          fold_fp16_resource_casts
      )
      if _litert_converter_flags:
        _apply_tfl_converter_flags(tfl_converter, dict(_litert_converter_flags))
      tfl_converter.convert(target_path)

      if output_file_path is None:
        with open(target_path, "rb") as f:
          content = f.read()
        return model.LiteRTModel(content)
      return model.LiteRTModel.load(target_path)
    finally:
      shutil.rmtree(temp_dir, ignore_errors=True)


class Converter:
  """A converter for converting PyTorch models using Converter V2 pipeline."""

  def __init__(self):
    self._signatures: list[signature_module.Signature] = []

  def signature(
      self,
      name: str,
      module: torch.nn.Module,
      sample_args=None,
      sample_kwargs=None,
      *,
      dynamic_shapes: dict[str, Any] | tuple[Any, ...] | None = None,
  ) -> Converter:
    """Functions as an alias to add_signature."""
    return self.add_signature(
        name, module, sample_args, sample_kwargs, dynamic_shapes=dynamic_shapes
    )

  def add_signature(
      self,
      name: str,
      module: torch.nn.Module,
      sample_args=None,
      sample_kwargs=None,
      *,
      dynamic_shapes: dict[str, Any] | tuple[Any, ...] | None = None,
  ) -> Converter:
    """Adds a new named signature to the converter."""
    if name in [sig.name for sig in self._signatures]:
      raise ValueError(
          f"A signature with the provided name ({name}) is already added."
      )

    if sample_args is None and sample_kwargs is None:
      raise ValueError("sample_args or sample_kwargs must be provided.")

    self._signatures.append(
        signature_module.Signature(
            name,
            module,
            sample_args,
            sample_kwargs,
            dynamic_shapes=dynamic_shapes,
        )
    )
    return self

  def convert(
      self,
      module: Optional[torch.nn.Module] = None,
      sample_args=None,
      sample_kwargs=None,
      *,
      strict_export: Literal["auto"] | bool = False,
      dynamic_shapes: dict[str, Any] | tuple[Any, ...] | None = None,
      export_dir: Optional[str] = None,
      output_file_path: Optional[str] = None,
      delete_in_memory_params: bool = False,
      fold_fp16_resource_casts: bool = True,
      allow_reuse_intermediates: bool = False,
      _litert_converter_flags: Optional[dict[str, Any]] = None,
  ) -> model.LiteRTModel:
    """Converts the PyTorch module(s) to a LiteRT model using V2 converter."""
    if module is not None:
      if sample_args is not None or sample_kwargs is not None:
        self.add_signature(
            model.DEFAULT_SIGNATURE_NAME,
            module,
            sample_args,
            sample_kwargs,
            dynamic_shapes=dynamic_shapes,
        )
      else:
        raise ValueError(
            "sample_args or sample_kwargs must be provided if a module is"
            " specified."
        )

    return convert_signatures_v2(
        self._signatures,
        strict_export=strict_export,
        export_dir=export_dir,
        output_file_path=output_file_path,
        delete_in_memory_params=delete_in_memory_params,
        fold_fp16_resource_casts=fold_fp16_resource_casts,
        allow_reuse_intermediates=allow_reuse_intermediates,
        _litert_converter_flags=_litert_converter_flags,
    )


def signature(
    name: str,
    module: torch.nn.Module,
    sample_args=None,
    sample_kwargs=None,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | None = None,
) -> Converter:
  """Initiates a Converter V2 object with the provided signature."""
  return Converter().signature(
      name, module, sample_args, sample_kwargs, dynamic_shapes=dynamic_shapes
  )


def convert(
    module: Optional[torch.nn.Module] = None,
    sample_args=None,
    sample_kwargs=None,
    *,
    strict_export: Literal["auto"] | bool = False,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | None = None,
    export_dir: Optional[str] = None,
    output_file_path: Optional[str] = None,
    delete_in_memory_params: bool = False,
    fold_fp16_resource_casts: bool = True,
    allow_reuse_intermediates: bool = False,
    _litert_converter_flags: Optional[dict[str, Any]] = None,
) -> model.LiteRTModel:
  """Converts a PyTorch model to an edge model using V2 converter."""
  return Converter().convert(
      module,
      sample_args,
      sample_kwargs,
      strict_export=strict_export,
      dynamic_shapes=dynamic_shapes,
      export_dir=export_dir,
      output_file_path=output_file_path,
      delete_in_memory_params=delete_in_memory_params,
      fold_fp16_resource_casts=fold_fp16_resource_casts,
      allow_reuse_intermediates=allow_reuse_intermediates,
      _litert_converter_flags=_litert_converter_flags,
  )
