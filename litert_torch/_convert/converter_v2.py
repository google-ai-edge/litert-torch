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

Experimental: this module is under active development and is subject to change
without notice, including breaking changes to its functions, arguments, and
intermediate file formats.
"""

from __future__ import annotations

from collections.abc import Callable
import gc
import json
import logging
import os
import shutil
import tempfile
import time
from typing import Any, Literal, Optional
import uuid

from litert_torch import backend
from litert_torch import fx_infra
from litert_torch import model
from litert_torch import progress
from litert_torch._convert import core
from litert_torch._convert import litert_converter
from litert_torch._convert import signature as signature_module
import torch

from tensorflow.compiler.mlir.lite import converter_flags_pb2
from tensorflow.lite.python import convert as tfl_convert


def _apply_tfl_converter_flags(
    flags: converter_flags_pb2.ConverterFlags,
    tfl_converter_flags: dict[str, Any],
) -> None:
  """Applies TFLite converter flags to ConverterFlags proto."""

  def _normalize_flag_name(name: str) -> str:
    if name.startswith("_experimental_"):
      name = name[len("_experimental_") :]
    if name == "unfold_batch_matmul":
      return "unfold_batchmatmul"
    return name

  def _set_converter_flag(path: list[Any]):
    if len(path) < 2:
      raise ValueError("Expecting at least two values in the path.")

    target_obj = flags
    for idx in range(len(path) - 2):
      target_obj = getattr(target_obj, _normalize_flag_name(path[idx]))

    attr_name = _normalize_flag_name(path[-2])
    val = path[-1]
    if attr_name == "model_origin_framework" and isinstance(val, str):
      val = converter_flags_pb2.ConverterFlags.ModelOriginFramework.Value(val)
    if (
        hasattr(target_obj, "debug_options")
        and hasattr(target_obj.debug_options, attr_name)
        and not hasattr(target_obj, attr_name)
    ):
      setattr(target_obj.debug_options, attr_name, val)
    else:
      setattr(target_obj, attr_name, val)

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


def _get_param_id(
    p: torch.Tensor, origin: torch.Tensor | None = None
) -> Any:
  """Returns a unique fingerprint for a parameter tensor to enable deduplication.

  Args:
    p: The tensor to fingerprint.
    origin: For meta tensors, the module parameter/buffer `p` was exported
      from. Meta tensors have no storage, so identity of the originating module
      tensor is the only reliable way to tell shared weights (same object
      reached via different signatures) from distinct weights that happen to
      share a relative FQN. Falls back to `p` itself.
  """
  if p.device.type == "meta":
    anchor = origin if origin is not None else p
    return ("meta", id(anchor), tuple(p.shape), p.dtype)
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
    return t.reshape(-1).view(torch.uint8).numpy().tobytes()
  return t.numpy().tobytes()


def _fallback_tensor_or_raise(
    p: torch.Tensor, param_name: str, cause: Exception | None = None
) -> torch.Tensor:
  """Returns `p` for a weight missing from weights_loader, or raises if meta."""
  if p.device.type != "meta":
    return p
  if "lifted_tensor" in param_name:
    msg = (
        f"'{param_name}' is a lifted constant (a tensor created inside"
        " forward()) on the meta device and was not found in weights_loader."
        " Its values cannot be recovered; provide it via weights_loader or"
        " create it on a real device."
    )
  else:
    msg = f"Parameter '{param_name}' not found in weights_loader."
  raise KeyError(msg) from cause


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
  meta_path = os.path.join(export_dir, "weights_metadata.json")
  if not os.path.exists(meta_path):
    return False
  try:
    with open(meta_path, "r") as f:
      meta = json.load(f)
    recorded_sigs = set(meta.get("signatures", {}).keys())
  except (OSError, ValueError, AttributeError):
    return False
  if not recorded_sigs:
    return False
  if signatures and recorded_sigs != {sig.name for sig in signatures}:
    return False
  return all(
      os.path.exists(os.path.join(export_dir, f"{sig_name}.mlirbc"))
      for sig_name in recorded_sigs
  )


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
    self.unique_params: list[tuple[Any, torch.Tensor, str]] = []
    self.id_to_offset: dict[Any, int] = {}
    self.current_offset: int = 0
    self.timings: dict[str, float] = {}
    self.metadata: dict[str, Any] = {
        "signatures": {},
        "signature_inputs": {},
        "signature_outputs": {},
    }
    # Signature that first registered each unique param (for error messages).
    self._param_signature: dict[Any, str] = {}
    # Keeps id()-anchored origin tensors alive so their ids cannot be reused.
    self._origin_refs: list[torch.Tensor] = []

  def register_signature(
      self,
      name: str,
      lowered: backend.export.MlirLowered,
      input_names: list[str] | None = None,
      output_names: list[str] | None = None,
      module: torch.nn.Module | None = None,
  ):
    """Registers a signature and its parameter mapping from lowered MLIR signature.

    Args:
      name: Signature name.
      lowered: The lowered MLIR signature.
      input_names: Optional user input names.
      output_names: Optional output names.
      module: The signature's root module. Used to anchor meta tensors to their
        originating parameters/buffers for deduplication.
    """
    sig_metadata = []
    module_tensors: dict[str, torch.Tensor] = {}
    if module is not None:
      module_tensors.update(module.named_parameters(remove_duplicate=False))
      module_tensors.update(module.named_buffers(remove_duplicate=False))

    for i, var_sig in enumerate(lowered.input_signature):
      if not var_sig.input_spec.is_user_input:
        # Parameter or buffer constant
        param_name = var_sig.input_spec.name
        tensor = lowered.state_dict.get(param_name)
        if tensor is None:
          raise ValueError(
              f"Signature '{name}': non-user input '{param_name}' (arg"
              f" {i}) has no backing tensor in the lowered state_dict; its"
              " parameter mapping cannot be recorded."
          )

        origin = None
        if tensor.device.type == "meta":
          origin = module_tensors.get(param_name)
          if origin is not None and (
              tuple(origin.shape) != tuple(tensor.shape)
              or origin.dtype != tensor.dtype
          ):
            origin = None
          if origin is not None:
            self._origin_refs.append(origin)
        param_id = _get_param_id(tensor, origin)
        if param_id not in self.id_to_offset:
          # Align offset to 64-byte boundary for SIMD / mmap alignment
          self.current_offset = (self.current_offset + 63) // 64 * 64
          offset = self.current_offset
          self.id_to_offset[param_id] = offset
          self.unique_params.append((param_id, tensor, param_name))
          self._param_signature[param_id] = name
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

  def save(
      self,
      bin_path: str,
      json_path: str,
      weights_loader: Optional[
          Callable[[str], torch.Tensor] | dict[str, torch.Tensor]
      ] = None,
  ):
    """Saves the parameter buffer and metadata JSON with 64-byte padding."""
    current_pos = 0
    # Maps each weights_loader key to the unique param it was used for.
    loader_key_owner: dict[str, Any] = {}
    with open(bin_path, "wb") as f:
      for param_id, p, param_name in self.unique_params:
        aligned_pos = (current_pos + 63) // 64 * 64
        expected_offset = self.id_to_offset[param_id]
        if aligned_pos != expected_offset:
          raise RuntimeError(
              f"params.bin offset mismatch for '{param_name}': writing at"
              f" {aligned_pos} but weights_metadata.json records"
              f" {expected_offset}."
          )
        if aligned_pos > current_pos:
          f.write(b"\x00" * (aligned_pos - current_pos))
          current_pos = aligned_pos

        if weights_loader is not None:
          actual_tensor = None
          loader_key = None
          if callable(weights_loader):
            try:
              actual_tensor = weights_loader(param_name)
              loader_key = param_name
            except (KeyError, TypeError):
              stripped = param_name.removeprefix("module.")
              try:
                actual_tensor = weights_loader(stripped)
                loader_key = stripped
              except (KeyError, TypeError) as e:
                actual_tensor = _fallback_tensor_or_raise(p, param_name, e)
          elif isinstance(weights_loader, dict):
            key = (
                param_name
                if param_name in weights_loader
                else param_name.removeprefix("module.")
            )
            if key in weights_loader:
              actual_tensor = weights_loader[key]
              loader_key = key
            else:
              actual_tensor = _fallback_tensor_or_raise(p, param_name)
          else:
            raise TypeError(
                f"Unsupported weights_loader type: {type(weights_loader)}"
            )
          if loader_key is not None:
            owner = loader_key_owner.setdefault(loader_key, param_id)
            if owner != param_id:
              raise ValueError(
                  f"weights_loader key '{loader_key}' is ambiguous: it"
                  " resolves distinct tensors in signatures"
                  f" '{self._param_signature.get(owner)}' and"
                  f" '{self._param_signature.get(param_id)}'. weights_loader"
                  " keys are FQNs relative to each signature's module; export"
                  " each signature from a module that holds the full model"
                  " (e.g. a thin wrapper that calls model.encoder) so FQNs"
                  " are unique."
              )

          if tuple(actual_tensor.shape) != tuple(p.shape):
            raise ValueError(
                f"weights_loader returned shape {tuple(actual_tensor.shape)}"
                f" for '{param_name}', but the exported program expects"
                f" {tuple(p.shape)}."
            )
          if actual_tensor.dtype != p.dtype:
            actual_tensor = actual_tensor.to(p.dtype)
          raw_bytes = _tensor_to_raw_bytes(actual_tensor)
          del actual_tensor
        else:
          raw_bytes = _tensor_to_raw_bytes(p)

        expected_nbytes = p.numel() * p.element_size()
        if len(raw_bytes) != expected_nbytes:
          raise RuntimeError(
              f"params.bin size mismatch for '{param_name}': got"
              f" {len(raw_bytes)} bytes but {expected_nbytes} were reserved"
              " during registration."
          )
        f.write(raw_bytes)
        current_pos += len(raw_bytes)

    with open(json_path, "w") as f:
      json.dump(self.metadata, f, indent=2)

  def delete_in_memory_params(self):
    """Releases physical memory of tracked PyTorch tensors and clears references."""
    with torch.no_grad():
      for _, p, _ in self.unique_params:
        if hasattr(p, "device") and p.device.type == "meta":
          continue
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


def _delete_signature_params(
    signatures: list[signature_module.Signature],
) -> None:
  """Releases PyTorch tensor storage associated with signatures to free RAM."""
  with torch.no_grad():
    for sig in signatures:
      if sig.module is not None:
        for p in sig.module.parameters():
          if hasattr(p, "device") and p.device.type == "meta":
            continue
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
          if hasattr(b, "device") and b.device.type == "meta":
            continue
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
  gc.collect()


def _close_weights_loader(weights_loader: Any) -> None:
  """Closes the weights loader if it exposes a close() method."""
  if weights_loader is None or not hasattr(weights_loader, "close"):
    return
  try:
    weights_loader.close()
  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception("[CONVERTER V2] Failed to close weights_loader.")


def _trim_memory():
  """Triggers Python GC and returns freed glibc heap memory to OS."""
  gc.collect()
  try:
    import ctypes  # pylint: disable=g-import-not-at-top

    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def export_to_dir(
    signatures: list[signature_module.Signature],
    export_dir: str,
    *,
    strict_export: Literal["auto"] | bool = False,
    delete_in_memory_params: bool = False,
    weights_loader: Optional[
        Callable[[str], torch.Tensor] | dict[str, torch.Tensor]
    ] = None,
) -> ParameterRegistry:
  """Exports PyTorch signatures to MLIR bytecode, params.bin, and weights_metadata.json."""
  os.makedirs(export_dir, exist_ok=True)
  registry = ParameterRegistry()

  t_torch_start = time.perf_counter()
  exported_programs = []
  for sig in signatures:
    with progress.task(f"Torch Export: {sig.name}"):
      exported_program = core.export_torch_signature(
          sig, strict_export=strict_export, run_decompositions=False
      )
    exported_programs.append(exported_program)
  t_torch_export = time.perf_counter() - t_torch_start

  t_python_rest_start = time.perf_counter()
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
        lowered,
        input_names=sig.flat_arg_names,
        output_names=output_names,
        module=sig.module,
    )

  # Save params.bin and metadata JSON
  with progress.task("Save Params Bin & Metadata"):
    registry.save(
        os.path.join(export_dir, "params.bin"),
        os.path.join(export_dir, "weights_metadata.json"),
        weights_loader=weights_loader,
    )

  if delete_in_memory_params:
    _delete_signature_params(signatures)
    registry.delete_in_memory_params()
    _close_weights_loader(weights_loader)
    _trim_memory()

  t_python_rest = time.perf_counter() - t_python_rest_start
  registry.timings["torch_export_s"] = t_torch_export
  registry.timings["python_rest_s"] = t_python_rest

  logging.info(
      "[STAGE TIMING] PyTorch Export: %.3f s | Rest of Python Export: %.3f s",
      t_torch_export,
      t_python_rest,
  )
  return registry


def _build_tfl_converter_flags(
    *,
    fold_fp16_resource_casts: bool = True,
    enable_debug: bool = False,
    debug_dir: Optional[str] = None,
    enable_timing: bool = False,
    print_ir_before: Optional[str] = None,
    print_ir_after: Optional[str] = None,
    elide_elementsattrs_if_larger: Optional[int] = None,
    _litert_converter_flags: Optional[dict[str, Any]] = None,
) -> converter_flags_pb2.ConverterFlags:
  """Builds ConverterFlags proto with standard V2 flags and user overrides."""
  flags = converter_flags_pb2.ConverterFlags()
  flags.model_origin_framework = converter_flags_pb2.ConverterFlags.PYTORCH
  flags.enable_composite_direct_lowering = True
  flags.fold_fp16_resource_casts = fold_fp16_resource_casts
  if enable_debug:
    flags.enable_debug = True
  if debug_dir:
    flags.debug_dir = debug_dir
  if enable_timing:
    flags.debug_options.enable_timing = True
  if print_ir_before is not None:
    flags.debug_options.print_ir_before = print_ir_before
  if print_ir_after is not None:
    flags.debug_options.print_ir_after = print_ir_after
  if elide_elementsattrs_if_larger is not None:
    flags.debug_options.elide_elementsattrs_if_larger = (
        elide_elementsattrs_if_larger
    )
  if _litert_converter_flags:
    _apply_tfl_converter_flags(flags, dict(_litert_converter_flags))
  return flags


def _log_benchmark_summary(
    torch_export_s: float,
    python_rest_s: float,
    mlir_pipeline_s: float,
    total_convert_s: float,
    target_dir: Optional[str] = None,
    enable_timing: bool = False,
) -> None:
  """Logs a structured benchmark breakdown across stages and writes to timing file."""
  pct_torch = (
      (torch_export_s / total_convert_s * 100) if total_convert_s > 0 else 0
  )
  pct_py_rest = (
      (python_rest_s / total_convert_s * 100) if total_convert_s > 0 else 0
  )
  pct_mlir = (
      (mlir_pipeline_s / total_convert_s * 100) if total_convert_s > 0 else 0
  )

  summary_msg = (
      "\n"
      "================================================================================\n"
      "                    LiteRT Converter V2 Benchmark Breakdown             "
      "       \n"
      "================================================================================\n"
      f"  1. PyTorch Export (torch.export)       : {torch_export_s:8.3f} s"
      f" ({pct_torch:5.1f} %)\n"
      f"  2. Rest of Python Convert              : {python_rest_s:8.3f} s"
      f" ({pct_py_rest:5.1f} %)\n"
      f"  3. MLIR Pipeline -> FlatBuffer Output  : {mlir_pipeline_s:8.3f} s"
      f" ({pct_mlir:5.1f} %)\n"
      "--------------------------------------------------------------------------------\n"
      f"  Total Convert Duration                 : {total_convert_s:8.3f} s"
      " (100.0 %)\n"
      "================================================================================"
  )
  if not enable_timing:
    return

  logging.info(summary_msg)

  if not target_dir:
    return

  os.makedirs(target_dir, exist_ok=True)
  mlir_timing_path = os.path.join(target_dir, "mlir_pass_timing.log")
  mlir_log_content = ""
  if os.path.exists(mlir_timing_path):
    try:
      with open(mlir_timing_path, "r") as f:
        mlir_log_content = f.read()
    except OSError:
      pass

  combined_timing = summary_msg.strip() + "\n\n"
  if mlir_log_content:
    combined_timing += (
        "================================================================================\n"
        "                           MLIR Compiler Pass Timings                 "
        "          \n"
        "================================================================================\n"
        + mlir_log_content
    )

  try:
    with open(mlir_timing_path, "w") as f:
      f.write(combined_timing)
  except OSError:
    pass

  timing_json_path = os.path.join(target_dir, "timing.json")
  try:
    with open(timing_json_path, "w") as f:
      json.dump(
          {
              "torch_export_seconds": torch_export_s,
              "python_rest_seconds": python_rest_s,
              "mlir_pipeline_seconds": mlir_pipeline_s,
              "total_convert_seconds": total_convert_s,
              "timestamp": time.time(),
          },
          f,
          indent=2,
      )
  except OSError:
    pass


def convert_signatures_v2(
    signatures: list[signature_module.Signature],
    *,
    strict_export: Literal["auto"] | bool = False,
    export_dir: Optional[str] = None,
    output_file_path: Optional[str] = None,
    delete_in_memory_params: bool = False,
    fold_fp16_resource_casts: bool = True,
    allow_reuse_intermediates: bool = False,
    weights_loader: Optional[
        Callable[[str], torch.Tensor] | dict[str, torch.Tensor]
    ] = None,
    enable_debug: bool = False,
    debug_dir: Optional[str] = None,
    enable_timing: bool = False,
    print_ir_before: Optional[str] = None,
    print_ir_after: Optional[str] = None,
    elide_elementsattrs_if_larger: Optional[int] = None,
    _litert_converter_flags: Optional[dict[str, Any]] = None,
) -> model.LiteRTModel:
  """Converts a list of signatures to a LiteRT model using the V2 bridge."""
  if not signatures and export_dir is None:
    raise ValueError("No signatures added to the converter.")

  core._warn_training_modules(signatures)
  fx_infra.silence_torch_logs()

  t_conv_start = time.perf_counter()
  t_torch_export = 0.0
  t_python_rest = 0.0

  if export_dir is not None:
    os.makedirs(export_dir, exist_ok=True)
    if _has_any_intermediates(export_dir, signatures):
      if not allow_reuse_intermediates:
        raise ValueError(
            f"Intermediate conversion files found in export_dir '{export_dir}'."
            " To overwrite them, remove existing files or use a clean"
            " directory. To reuse existing per-signature intermediates without"
            " re-tracing, set allow_reuse_intermediates=True."
        )
      if not _has_complete_intermediates(export_dir, signatures):
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
      if delete_in_memory_params:
        if signatures:
          _delete_signature_params(signatures)
        _close_weights_loader(weights_loader)
        _trim_memory()
    else:
      if not signatures:
        raise ValueError("No signatures added to the converter.")
      # export_to_dir() already releases the weights loader and trims memory
      # when delete_in_memory_params is set.
      registry = export_to_dir(
          signatures,
          export_dir,
          strict_export=strict_export,
          delete_in_memory_params=delete_in_memory_params,
          weights_loader=weights_loader,
      )
      t_torch_export = registry.timings.get("torch_export_s", 0.0)
      t_python_rest = registry.timings.get("python_rest_s", 0.0)

    target_path = output_file_path or os.path.join(export_dir, "model.tflite")
    if os.path.dirname(target_path):
      os.makedirs(os.path.dirname(target_path), exist_ok=True)
    conversion_flags = _build_tfl_converter_flags(
        fold_fp16_resource_casts=fold_fp16_resource_casts,
        enable_debug=enable_debug,
        debug_dir=debug_dir or export_dir,
        enable_timing=enable_timing,
        print_ir_before=print_ir_before,
        print_ir_after=print_ir_after,
        elide_elementsattrs_if_larger=elide_elementsattrs_if_larger,
        _litert_converter_flags=_litert_converter_flags,
    )
    t_mlir_start = time.perf_counter()
    with progress.task("MLIR Pipeline Execution"):
      tfl_convert.convert_mlir_bytecode(
          conversion_flags, export_dir, target_path
      )
    t_mlir_duration = time.perf_counter() - t_mlir_start
    t_total_convert = time.perf_counter() - t_conv_start

    _log_benchmark_summary(
        t_torch_export,
        t_python_rest,
        t_mlir_duration,
        t_total_convert,
        target_dir=debug_dir or export_dir,
        enable_timing=enable_timing,
    )
    return model.LiteRTModel(model.FileExporter(target_path))
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
    if os.path.dirname(target_path):
      os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
      registry = export_to_dir(
          signatures,
          temp_dir,
          strict_export=strict_export,
          delete_in_memory_params=delete_in_memory_params,
          weights_loader=weights_loader,
      )
      t_torch_export = registry.timings.get("torch_export_s", 0.0)
      t_python_rest = registry.timings.get("python_rest_s", 0.0)

      conversion_flags = _build_tfl_converter_flags(
          fold_fp16_resource_casts=fold_fp16_resource_casts,
          enable_debug=enable_debug,
          debug_dir=debug_dir,
          enable_timing=enable_timing,
          print_ir_before=print_ir_before,
          print_ir_after=print_ir_after,
          elide_elementsattrs_if_larger=elide_elementsattrs_if_larger,
          _litert_converter_flags=_litert_converter_flags,
      )
      t_mlir_start = time.perf_counter()
      with progress.task("MLIR Pipeline Execution"):
        tfl_convert.convert_mlir_bytecode(
            conversion_flags, temp_dir, target_path
        )
      t_mlir_duration = time.perf_counter() - t_mlir_start
      t_total_convert = time.perf_counter() - t_conv_start

      _log_benchmark_summary(
          t_torch_export,
          t_python_rest,
          t_mlir_duration,
          t_total_convert,
          target_dir=debug_dir,
          enable_timing=enable_timing,
      )

      if output_file_path is None:
        with open(target_path, "rb") as f:
          content = f.read()
        return model.LiteRTModel(content)
      return model.LiteRTModel(model.FileExporter(target_path))
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
      weights_loader: Optional[
          Callable[[str], torch.Tensor] | dict[str, torch.Tensor]
      ] = None,
      enable_debug: bool = False,
      debug_dir: Optional[str] = None,
      enable_timing: bool = False,
      print_ir_before: Optional[str] = None,
      print_ir_after: Optional[str] = None,
      elide_elementsattrs_if_larger: Optional[int] = None,
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
        weights_loader=weights_loader,
        enable_debug=enable_debug,
        debug_dir=debug_dir,
        enable_timing=enable_timing,
        print_ir_before=print_ir_before,
        print_ir_after=print_ir_after,
        elide_elementsattrs_if_larger=elide_elementsattrs_if_larger,
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
    weights_loader: Optional[
        Callable[[str], torch.Tensor] | dict[str, torch.Tensor]
    ] = None,
    enable_debug: bool = False,
    debug_dir: Optional[str] = None,
    enable_timing: bool = False,
    print_ir_before: Optional[str] = None,
    print_ir_after: Optional[str] = None,
    elide_elementsattrs_if_larger: Optional[int] = None,
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
      weights_loader=weights_loader,
      enable_debug=enable_debug,
      debug_dir=debug_dir,
      enable_timing=enable_timing,
      print_ir_before=print_ir_before,
      print_ir_after=print_ir_after,
      elide_elementsattrs_if_larger=elide_elementsattrs_if_larger,
      _litert_converter_flags=_litert_converter_flags,
  )
