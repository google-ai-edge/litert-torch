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
"""The pre-lower pass to make exported program's constant inputs into MLIR ElementsAttrs."""

from typing import Any
from litert_torch.backend import lowerings
from litert_torch.backend.lowerings import utils as lowering_utils
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import arith
import numpy as np
import torch
from ai_edge_litert.mlir._mlir_libs import converter_api_ext

LAZY_CONSTANTS_NUMEL_THRESHOLD = 512 * 1024  # 512KB
LAZY_CONSTANTS_GETTER_CHUNK_SIZE = 1024 * 1024  # 1MB


def _tensor_fingerprint(tensor: torch.Tensor) -> int:
  return (
      str(tensor.device),
      tensor.shape,
      tensor.stride(),
      tensor.untyped_storage().data_ptr(),
  )


def _tensor_to_mlir_compatible_array(tensor: torch.Tensor) -> np.ndarray:
  """Converts a tensor to a numpy array that is compatible with MLIR contiguity and endianness."""
  if hasattr(tensor, 'detach'):
    arr = tensor.detach().cpu().numpy()
  else:
    arr = np.array(tensor)

  if arr.dtype == bool or arr.dtype == np.bool_:
    # packbits returns uint8; bitorder='little' is crucial for MLIR
    packed = np.packbits(arr, axis=None, bitorder='little')
    return packed

  target_dtype = {
      # Floating point
      np.float16: '<f2',
      np.float32: '<f4',
      np.float64: '<f8',
      # Signed Integers
      np.int8: '<i1',
      np.int16: '<i2',
      np.int32: '<i4',
      np.int64: '<i8',
      # Unsigned Integers
      np.uint8: '<u1',
      np.uint16: '<u2',
      np.uint32: '<u4',
      np.uint64: '<u8',
  }.get(arr.dtype.type)

  if target_dtype is None:
    raise TypeError(f'Unsupported dtype for MLIR conversion: {arr.dtype}')

  # Ensure C-contiguity and the specific bit-width/endianness
  return np.ascontiguousarray(arr, dtype=target_dtype)


def _get_tensor_uniform_value(tensor: torch.Tensor):
  """Returns the uniform value of a tensor in NumPy if uniform, else None."""

  # Handle empty tensors
  if tensor.numel() == 0:
    return None

  # Handle single-element tensors (always uniform)
  if tensor.numel() == 1:
    return tensor.detach().cpu().numpy()

  # Handle types that don't support min/max (Complex numbers)
  if tensor.is_complex():
    first_val = tensor[0]
    if (tensor == first_val).all():
      return first_val.detach().cpu().numpy()
    return None

  # Fast path for most numeric types
  mn, mx = tensor.min(), tensor.max()

  # Handle NaNs
  if torch.isnan(mn) and torch.isnan(mx):
    if torch.isnan(tensor).all():
      return mn.detach().cpu().numpy()
    return None

  # Check equality and return as NumPy scalar
  if (mn == mx).item():
    return mn.detach().cpu().numpy()

  return None


def _clamp_inf_values(tensor: torch.Tensor):
  """Clamps a tensor to the min/max value for float tensors."""
  if torch.is_floating_point(tensor):
    info = torch.finfo(tensor.dtype)
    tensor = torch.clamp(tensor, info.min, info.max)
  return tensor


def _build_const(attr: ir.Attribute, tensor_type: ir.RankedTensorType):
  return arith.ConstantOp(tensor_type, attr).results[0]


def get_tensor_lowering_placeholder(
    constant_cache: dict[Any, ir.Attribute],
    enable_lazy_constants: bool,
):
  """Returns a placeholder that will be lowered to elements attr."""

  def placeholder(x: torch.Tensor):
    return x

  @lowerings.lower(placeholder)
  def tensor_lowering(lctx, x: torch.Tensor):
    del lctx
    x_fingerprint = _tensor_fingerprint(x)
    elty = lowering_utils.torch_dtype_to_ir_element_type(x.dtype)
    tensor_type = ir.RankedTensorType.get(x.shape, elty)

    # If the tensor is already in the cache, return it.
    cached_attr = constant_cache.get(x_fingerprint)
    if cached_attr is not None:
      return _build_const(cached_attr, tensor_type)

    use_lazy_attr = enable_lazy_constants
    if x.dtype not in [torch.float32]:
      use_lazy_attr = False

    # 512KB as the threshold for using custom dense elements attr.
    if x.numel() * x.element_size() < LAZY_CONSTANTS_NUMEL_THRESHOLD:
      use_lazy_attr = False

    # If not using lazy attr, clamp inf values to the min/max value of the
    # tensor's dtype. Otherwise, rely on the bytes getter to clamp values
    # lazily.
    if not use_lazy_attr:
      x = _clamp_inf_values(x)

    # If the tensor is uniform, use a splat constant.
    uniform_value = _get_tensor_uniform_value(x)
    if uniform_value is not None:
      use_lazy_attr = False

    if uniform_value is not None:
      attr = lowering_utils.splat_attr(
          uniform_value.item(),
          tensor_type.element_type,
          tensor_type.shape,
      )
    elif use_lazy_attr:

      def chunk_iterator_factory():
        nonlocal x
        element_size = x.element_size()
        elements_per_chunk = LAZY_CONSTANTS_GETTER_CHUNK_SIZE // element_size

        # x.view(-1) is a metadata-only operation (0 bytes allocated)
        flat_x = x.view(-1)
        numel = flat_x.numel()

        for i in range(0, numel, elements_per_chunk):
          chunk_view = flat_x[i : i + elements_per_chunk]
          chunk_data = _tensor_to_mlir_compatible_array(chunk_view)
          yield chunk_data.tobytes()

      attr = converter_api_ext.get_py_chunked_callback_resource_attr(
          tensor_type, chunk_iterator_factory
      )
    else:
      arr = _tensor_to_mlir_compatible_array(x)
      attr = ir.DenseElementsAttr.get(arr, type=tensor_type)

    constant_cache[x_fingerprint] = attr
    return _build_const(attr, tensor_type)

  return placeholder


def inline_consts(
    exported_program: torch.export.ExportedProgram,
    constant_cache: dict[Any, ir.Attribute] | None = None,
    enable_lazy_constants: bool = False,
):
  """Inlines exported program's constant inputs by replacing with lazy_tensor_placeholder."""
  flat_user_inputs, _ = exported_program._get_flat_args_with_check(
      *exported_program.example_inputs
  )
  flat_inputs = exported_program._graph_module_flat_inputs(
      *exported_program.example_inputs
  )
  if flat_user_inputs:
    flat_consts = flat_inputs[: -len(flat_user_inputs)]
  else:
    flat_consts = flat_inputs

  if constant_cache is None:
    if enable_lazy_constants:
      raise ValueError(
          'constant_cache must be provided when enable_lazy_constants is True'
          ' to enforce constant deduplication.'
      )
    constant_cache = {}

  lowering_placeholder = get_tensor_lowering_placeholder(
      constant_cache=constant_cache,
      enable_lazy_constants=enable_lazy_constants,
  )

  for i, (tensor, node) in enumerate(
      zip(flat_consts, exported_program.graph.nodes)
  ):
    if node.op != 'placeholder':
      raise ValueError(
          'Expect FX graph nodes to start with placeholder nodes, but got'
          f'{i}-th node: {node}'
      )
    node.op = 'call_function'
    node.target = lowering_placeholder
    node.args = (tensor,)

  exported_program.graph_signature.input_specs = (
      exported_program.graph_signature.input_specs[len(flat_consts) :]
  )
