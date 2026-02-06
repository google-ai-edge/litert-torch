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
"""Utilities for building MLIR lowerings."""

from collections.abc import Callable
import functools
import numbers
from typing import Any, Optional, Sequence, Union
from litert_torch.backend import export_utils
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import stablehlo
import numpy as np
import torch
import torch.utils._pytree as pytree


def torch_dtype_to_ir_element_type(dtype) -> ir.Type:
  """Builds ir.Type from torch dtype."""
  ty_get = {
      torch.double: ir.F64Type.get,
      torch.float32: ir.F32Type.get,
      torch.half: ir.F16Type.get,
      torch.long: functools.partial(ir.IntegerType.get_signless, 64),
      torch.int32: functools.partial(ir.IntegerType.get_signless, 32),
      torch.int16: functools.partial(ir.IntegerType.get_signless, 16),
      torch.int8: functools.partial(ir.IntegerType.get_signless, 8),
      torch.bool: functools.partial(ir.IntegerType.get_signless, 1),
      torch.bfloat16: ir.BF16Type.get,
  }[dtype]
  return ty_get()


def node_meta_to_ir_types(node: torch.fx.Node) -> list[ir.Type]:
  """Builds IR result types from torch FX node meta."""
  tensor_meta = node.meta.get("tensor_meta") or node.meta.get("val")
  if not tensor_meta:
    raise RuntimeError(f"{node.name} does not have tensor meta")

  tensor_meta_list, _ = pytree.tree_flatten(
      [tensor_meta],
      is_leaf=lambda x: hasattr(x, "dtype") and hasattr(x, "shape"),
  )
  results = []
  for meta in tensor_meta_list:
    shape = [
        export_utils.IR_DYNAMIC if export_utils.is_torch_dynamic(dim) else dim
        for dim in meta.shape
    ]
    elty = torch_dtype_to_ir_element_type(meta.dtype)
    results.append(ir.RankedTensorType.get(shape, elty))
  return results


def splat(val, ty, shape=tuple(), *, loc: Optional[Any] = None):
  if isinstance(ty, ir.IntegerType):
    if ty.width == 1:
      attr = ir.BoolAttr.get(bool(val))
    else:
      attr = ir.IntegerAttr.get(ty, int(val))
  elif isinstance(ty, ir.FloatType):
    attr = ir.FloatAttr.get(ty, val)
  else:
    raise ValueError("Unsupported type: %s" % str(ty))

  return stablehlo.constant(
      ir.DenseElementsAttr.get_splat(
          ir.RankedTensorType.get(shape, ty),
          attr,
      ),
      loc=loc,
  )


def get_common_broadcast_shape(
    shape_1: list[int], shape_2: list[int]
) -> Optional[list[int]]:
  if not shape_1 and not shape_2:
    return None

  shape_1 = shape_1 if shape_1 else [1]
  shape_2 = shape_2 if shape_2 else [1]

  length_diff = abs(len(shape_1) - len(shape_2))
  if len(shape_1) < len(shape_2):
    shape_1 = [1] * length_diff + shape_1
  elif len(shape_1) > len(shape_2):
    shape_2 = [1] * length_diff + shape_2

  common_broadcast_shape = []
  for idx in reversed(range(len(shape_1))):
    dim_size1 = shape_1[idx]
    dim_size2 = shape_2[idx]

    if dim_size1 == dim_size2:
      common_broadcast_shape.insert(0, dim_size1)
    elif dim_size1 == 1 or dim_size2 == 1:
      common_broadcast_shape.insert(0, max(dim_size1, dim_size2))
    else:
      return None

  return common_broadcast_shape


def get_broadcast_dimensions(
    shape_from: list[int], shape_to: list[int]
) -> list[int]:
  assert get_common_broadcast_shape(shape_from, shape_to) == shape_to

  ret = []
  for val in range(len(shape_to) - len(shape_from), len(shape_to)):
    ret.append(val)

  return ir.DenseI64ArrayAttr.get(np.asarray(ret, np.int64))


def broadcast_args_if_needed(
    val_1: ir.Value, val_2: ir.Value
) -> tuple[Optional[ir.Value], Optional[ir.Value]]:
  broadcast_shape = get_common_broadcast_shape(
      val_1.type.shape, val_2.type.shape
  )
  if broadcast_shape is None:
    return None, None

  new_val_1, new_val_2 = val_1, val_2

  if val_1.type.shape != broadcast_shape:
    new_val_1 = stablehlo.broadcast_in_dim(
        result=ir.RankedTensorType.get(
            broadcast_shape, val_1.type.element_type
        ),
        operand=val_1,
        broadcast_dimensions=get_broadcast_dimensions(
            val_1.type.shape, broadcast_shape
        ),
    )
  if val_2.type.shape != broadcast_shape:
    new_val_2 = stablehlo.broadcast_in_dim(
        result=ir.RankedTensorType.get(
            broadcast_shape, val_2.type.element_type
        ),
        operand=val_2,
        broadcast_dimensions=get_broadcast_dimensions(
            val_2.type.shape, broadcast_shape
        ),
    )
  return new_val_1, new_val_2


def upcast_to_same_type(*vals: ir.Value):
  if not vals:
    return None
  if len(vals) == 1:
    return vals[0]

  def get_priority(ty: ir.Type):
    priorities = [
        ir.IntegerType.get_signless(1),
        ir.IntegerType.get_signless(16),
        ir.IntegerType.get_signless(32),
        ir.IntegerType.get_signless(64),
        ir.F16Type,
        ir.F32Type,
        ir.F64Type,
    ]
    for i, tycls in enumerate(priorities):
      if tycls.isinstance(ty):
        return i
    raise ValueError("Unsupported type: %s" % str(ty))

  cast_tycls = type(max([v.type.element_type for v in vals], key=get_priority))
  new_vals = []
  for val in vals:
    if not cast_tycls.isinstance(val.type.element_type):
      val = stablehlo.convert(
          ir.RankedTensorType.get(val.type.shape, cast_tycls.get()), val
      )
    new_vals.append(val)
  return tuple(new_vals)


def minmax(ty: ir.Type) -> tuple[numbers.Number, numbers.Number]:
  if isinstance(ty, ir.IntegerType):
    if ty.is_unsigned:
      return (0, 1 << ty.width)
    else:
      return (-(1 << (ty.width - 1)), (1 << (ty.width - 1)) - 1)
  elif isinstance(ty, ir.F16Type):
    return (np.finfo(np.float16).min, np.finfo(np.float16).max)
  elif isinstance(ty, ir.F32Type):
    return (np.finfo(np.float32).min, np.finfo(np.float32).max)
  elif isinstance(ty, ir.F64Type):
    return (np.finfo(np.float64).min, np.finfo(np.float64).max)
  else:
    raise ValueError("Unsupported type: %s" % ty)


def convert_int_to_float(t: ir.Value) -> ir.Value:
  """Converts an input with type ir.IntegerType to an ir.FloatType of equivalent width."""
  elty = t.type.element_type
  if not isinstance(elty, ir.IntegerType):
    raise ValueError(
        "Expected input with integer type, received %s" % type(elty)
    )

  if elty.width == 32:
    return stablehlo.convert(
        ir.RankedTensorType.get(t.type.shape, ir.F32Type.get()), t
    )
  elif elty.width == 64:
    return stablehlo.convert(
        ir.RankedTensorType.get(t.type.shape, ir.F64Type.get()), t
    )


# IR Helpers
IrValues = Union[ir.Value, tuple[ir.Value, ...]]


# Non-canonicalized dtype to IR type mapping.
_numpy_dtype_to_ir_type: dict[np.dtype, Callable[[], ir.Type]] = {
    np.dtype(np.bool_): functools.partial(ir.IntegerType.get_signless, 1),
    np.dtype(np.int8): functools.partial(ir.IntegerType.get_signless, 8),
    np.dtype(np.int16): functools.partial(ir.IntegerType.get_signless, 16),
    np.dtype(np.int32): functools.partial(ir.IntegerType.get_signless, 32),
    np.dtype(np.int64): functools.partial(ir.IntegerType.get_signless, 64),
    np.dtype(np.uint8): functools.partial(ir.IntegerType.get_unsigned, 8),
    np.dtype(np.uint16): functools.partial(ir.IntegerType.get_unsigned, 16),
    np.dtype(np.uint32): functools.partial(ir.IntegerType.get_unsigned, 32),
    np.dtype(np.uint64): functools.partial(ir.IntegerType.get_unsigned, 64),
    np.dtype(np.float16): ir.F16Type.get,
    np.dtype(np.float32): ir.F32Type.get,
    np.dtype(np.float64): ir.F64Type.get,
    np.dtype(np.complex64): lambda: ir.ComplexType.get(ir.F32Type.get()),
    np.dtype(np.complex128): lambda: ir.ComplexType.get(ir.F64Type.get()),
}


def numpy_dtype_to_ir_type(dtype: np.dtype | np.generic) -> ir.Type:
  assert isinstance(dtype, (np.dtype, np.generic)), type(dtype)
  dtype = np.dtype(dtype)
  try:
    ir_type_factory = _numpy_dtype_to_ir_type[dtype]
  except KeyError as err:
    raise TypeError(
        f"No numpy_dtype_to_ir_type handler for dtype: {dtype}"
    ) from err
  return ir_type_factory()


def numpy_array_constant(x: np.ndarray | np.generic) -> IrValues:
  element_type = numpy_dtype_to_ir_type(x.dtype)
  shape = x.shape
  if x.dtype == np.bool_:
    x = np.packbits(x, bitorder="little")  # type: ignore
  x = np.ascontiguousarray(x)
  attr = ir.DenseElementsAttr.get(x, type=element_type, shape=shape)  # type: ignore
  return stablehlo.constant(attr)


def convert_to_ir_value(
    value: ir.Value | int | float | np.ndarray | np.generic,
) -> ir.Value:
  if isinstance(value, (np.ndarray, np.generic)):
    return numpy_array_constant(value)
  if isinstance(value, (int, float)):
    dtype = np.float32 if isinstance(value, float) else np.int32
    return numpy_array_constant(np.array([value], dtype=dtype))
  if isinstance(value, ir.Value):
    return value
  raise TypeError(f"Unsupported type for conversion to ir.Value: {type(value)}")


def convert_shape_to_ir_value(
    shape: Sequence[int],
) -> ir.Value:
  # Check if all elements in the shape sequence are integers.
  if not shape or all(isinstance(dim, int) for dim in shape):
    # If all are integers, create a constant numpy array.
    # Assuming int32 is the required type for TFLite shape tensors.
    shape_ir_value = numpy_array_constant(np.array(shape, dtype=np.int32))
  else:
    # Handle mixed int and ir.Value shape sequence
    processed_dims = []
    for dim in shape:
      if isinstance(dim, int):
        # Convert int to a constant 1D tensor
        shape_ir_value = numpy_array_constant(np.array([dim], dtype=np.int32))
        processed_dims.append(shape_ir_value)
      else:
        assert isinstance(dim, ir.Value)
        # Convert ir.Value to a constant 1D tensor
        new_type = ir.RankedTensorType.get([1], dim.type.element_type)
        reshape_dim = stablehlo.reshape(new_type, dim)
        processed_dims.append(reshape_dim)

    shape_ir_value = stablehlo.concatenate(
        processed_dims,
        dimension=0,
    )
  return shape_ir_value
