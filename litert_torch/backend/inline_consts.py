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

import dataclasses
import logging
import math
from litert_torch import _config
from litert_torch.backend import lowerings
from litert_torch.backend.lowerings import utils as lowering_utils
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import arith
import numpy as np
import torch

config = _config.config


@dataclasses.dataclass
class InlineConstsContext(lowerings.context.LoweringContextPlugin):
  """The context object for inlining constants."""

  enable_resource_constants: bool = False
  constant_cache: dict[int, ir.Attribute] = dataclasses.field(
      default_factory=dict
  )

  @classmethod
  def get(
      cls, lctx: lowerings.context.LoweringContext
  ) -> 'InlineConstsContext':
    if not lctx.has_plugin(cls):
      logging.warning(
          'InlineConstsContext is not registered in the lowering context.'
          ' Constants may not be shared between exported program lowerings.'
      )
      lctx.add_plugin(cls())
    return lctx.get_plugin(cls)


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
    arr = tensor.contiguous().detach().cpu().numpy()
  else:
    arr = np.array(tensor)

  # Ensure C-contiguity
  return np.ascontiguousarray(arr)


def _get_tensor_uniform_value(tensor: torch.Tensor):
  """Returns the uniform value of a tensor if it is uniform, otherwise None."""
  flat = tensor.view(-1)
  numel = flat.numel()

  if numel == 0:
    return None

  # Extract the first value as a Python scalar early
  first_val = flat[0].item()
  if numel == 1:
    return first_val

  # The Heuristic "Fast-Fail"
  # If the tensor is large, check small chunks at the start and end.
  # This catches most non-uniform cases in constant time O(1).
  if numel > 128:
    if not torch.all(flat[:64] == first_val):
      return None
    if not torch.all(flat[-64:] == first_val):
      return None

  # The Full Scan
  if torch.all(flat == first_val):
    return first_val

  # The NaN Edge Case
  if isinstance(first_val, float) and math.isnan(first_val):
    if torch.isnan(flat).all():
      return first_val

  return None


def _clamp_inf_values(tensor: torch.Tensor):
  """Clamps a tensor to the min/max value for float tensors."""
  if torch.is_floating_point(tensor):
    info = torch.finfo(tensor.dtype)
    tensor.clamp_(min=info.min, max=info.max)
  return tensor


def _build_const(attr: ir.Attribute, tensor_type: ir.RankedTensorType):
  return arith.ConstantOp(tensor_type, attr).results[0]


def tensor_lowering_placeholder(*args, **kwargs):  # pylint: disable=g-doc-args
  """The placeholder function to be plugged into the exported program to be lowered to a constant op.

  Raises:
    RuntimeError: This function should not be called directly.
  """
  del args, kwargs
  raise RuntimeError(
      'tensor_lowering_placeholder should not be called directly.'
  )


@lowerings.lower(tensor_lowering_placeholder)
def tensor_lowering_placeholder_lowering(
    lctx: lowerings.context.LoweringContext,
    x: torch.Tensor,
):
  """Lower the placeholder function to a constant op."""
  const_ctx = InlineConstsContext.get(lctx)
  x = x.contiguous().detach().cpu()

  x_fingerprint = _tensor_fingerprint(x)
  elty = lowering_utils.torch_dtype_to_ir_element_type(x.dtype)
  tensor_type = ir.RankedTensorType.get(x.shape, elty)

  # If the tensor is already in the cache, return it.
  cached_attr = const_ctx.constant_cache.get(x_fingerprint)
  if cached_attr is not None:
    return _build_const(cached_attr, tensor_type)

  use_resource_attr = const_ctx.enable_resource_constants
  if x.dtype not in [torch.float32, torch.int32]:
    use_resource_attr = False

  # If the tensor is too small, just use a dense elements attr.
  if x.numel() * x.element_size() < config.resource_constant_numel_threshold:
    use_resource_attr = False

  x = _clamp_inf_values(x)

  # If the tensor is uniform, use a splat constant.
  uniform_value = _get_tensor_uniform_value(x)
  if uniform_value is not None:
    use_resource_attr = False

  if uniform_value is not None:
    attr = lowering_utils.splat_attr(
        uniform_value,
        tensor_type.element_type,
        tensor_type.shape,
    )
  elif use_resource_attr:
    arr = _tensor_to_mlir_compatible_array(x)
    attr = ir.DenseResourceElementsAttr.get_from_buffer(
        memoryview(arr),
        f'TENSOR_{x_fingerprint}',
        tensor_type,
    )
  else:
    arr = _tensor_to_mlir_compatible_array(x)
    attr = ir.DenseElementsAttr.get(arr, type=tensor_type)

  const_ctx.constant_cache[x_fingerprint] = attr
  return _build_const(attr, tensor_type)


def inline_consts(exported_program: torch.export.ExportedProgram) -> None:
  """Inlines exported program's constant inputs by replacing with resource_tensor_placeholder."""
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

  for i, (tensor, node) in enumerate(
      zip(flat_consts, exported_program.graph.nodes)
  ):
    if node.op != 'placeholder':
      raise ValueError(
          'Expect FX graph nodes to start with placeholder nodes, but got'
          f'{i}-th node: {node}'
      )
    node.op = 'call_function'
    node.target = tensor_lowering_placeholder
    node.args = (tensor,)

  exported_program.graph_signature.input_specs = (
      exported_program.graph_signature.input_specs[len(flat_consts) :]
  )
