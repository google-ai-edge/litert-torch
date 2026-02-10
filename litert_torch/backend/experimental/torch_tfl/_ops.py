# Copyright 2025 The LiteRT Torch Authors.
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
"""Torch-TFL op definitions and fake implementations."""

import re
from typing import Any, Sequence

from litert_torch.backend.experimental.torch_tfl import torch_library_utils
import numpy as np
import torch


custom_op_with_fake = torch_library_utils.custom_op_with_fake


@custom_op_with_fake("tfl::batch_matmul")
def tfl_batch_matmul(
    x: torch.Tensor, y: torch.Tensor, adj_x: bool = False, adj_y: bool = False
) -> torch.Tensor:
  if x.ndim < 2 or y.ndim < 2:
    raise ValueError("Input tensors must have at least 2 dimensions.")
  if adj_x:
    x = torch.transpose(x, -1, -2)
  if adj_y:
    y = torch.transpose(y, -1, -2)
  return torch.matmul(x, y)


@custom_op_with_fake("tfl::add", schema="(Tensor x, Any y) -> Tensor")
def tfl_add(x: torch.Tensor, y: Any) -> torch.Tensor:
  return torch.add(x, y)


@custom_op_with_fake("tfl::sub", schema="(Tensor x, Any y) -> Tensor")
def tfl_sub(x: torch.Tensor, y: Any) -> torch.Tensor:
  return torch.sub(x, y)


@custom_op_with_fake("tfl::mul", schema="(Tensor x, Any y) -> Tensor")
def tfl_mul(x: torch.Tensor, y: Any) -> torch.Tensor:
  return torch.mul(x, y)


@custom_op_with_fake("tfl::div", schema="(Tensor x, Any y) -> Tensor")
def tfl_div(x: torch.Tensor, y: Any) -> torch.Tensor:
  return torch.div(x, y)


@custom_op_with_fake("tfl::pow", schema="(Any x, Any y) -> Tensor")
def tfl_pow(x: Any, y: Any) -> torch.Tensor:
  return torch.pow(x, y)


@custom_op_with_fake("tfl::logical_and")
def tfl_logical_and(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch.logical_and(x, y)


@custom_op_with_fake(
    "tfl::mean", schema="(Tensor x, Any dims, bool keepdim) -> Tensor"
)
def tfl_mean(x: torch.Tensor, dims: Any, keepdim: bool = False) -> torch.Tensor:
  return torch.mean(x, dims, keepdim)


@custom_op_with_fake("tfl::greater")
def tfl_greater(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch.gt(x, y)


@custom_op_with_fake("tfl::less")
def tfl_less(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch.lt(x, y)


@custom_op_with_fake("tfl::maximum")
def tfl_maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch.maximum(x, y)


@custom_op_with_fake("tfl::minimum")
def tfl_minimum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  return torch.minimum(x, y)


@custom_op_with_fake("tfl::sin")
def tfl_sin(x: torch.Tensor) -> torch.Tensor:
  return torch.sin(x)


@custom_op_with_fake("tfl::cos")
def tfl_cos(x: torch.Tensor) -> torch.Tensor:
  return torch.cos(x)


@custom_op_with_fake("tfl::rsqrt")
def tfl_rsqrt(x: torch.Tensor) -> torch.Tensor:
  return torch.rsqrt(x)


@custom_op_with_fake("tfl::neg")
def tfl_neg(x: torch.Tensor) -> torch.Tensor:
  return torch.neg(x)


@custom_op_with_fake("tfl::gelu")
def tfl_gelu(x: torch.Tensor, approximate: bool = False) -> torch.Tensor:
  gelu_approximate = "tanh" if approximate else "none"
  return torch.nn.functional.gelu(x, approximate=gelu_approximate)


@custom_op_with_fake("tfl::transpose")
def tfl_transpose(input: torch.Tensor, perm: Sequence[int]) -> torch.Tensor:
  assert len(perm) == input.ndim

  return torch.permute(input, perm).clone()


@custom_op_with_fake("tfl::concatenation")
def tfl_concatenation(
    tensors: Sequence[torch.Tensor], dim: int
) -> torch.Tensor:
  return torch.cat(tensors, dim=dim)


@custom_op_with_fake("tfl::fill", schema="(SymInt[] x, Any y) -> Tensor")
def tfl_fill(dims: Sequence[torch.SymInt], fill_value: Any) -> torch.Tensor:
  return torch.full(dims, fill_value)


def _normalize_shape(
    tensor_input: torch.Tensor, shape: Sequence[int]
) -> Sequence[int]:
  """Normalize the size for the -1 dimension in the "shape".

  Args:
      tensor_input: The input tensor.
      shape: The desired shape, which may contain a -1 to indicate an inferred
        dimension.

  Returns:
      The inferred shape.

  Raises:
      ValueError: If the shape is invalid or cannot be inferred.
  """
  inferred_shape = list(shape)
  if -1 in inferred_shape:
    numel = tensor_input.numel()
    product = 1
    neg_one_idx = -1
    for i, dim in enumerate(inferred_shape):
      if dim == -1:
        if neg_one_idx != -1:
          raise ValueError("Only one dimension can be inferred (-1)")
        neg_one_idx = i
      elif dim >= 0:
        product *= dim
      else:
        raise ValueError(
            "Shape dimensions must be non-negative or -1 for inference"
        )

    if neg_one_idx != -1:
      if product == 0:
        if numel != 0:
          raise ValueError(
              "Cannot infer dimension for non-zero input size when other"
              " dimensions multiply to zero"
          )
        inferred_shape[neg_one_idx] = 0
      else:
        if numel % product != 0:
          raise ValueError(
              f"Input size {numel} not divisible by product of known dimensions"
              f" {product}"
          )
        inferred_shape[neg_one_idx] = numel // product

  # Ensure the inferred shape still matches the total number of elements
  if np.prod(inferred_shape) != tensor_input.numel():
    raise ValueError(
        f"Calculated shape {inferred_shape} does not match input numel"
        f" {tensor_input.numel()}"
    )

  return inferred_shape


@torch.library.custom_op("tfl::reshape", mutates_args=())
def tfl_reshape(input: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
  inferred_shape = _normalize_shape(input, shape)
  return input.view(inferred_shape).clone()


# Use explicit fake implementation for tfl.reshape because dynamo cannot
# derive the output's symbolic shape from the impl above.
@torch.library.register_fake("tfl::reshape")
def tfl_reshape_fake(input: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
  inferred_shape = _normalize_shape(input, shape)
  return torch.empty(inferred_shape, dtype=input.dtype)


@custom_op_with_fake(
    "tfl::range", schema="(Scalar start, Scalar limit, Scalar delta) -> Tensor"
)
def tfl_range(
    start: int | float, limit: int | float, delta: int | float
) -> torch.Tensor:
  return torch.arange(start, limit, delta)


@custom_op_with_fake(
    "tfl::split_v", schema="(Tensor x, SymInt[] y, int z) -> Tensor[]"
)
def tfl_split_v(
    input: torch.Tensor, size_splits: Sequence[torch.SymInt], split_dim: int
) -> Sequence[torch.Tensor]:
  # Clone the output tensors to avoid aliasing issues.
  return [t.clone() for t in torch.split(input, size_splits, dim=split_dim)]


@custom_op_with_fake("tfl::expand_dims")
def tfl_expand_dims(x: torch.Tensor, dim: int) -> torch.Tensor:
  return torch.unsqueeze(x, dim).clone()


@custom_op_with_fake("tfl::broadcast_to")
def tfl_broadcast_to(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
  return x.expand(shape).clone()


@custom_op_with_fake("tfl::squeeze")
def tfl_squeeze(x: torch.Tensor, squeeze_dims: Sequence[int]) -> torch.Tensor:
  return torch.squeeze(x, squeeze_dims).clone()


@custom_op_with_fake("tfl::strided_slice")
def tfl_strided_slice(
    input: torch.Tensor,
    begin: Sequence[int],
    end: Sequence[int],
    strides: Sequence[int],
) -> torch.Tensor:
  assert (
      len(begin) == len(end) == len(strides) == input.ndim
  ), "Dimension mismatch"

  slices = []

  for i in range(input.ndim):
    b = begin[i]
    e = end[i]
    s = strides[i]
    slices.append(slice(b, e, s))

  result = input[tuple(slices)].clone()

  return result


@custom_op_with_fake("tfl::select_v2")
def tfl_select_v2(
    condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
  return torch.where(condition, x, y)


@custom_op_with_fake("tfl::embedding_lookup")
def tfl_embedding_lookup(
    indices: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
  return torch.nn.functional.embedding(indices, weight)


@custom_op_with_fake("tfl::gather")
def tfl_gather(
    input: torch.Tensor, indices: torch.Tensor, axis: int
) -> torch.Tensor:
  return torch.index_select(input, axis, indices)


@custom_op_with_fake("tfl::softmax")
def tfl_softmax(x: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.softmax(x, dim=-1)


@custom_op_with_fake("tfl::topk_v2")
def tfl_topk_v2(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
  out, indices = torch.topk(x, k, dim=-1, largest=True, sorted=True)
  indices = indices.to(torch.int32)
  return out, indices


@custom_op_with_fake("tfl::multinomial")
def tfl_multinomial(
    logits: torch.Tensor, num_samples: int, replacement: bool = False
) -> torch.Tensor:
  indices = torch.multinomial(
      torch.nn.functional.softmax(logits, dim=-1),
      num_samples,
      replacement=replacement,
  )
  return indices


@custom_op_with_fake(
    "tfl::slice", schema="(Tensor x, SymInt[] begin, SymInt[] size) -> Tensor"
)
def tfl_slice(
    input: torch.Tensor,
    begin: Sequence[torch.SymInt],
    size: Sequence[torch.SymInt],
) -> torch.Tensor:
  assert len(begin) == len(size) == input.ndim

  slices = [slice(i, i + l) for i, l in zip(begin, size)]
  return input[tuple(slices)].clone()


@torch.library.custom_op("tfl::slice.tensor", mutates_args=())
def tfl_slice_tensor(
    input: torch.Tensor,
    begin: torch.Tensor,
    size: torch.Tensor,
    *,
    shape: str = "",
) -> torch.Tensor:
  assert begin.ndim == size.ndim == 1
  assert begin.numel() == size.numel() == input.ndim
  assert begin.dtype == torch.int32 and size.dtype == torch.int32
  assert not shape or shape.count(",") == input.ndim - 1

  slices = [slice(i, i + l) for i, l in zip(begin.tolist(), size.tolist())]
  return input[tuple(slices)].clone()


@torch.library.register_fake("tfl::slice.tensor")
def tfl_slice_tensor_fake(
    input: torch.Tensor,
    begin: torch.Tensor,
    size: torch.Tensor,
    *,
    shape: str = "",
) -> torch.Tensor:
  ctx = torch.library.get_ctx()
  shape_str = shape
  if not shape_str:
    shape_str = ",".join(["?" for _ in range(input.ndim)])

  shape = []
  shape_symbols = shape_str.split(",")
  for sym in shape_symbols:
    if re.match(r"\d+", sym):
      shape.append(int(sym))
    else:
      nnz = ctx.new_dynamic_size()
      shape.append(nnz)
  return input.new_empty(shape, dtype=input.dtype)
