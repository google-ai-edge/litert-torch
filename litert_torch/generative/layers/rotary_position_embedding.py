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
# Implementation for Rotary Position embedding. https://arxiv.org/pdf/2104.09864.pdf

from typing import Tuple
import torch


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
  """Computes rotary positional embedding.

  Args:
    x: the input tensor.
    cos: cosine value for the rope.
    sin: sin value for the rope.

  Returns:
    output tensor of RoPE.
  """
  x = x.transpose(1, 2)
  rope_size = cos.size(-1)
  x_splited = torch.split(x, rope_size, dim=-1)
  left = x_splited[0] * cos - x_splited[1] * sin
  right = x_splited[1] * cos + x_splited[0] * sin
  roped = torch.cat((left, right) + x_splited[2:], dim=-1)
  return roped.transpose(1, 2).type_as(x)


def build_rope(
    input_pos: torch.Tensor,
    n_elem: int,
    base: int = 10_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Computes rotary positional embedding cosine and sine tensors.

  Args:
    input_pos: the sequence indices for the query and key
    n_elem: number of elements of the head dimension for RoPE computation
    base: the base of the exponentiated value for RoPE.

  Returns:
    cos, sin tensors
  """

  if n_elem <= 0:
    return None, None

  freq_exponents = (2.0 / n_elem) * torch.arange(
      n_elem // 2, dtype=torch.float32, device=input_pos.device
  )
  timescale = float(base) ** freq_exponents
  radians = input_pos.clone().unsqueeze(0).unsqueeze(-1) / timescale.unsqueeze(
      0
  ).unsqueeze(0)
  cos = torch.cos(radians)
  sin = torch.sin(radians)
  return cos, sin


def apply_rope_inline(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Computes rotary positional embedding inline for a query and key.

  Args:
    q: the query tensor.
    k: the key tensor.
    cos: the cosine tensor.
    sin: the sine tensor.

  Returns:
    output the RoPE'd query and key.
  """

  q_roped = apply_rope(q, cos, sin)
  k_roped = apply_rope(k, cos, sin)
  return q_roped, k_roped
