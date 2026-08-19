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
"""Optimized GPU Rotary Position Embedding (RoPE) operation compatible with MLDrift."""

from litert_torch.backend import composite
from litert_torch.generative.layers import rotary_position_embedding as rotary_pos_emb
import torch


def apply_mldrift_compatible_rope(
    x: torch.Tensor,
    position: torch.Tensor,
    base: float = 10000.0,
    head_dim: int | None = None,
) -> torch.Tensor:
  """Computes RoPE matching MLDrift's existing 2-input/1-output SplitRoPEConcat format.

  Args:
    x: Input tensor (Query or Key) with shape [B, T, N, H] or [B, N, T, H].
    position: 1D or 2D position indices tensor [B, T] or [T].
    base: RoPE theta base value (e.g. 10000.0 for standard, 500000.0 for Gemma
      3/4).
    head_dim: Head dimension size (defaults to x.shape[-1]).

  Returns:
    x_roped: Rotated output tensor of identical shape and dtype.
  """
  attrs = {
      "min_timescale": 1.0,
      "max_timescale": float(base),
      "proportion": 1.0,
  }
  builder = composite.StableHLOCompositeBuilder(name="odml.rope", attr=attrs)
  x, position = builder.mark_inputs(x, position)

  # Fallback PyTorch execution during export tracing:
  head_dim_size = head_dim if head_dim is not None else x.shape[-1]
  pos = position[0] if position.ndim > 1 else position
  cos, sin = rotary_pos_emb.build_rope(
      pos, n_elem=head_dim_size, base=int(base)
  )
  if cos is not None and cos.ndim == 3:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

  x_roped = rotary_pos_emb.apply_rope(x, cos, sin)
  x_roped = builder.mark_outputs(x_roped)
  return x_roped


def apply_rope_composite(
    x: torch.Tensor,
    position: torch.Tensor,
    base: float = 10000.0,
    head_dim: int | None = None,
    **kwargs,
) -> torch.Tensor:
  """Computes rotary positional embedding inline using MLDrift 'rope' Composite."""
  return apply_mldrift_compatible_rope(
      x=x, position=position, base=base, head_dim=head_dim
  )
