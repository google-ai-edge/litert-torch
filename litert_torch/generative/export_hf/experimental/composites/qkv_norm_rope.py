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
"""Optimized GPU QKV layout, QK-RMSNorm, and RoPE composite operation compatible with MLDrift."""

from litert_torch.backend import composite
from litert_torch.generative.layers import rotary_position_embedding as rotary_pos_emb
import torch


def apply_qkv_norm_rope(
    qkv: torch.Tensor,
    position: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    base: float = 1000000.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Computes fused QKV split, RMSNorm on Q & K, and RoPE on Q & K from linear qkv tensor.

  Args:
    qkv: Fused QKV tensor with shape [B, T, (num_heads + 2 * num_kv_heads) * head_dim].
    position: 1D or 2D position tensor [B, T] or [T].
    q_weight: Weight gamma tensor for Q RMSNorm.
    k_weight: Weight gamma tensor for K RMSNorm.
    num_heads: Number of query attention heads.
    num_kv_heads: Number of key/value attention heads.
    head_dim: Dimension of each attention head.
    base: RoPE theta base value (default 1000000.0).
    eps: Epsilon for RMSNorm (default 1e-6).

  Returns:
    q_out: Transformed, normed, and roped query states [B, num_heads, T, head_dim].
    k_out: Transformed, normed, and roped key states [B, num_kv_heads, T, head_dim].
    v_out: Transformed value states [B, num_kv_heads, T, head_dim].
  """
  attrs = {
      "num_heads": int(num_heads),
      "num_kv_heads": int(num_kv_heads),
      "head_dim": int(head_dim),
      "min_timescale": 1.0,
      "max_timescale": float(base),
      "proportion": 1.0,
      "epsilon": float(eps),
  }
  builder = composite.StableHLOCompositeBuilder(
      name="odml.qkv_norm_rope", attr=attrs
  )
  qkv, position, q_weight, k_weight = builder.mark_inputs(
      qkv, position, q_weight, k_weight
  )

  # Fallback PyTorch execution during export tracing:
  q_size = num_heads * head_dim
  kv_size = num_kv_heads * head_dim
  q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

  input_shape = q.shape[:-1]
  hidden_shape_q = (*input_shape, num_heads, head_dim)
  hidden_shape_kv = (*input_shape, num_kv_heads, head_dim)

  q_reshaped = q.view(hidden_shape_q)
  k_reshaped = k.view(hidden_shape_kv)
  v_reshaped = v.view(hidden_shape_kv)

  def _rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + epsilon) * weight

  q_normed = _rms_norm(q_reshaped, q_weight, eps).transpose(1, 2)
  k_normed = _rms_norm(k_reshaped, k_weight, eps).transpose(1, 2)
  v_out = v_reshaped.transpose(1, 2)

  pos = position[0] if position.ndim > 1 else position
  cos, sin = rotary_pos_emb.build_rope(
      pos, n_elem=head_dim, base=int(base)
  )
  if cos is not None and cos.ndim == 3:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

  q_out = rotary_pos_emb.apply_rope(q_normed, cos, sin)
  k_out = rotary_pos_emb.apply_rope(k_normed, cos, sin)

  q_out, k_out, v_out = builder.mark_outputs(q_out, k_out, v_out)
  return q_out, k_out, v_out
