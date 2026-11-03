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
    has_v_norm: bool = False,
    proportion: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Computes fused QKV split, RMSNorm on Q & K (and optionally V), and RoPE on Q & K.

  Args:
    qkv: Fused QKV tensor with shape [B, T, C], C >= (num_heads + 2 *
      num_kv_heads) * head_dim. q, k and v are read from the leading channels;
      any trailing channels (e.g. an attention output gate fused into the same
      projection) are ignored, so callers don't need to slice them off.
    position: 1D or 2D position tensor [B, T] or [T].
    q_weight: Weight gamma tensor for Q RMSNorm.
    k_weight: Weight gamma tensor for K RMSNorm.
    num_heads: Number of query attention heads.
    num_kv_heads: Number of key/value attention heads.
    head_dim: Dimension of each attention head.
    base: RoPE theta base value (default 1000000.0).
    eps: Epsilon for RMSNorm (default 1e-6).
    has_v_norm: Whether to apply unscaled RMSNorm to V (e.g. Gemma 4).
    proportion: Fraction of head_dim to apply RoPE to (default 1.0).

  Returns:
    q_out: Transformed, normed, and roped query states [B, num_heads, T,
    head_dim].
    k_out: Transformed, normed, and roped key states [B, num_kv_heads, T,
    head_dim].
    v_out: Transformed value states [B, num_kv_heads, T, head_dim].
  """
  attrs = {
      "num_heads": int(num_heads),
      "num_kv_heads": int(num_kv_heads),
      "head_dim": int(head_dim),
      "min_timescale": 1.0,
      "max_timescale": float(base),
      "proportion": float(proportion),
      "epsilon": float(eps),
      "has_v_norm": bool(has_v_norm),
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
  q = qkv[..., :q_size]
  k = qkv[..., q_size : q_size + kv_size]
  v = qkv[..., q_size + kv_size : q_size + 2 * kv_size]

  input_shape = q.shape[:-1]
  hidden_shape_q = (*input_shape, num_heads, head_dim)
  hidden_shape_kv = (*input_shape, num_kv_heads, head_dim)

  q_reshaped = q.view(hidden_shape_q)
  k_reshaped = k.view(hidden_shape_kv)
  v_reshaped = v.view(hidden_shape_kv)

  def _rms_norm(
      x: torch.Tensor, weight: torch.Tensor, epsilon: float
  ) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + epsilon) * weight

  q_normed = _rms_norm(q_reshaped, q_weight, eps).transpose(1, 2)
  k_normed = _rms_norm(k_reshaped, k_weight, eps).transpose(1, 2)
  if has_v_norm:
    v_variance = v_reshaped.pow(2).mean(-1, keepdim=True)
    v_reshaped = v_reshaped * torch.rsqrt(v_variance + eps)
  v_out = v_reshaped.transpose(1, 2)

  pos = position[0] if position.ndim > 1 else position
  cos, sin = rotary_pos_emb.build_rope(pos, n_elem=head_dim, base=int(base))
  if proportion < 1.0 and cos is not None:
    rotary_dim = int(proportion * (head_dim // 2))
    cos = torch.cat(
        [cos[..., :rotary_dim], torch.ones_like(cos[..., rotary_dim:])], dim=-1
    )
    sin = torch.cat(
        [sin[..., :rotary_dim], torch.zeros_like(sin[..., rotary_dim:])], dim=-1
    )
  if cos is not None and cos.ndim == 3:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

  q_out = rotary_pos_emb.apply_rope(q_normed, cos, sin)
  k_out = rotary_pos_emb.apply_rope(k_normed, cos, sin)

  q_out, k_out, v_out = builder.mark_outputs(q_out, k_out, v_out)
  return q_out, k_out, v_out


def apply_q_norm_rope(
    q: torch.Tensor,
    position: torch.Tensor,
    q_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    base: float = 1000000.0,
    eps: float = 1e-6,
    proportion: float = 1.0,
) -> torch.Tensor:
  """Computes fused Q reshape, RMSNorm on Q, and RoPE on Q for KV-shared layers.

  Args:
    q: Query tensor with shape [B, T, num_heads * head_dim].
    position: 1D or 2D position tensor [B, T] or [T].
    q_weight: Weight gamma tensor for Q RMSNorm.
    num_heads: Number of query attention heads.
    head_dim: Dimension of each attention head.
    base: RoPE theta base value (default 1000000.0).
    eps: Epsilon for RMSNorm (default 1e-6).
    proportion: Fraction of head_dim to apply RoPE to (default 1.0).

  Returns:
    q_out: Transformed, normed, and roped query states [B, num_heads, T, head_dim].
  """
  attrs = {
      "num_heads": int(num_heads),
      "num_kv_heads": 0,
      "head_dim": int(head_dim),
      "min_timescale": 1.0,
      "max_timescale": float(base),
      "proportion": float(proportion),
      "epsilon": float(eps),
      "has_v_norm": False,
  }
  builder = composite.StableHLOCompositeBuilder(
      name="odml.qkv_norm_rope", attr=attrs
  )
  q, position, q_weight = builder.mark_inputs(q, position, q_weight)

  input_shape = q.shape[:-1]
  q_reshaped = q.view(*input_shape, num_heads, head_dim)
  variance = q_reshaped.pow(2).mean(-1, keepdim=True)
  q_normed = (q_reshaped * torch.rsqrt(variance + eps) * q_weight).transpose(
      1, 2
  )

  pos = position[0] if position.ndim > 1 else position
  cos, sin = rotary_pos_emb.build_rope(pos, n_elem=head_dim, base=int(base))
  if proportion < 1.0 and cos is not None:
    rotary_dim = int(proportion * (head_dim // 2))
    cos = torch.cat(
        [cos[..., :rotary_dim], torch.ones_like(cos[..., rotary_dim:])], dim=-1
    )
    sin = torch.cat(
        [sin[..., :rotary_dim], torch.zeros_like(sin[..., rotary_dim:])], dim=-1
    )
  if cos is not None and cos.ndim == 3:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

  q_out = rotary_pos_emb.apply_rope(q_normed, cos, sin)
  q_out = builder.mark_outputs(q_out)
  return q_out
