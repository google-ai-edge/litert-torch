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
"""Custom operators and lowerings for Gated Delta Net linear attention."""

from flatbuffers import flexbuffers
from litert_torch.backend import lowerings
from litert_torch.backend.lowerings import utils as lowering_utils
from litert_converter.mlir import ir
import torch
import torch.nn.functional as F


def _gated_delta_update_custom_options(
    *,
    mode: int,
) -> bytes:
  return bytes(
      flexbuffers.Dumps({
          "mode": int(mode),
      })
  )


def _const_bytes_attr(data: bytes) -> ir.Attribute:
  return ir.Attribute.parse(f'#tfl<const_bytes: "0x{data.hex()}">')


@torch.library.custom_op("litert_torch::gdn_tril_inv", mutates_args=())
def gdn_tril_inv(attn: torch.Tensor) -> torch.Tensor:
  """Reference PyTorch implementation of lower triangular inverse for Gated Delta Net."""
  attn = attn.clone()
  chunk_size = attn.shape[-1]
  idx = torch.arange(chunk_size, device=attn.device, dtype=torch.int32)
  eye_mask = idx.unsqueeze(1) == idx.unsqueeze(0)
  for i in range(1, chunk_size):
    row = attn[..., i, :i]
    sub = attn[..., :i, :i]
    attn[..., i, :i] = row + (row.unsqueeze(-2) @ sub).squeeze(-2)
  attn = torch.where(eye_mask, attn + 1.0, attn)
  return attn


@gdn_tril_inv.register_fake
def _gdn_tril_inv_fake(attn: torch.Tensor) -> torch.Tensor:
  """Fake implementation for shape inference."""
  return torch.empty_like(attn)


@lowerings.lower(torch.ops.litert_torch.gdn_tril_inv)
def _gdn_tril_inv_lower(lctx, attn: ir.Value):
  """Lowers gdn_tril_inv to tfl.custom."""
  op = ir.Operation.create(
      "tfl.custom",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[attn],
      attributes={
          "custom_code": ir.StringAttr.get("gdn_tril_inv"),
          "custom_option": ir.Attribute.parse('#tfl<const_bytes: "0x">'),
      },
  )
  return op.results[0]


@torch.library.custom_op("litert_torch::gated_delta_update", mutates_args=())
def gated_delta_update(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    beta_t: torch.Tensor,
    g_t: torch.Tensor,
    recurrent_state: torch.Tensor,
    mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reference PyTorch implementation of Gated Delta Update."""
  del mode
  batch_size, num_v_heads, seq_len, _ = q_t.shape

  # Recurrent path
  new_recurrent_state = recurrent_state.clone()
  core_attn_out = torch.zeros_like(v_t)
  for t in range(seq_len):
    g_step = g_t[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
    beta_step = beta_t[:, :, t].unsqueeze(-1)
    new_recurrent_state = new_recurrent_state * g_step
    kv_mem = (new_recurrent_state * k_t[:, :, t].unsqueeze(-1)).sum(dim=-2)
    delta = (v_t[:, :, t] - kv_mem) * beta_step
    new_recurrent_state = new_recurrent_state + k_t[:, :, t].unsqueeze(
        -1
    ) * delta.unsqueeze(-2)
    core_attn_out[:, :, t] = (
        new_recurrent_state * q_t[:, :, t].unsqueeze(-1)
    ).sum(dim=-2)
  return core_attn_out, new_recurrent_state


@gated_delta_update.register_fake
def _gated_delta_update_fake(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    beta_t: torch.Tensor,
    g_t: torch.Tensor,
    recurrent_state: torch.Tensor,
    mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Fake implementation for shape inference."""
  del k_t, beta_t, g_t, mode
  batch_size, num_v_heads, seq_len, _ = q_t.shape
  head_v_dim = v_t.shape[-1]
  out1 = torch.empty(
      (batch_size, num_v_heads, seq_len, head_v_dim),
      dtype=q_t.dtype,
      device=q_t.device,
  )
  out2 = torch.empty_like(recurrent_state)
  return out1, out2


@lowerings.lower(torch.ops.litert_torch.gated_delta_update)
def _gated_delta_update_lower(
    lctx,
    q_t: ir.Value,
    k_t: ir.Value,
    v_t: ir.Value,
    beta_t: ir.Value,
    g_t: ir.Value,
    recurrent_state: ir.Value,
    mode: int = 0,
):
  """Lowers gdn_attention to tfl.custom."""
  op = ir.Operation.create(
      "tfl.custom",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=[q_t, k_t, v_t, beta_t, g_t, recurrent_state],
      attributes={
          "custom_code": ir.StringAttr.get("gated_delta_update"),
          "custom_option": _const_bytes_attr(
              _gated_delta_update_custom_options(mode=mode)
          ),
      },
  )
  return tuple(op.results)


def gated_delta_net(
    mixed_qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    valid_mask: torch.Tensor,
    head_k_dim: int = 128,
    head_v_dim: int = 128,
    num_k_heads: int = 16,
    num_v_heads: int = 16,
    rms_norm_eps: float = 1e-6,
    mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Helper function to apply Gated Delta Net with conv1d outside and custom gated_delta_update op."""
  batch_size, seq_len, _ = mixed_qkv.shape
  conv_dim = conv_weight.shape[0]
  key_dim = head_k_dim * num_k_heads
  value_dim = head_v_dim * num_v_heads
  state_len = conv_state.shape[-1]

  mixed_qkv_t = mixed_qkv.transpose(1, 2)
  if valid_mask is not None and valid_mask.numel() > 0:
    mixed_qkv_masked = mixed_qkv_t * valid_mask.view(batch_size, 1, seq_len).to(
        mixed_qkv_t.dtype
    )
  else:
    mixed_qkv_masked = mixed_qkv_t

  full_qkv = torch.cat([conv_state, mixed_qkv_masked], dim=-1)
  conv_out = F.conv1d(
      full_qkv,
      conv_weight,
      None,
      padding=0,
      groups=conv_dim,
  )
  conv_out = F.silu(conv_out[:, :, -seq_len:]).transpose(1, 2)

  if seq_len > 1 and valid_mask is not None and valid_mask.numel() > 0:
    num_real = valid_mask[0].to(torch.int32).sum(dtype=torch.int32)
    idx = (
        torch.arange(state_len, device=full_qkv.device, dtype=torch.int64)
        + num_real
    )
    new_conv_state = full_qkv[:, :, idx]
  else:
    new_conv_state = full_qkv[:, :, -state_len:]

  query, key, value = torch.split(
      conv_out, [key_dim, key_dim, value_dim], dim=-1
  )
  query = query.reshape(batch_size, seq_len, -1, head_k_dim)
  key = key.reshape(batch_size, seq_len, -1, head_k_dim)
  value = value.reshape(batch_size, seq_len, -1, head_v_dim)

  beta = b.sigmoid()
  g = -a_log.to(torch.float32).exp() * F.softplus(a.to(torch.float32) + dt_bias)

  if num_v_heads // num_k_heads > 1:
    query = query.repeat_interleave(num_v_heads // num_k_heads, dim=2)
    key = key.repeat_interleave(num_v_heads // num_k_heads, dim=2)

  if valid_mask is not None and valid_mask.numel() > 0:
    vm_4d = valid_mask.view(batch_size, seq_len, 1, 1).to(query.dtype)
    vm_3d = valid_mask.view(batch_size, seq_len, 1).to(query.dtype)
    query = query * vm_4d
    key = key * vm_4d
    value = value * vm_4d
    beta = beta * vm_3d
    g = g * vm_3d

  # Normalize query and key with l2norm and scale query
  query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
  key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
  query = query * (1.0 / (head_k_dim**0.5))

  # Transpose to [B, H, N, D] for gated_delta_update
  q_t = query.transpose(1, 2)
  k_t = key.transpose(1, 2)
  v_t = value.transpose(1, 2)
  beta_t = beta.transpose(1, 2)
  g_t = g.transpose(1, 2)

  # Call custom op
  core_out, new_recurrent_state = torch.ops.litert_torch.gated_delta_update(
      q_t, k_t, v_t, beta_t, g_t, recurrent_state, mode=mode
  )

  # Transpose output back to [B, N, H, D_v]
  core_out = core_out.transpose(1, 2)

  # Apply RMSNorm
  variance = core_out.to(torch.float32).pow(2).mean(-1, keepdim=True)
  core_out = core_out * torch.rsqrt(variance + rms_norm_eps) * norm_weight

  # Gate with z (z is [B, N, H*D_v])
  z_reshaped = z.reshape(batch_size, seq_len, num_v_heads, head_v_dim)
  out = F.silu(z_reshaped) * core_out

  # Reshape to [B, N, H*D_v]
  out = out.reshape(batch_size, seq_len, -1)

  return out, new_conv_state, new_recurrent_state
