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
from litert_torch.backend.torch_tfl import _ops as _
from litert_torch.backend.lowerings import utils as lowering_utils
from litert_converter.mlir import ir
import torch


def _gated_delta_update_custom_options(
    *,
    mode: int,
    state_dtype: str = "float32",
) -> bytes:
  return bytes(
      flexbuffers.Dumps({
          "mode": int(mode),
          "state_dtype": str(state_dtype),
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


def _gated_delta_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Single recurrent step for Gated Delta Net."""
  old_state = state

  # decay
  decay = g.exp().unsqueeze(-1).unsqueeze(-1)
  state = state * decay

  # kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
  kv_mem = torch.sum(state * k.unsqueeze(-1), dim=-2)

  # delta = (v - kv_mem) * beta.unsqueeze(-1)
  delta = (v - kv_mem) * beta.unsqueeze(-1)

  # state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
  state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)

  # y = (state * q.unsqueeze(-1)).sum(dim=-2)
  y = torch.sum(state * q.unsqueeze(-1), dim=-2)

  if mask is not None:
    mask_state = (mask > 0).view(-1, 1, 1, 1)
    state = torch.where(mask_state, state, old_state)

    mask_y = (mask > 0).view(-1, 1, 1)
    y = torch.where(mask_y, y, torch.zeros_like(y))

  return y, state


def recurrent_gated_delta_rule(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    beta_t: torch.Tensor,
    g_t: torch.Tensor,
    recurrent_state: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    valid_len: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Recurrent Gated Delta Rule implementation (single-token or unrolled loop)."""
  batch_size, _, seq_len, _ = q_t.shape
  has_valid_mask = valid_mask is not None and valid_mask.numel() > 0

  if seq_len == 1 and not has_valid_mask and valid_len is None:
    g_step = g_t[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
    beta_step = beta_t[:, :, 0].unsqueeze(-1)
    new_recurrent_state = recurrent_state.to(q_t.dtype) * g_step
    kv_mem = (new_recurrent_state * k_t[:, :, 0].unsqueeze(-1)).sum(dim=-2)
    delta = (v_t[:, :, 0] - kv_mem) * beta_step
    new_recurrent_state = new_recurrent_state + k_t[:, :, 0].unsqueeze(
        -1
    ) * delta.unsqueeze(-2)
    core_attn_out = (
        (new_recurrent_state * q_t[:, :, 0].unsqueeze(-1))
        .sum(dim=-2)
        .unsqueeze(2)
    )
    return core_attn_out.to(q_t.dtype), new_recurrent_state.to(torch.float32)

  new_recurrent_state = recurrent_state.to(q_t.dtype)
  ys = []
  for t in range(seq_len):
    step_mask = None
    if has_valid_mask:
      assert valid_mask is not None
      step_mask = valid_mask[:, t]
    elif valid_len is not None:
      step_mask = t < valid_len.reshape(batch_size)

    y_step, new_recurrent_state = _gated_delta_step(
        q_t[:, :, t],
        k_t[:, :, t],
        v_t[:, :, t],
        g_t[:, :, t],
        beta_t[:, :, t],
        new_recurrent_state,
        step_mask,
    )
    ys.append(y_step)

  core_attn_out = torch.stack(ys, dim=2)
  return core_attn_out.to(q_t.dtype), new_recurrent_state.to(torch.float32)


def chunk_gated_delta_rule(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    beta_t: torch.Tensor,
    g_t: torch.Tensor,
    recurrent_state: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Chunked decomposed Gated Delta Rule prefill implementation."""
  batch_size, num_v_heads, seq_len, _ = q_t.shape
  head_v_dim = v_t.shape[-1]
  num_chunks = seq_len // chunk_size

  v_beta = v_t * beta_t.unsqueeze(-1)
  k_beta = k_t * beta_t.unsqueeze(-1)

  q_c, k_c, _, k_beta_c, v_beta_c = [
      x.reshape(
          batch_size,
          num_v_heads,
          num_chunks,
          chunk_size,
          x.shape[-1],
      )
      for x in (q_t, k_t, v_t, k_beta, v_beta)
  ]
  g_c = g_t.reshape(batch_size, num_v_heads, num_chunks, chunk_size)

  idx = torch.arange(chunk_size, device=q_t.device, dtype=torch.int32)
  mask_triu = idx.unsqueeze(1) <= idx.unsqueeze(0)
  mask_tril = idx.unsqueeze(1) >= idx.unsqueeze(0)
  eye_mask = idx.unsqueeze(1) == idx.unsqueeze(0)

  g_cumsum = g_c.cumsum(dim=-1)
  diff = g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)
  decay_mask = torch.where(
      mask_tril, torch.where(mask_tril, diff, 0.0).exp(), 0.0
  )

  attn_in = -((k_beta_c @ k_c.transpose(-1, -2)) * decay_mask).masked_fill(
      mask_triu, 0
  )

  attn = attn_in.clone()
  for i in range(1, chunk_size):
    row = attn[..., i, :i]
    sub = attn[..., :i, :i]
    attn[..., i, :i] = row + (row.unsqueeze(-2) @ sub).squeeze(-2)

  attn = torch.where(eye_mask, attn + 1.0, attn)
  v_local = attn @ v_beta_c

  k_beta_exp = k_beta_c * g_cumsum.exp().unsqueeze(-1)
  k_cumdecay = attn @ k_beta_exp

  new_recurrent_state = recurrent_state.to(q_t.dtype)
  core_attn_outs = []
  for i in range(num_chunks):
    q_i = q_c[:, :, i]
    k_i = k_c[:, :, i]
    v_i = v_local[:, :, i]

    exp_g = g_cumsum[:, :, i].exp()
    attn_inter = (q_i * exp_g.unsqueeze(-1)) @ new_recurrent_state

    v_prime = k_cumdecay[:, :, i] @ new_recurrent_state
    v_new = v_i - v_prime

    attn_i = (q_i @ k_i.transpose(-1, -2)) * decay_mask[:, :, i]

    core_attn_outs.append(attn_inter + attn_i @ v_new)

    exp_last = g_cumsum[:, :, i, -1, None, None].exp()
    exp_diff = (
        (g_cumsum[:, :, i, -1, None] - g_cumsum[:, :, i]).exp().unsqueeze(-1)
    )
    k_scaled = k_i * exp_diff

    new_recurrent_state = (
        new_recurrent_state * exp_last + k_scaled.transpose(-1, -2) @ v_new
    )

  core_attn_out = torch.stack(core_attn_outs, dim=2)
  core_attn_out = core_attn_out.reshape(
      batch_size, num_v_heads, seq_len, head_v_dim
  )
  return core_attn_out, new_recurrent_state


@torch.library.custom_op("litert_torch::gated_delta_update", mutates_args=())
def gated_delta_update(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    beta_t: torch.Tensor,
    g_t: torch.Tensor,
    recurrent_state: torch.Tensor,
    valid_len: torch.Tensor | None = None,
    mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reference PyTorch implementation of Gated Delta Update."""
  del mode
  batch_size, num_v_heads, _, _ = v_t.shape
  if valid_len is not None and valid_len.shape != (batch_size,):
    raise ValueError(
        f"valid_len must be a 1D tensor of shape [{batch_size}], got"
        f" {tuple(valid_len.shape)}"
    )
  num_k_heads = q_t.shape[1]
  if num_v_heads != num_k_heads:
    g_ratio = num_v_heads // num_k_heads
    q_t = q_t.repeat_interleave(g_ratio, dim=1)
    k_t = k_t.repeat_interleave(g_ratio, dim=1)

  return recurrent_gated_delta_rule(
      q_t, k_t, v_t, beta_t, g_t, recurrent_state, valid_len=valid_len
  )


@gated_delta_update.register_fake
def _gated_delta_update_fake(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    beta_t: torch.Tensor,
    g_t: torch.Tensor,
    recurrent_state: torch.Tensor,
    valid_len: torch.Tensor | None = None,
    mode: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Fake implementation for shape inference."""
  del k_t, beta_t, g_t, mode
  batch_size, num_v_heads, seq_len, head_v_dim = v_t.shape
  if valid_len is not None and valid_len.shape != (batch_size,):
    raise ValueError(
        f"valid_len must be a 1D tensor of shape [{batch_size}], got"
        f" {tuple(valid_len.shape)}"
    )
  out1 = torch.empty(
      (batch_size, num_v_heads, seq_len, head_v_dim),
      dtype=q_t.dtype,
      device=q_t.device,
  )
  out2 = torch.empty(
      recurrent_state.shape,
      dtype=torch.float32,
      device=recurrent_state.device,
  )
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
    valid_len: ir.Value | None = None,
    mode: int = 0,
):
  """Lowers gdn_attention to tfl.custom."""
  operands = [q_t, k_t, v_t, beta_t, g_t, recurrent_state]
  if valid_len is not None:
    operands.append(valid_len)
  op = ir.Operation.create(
      "tfl.custom",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=operands,
      attributes={
          "custom_code": ir.StringAttr.get("gated_delta_update"),
          "custom_option": _const_bytes_attr(
              _gated_delta_update_custom_options(mode=mode)
          ),
      },
  )
  return tuple(op.results)


def apply_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    recurrent_state: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    valid_len: torch.Tensor | None = None,
    *,
    use_fused_gdn: bool = True,
    gdn_mode: int = 0,
    use_fp32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Dispatches Gated Delta Rule update to custom op or decomposed PyTorch implementation."""
  if use_fused_gdn:
    initial_dtype = query.dtype
    compute_dtype = torch.float32 if use_fp32 else initial_dtype
    q_t, k_t, v_t, beta_t, g_t = [
        x.transpose(1, 2).contiguous().to(compute_dtype)
        for x in (query, key, value, beta, g)
    ]
    return gated_delta_update(
        q_t,
        k_t,
        v_t,
        beta_t,
        g_t,
        recurrent_state.to(torch.float32),
        valid_len=valid_len,
        mode=gdn_mode,
    )

  batch_size, seq_len, num_k_heads, head_k_dim = query.shape
  num_v_heads = value.shape[2]
  g_ratio = num_v_heads // num_k_heads
  if g_ratio > 1:
    # Concat, not repeat_interleave: the latter lowers to BROADCAST_TO, which the
    # ML Drift GPU delegate does not support (forces a CPU partition).
    def _repeat_heads(x):
      x = x.reshape(batch_size * seq_len, num_k_heads, 1, head_k_dim)
      x = torch.cat([x] * g_ratio, dim=2)
      return x.reshape(batch_size, seq_len, num_v_heads, head_k_dim)

    query = _repeat_heads(query)
    key = _repeat_heads(key)

  q_t, k_t, v_t, beta_t, g_t = [
      x.transpose(1, 2).contiguous().to(torch.float32)
      for x in (query, key, value, beta, g)
  ]
  chunk_size = 64
  if seq_len > 1 and gdn_mode == 1 and seq_len >= chunk_size:
    return chunk_gated_delta_rule(
        q_t, k_t, v_t, beta_t, g_t, recurrent_state, chunk_size=chunk_size
    )
  return recurrent_gated_delta_rule(
      q_t,
      k_t,
      v_t,
      beta_t,
      g_t,
      recurrent_state,
      valid_mask=valid_mask,
      valid_len=valid_len,
  )
