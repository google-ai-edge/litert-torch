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
from litert_torch.backend.experimental.torch_tfl import _ops as _
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
  _, num_v_heads, seq_len, _ = v_t.shape
  num_k_heads = q_t.shape[1]
  if num_v_heads != num_k_heads:
    g_ratio = num_v_heads // num_k_heads
    q_t = q_t.repeat_interleave(g_ratio, dim=1)
    k_t = k_t.repeat_interleave(g_ratio, dim=1)

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
  batch_size, num_v_heads, seq_len, head_v_dim = v_t.shape
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

