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
"""TFLite custom op wrapper for LiteRT MoE expert dispatch."""

from __future__ import annotations

from flatbuffers import flexbuffers
from litert_torch.backend import lowerings
from litert_torch.backend.lowerings import utils as lowering_utils
from litert_converter.mlir import ir
import torch
from torch.nn import functional as F

_MOE_CUSTOM_OP_NAME = "moe"


def flatten_expert_weight(weight: torch.Tensor) -> torch.Tensor:
  """Converts [experts, out_channels, in_channels] to OHWI delegate layout."""
  if weight.dim() != 3:
    raise ValueError(
        "MoE expert weights must have shape "
        "[num_experts, out_channels, in_channels]."
    )
  num_experts, out_channels, in_channels = weight.shape
  return (
      weight.permute(1, 0, 2)
      .contiguous()
      .reshape(out_channels, num_experts, 1, in_channels)
  )


def flatten_expert_scale(scale: torch.Tensor) -> torch.Tensor:
  """Converts [experts, out_channels] scales to OHWI delegate layout."""
  if scale.dim() != 2:
    raise ValueError(
        "MoE expert quantization scales must have shape "
        "[num_experts, out_channels]."
    )
  return (
      scale.transpose(0, 1)
      .contiguous()
      .reshape(scale.shape[1], scale.shape[0], 1, 1)
  )


def _restore_src_rank(src: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
  if src.dim() == 4:
    return output.reshape(src.shape)
  return output.reshape(src.shape)


def _expand_weight(
    weight: torch.Tensor,
    num_experts: int,
    out_channels: int,
    in_channels: int,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
  weight = (
      weight.reshape(out_channels, num_experts, 1, in_channels)[:, :, 0, :]
      .permute(1, 0, 2)
      .to(torch.float32)
  )
  if scale is not None:
    scale = (
        scale.reshape(out_channels, num_experts, 1, 1)[:, :, 0, 0]
        .permute(1, 0)
        .reshape(num_experts, out_channels, 1)
        .to(torch.float32)
    )
    weight = weight * scale
  return weight


def _moe_experts_reference(
    src: torch.Tensor,
    top_weights: torch.Tensor,
    top_indices: torch.Tensor,
    ff_gate_weight: torch.Tensor,
    ff1_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    per_expert_scale: torch.Tensor,
    num_experts: int,
    num_active_experts: int,
    model_dim: int,
    hidden_dim: int,
    ff_gate_scale: torch.Tensor | None = None,
    ff1_scale: torch.Tensor | None = None,
    linear_scale: torch.Tensor | None = None,
) -> torch.Tensor:
  src_flat = src.reshape(1, -1, model_dim).to(torch.float32)
  weights_flat = top_weights.reshape(1, -1, num_active_experts).to(
      torch.float32
  )
  indices = top_indices.reshape(1, -1, num_active_experts).to(torch.long)

  gate_w = _expand_weight(
      ff_gate_weight, num_experts, hidden_dim, model_dim, ff_gate_scale
  )
  ff1_w = _expand_weight(
      ff1_weight, num_experts, hidden_dim, model_dim, ff1_scale
  )
  linear_w = _expand_weight(
      linear_weight, num_experts, model_dim, hidden_dim, linear_scale
  )

  selected_gate = gate_w[indices]
  selected_ff1 = ff1_w[indices]
  selected_linear = linear_w[indices]
  gate = torch.einsum("bskhd,bsd->bskh", selected_gate, src_flat)
  ff1 = torch.einsum("bskhd,bsd->bskh", selected_ff1, src_flat)
  hidden = F.gelu(gate) * ff1
  expert_output = torch.einsum("bskdh,bskh->bskd", selected_linear, hidden)
  scale = per_expert_scale.reshape(num_experts)[indices].to(torch.float32)
  expert_output = expert_output * scale.unsqueeze(-1)
  output = (expert_output * weights_flat.unsqueeze(-1)).sum(dim=2)
  return _restore_src_rank(src, output.to(src.dtype))


def _moe_custom_options(
    *,
    num_experts: int,
    num_active_experts: int,
    model_dim: int,
    hidden_dim: int,
    weight_type: str,
) -> bytes:
  return bytes(
      flexbuffers.Dumps({
          "num_experts": num_experts,
          "num_active_experts": num_active_experts,
          "model_dim": model_dim,
          "hidden_dim": hidden_dim,
          "activation": "gelu",
          "weight_type": weight_type,
          "renormalized_top_weights": True,
      })
  )


def _const_bytes_attr(data: bytes) -> ir.Attribute:
  return ir.Attribute.parse(f'#tfl<const_bytes: "0x{data.hex()}">')


def _tfl_custom_moe(
    lctx,
    operands: list[ir.Value],
    *,
    num_experts: int,
    num_active_experts: int,
    model_dim: int,
    hidden_dim: int,
    weight_type: str,
) -> ir.Value:
  op = ir.Operation.create(
      "tfl.custom",
      results=lowering_utils.node_meta_to_ir_types(lctx.node),
      operands=operands,
      attributes={
          "custom_code": ir.StringAttr.get(_MOE_CUSTOM_OP_NAME),
          "custom_option": _const_bytes_attr(
              _moe_custom_options(
                  num_experts=num_experts,
                  num_active_experts=num_active_experts,
                  model_dim=model_dim,
                  hidden_dim=hidden_dim,
                  weight_type=weight_type,
              )
          ),
      },
  )
  return op.results[0]


@torch.library.custom_op("litert_torch::moe_fp32", mutates_args=())
def _moe_fp32(
    src: torch.Tensor,
    top_weights: torch.Tensor,
    top_indices: torch.Tensor,
    ff_gate_weight: torch.Tensor,
    ff1_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    per_expert_scale: torch.Tensor,
    num_experts: int,
    num_active_experts: int,
    model_dim: int,
    hidden_dim: int,
) -> torch.Tensor:
  return _moe_experts_reference(
      src,
      top_weights,
      top_indices,
      ff_gate_weight,
      ff1_weight,
      linear_weight,
      per_expert_scale,
      num_experts,
      num_active_experts,
      model_dim,
      hidden_dim,
  )


@_moe_fp32.register_fake
def _(
    src,
    top_weights,
    top_indices,
    ff_gate_weight,
    ff1_weight,
    linear_weight,
    per_expert_scale,
    num_experts,
    num_active_experts,
    model_dim,
    hidden_dim,
):
  del (
      top_weights,
      top_indices,
      ff_gate_weight,
      ff1_weight,
      linear_weight,
      per_expert_scale,
      num_experts,
      num_active_experts,
      model_dim,
      hidden_dim,
  )
  return torch.empty_like(src)


@lowerings.lower(torch.ops.litert_torch.moe_fp32)
def _moe_fp32_lower(
    lctx,
    src: ir.Value,
    top_weights: ir.Value,
    top_indices: ir.Value,
    ff_gate_weight: ir.Value,
    ff1_weight: ir.Value,
    linear_weight: ir.Value,
    per_expert_scale: ir.Value,
    num_experts: int,
    num_active_experts: int,
    model_dim: int,
    hidden_dim: int,
):
  return _tfl_custom_moe(
      lctx,
      [
          src,
          top_weights,
          top_indices,
          ff_gate_weight,
          ff1_weight,
          linear_weight,
          per_expert_scale,
      ],
      num_experts=num_experts,
      num_active_experts=num_active_experts,
      model_dim=model_dim,
      hidden_dim=hidden_dim,
      weight_type="fp32",
  )


@torch.library.custom_op("litert_torch::moe_int8", mutates_args=())
def _moe_int8(
    src: torch.Tensor,
    top_weights: torch.Tensor,
    top_indices: torch.Tensor,
    ff_gate_weight: torch.Tensor,
    ff_gate_scale: torch.Tensor,
    ff1_weight: torch.Tensor,
    ff1_scale: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_scale: torch.Tensor,
    per_expert_scale: torch.Tensor,
    num_experts: int,
    num_active_experts: int,
    model_dim: int,
    hidden_dim: int,
) -> torch.Tensor:
  return _moe_experts_reference(
      src,
      top_weights,
      top_indices,
      ff_gate_weight,
      ff1_weight,
      linear_weight,
      per_expert_scale,
      num_experts,
      num_active_experts,
      model_dim,
      hidden_dim,
      ff_gate_scale=ff_gate_scale,
      ff1_scale=ff1_scale,
      linear_scale=linear_scale,
  )


@_moe_int8.register_fake
def _(
    src,
    top_weights,
    top_indices,
    ff_gate_weight,
    ff_gate_scale,
    ff1_weight,
    ff1_scale,
    linear_weight,
    linear_scale,
    per_expert_scale,
    num_experts,
    num_active_experts,
    model_dim,
    hidden_dim,
):
  del (
      top_weights,
      top_indices,
      ff_gate_weight,
      ff_gate_scale,
      ff1_weight,
      ff1_scale,
      linear_weight,
      linear_scale,
      per_expert_scale,
      num_experts,
      num_active_experts,
      model_dim,
      hidden_dim,
  )
  return torch.empty_like(src)


@lowerings.lower(torch.ops.litert_torch.moe_int8)
def _moe_int8_lower(
    lctx,
    src: ir.Value,
    top_weights: ir.Value,
    top_indices: ir.Value,
    ff_gate_weight: ir.Value,
    ff_gate_scale: ir.Value,
    ff1_weight: ir.Value,
    ff1_scale: ir.Value,
    linear_weight: ir.Value,
    linear_scale: ir.Value,
    per_expert_scale: ir.Value,
    num_experts: int,
    num_active_experts: int,
    model_dim: int,
    hidden_dim: int,
):
  return _tfl_custom_moe(
      lctx,
      [
          src,
          top_weights,
          top_indices,
          ff_gate_weight,
          ff_gate_scale,
          ff1_weight,
          ff1_scale,
          linear_weight,
          linear_scale,
          per_expert_scale,
      ],
      num_experts=num_experts,
      num_active_experts=num_active_experts,
      model_dim=model_dim,
      hidden_dim=hidden_dim,
      weight_type="int8",
  )


def moe_experts(
    src: torch.Tensor,
    top_weights: torch.Tensor,
    top_indices: torch.Tensor,
    ff_gate_weight: torch.Tensor,
    ff1_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    per_expert_scale: torch.Tensor,
    *,
    num_experts: int,
    num_active_experts: int,
    model_dim: int,
    hidden_dim: int,
    weight_type: str = "fp32",
    ff_gate_scale: torch.Tensor | None = None,
    ff1_scale: torch.Tensor | None = None,
    linear_scale: torch.Tensor | None = None,
) -> torch.Tensor:
  """Builds the `moe` TFLite custom op."""
  if weight_type == "fp32":
    return _moe_fp32(
        src,
        top_weights,
        top_indices,
        ff_gate_weight,
        ff1_weight,
        linear_weight,
        per_expert_scale,
        num_experts,
        num_active_experts,
        model_dim,
        hidden_dim,
    )
  if weight_type == "int8":
    if ff_gate_scale is None or ff1_scale is None or linear_scale is None:
      raise ValueError("int8 MoE experts requires all weight scale tensors.")
    return _moe_int8(
        src,
        top_weights,
        top_indices,
        ff_gate_weight,
        ff_gate_scale,
        ff1_weight,
        ff1_scale,
        linear_weight,
        linear_scale,
        per_expert_scale,
        num_experts,
        num_active_experts,
        model_dim,
        hidden_dim,
    )
  raise ValueError(f"Unsupported MoE expert weight_type: {weight_type}")


def litert_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
  gate_weight, ff1_weight = self.gate_up_proj.chunk(2, dim=1)
  per_expert_scale = torch.ones(
      (1, 1, 1, self.num_experts), dtype=torch.float32
  )
  output = moe_experts(
      hidden_states.reshape(1, -1, self.hidden_dim),
      top_k_weights.reshape(1, -1, self.config.top_k_experts),
      top_k_index.reshape(1, -1, self.config.top_k_experts).to(torch.int32),
      flatten_expert_weight(gate_weight),
      flatten_expert_weight(ff1_weight),
      flatten_expert_weight(self.down_proj),
      per_expert_scale,
      num_experts=self.num_experts,
      num_active_experts=self.config.top_k_experts,
      model_dim=self.hidden_dim,
      hidden_dim=self.intermediate_dim,
  )
  return output.reshape(hidden_states.shape)
