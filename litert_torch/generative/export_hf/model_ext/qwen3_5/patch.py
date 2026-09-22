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
"""Patches for Qwen3.5 model."""

import contextlib
from typing import Any
from litert_torch.generative.export_hf.experimental.composites import rope as rope_composite
from litert_torch.generative.export_hf.experimental.composites import swiglu as swiglu_composite
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.layers import normalization
import torch
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5 import modular_qwen3_5


class Qwen3_5RMSNorm(torch.nn.Module):
  """Fused RMSNorm Layer for Qwen3.5 with pre-shifted (1 + weight) scaling."""

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.ones(dim))
    self.hidden_size = dim

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    dtype = hidden_states.dtype
    hidden_states_fp32 = hidden_states.to(torch.float32)
    return normalization.rms_norm_with_hlfb(
        hidden_states_fp32,
        self.weight,
        self.eps,
        torch.ones(
            (self.hidden_size,),
            dtype=torch.float32,
            device=hidden_states.device,
        ),
    ).to(dtype)

  def extra_repr(self) -> str:
    return f"{tuple(self.weight.shape)}, eps={self.eps}"


class FusedQwen3_5MLP(torch.nn.Module):
  """Fused Gate-Up MLP Layer for Qwen3.5 model."""

  def __init__(
      self,
      original_mlp: (
          modeling_qwen3_5.Qwen3_5MLP | modular_qwen3_5.Qwen3_5MLP
      ),
      use_swiglu_composite: bool = False,
  ):
    super().__init__()
    self.config = getattr(original_mlp, "config", None)
    self.hidden_size = getattr(original_mlp, "hidden_size", 0)
    self.intermediate_size = getattr(original_mlp, "intermediate_size", 0)
    self.act_fn = original_mlp.act_fn
    self.down_proj = original_mlp.down_proj
    self.use_swiglu_composite = use_swiglu_composite

    gate_w = original_mlp.gate_proj.weight.data
    up_w = original_mlp.up_proj.weight.data
    fused_weight = torch.cat([gate_w, up_w], dim=0)

    self.gate_up_proj = torch.nn.Linear(
        self.hidden_size,
        2 * self.intermediate_size,
        bias=False,
        device=fused_weight.device,
        dtype=fused_weight.dtype,
    )
    self.gate_up_proj.weight = torch.nn.Parameter(fused_weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_up = self.gate_up_proj(x)
    if self.use_swiglu_composite:
      return self.down_proj(swiglu_composite.apply_swiglu(gate_up))
    gate, up = gate_up.chunk(2, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


class FusedQwen3_5Attention(torch.nn.Module):
  """Fused QKV Attention Layer for Qwen3.5 model (with gated query projection)."""

  def __init__(
      self,
      original_attn: modeling_qwen3_5.Qwen3_5Attention,
      fuse_qkv: bool = True,
      use_rope_composite: bool = False,
  ):
    super().__init__()
    self.config = original_attn.config
    self.layer_idx = getattr(original_attn, "layer_idx", 0)
    self.head_dim = getattr(
        original_attn,
        "head_dim",
        getattr(original_attn.config, "head_dim", 128),
    )
    self.num_heads = getattr(
        original_attn,
        "num_heads",
        getattr(original_attn.config, "num_attention_heads", 16),
    )
    self.num_key_value_heads = getattr(
        original_attn,
        "num_key_value_heads",
        getattr(original_attn.config, "num_key_value_heads", 8),
    )
    self.num_key_value_groups = getattr(
        original_attn,
        "num_key_value_groups",
        getattr(original_attn.config, "num_key_value_groups", 2),
    )
    self.scaling = getattr(original_attn, "scaling", self.head_dim**-0.5)
    self.is_causal = getattr(original_attn, "is_causal", True)
    self.attention_dropout = getattr(original_attn, "attention_dropout", 0.0)

    self.q_norm = original_attn.q_norm
    self.k_norm = original_attn.k_norm
    self.o_proj = original_attn.o_proj
    self.fuse_qkv = fuse_qkv
    self.use_rope_composite = use_rope_composite

    self.q_size = original_attn.q_proj.out_features
    self.k_size = original_attn.k_proj.out_features
    self.v_size = original_attn.v_proj.out_features

    if fuse_qkv:
      q_w = original_attn.q_proj.weight.data
      k_w = original_attn.k_proj.weight.data
      v_w = original_attn.v_proj.weight.data
      fused_weight = torch.cat([q_w, k_w, v_w], dim=0)
      has_bias = original_attn.q_proj.bias is not None

      total_out_dim = self.q_size + self.k_size + self.v_size
      self.qkv_proj = torch.nn.Linear(
          self.config.hidden_size,
          total_out_dim,
          bias=has_bias,
          device=fused_weight.device,
          dtype=fused_weight.dtype,
      )
      self.qkv_proj.weight = torch.nn.Parameter(fused_weight)
      if has_bias:
        q_b = original_attn.q_proj.bias.data
        k_b = original_attn.k_proj.bias.data
        v_b = original_attn.v_proj.bias.data
        self.qkv_proj.bias = torch.nn.Parameter(torch.cat([q_b, k_b, v_b], dim=0))
    else:
      self.q_proj = original_attn.q_proj
      self.k_proj = original_attn.k_proj
      self.v_proj = original_attn.v_proj

  def _get_rope_params(self) -> tuple[float, float]:
    rope_base = 1000000.0
    partial_factor = 0.25
    rope_params = getattr(self.config, "rope_parameters", None)
    if isinstance(rope_params, dict):
      rope_base = float(rope_params.get("rope_theta", rope_base))
      partial_factor = float(
          rope_params.get("partial_rotary_factor", partial_factor)
      )
    elif rope_params is not None:
      rope_base = float(getattr(rope_params, "rope_theta", rope_base))
      partial_factor = float(
          getattr(rope_params, "partial_rotary_factor", partial_factor)
      )
    else:
      rope_base = float(getattr(self.config, "rope_theta", rope_base))
      partial_factor = float(
          getattr(self.config, "partial_rotary_factor", partial_factor)
      )
    return rope_base, partial_factor

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple[torch.Tensor, torch.Tensor],
      attention_mask: torch.Tensor | None = None,
      past_key_values: transformers.cache_utils.Cache | None = None,
      **kwargs: Any,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    if self.fuse_qkv:
      qkv = self.qkv_proj(hidden_states)
      q_raw, k_raw, v_raw = qkv.split(
          [self.q_size, self.k_size, self.v_size], dim=-1
      )
    else:
      q_raw = self.q_proj(hidden_states)
      k_raw = self.k_proj(hidden_states)
      v_raw = self.v_proj(hidden_states)

    query_states, gate = torch.chunk(
        q_raw.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(k_raw.view(hidden_shape)).transpose(1, 2)
    value_states = v_raw.view(hidden_shape).transpose(1, 2)

    if getattr(self, "use_rope_composite", False):
      position_ids = kwargs.get("position_ids", None)
      if position_ids is None:
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(
            seq_len, device=hidden_states.device
        ).unsqueeze(0)
      rope_base, partial_factor = self._get_rope_params()
      rotary_dim = int(self.head_dim * partial_factor)
      if rotary_dim == self.head_dim:
        query_states = rope_composite.apply_mldrift_compatible_rope(
            query_states, position_ids, base=rope_base, head_dim=self.head_dim
        )
        key_states = rope_composite.apply_mldrift_compatible_rope(
            key_states, position_ids, base=rope_base, head_dim=self.head_dim
        )
      else:
        q_rot, q_pass = (
            query_states[..., :rotary_dim],
            query_states[..., rotary_dim:],
        )
        k_rot, k_pass = (
            key_states[..., :rotary_dim],
            key_states[..., rotary_dim:],
        )
        q_rot = rope_composite.apply_mldrift_compatible_rope(
            q_rot, position_ids, base=rope_base, head_dim=rotary_dim
        )
        k_rot = rope_composite.apply_mldrift_compatible_rope(
            k_rot, position_ids, base=rope_base, head_dim=rotary_dim
        )
        query_states = torch.cat([q_rot, q_pass], dim=-1)
        key_states = torch.cat([k_rot, k_pass], dim=-1)
    else:
      cos, sin = position_embeddings
      query_states, key_states = modeling_qwen3_5.apply_rotary_pos_emb(
          query_states, key_states, cos, sin
      )

    if past_key_values is not None:
      key_states, value_states = past_key_values.update(
          key_states, value_states, self.layer_idx
      )

    attention_interface = modeling_qwen3_5.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        modeling_qwen3_5.eager_attention_forward,
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def apply_qwen3_5_model_patches(
    model: torch.nn.Module, export_config: Any
) -> list[tuple[torch.nn.Module, str, torch.nn.Module]]:
  """Replaces Qwen3.5 submodules in-place with fused / composite equivalents."""
  fuse_gate_up = getattr(export_config, "fuse_gate_up", False)
  fuse_qkv = getattr(export_config, "fuse_qkv", False)
  use_rope = getattr(export_config, "use_rope_composite", False)
  use_swiglu = getattr(export_config, "use_swiglu_composite", False)
  extra_kwargs = getattr(export_config, "extra_kwargs", None) or {}
  use_fused_gdn = extra_kwargs.get(
      "use_fused_gdn", getattr(export_config, "use_fused_gdn", True)
  )
  gdn_mode = extra_kwargs.get(
      "gdn_mode", getattr(export_config, "gdn_mode", 0)
  )
  gdn_use_fp32 = extra_kwargs.get(
      "gdn_use_fp32", getattr(export_config, "gdn_use_fp32", True)
  )

  replaced_modules: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []

  def replace_modules(module: torch.nn.Module) -> None:
    if hasattr(module, "use_fused_gdn") and hasattr(module, "gdn_mode"):
      module.use_fused_gdn = bool(use_fused_gdn)
      module.gdn_mode = int(gdn_mode)
    if hasattr(module, "use_fp32"):
      module.use_fp32 = bool(gdn_use_fp32)
    for child_name, child in module.named_children():
      if isinstance(
          child,
          (modeling_qwen3_5.Qwen3_5RMSNorm, modular_qwen3_5.Qwen3_5RMSNorm),
      ):
        dim = child.weight.shape[0]
        fused_norm = Qwen3_5RMSNorm(dim, eps=child.eps)
        with torch.no_grad():
          fused_norm.weight = torch.nn.Parameter(
              child.weight.to(torch.float32) + 1.0
          )
        setattr(module, child_name, fused_norm)
        replaced_modules.append((module, child_name, child))
      elif (fuse_gate_up or use_swiglu) and isinstance(
          child, (modeling_qwen3_5.Qwen3_5MLP, modular_qwen3_5.Qwen3_5MLP)
      ):
        fused = FusedQwen3_5MLP(child, use_swiglu_composite=use_swiglu)
        setattr(module, child_name, fused)
        replaced_modules.append((module, child_name, child))
      elif isinstance(child, modeling_qwen3_5.Qwen3_5Attention):
        replace_modules(child)
        if fuse_qkv or use_rope:
          fused = FusedQwen3_5Attention(
              child,
              fuse_qkv=fuse_qkv,
              use_rope_composite=use_rope,
          )
          setattr(module, child_name, fused)
          replaced_modules.append((module, child_name, child))
      else:
        replace_modules(child)

  replace_modules(model)
  return replaced_modules


@patches_lib.register_patch(
    ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"]
)
@contextlib.contextmanager
def qwen3_5_litert_patch():
  """Qwen3.5 class-level patch."""
  yield


@patches_lib.register_model_patch(
    ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"]
)
@contextlib.contextmanager
def patch_qwen3_5_model(model: torch.nn.Module, export_config: Any):
  """Dynamic model patch for Qwen3.5 export."""
  # For Qwen3.5 export, Qwen3_5ExportableMixin wraps the source HF model into
  # Qwen3_5StaticForCausalLM.from_hf_model(hf_model) and then applies
  # apply_qwen3_5_model_patches to model.static_model. Mutating the source HF
  # model before from_hf_model() runs causes state_dict keys (gate_up_proj,
  # qkv_proj) to mismatch the unfused static model during load_state_dict().
  if type(model).__name__ == "Qwen3_5StaticForCausalLM":
    replaced_modules = apply_qwen3_5_model_patches(model, export_config)
  else:
    replaced_modules = []
  try:
    yield
  finally:
    for module, name, original in reversed(replaced_modules):
      setattr(module, name, original)
