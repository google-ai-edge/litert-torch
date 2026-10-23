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
"""Patches for Flint model for on-device deployment."""

import contextlib

# Registers torch.ops.tfl.*
from litert_torch.backend.torch_tfl import _ops as _
from litert_torch.generative.export_hf.experimental.composites import qkv_norm_rope
from litert_torch.generative.export_hf.experimental.composites import rope as rope_composite
from litert_torch.generative.export_hf.experimental.composites import swiglu as swiglu_composite
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.flint import configuration_flint
from litert_torch.generative.export_hf.model_ext.flint import modeling_flint
from litert_torch.generative.layers import normalization
import torch
import torch.nn.functional as F
import transformers


def _register_flint_with_transformers():
  """Registers FlintConfig, FlintModel, and FlintForCausalLM with Auto* classes."""
  try:
    transformers.AutoConfig.register(
        "flint", configuration_flint.FlintConfig, exist_ok=True
    )
  except TypeError:
    try:
      transformers.AutoConfig.register("flint", configuration_flint.FlintConfig)
    except ValueError:
      pass
  for auto_cls, model_cls in (
      (transformers.AutoModel, modeling_flint.FlintModel),
      (transformers.AutoModelForCausalLM, modeling_flint.FlintForCausalLM),
  ):
    try:
      auto_cls.register(
          configuration_flint.FlintConfig, model_cls, exist_ok=True
      )
    except TypeError:
      try:
        auto_cls.register(configuration_flint.FlintConfig, model_cls)
      except ValueError:
        pass
  try:
    from transformers import conversion_mapping  # pylint: disable=g-import-not-at-top

    conversion_mapping.register_checkpoint_conversion_mapping(
        "flint",
        [
            conversion_mapping.WeightRenaming(
                source_patterns=r"self_attn\.softmax_offset$",
                target_patterns="self_attn.sinks",
            ),
            conversion_mapping.WeightConverter(
                source_patterns=[
                    "mlp.experts.*.gate_proj.weight",
                    "mlp.experts.*.up_proj.weight",
                ],
                target_patterns="mlp.experts.gate_up_proj",
                operations=[
                    conversion_mapping.MergeModulelist(dim=0),
                    conversion_mapping.Concatenate(dim=1),
                ],
            ),
            conversion_mapping.WeightConverter(
                source_patterns="mlp.experts.*.down_proj.weight",
                target_patterns="mlp.experts.down_proj",
                operations=[conversion_mapping.MergeModulelist(dim=0)],
            ),
        ],
        overwrite=True,
    )
  except Exception:  # pylint: disable=broad-except
    pass


_register_flint_with_transformers()


class FlintRMSNorm(torch.nn.Module):
  """LiteRT RMSNorm Layer for Flint."""

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.hidden_size = hidden_size

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    original_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    return normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight,
        self.variance_epsilon,
        torch.ones(
            (self.hidden_size,),
            dtype=torch.float32,
            device=hidden_states.device,
        ),
    ).to(original_dtype)

  def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LiteRTFlintTopKRouter(torch.nn.Module):
  """LiteRT compatible Flint Top-K Sigmoid Router."""

  def __init__(self, config):
    super().__init__()
    self.top_k = config.num_experts_per_tok
    self.num_experts = config.num_experts
    self.norm_topk_prob = config.norm_topk_prob
    self.hidden_dim = config.hidden_size
    self.weight = torch.nn.Parameter(
        torch.zeros(self.num_experts, self.hidden_dim)
    )
    # Set by `moe.bind_per_expert_scale` when experts are lowered to the `moe`
    # custom op so top-k indices stay in int32 end-to-end.
    self.use_int32_indices = False

  def forward(self, hidden_states: torch.Tensor):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states.float(), self.weight.float())
    scores = router_logits.sigmoid()
    if self.use_int32_indices:
      routing_weights, selected_experts = torch.ops.tfl.topk_v2(
          scores, self.top_k
      )
    else:
      routing_weights, selected_experts = torch.topk(scores, self.top_k, dim=-1)
    if self.norm_topk_prob:
      routing_weights = routing_weights / routing_weights.sum(
          dim=-1, keepdim=True
      )
    return router_logits, routing_weights, selected_experts


class FusedFlintMLP(torch.nn.Module):
  """Fused Gate + Up MLP layer for Flint."""

  def __init__(
      self,
      original_mlp: modeling_flint.FlintMLP,
      use_swiglu_composite: bool = False,
  ):
    super().__init__()
    self.config = original_mlp.config
    self.hidden_size = original_mlp.hidden_size
    self.intermediate_size = original_mlp.intermediate_size
    self.gate_proj = original_mlp.gate_proj
    self.up_proj = original_mlp.up_proj
    self.down_proj = original_mlp.down_proj
    self.act_fn = original_mlp.act_fn
    self.use_swiglu_composite = use_swiglu_composite

    gate_out_features = self.gate_proj.out_features
    up_out_features = self.up_proj.out_features
    param_device = self.gate_proj.weight.device
    param_dtype = self.gate_proj.weight.dtype

    self.gate_up_proj = torch.nn.Linear(
        self.gate_proj.in_features,
        gate_out_features + up_out_features,
        bias=self.gate_proj.bias is not None,
        device=param_device,
        dtype=param_dtype,
    )

    if param_device.type != "meta":
      with torch.no_grad():
        self.gate_up_proj.weight.copy_(
            torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        )
        if self.gate_up_proj.bias is not None:
          self.gate_up_proj.bias.copy_(
              torch.cat([self.gate_proj.bias, self.up_proj.bias], dim=0)
          )

    self.gate_size = gate_out_features

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_up = self.gate_up_proj(x)
    if self.use_swiglu_composite:
      return self.down_proj(
          swiglu_composite.apply_swiglu(
              gate_up,
              gate_size=self.gate_size,
              activation="silu",
          )
      )
    gate, up = gate_up.split(
        [self.gate_size, gate_up.shape[-1] - self.gate_size], dim=-1
    )
    return self.down_proj(self.act_fn(gate) * up)


class FusedFlintAttention(torch.nn.Module):
  """Fused QKV Attention layer for Flint."""

  def __init__(
      self,
      original_attn: modeling_flint.FlintAttention,
      fuse_qkv: bool = False,
      use_rope_composite: bool = False,
      use_qkv_norm_rope_composite: bool = False,
  ):
    super().__init__()
    self.config = original_attn.config
    self.layer_idx = original_attn.layer_idx
    self.head_dim = original_attn.head_dim
    self.num_heads = self.config.num_attention_heads
    self.num_key_value_heads = self.config.num_key_value_heads
    self.num_key_value_groups = original_attn.num_key_value_groups
    self.scaling = original_attn.scaling
    self.attention_dropout = original_attn.attention_dropout
    self.is_causal = original_attn.is_causal
    self.sliding_window = original_attn.sliding_window

    self.q_proj = original_attn.q_proj
    self.k_proj = original_attn.k_proj
    self.v_proj = original_attn.v_proj
    self.o_proj = original_attn.o_proj
    self.q_norm = original_attn.q_norm
    self.k_norm = original_attn.k_norm
    self.sinks = original_attn.sinks

    self.use_qkv_norm_rope_composite = use_qkv_norm_rope_composite
    self.fuse_qkv = fuse_qkv or use_qkv_norm_rope_composite
    self.use_rope_composite = use_rope_composite

    self.q_size = self.q_proj.out_features
    self.k_size = self.k_proj.out_features
    self.v_size = self.v_proj.out_features

    if self.fuse_qkv:
      param_device = self.q_proj.weight.device
      param_dtype = self.q_proj.weight.dtype
      self.qkv_proj = torch.nn.Linear(
          self.q_proj.in_features,
          self.q_size + self.k_size + self.v_size,
          bias=self.q_proj.bias is not None,
          device=param_device,
          dtype=param_dtype,
      )
      if param_device.type != "meta":
        with torch.no_grad():
          self.qkv_proj.weight.copy_(
              torch.cat(
                  [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
                  dim=0,
              )
          )
          if self.qkv_proj.bias is not None:
            self.qkv_proj.bias.copy_(
                torch.cat(
                    [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias],
                    dim=0,
                )
            )

  def _get_rope_base(self) -> float:
    rope_base = 10000.0
    if hasattr(self.config, "rope_parameters") and self.config.rope_parameters:
      if isinstance(self.config.rope_parameters, dict):
        val = self.config.rope_parameters.get("rope_theta", rope_base)
        if val is not None:
          rope_base = float(val)
      elif hasattr(self.config.rope_parameters, "rope_theta"):
        rope_base = float(
            getattr(self.config.rope_parameters, "rope_theta", rope_base)
        )
    elif hasattr(self.config, "rope_theta"):
      rope_base = float(getattr(self.config, "rope_theta", rope_base))
    return rope_base

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple[torch.Tensor, torch.Tensor],
      attention_mask: torch.Tensor | None,
      past_key_values=None,
      **kwargs,
  ):
    input_shape = hidden_states.shape[:-1]
    hidden_shape_q = (*input_shape, self.num_heads, self.head_dim)
    hidden_shape_kv = (*input_shape, self.num_key_value_heads, self.head_dim)
    has_rope = self.sliding_window is not None

    if (
        has_rope
        and getattr(self, "use_qkv_norm_rope_composite", False)
        and self.fuse_qkv
    ):
      position_ids = kwargs.get("position_ids", None)
      if position_ids is None:
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(
            seq_len, device=hidden_states.device
        ).unsqueeze(0)
      rope_base = self._get_rope_base()
      qkv = self.qkv_proj(hidden_states)
      query_states, key_states, value_states = (
          qkv_norm_rope.apply_qkv_norm_rope(
              qkv,
              position_ids,
              self.q_norm.weight,
              self.k_norm.weight,
              num_heads=self.num_heads,
              num_kv_heads=self.num_key_value_heads,
              head_dim=self.head_dim,
              base=rope_base,
              eps=float(self.q_norm.variance_epsilon),
              has_v_norm=False,
          )
      )
    else:
      if self.fuse_qkv:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
      else:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

      query_states = self.q_norm(q.view(hidden_shape_q)).transpose(1, 2)
      key_states = self.k_norm(k.view(hidden_shape_kv)).transpose(1, 2)
      value_states = v.view(hidden_shape_kv).transpose(1, 2)

      # Only sliding-window layers apply RoPE; full-attention layers use no
      # positional encoding.
      if has_rope:
        if getattr(self, "use_rope_composite", False):
          position_ids = kwargs.get("position_ids", None)
          if position_ids is None:
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(
                seq_len, device=hidden_states.device
            ).unsqueeze(0)
          rope_base = self._get_rope_base()
          query_states = rope_composite.apply_mldrift_compatible_rope(
              query_states, position_ids, base=rope_base, head_dim=self.head_dim
          )
          key_states = rope_composite.apply_mldrift_compatible_rope(
              key_states, position_ids, base=rope_base, head_dim=self.head_dim
          )
        else:
          cos, sin = position_embeddings
          query_states, key_states = modeling_flint.apply_rotary_pos_emb(
              query_states, key_states, cos, sin
          )

    if past_key_values is not None:
      key_states, value_states = past_key_values.update(
          key_states, value_states, self.layer_idx
      )

    attention_interface = modeling_flint.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,  # pylint: disable=protected-access
        modeling_flint.eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        s_aux=self.sinks,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@patches_lib.register_patch(["flint"])
@contextlib.contextmanager
def flint_litert_patch():
  """Flint static class patch."""
  original_norm = modeling_flint.FlintRMSNorm
  modeling_flint.FlintRMSNorm = FlintRMSNorm

  original_router = modeling_flint.FlintTopKRouter
  modeling_flint.FlintTopKRouter = LiteRTFlintTopKRouter

  try:
    yield
  finally:
    modeling_flint.FlintRMSNorm = original_norm
    modeling_flint.FlintTopKRouter = original_router


@patches_lib.register_model_patch(["flint"])
@contextlib.contextmanager
def patch_flint_model(model, export_config):
  """Dynamic model patch for Flint export."""
  fuse_gate_up = getattr(export_config, "fuse_gate_up", False)
  fuse_qkv = getattr(export_config, "fuse_qkv", False)
  use_rope = getattr(export_config, "use_rope_composite", False)
  use_qkv_norm_rope = getattr(
      export_config, "use_qkv_norm_rope_composite", False
  )
  use_swiglu = getattr(export_config, "use_swiglu_composite", False)

  replaced_modules = []

  def replace_modules(module):
    for child_name, child in module.named_children():
      if (fuse_gate_up or use_swiglu) and isinstance(
          child, modeling_flint.FlintMLP
      ):
        fused = FusedFlintMLP(child, use_swiglu_composite=use_swiglu)
        setattr(module, child_name, fused)
        replaced_modules.append((module, child_name, child))
      elif isinstance(child, modeling_flint.FlintAttention):
        if fuse_qkv or use_rope or use_qkv_norm_rope:
          fused = FusedFlintAttention(
              child,
              fuse_qkv=fuse_qkv or use_qkv_norm_rope,
              use_rope_composite=use_rope,
              use_qkv_norm_rope_composite=use_qkv_norm_rope,
          )
          setattr(module, child_name, fused)
          replaced_modules.append((module, child_name, child))
      else:
        replace_modules(child)

  replace_modules(model)
  try:
    yield
  finally:
    for module, name, original in reversed(replaced_modules):
      setattr(module, name, original)
