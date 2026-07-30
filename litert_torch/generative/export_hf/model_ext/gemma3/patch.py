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
"""Patches for Gemma3."""

import contextlib
from litert_torch.generative.export_hf.experimental.composites import rope as rope_composite
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.layers import normalization
import torch
from transformers.models.gemma3 import modeling_gemma3


class Gemma3RMSNorm(torch.nn.Module):
  """RMSNorm Layer."""

  def __init__(self, dim: int, eps: float = 1e-6):
    """RMSNorm Layer."""
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(dim))
    self.variance_epsilon = eps
    self.hidden_size = dim

  def forward(self, hidden_states):
    return normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight + 1.0,
        self.variance_epsilon,
        torch.ones((self.hidden_size,), dtype=torch.float32),
    )

  def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class FusedGemma3MLP(torch.nn.Module):
  """Fused Gate + Up MLP layer."""

  def __init__(self, original_mlp: modeling_gemma3.Gemma3MLP):
    super().__init__()
    self.gate_proj = original_mlp.gate_proj
    self.up_proj = original_mlp.up_proj
    self.down_proj = original_mlp.down_proj
    self.act_fn = original_mlp.act_fn

    # Fuse gate and up projections
    gate_out_features = self.gate_proj.out_features
    up_out_features = self.up_proj.out_features

    self.gate_up_proj = torch.nn.Linear(
        self.gate_proj.in_features,
        gate_out_features + up_out_features,
        bias=self.gate_proj.bias is not None,
    )

    # Copy weights and biases
    with torch.no_grad():
      self.gate_up_proj.weight.copy_(
          torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
      )
      if self.gate_up_proj.bias is not None:
        self.gate_up_proj.bias.copy_(
            torch.cat([self.gate_proj.bias, self.up_proj.bias], dim=0)
        )

    self.gate_size = gate_out_features

  def forward(self, x):
    gate_up = self.gate_up_proj(x)
    gate, up = gate_up.split(
        [self.gate_size, gate_up.shape[-1] - self.gate_size], dim=-1
    )
    return self.down_proj(self.act_fn(gate) * up)


class FusedGemma3Attention(torch.nn.Module):
  """Fused Attention layer (Q + K + V)."""

  def __init__(
      self,
      original_attn: modeling_gemma3.Gemma3Attention,
      fuse_qkv: bool = False,
      use_rope_composite: bool = False,
  ):
    super().__init__()
    self.o_proj = original_attn.o_proj
    self.q_norm = original_attn.q_norm
    self.k_norm = original_attn.k_norm

    self.config = original_attn.config
    self.layer_idx = original_attn.layer_idx
    self.head_dim = original_attn.head_dim
    self.num_key_value_groups = original_attn.num_key_value_groups
    self.scaling = original_attn.scaling
    self.attention_dropout = original_attn.attention_dropout
    self.is_causal = original_attn.is_causal
    self.attn_logit_softcapping = original_attn.attn_logit_softcapping
    self.sliding_window = original_attn.sliding_window
    self.is_sliding = original_attn.is_sliding

    self.fuse_qkv = fuse_qkv
    self.use_rope_composite = use_rope_composite

    self.q_proj = original_attn.q_proj
    self.k_proj = original_attn.k_proj
    self.v_proj = original_attn.v_proj

    if self.fuse_qkv:
      q_out_features = self.q_proj.out_features
      k_out_features = self.k_proj.out_features
      v_out_features = self.v_proj.out_features

      self.qkv_proj = torch.nn.Linear(
          self.q_proj.in_features,
          q_out_features + k_out_features + v_out_features,
          bias=self.q_proj.bias is not None,
      )

      # Copy weights and biases
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
                  [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
              )
          )

      self.q_size = q_out_features
      self.k_size = k_out_features
      self.v_size = v_out_features

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: torch.Tensor = None,
      attention_mask: torch.Tensor | None = None,
      past_key_values=None,
      **kwargs,
  ):
    input_shape = hidden_states.shape[:-1]

    if self.fuse_qkv:
      qkv = self.qkv_proj(hidden_states)
      qkv_reshaped = qkv.view(*input_shape, -1, self.head_dim)
      num_q_heads = self.q_size // self.head_dim
      num_k_heads = self.k_size // self.head_dim
      q_view = qkv_reshaped[..., :num_q_heads, :]
      k_view = qkv_reshaped[..., num_q_heads : num_q_heads + num_k_heads, :]
      v_view = qkv_reshaped[..., num_q_heads + num_k_heads :, :]
    else:
      q = self.q_proj(hidden_states)
      k = self.k_proj(hidden_states)
      v = self.v_proj(hidden_states)
      q_view = q.view(*input_shape, -1, self.head_dim)
      k_view = k.view(*input_shape, -1, self.head_dim)
      v_view = v.view(*input_shape, -1, self.head_dim)

    query_states = q_view.transpose(1, 2)
    key_states = k_view.transpose(1, 2)
    value_states = v_view.transpose(1, 2)

    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    if getattr(self, "use_rope_composite", False):
      position_ids = kwargs.get("position_ids", None)
      if position_ids is None:
        seq_len = query_states.shape[2]
        position_ids = torch.arange(
            seq_len, device=query_states.device
        ).unsqueeze(0)

      rope_base = 500000.0
      if hasattr(self.config, "rope_parameters") and self.config.rope_parameters:
        if isinstance(self.config.rope_parameters, dict):
          rope_base = float(
              self.config.rope_parameters.get("rope_theta", rope_base)
          )
        elif hasattr(self.config.rope_parameters, "rope_theta"):
          rope_base = float(
              getattr(self.config.rope_parameters, "rope_theta", rope_base)
          )
      elif hasattr(self.config, "rope_theta"):
        rope_base = float(getattr(self.config, "rope_theta", rope_base))

      is_local = getattr(self, "is_sliding", False)
      num_local = getattr(self.config, "num_local_layers_per_global", 0)
      if num_local > 0 and (self.layer_idx + 1) % (num_local + 1) != 0:
        is_local = True
      elif hasattr(self.config, "layer_types") and self.config.layer_types:
        if (
            self.layer_idx < len(self.config.layer_types)
            and self.config.layer_types[self.layer_idx] == "sliding_attention"
        ):
          is_local = True

      if is_local:
        rope_base = float(
            getattr(
                self.config,
                "rope_local_base_freq",
                getattr(self.config, "local_rope_theta", 10000.0),
            )
        )

      query_states = rope_composite.apply_mldrift_compatible_rope(
          query_states, position_ids, base=rope_base, head_dim=self.head_dim
      )
      key_states = rope_composite.apply_mldrift_compatible_rope(
          key_states, position_ids, base=rope_base, head_dim=self.head_dim
      )
    else:
      cos, sin = position_embeddings
      query_states, key_states = modeling_gemma3.apply_rotary_pos_emb(
          query_states, key_states, cos, sin
      )

    if past_key_values is not None:
      key_states, value_states = past_key_values.update(
          key_states, value_states, self.layer_idx
      )

    # pytype: disable=attribute-error
    attention_interface = modeling_gemma3.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,  # pylint: disable=protected-access
        modeling_gemma3.eager_attention_forward,
    )
    # pytype: enable=attribute-error

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@patches_lib.register_patch(["gemma3", "gemma3_text"])
@contextlib.contextmanager
def gemma3_litert_patch():
  """Gemma3 patch."""
  print("Gemma3 patch applied.")
  original_norm = modeling_gemma3.Gemma3RMSNorm
  modeling_gemma3.Gemma3RMSNorm = Gemma3RMSNorm  # pyrefly: ignore[bad-assignment]

  try:
    yield
  finally:
    modeling_gemma3.Gemma3RMSNorm = original_norm


@patches_lib.register_model_patch(["gemma3", "gemma3_text"])
@contextlib.contextmanager
def patch_gemma3_model(model, export_config):
  """Dynamic model patch for Gemma3 export."""
  fuse_gate_up = export_config.fuse_gate_up
  fuse_qkv = export_config.fuse_qkv
  use_rope = export_config.use_rope_composite
  print(
      "Gemma3 model patch applied. "
      f"fuse_gate_up={fuse_gate_up}, fuse_qkv={fuse_qkv}, "
      f"use_rope_composite={use_rope}"
  )

  replaced_modules = []

  def replace_modules(module):
    for child_name, child in module.named_children():
      if fuse_gate_up and isinstance(child, modeling_gemma3.Gemma3MLP):
        print(f"Fusing MLP: {child_name}")
        fused = FusedGemma3MLP(child)
        setattr(module, child_name, fused)
        replaced_modules.append((module, child_name, child))
      elif isinstance(child, modeling_gemma3.Gemma3Attention):
        if fuse_qkv or use_rope:
          print(
              f"Replacing Attention: {child_name} "
              f"(fuse_qkv={fuse_qkv}, use_rope={use_rope})"
          )
          fused = FusedGemma3Attention(
              child,
              fuse_qkv=fuse_qkv,
              use_rope_composite=use_rope,
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
