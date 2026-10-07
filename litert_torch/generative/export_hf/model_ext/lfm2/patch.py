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
"""Patch for LFM2."""

import contextlib
from litert_torch.generative.export_hf.experimental.composites import qkv_norm_rope as qkv_norm_rope_composite
from litert_torch.generative.export_hf.experimental.composites import rope as rope_composite
from litert_torch.generative.export_hf.experimental.composites import swiglu as swiglu_composite
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.lfm2 import short_conv as short_conv_lib
from litert_torch.generative.layers import normalization
import torch
import transformers
from transformers.models.lfm2 import modeling_lfm2


class Lfm2RMSNorm(torch.nn.Module):
  """RMSNorm Layer with HLFB composite support for LFM2."""

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.hidden_size = hidden_size

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight,
        self.variance_epsilon,
        torch.ones(
            (self.hidden_size,),
            dtype=torch.float32,
            device=hidden_states.device,
        ),
    )

  def extra_repr(self):
    return f"{self.hidden_size}, eps={self.variance_epsilon}"


class FusedLfm2MLP(torch.nn.Module):
  """Fused Gate-Up MLP Layer for LFM2 model."""

  def __init__(
      self,
      original_mlp: modeling_lfm2.Lfm2MLP,
      use_swiglu_composite: bool = False,
  ):
    super().__init__()
    self.w2 = original_mlp.w2
    self.use_swiglu_composite = use_swiglu_composite

    w1_w = original_mlp.w1.weight.data
    w3_w = original_mlp.w3.weight.data
    fused_weight = torch.cat([w1_w, w3_w], dim=0)

    hidden_size = w1_w.shape[1]
    fused_out_dim = fused_weight.shape[0]

    self.gate_up_proj = torch.nn.Linear(
        hidden_size, fused_out_dim, bias=False
    )
    self.gate_up_proj.weight = torch.nn.Parameter(fused_weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_up = self.gate_up_proj(x)
    if self.use_swiglu_composite:
      return self.w2(swiglu_composite.apply_swiglu(gate_up))
    gate, up = gate_up.chunk(2, dim=-1)
    return self.w2(torch.nn.functional.silu(gate) * up)


class FusedLfm2Attention(torch.nn.Module):
  """Fused QKV Attention Layer for LFM2 model."""

  def __init__(
      self,
      original_attn: modeling_lfm2.Lfm2Attention,
      fuse_qkv: bool = True,
      use_rope_composite: bool = False,
      use_qkv_norm_rope_composite: bool = False,
  ):
    super().__init__()
    self.config = original_attn.config
    self.layer_idx = getattr(original_attn, "layer_idx", 0)
    self.head_dim = getattr(
        original_attn,
        "head_dim",
        getattr(
            original_attn.config,
            "head_dim",
            original_attn.config.hidden_size
            // original_attn.config.num_attention_heads,
        ),
    )
    self.num_heads = getattr(
        original_attn,
        "num_heads",
        getattr(original_attn.config, "num_attention_heads", 32),
    )
    self.num_key_value_heads = getattr(
        original_attn,
        "num_key_value_heads",
        getattr(original_attn.config, "num_key_value_heads", 8),
    )
    self.num_key_value_groups = getattr(
        original_attn,
        "num_key_value_groups",
        self.num_heads // self.num_key_value_heads,
    )
    self.scaling = getattr(original_attn, "scaling", self.head_dim**-0.5)
    self.is_causal = getattr(original_attn, "is_causal", True)

    self.q_layernorm = original_attn.q_layernorm
    self.k_layernorm = original_attn.k_layernorm
    self.out_proj = original_attn.out_proj
    self.fuse_qkv = fuse_qkv
    self.use_rope_composite = use_rope_composite
    self.use_qkv_norm_rope_composite = use_qkv_norm_rope_composite

    self.q_size = self.num_heads * self.head_dim
    self.k_size = self.num_key_value_heads * self.head_dim
    self.v_size = self.num_key_value_heads * self.head_dim

    if fuse_qkv:
      q_w = original_attn.q_proj.weight.data
      k_w = original_attn.k_proj.weight.data
      v_w = original_attn.v_proj.weight.data
      fused_weight = torch.cat([q_w, k_w, v_w], dim=0)

      total_out_dim = self.q_size + self.k_size + self.v_size
      self.qkv_proj = torch.nn.Linear(
          self.config.hidden_size, total_out_dim, bias=False
      )
      self.qkv_proj.weight = torch.nn.Parameter(fused_weight)
    else:
      self.q_proj = original_attn.q_proj
      self.k_proj = original_attn.k_proj
      self.v_proj = original_attn.v_proj

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
      attention_mask: torch.Tensor | None = None,
      past_key_values: transformers.cache_utils.Cache | None = None,
      position_ids: torch.LongTensor | None = None,
      **kwargs,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape_q = (*input_shape, self.num_heads, self.head_dim)
    hidden_shape_kv = (*input_shape, self.num_key_value_heads, self.head_dim)

    if getattr(self, "use_qkv_norm_rope_composite", False) and self.fuse_qkv:
      if position_ids is None:
        position_ids = kwargs.get("position_ids", None)
      if position_ids is None:
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(
            seq_len, device=hidden_states.device
        ).unsqueeze(0)

      rope_base = 1000000.0
      if (
          hasattr(self.config, "rope_parameters")
          and self.config.rope_parameters
      ):
        if isinstance(self.config.rope_parameters, dict):
          val = self.config.rope_parameters.get("rope_theta", rope_base)
          rope_base = float(val) if val is not None else rope_base
        elif hasattr(self.config.rope_parameters, "rope_theta"):
          rope_base = float(
              getattr(self.config.rope_parameters, "rope_theta", rope_base)
          )
      elif hasattr(self.config, "rope_theta"):
        rope_base = float(getattr(self.config, "rope_theta", rope_base))
      elif hasattr(self.config, "default_theta"):
        rope_base = float(getattr(self.config, "default_theta", rope_base))

      qkv = self.qkv_proj(hidden_states)
      eps = float(
          getattr(
              self.q_layernorm,
              "variance_epsilon",
              getattr(self.config, "norm_eps", 1e-6),
          )
      )
      query_states, key_states, value_states = (
          qkv_norm_rope_composite.apply_qkv_norm_rope(
              qkv,
              position_ids,
              self.q_layernorm.weight,
              self.k_layernorm.weight,
              num_heads=self.num_heads,
              num_kv_heads=self.num_key_value_heads,
              head_dim=self.head_dim,
              base=rope_base,
              eps=eps,
          )
      )
    elif getattr(self, "use_rope_composite", False):
      if position_ids is None:
        position_ids = kwargs.get("position_ids", None)
      if position_ids is None:
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(
            seq_len, device=hidden_states.device
        ).unsqueeze(0)

      rope_base = 1000000.0
      if (
          hasattr(self.config, "rope_parameters")
          and self.config.rope_parameters
      ):
        if isinstance(self.config.rope_parameters, dict):
          val = self.config.rope_parameters.get("rope_theta", rope_base)
          rope_base = float(val) if val is not None else rope_base
        elif hasattr(self.config.rope_parameters, "rope_theta"):
          rope_base = float(
              getattr(self.config.rope_parameters, "rope_theta", rope_base)
          )
      elif hasattr(self.config, "rope_theta"):
        rope_base = float(getattr(self.config, "rope_theta", rope_base))
      elif hasattr(self.config, "default_theta"):
        rope_base = float(getattr(self.config, "default_theta", rope_base))

      if self.fuse_qkv:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
      else:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

      query_states = self.q_layernorm(q.view(hidden_shape_q)).transpose(1, 2)
      key_states = self.k_layernorm(k.view(hidden_shape_kv)).transpose(1, 2)
      value_states = v.view(hidden_shape_kv).transpose(1, 2)

      query_states = rope_composite.apply_mldrift_compatible_rope(
          query_states, position_ids, base=rope_base, head_dim=self.head_dim
      )
      key_states = rope_composite.apply_mldrift_compatible_rope(
          key_states, position_ids, base=rope_base, head_dim=self.head_dim
      )
    else:
      if self.fuse_qkv:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
      else:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

      query_states = self.q_layernorm(q.view(hidden_shape_q)).transpose(1, 2)
      key_states = self.k_layernorm(k.view(hidden_shape_kv)).transpose(1, 2)
      value_states = v.view(hidden_shape_kv).transpose(1, 2)

      if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = modeling_lfm2.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if past_key_values is not None:
      key_states, value_states = past_key_values.update(
          key_states, value_states, self.layer_idx
      )

    attention_interface = modeling_lfm2.ALL_ATTENTION_FUNCTIONS.get_interface(
        getattr(self.config, "_attn_implementation", "eager"),
        modeling_lfm2.eager_attention_forward,
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    output = self.out_proj(attn_output)
    return output, attn_weights


class PatchedLfm2DecoderLayer(modeling_lfm2.Lfm2DecoderLayer):
  """Patched Lfm2DecoderLayer supporting past_key_values and position_ids."""

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings=None,
      attention_mask=None,
      position_ids=None,
      past_key_values=None,
      **kwargs,
  ) -> torch.Tensor:
    residual = hidden_states
    if self.is_attention_layer:
      hidden_states, _ = self.self_attn(
          hidden_states=self.operator_norm(hidden_states),
          position_embeddings=position_embeddings,
          attention_mask=attention_mask,
          position_ids=position_ids,
          past_key_values=past_key_values,
          **kwargs,
      )
    else:
      cache_position = (
          position_ids.squeeze(0) if position_ids is not None else None
      )
      valid_mask = kwargs.get("valid_mask", None)
      hidden_states = self.conv(
          hidden_states=self.operator_norm(hidden_states),
          past_key_values=past_key_values,
          attention_mask=attention_mask,
          cache_position=cache_position,
          valid_mask=valid_mask,
      )
    hidden_states = hidden_states + residual
    hidden_states = hidden_states + self.feed_forward(
        self.ffn_norm(hidden_states)
    )

    return hidden_states


@patches_lib.register_patch(["lfm2"])
@contextlib.contextmanager
def lfm2_litert_patch():
  """Context manager for patching LFM2 modules for LiteRT export."""
  print("LFM2 patch applied.")
  original_short_conv = modeling_lfm2.Lfm2ShortConv
  modeling_lfm2.Lfm2ShortConv = short_conv_lib.Lfm2ShortConv

  original_decoder_layer = modeling_lfm2.Lfm2DecoderLayer
  modeling_lfm2.Lfm2DecoderLayer = PatchedLfm2DecoderLayer

  original_norm = modeling_lfm2.Lfm2RMSNorm
  modeling_lfm2.Lfm2RMSNorm = Lfm2RMSNorm  # pyrefly: ignore[bad-assignment]

  try:
    yield
  finally:
    modeling_lfm2.Lfm2ShortConv = original_short_conv
    modeling_lfm2.Lfm2DecoderLayer = original_decoder_layer
    modeling_lfm2.Lfm2RMSNorm = original_norm


@patches_lib.register_model_patch(["lfm2", "lfm2_vl"])
@contextlib.contextmanager
def patch_lfm2_model(model, export_config):
  """Dynamic model patch for LFM2 export."""
  fuse_gate_up = getattr(export_config, "fuse_gate_up", False)
  fuse_qkv = getattr(export_config, "fuse_qkv", False)
  use_rope = getattr(export_config, "use_rope_composite", False)
  use_swiglu = getattr(export_config, "use_swiglu_composite", False)
  use_qkv_norm_rope = getattr(
      export_config, "use_qkv_norm_rope_composite", False
  )
  use_short_conv = getattr(
      export_config, "use_short_conv_composite", False
  )
  print(
      "LFM2 model patch applied. "
      f"fuse_gate_up={fuse_gate_up}, fuse_qkv={fuse_qkv}, "
      f"use_rope_composite={use_rope}, use_swiglu_composite={use_swiglu}, "
      f"use_qkv_norm_rope_composite={use_qkv_norm_rope}, "
      f"use_short_conv_composite={use_short_conv}"
  )

  replaced_modules = []

  def replace_modules(module):
    for child_name, child in module.named_children():
      if (fuse_gate_up or use_swiglu) and isinstance(
          child, modeling_lfm2.Lfm2MLP
      ):
        fused = FusedLfm2MLP(child, use_swiglu_composite=use_swiglu)
        setattr(module, child_name, fused)
        replaced_modules.append((module, child_name, child))
      elif isinstance(child, modeling_lfm2.Lfm2Attention):
        if fuse_qkv or use_rope or use_qkv_norm_rope:
          fused = FusedLfm2Attention(
              child,
              fuse_qkv=fuse_qkv or use_qkv_norm_rope,
              use_rope_composite=use_rope,
              use_qkv_norm_rope_composite=use_qkv_norm_rope,
          )
          setattr(module, child_name, fused)
          replaced_modules.append((module, child_name, child))
      elif isinstance(child, (short_conv_lib.Lfm2ShortConv, modeling_lfm2.Lfm2ShortConv)):
        child.use_short_conv_composite = use_short_conv
      else:
        replace_modules(child)

  replace_modules(model)
  try:
    yield
  finally:
    for module, name, original in reversed(replaced_modules):
      setattr(module, name, original)
