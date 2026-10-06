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

"""Patches for Qwen3 model."""

import contextlib
from litert_torch.generative.export_hf.core.speech import asr_model
from litert_torch.generative.export_hf.experimental.composites import qkv_norm_rope as qkv_norm_rope_composite
from litert_torch.generative.export_hf.experimental.composites import rope as rope_composite
from litert_torch.generative.export_hf.experimental.composites import swiglu as swiglu_composite
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.layers import normalization
import torch
import transformers
from transformers.models.qwen3 import modeling_qwen3


class Qwen3RMSNorm(torch.nn.Module):
  """RMSNorm Layer."""

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.hidden_size = hidden_size

  def forward(self, hidden_states):
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


class FusedQwen3MLP(torch.nn.Module):
  """Fused Gate-Up MLP Layer for Qwen3 model."""

  def __init__(
      self,
      original_mlp: modeling_qwen3.Qwen3MLP,
      use_swiglu_composite: bool = False,
  ):
    super().__init__()
    self.config = original_mlp.config
    self.hidden_size = getattr(original_mlp, "hidden_size", 0)
    self.intermediate_size = getattr(original_mlp, "intermediate_size", 0)
    self.act_fn = original_mlp.act_fn
    self.down_proj = original_mlp.down_proj
    self.use_swiglu_composite = use_swiglu_composite

    gate_w = original_mlp.gate_proj.weight.data
    up_w = original_mlp.up_proj.weight.data
    fused_weight = torch.cat([gate_w, up_w], dim=0)

    self.gate_up_proj = torch.nn.Linear(
        self.hidden_size, 2 * self.intermediate_size, bias=False
    )
    self.gate_up_proj.weight = torch.nn.Parameter(fused_weight)

  def forward(self, x):
    gate_up = self.gate_up_proj(x)
    if self.use_swiglu_composite:
      return self.down_proj(swiglu_composite.apply_swiglu(gate_up))
    gate, up = gate_up.chunk(2, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


class FusedQwen3Attention(torch.nn.Module):
  """Fused QKV Attention Layer for Qwen3 model."""

  def __init__(
      self,
      original_attn: modeling_qwen3.Qwen3Attention,
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
    self.sliding_window = getattr(original_attn, "sliding_window", None)

    self.q_norm = original_attn.q_norm
    self.k_norm = original_attn.k_norm
    self.o_proj = original_attn.o_proj
    self.rotary_emb = getattr(original_attn, "rotary_emb", None)
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
      cache_position: torch.LongTensor | None = None,
      **kwargs,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape_q = (*input_shape, self.num_heads, self.head_dim)
    hidden_shape_kv = (*input_shape, self.num_key_value_heads, self.head_dim)

    if getattr(self, "use_qkv_norm_rope_composite", False) and self.fuse_qkv:
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

      is_local = False
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
        rope_base = float(getattr(self.config, "local_rope_theta", 10000.0))

      qkv = self.qkv_proj(hidden_states)
      query_states, key_states, value_states = (
          qkv_norm_rope_composite.apply_qkv_norm_rope(
              qkv,
              position_ids,
              self.q_norm.weight,
              self.k_norm.weight,
              num_heads=self.num_heads,
              num_kv_heads=self.num_key_value_heads,
              head_dim=self.head_dim,
              base=rope_base,
              eps=float(self.q_norm.variance_epsilon),
          )
      )
    elif getattr(self, "use_rope_composite", False):
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

      is_local = False
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
        rope_base = float(getattr(self.config, "local_rope_theta", 10000.0))

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

      query_states = self.q_norm(q.view(hidden_shape_q)).transpose(1, 2)
      key_states = self.k_norm(k.view(hidden_shape_kv)).transpose(1, 2)
      value_states = v.view(hidden_shape_kv).transpose(1, 2)

      if position_embeddings is not None:
        cos, sin = position_embeddings
      elif self.rotary_emb is not None:
        cos, sin = self.rotary_emb(
            value_states, position_ids=kwargs.get("position_ids", None)
        )
      else:
        cos, sin = None, None
      if cos is not None and sin is not None:
        query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

    if past_key_values is not None:
      key_states, value_states = past_key_values.update(
          key_states, value_states, self.layer_idx
      )

    attention_interface = modeling_qwen3.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        modeling_qwen3.eager_attention_forward,
    )

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


@patches_lib.register_patch(["qwen3", "qwen3_text", "qwen3_asr"])
@contextlib.contextmanager
def qwen3_litert_patch():
  """Qwen3 patch."""
  print("Qwen3 patch applied.")
  original_norm = modeling_qwen3.Qwen3RMSNorm
  modeling_qwen3.Qwen3RMSNorm = Qwen3RMSNorm  # pyrefly: ignore[bad-assignment]

  attn_funs = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
  original_sdpa = attn_funs.get("sdpa")
  attn_funs["sdpa"] = asr_model._sdpa

  try:
    yield
  finally:
    modeling_qwen3.Qwen3RMSNorm = original_norm
    attn_funs["sdpa"] = original_sdpa


@patches_lib.register_model_patch(["qwen3", "qwen3_text", "qwen3_asr"])
@contextlib.contextmanager
def patch_qwen3_model(model, export_config):
  """Dynamic model patch for Qwen3 export."""
  fuse_gate_up = getattr(export_config, "fuse_gate_up", False)
  fuse_qkv = getattr(export_config, "fuse_qkv", False)
  use_rope = getattr(export_config, "use_rope_composite", False)
  use_swiglu = getattr(export_config, "use_swiglu_composite", False)
  use_qkv_norm_rope = getattr(
      export_config, "use_qkv_norm_rope_composite", False
  )
  print(
      "Qwen3 model patch applied. "
      f"fuse_gate_up={fuse_gate_up}, fuse_qkv={fuse_qkv}, "
      f"use_rope_composite={use_rope}, use_swiglu_composite={use_swiglu}, "
      f"use_qkv_norm_rope_composite={use_qkv_norm_rope}, "
      f""
  )

  replaced_modules = []

  def replace_modules(module):
    for child_name, child in module.named_children():
      if (fuse_gate_up or use_swiglu) and isinstance(
          child, modeling_qwen3.Qwen3MLP
      ):
        fused = FusedQwen3MLP(child, use_swiglu_composite=use_swiglu)
        setattr(module, child_name, fused)
        replaced_modules.append((module, child_name, child))
      elif isinstance(child, modeling_qwen3.Qwen3Attention):
        if fuse_qkv or use_rope or use_qkv_norm_rope:
          fused = FusedQwen3Attention(
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
