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

"""Reauthored Qwen3.5 PyTorch model with static shapes aligned to LiteRT-LM inference schema.

Features:
- Stateless functional design (`use_cache=True` equivalent).
- Explicit state passing: both `prefill` and `decode` static shape graphs take
identical state tensors in shape and order.
- Supports chunked prefill (e.g. 2 x 128 tokens) with right padding cleanly
handled so `conv_state`, `recurrent_state`, and `kv_cache` reflect only valid
tokens.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union
from litert_torch.generative.export_hf.model_ext.qwen3_5 import gated_delta_rule
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from transformers.models.qwen3_5.modular_qwen3_5 import Qwen3_5MLP
from transformers.models.qwen3_5.modular_qwen3_5 import Qwen3_5RMSNorm


def _unwrap_config(config: Any) -> Any:
  if (
      hasattr(config, "text_config")
      and config.text_config is not None
      and not isinstance(config.text_config, dict)
  ):
    return config.text_config
  return config


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
  """Aligns with l2norm in FLA / Qwen3.5 linear attention."""
  inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
  return x * inv_norm


def _gated_delta_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Applies RoPE."""
  cos = cos.unsqueeze(1)
  sin = sin.unsqueeze(1)
  rotary_dim = cos.shape[-1]
  q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
  k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

  def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

  q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
  k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
  return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


# Reused upstream Qwen3_5Attention


def recurrent_gated_delta_rule(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    beta_t: torch.Tensor,
    g_t: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Single-token recurrent gated delta rule standalone function."""
  g_step = g_t[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
  beta_step = beta_t[:, :, 0].unsqueeze(-1)
  new_recurrent_state = recurrent_state * g_step
  kv_mem = (new_recurrent_state * k_t[:, :, 0].unsqueeze(-1)).sum(dim=-2)
  delta = (v_t[:, :, 0] - kv_mem) * beta_step
  new_recurrent_state = new_recurrent_state + k_t[:, :, 0].unsqueeze(-1) * delta.unsqueeze(-2)
  core_attn_out = (new_recurrent_state * q_t[:, :, 0].unsqueeze(-1)).sum(dim=-2).unsqueeze(2)
  return core_attn_out, new_recurrent_state


def chunk_gated_delta_rule(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    beta_t: torch.Tensor,
    g_t: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Multi-token chunked gated delta rule standalone function (GPU-delegate optimized, rank <= 4)."""
  batch_size, num_v_heads, seq_len, _ = q_t.shape
  chunk_size = min(64, seq_len) if seq_len >= 64 and seq_len % 64 == 0 else seq_len

  if seq_len % chunk_size != 0:
    pad_size = chunk_size - (seq_len % chunk_size)
    q_t = F.pad(q_t, (0, 0, 0, pad_size))
    k_t = F.pad(k_t, (0, 0, 0, pad_size))
    v_t = F.pad(v_t, (0, 0, 0, pad_size))
    beta_t = F.pad(beta_t, (0, pad_size))
    g_t = F.pad(g_t, (0, pad_size))
    total_len = seq_len + pad_size
  else:
    pad_size = 0
    total_len = seq_len

  B_H = batch_size * num_v_heads
  num_chunks = total_len // chunk_size

  v_beta = v_t * beta_t.unsqueeze(-1)
  k_beta = k_t * beta_t.unsqueeze(-1)

  # Reshape to 4D tensors [B_H, num_chunks, chunk_size, head_dim] so all GPU delegate ops stay rank <= 4
  q_c = q_t.reshape(B_H, num_chunks, chunk_size, -1)
  k_c = k_t.reshape(B_H, num_chunks, chunk_size, -1)
  v_c = v_t.reshape(B_H, num_chunks, chunk_size, -1)
  k_beta_c = k_beta.reshape(B_H, num_chunks, chunk_size, -1)
  v_beta_c = v_beta.reshape(B_H, num_chunks, chunk_size, -1)
  g_c = g_t.reshape(B_H, num_chunks, chunk_size)

  idx = torch.arange(chunk_size, device=q_t.device, dtype=torch.int32)
  mask_triu = idx.unsqueeze(1) <= idx.unsqueeze(0)
  mask_tril = idx.unsqueeze(1) >= idx.unsqueeze(0)
  eye_mask = idx.unsqueeze(1) == idx.unsqueeze(0)

  g_cumsum = g_c.cumsum(dim=-1)
  # diff is 4D: [B_H, num_chunks, chunk_size, 1] - [B_H, num_chunks, 1, chunk_size]
  diff = g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)
  decay_mask = torch.where(
      mask_tril, torch.where(mask_tril, diff, 0.0).exp(), 0.0
  )
  # attn is 4D: [B_H, num_chunks, chunk_size, chunk_size]
  attn = -((k_beta_c @ k_c.transpose(-1, -2)) * decay_mask).masked_fill(
      mask_triu, 0
  )
  for i in range(1, chunk_size):
    row = attn[..., i, :i]
    sub = attn[..., :i, :i]
    attn[..., i, :i] = row + (row.unsqueeze(-2) @ sub).squeeze(-2)
  attn = torch.where(eye_mask, attn + 1.0, attn)
  v_local = attn @ v_beta_c
  k_cumdecay = attn @ (k_beta_c * g_cumsum.exp().unsqueeze(-1))

  # Collapse recurrent_state [batch, num_v_heads, head_k_dim, head_v_dim] to 3D [B_H, head_k_dim, head_v_dim]
  new_recurrent_state = recurrent_state.to(v_local.dtype).reshape(B_H, -1, v_t.shape[-1])
  core_attn_out = torch.zeros_like(v_local)

  for i in range(num_chunks):
    # q_i, k_i, v_i are 3D: [B_H, chunk_size, head_dim]
    q_i, k_i, v_i = q_c[:, i], k_c[:, i], v_local[:, i]
    attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, i]
    v_prime = k_cumdecay[:, i] @ new_recurrent_state
    v_new = v_i - v_prime
    attn_inter = (q_i * g_cumsum[:, i, :, None].exp()) @ new_recurrent_state
    core_attn_out[:, i] = attn_inter + attn_i @ v_new
    new_recurrent_state = (
        new_recurrent_state * g_cumsum[:, i, -1, None, None].exp()
        + (k_i * (g_cumsum[:, i, -1, None] - g_cumsum[:, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    )

  # Reshape back to [batch_size, num_v_heads, seq_len, head_v_dim] and recurrent_state back to [batch, num_v_heads, head_k_dim, head_v_dim]
  core_attn_out = core_attn_out.reshape(batch_size, num_v_heads, total_len, -1)[:, :, :seq_len]
  new_recurrent_state = new_recurrent_state.reshape(batch_size, num_v_heads, -1, v_t.shape[-1])
  return core_attn_out, new_recurrent_state


class Qwen3_5StaticGatedDeltaNet(nn.Module):
  """Static shape functional linear attention (`GatedDeltaNet`) module."""

  def __init__(
      self,
      hidden_size: int,
      num_v_heads: int,
      num_k_heads: int,
      head_k_dim: int,
      head_v_dim: int,
      conv_kernel_size: int = 4,
      rms_norm_eps: float = 1e-6,
      layer_idx: int = 0,
  ):
    super().__init__()
    self.layer_idx = layer_idx
    self.hidden_size = hidden_size
    self.num_v_heads = num_v_heads
    self.num_k_heads = num_k_heads
    self.head_k_dim = head_k_dim
    self.head_v_dim = head_v_dim
    self.key_dim = head_k_dim * num_k_heads
    self.value_dim = head_v_dim * num_v_heads
    self.conv_kernel_size = conv_kernel_size
    self.conv_dim = self.key_dim * 2 + self.value_dim

    self.conv1d = nn.Conv1d(
        in_channels=self.conv_dim,
        out_channels=self.conv_dim,
        bias=False,
        kernel_size=conv_kernel_size,
        groups=self.conv_dim,
        padding=conv_kernel_size - 1,
    )
    self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
    A = torch.empty(num_v_heads).uniform_(0, 16)
    self.A_log = nn.Parameter(torch.log(A))
    self.norm = Qwen3_5RMSNormGated(head_v_dim, eps=rms_norm_eps)
    self.rms_norm_eps = float(rms_norm_eps)
    self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
    self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
    self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
    self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)

  def forward(
      self,
      hidden_states: torch.Tensor,
      positions: Optional[torch.Tensor] = None,
      past_key_values: Optional[Any] = None,
      valid_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.Tensor] = None,
      **kwargs,
  ) -> torch.Tensor:
    if positions is None:
      positions = position_ids
    batch_size, seq_len, _ = hidden_states.shape
    state_len = self.conv_kernel_size - 1

    if past_key_values is not None and hasattr(past_key_values, "layers"):
      layer_cache = past_key_values.layers[self.layer_idx]
      conv_state = layer_cache.conv_states
      recurrent_state = layer_cache.recurrent_states
    else:
      conv_state = torch.zeros(
          batch_size, self.conv_dim, state_len,
          dtype=hidden_states.dtype, device=hidden_states.device
      )
      recurrent_state = torch.zeros(
          batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim,
          dtype=hidden_states.dtype, device=hidden_states.device
      )
    if getattr(self, "use_fused_gdn", True):
      mask_arg = (
          valid_mask
          if valid_mask is not None
          else torch.ones(
              (batch_size, seq_len),
              dtype=torch.int32,
              device=hidden_states.device,
          )
      )
      out, new_conv_state, new_recurrent_state = (
          gated_delta_rule.gated_delta_net(
              self.in_proj_qkv(hidden_states),
              self.in_proj_z(hidden_states),
              self.in_proj_b(hidden_states),
              self.in_proj_a(hidden_states),
              conv_state,
              recurrent_state,
              self.conv1d.weight,
              self.A_log,
              self.dt_bias,
              self.norm.weight,
              mask_arg,
              head_k_dim=self.head_k_dim,
              head_v_dim=self.head_v_dim,
              num_k_heads=self.num_k_heads,
              num_v_heads=self.num_v_heads,
              rms_norm_eps=self.rms_norm_eps,
              mode=getattr(self, "gdn_mode", 0),
          )
      )
      if past_key_values is not None:
        layer_cache = past_key_values.layers[self.layer_idx]
        layer_cache.conv_states.copy_(new_conv_state)  # pyrefly: ignore[missing-attribute]
        layer_cache.recurrent_states.copy_(new_recurrent_state)  # pyrefly: ignore[missing-attribute]
      return self.out_proj(out)

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(
        1, 2
    )  # [batch, conv_dim, seq_len]
    z = self.in_proj_z(hidden_states).reshape(
        batch_size, seq_len, -1, self.head_v_dim
    )
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # 1. Causal 1D Conv with static conv_state
    if valid_mask is not None:
      mixed_qkv_masked = mixed_qkv * valid_mask.view(batch_size, 1, seq_len).to(
          mixed_qkv.dtype
      )
    else:
      mixed_qkv_masked = mixed_qkv

    full_qkv = torch.cat(
        [conv_state, mixed_qkv_masked], dim=-1
    )  # [batch, conv_dim, state_len + seq_len]

    conv_out = F.conv1d(
        full_qkv,
        self.conv1d.weight,
        self.conv1d.bias,
        padding=0,
        groups=self.conv_dim,
    )
    conv_out = F.silu(conv_out[:, :, -seq_len:]).transpose(
        1, 2
    )  # [batch, seq_len, conv_dim]

    query, key, value = torch.split(
        conv_out, [self.key_dim, self.key_dim, self.value_dim], dim=-1
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.to(torch.float32).exp() * F.softplus(
        a.to(torch.float32) + self.dt_bias
    )
    if self.num_v_heads // self.num_k_heads > 1:
      query = query.repeat_interleave(
          self.num_v_heads // self.num_k_heads, dim=2
      )
      key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if valid_mask is not None:
      vm_4d = valid_mask.view(batch_size, seq_len, 1, 1).to(query.dtype)
      vm_3d = valid_mask.view(batch_size, seq_len, 1).to(query.dtype)
      query = query * vm_4d
      key = key * vm_4d
      value = value * vm_4d
      beta = beta * vm_3d
      g = g * vm_3d

    # 2. Gated Delta Rule (supports single token decode or chunked prefill)
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    q_t, k_t, v_t, beta_t, g_t = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    q_t = q_t * scale

    if seq_len == 1:
      # Single token recurrent delta rule
      g_step = g_t[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
      beta_step = beta_t[:, :, 0].unsqueeze(-1)
      new_recurrent_state = recurrent_state * g_step
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
    else:
      # Recurrent delta rule with loop unrolling (non-chunked)
      new_recurrent_state = recurrent_state.to(q_t.dtype)
      ys = []
      for t in range(seq_len):
        q_t_step = q_t[:, :, t]
        k_t_step = k_t[:, :, t]
        v_t_step = v_t[:, :, t]
        g_t_step = g_t[:, :, t]
        beta_t_step = beta_t[:, :, t]

        step_mask = None
        if valid_mask is not None:
          step_mask = valid_mask[:, t]

        y_step, new_recurrent_state = _gated_delta_step(
            q_t_step,
            k_t_step,
            v_t_step,
            g_t_step,
            beta_t_step,
            new_recurrent_state,
            step_mask,
        )
        ys.append(y_step)

      core_attn_out = torch.stack(ys, dim=2)

    core_attn_out = (
        core_attn_out.transpose(1, 2).contiguous().to(hidden_states.dtype)
    )
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z_flat = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z_flat).reshape(
        batch_size, seq_len, -1
    )
    output = self.out_proj(core_attn_out)

    if past_key_values is not None:
      past_key_values.update_conv_state(
          mixed_qkv, self.layer_idx, valid_mask=valid_mask
      )
      past_key_values.update_recurrent_state(
          new_recurrent_state, self.layer_idx
      )

    return output


def get_head_dim(config: Qwen3_5Config) -> int:
  hd = getattr(config, "head_dim", None)
  if hd is not None:
    return int(hd)
  return int(config.hidden_size // config.num_attention_heads)


class Qwen3_5StaticDecoderLayer(nn.Module):

  def __init__(self, config: Qwen3_5Config, layer_idx: int):
    super().__init__()
    config = _unwrap_config(config)
    self.layer_idx = layer_idx
    self.hidden_size = config.hidden_size
    self.block_type = config.layer_types[layer_idx]

    if self.block_type == "linear_attention":
      self.linear_attn = Qwen3_5StaticGatedDeltaNet(
          hidden_size=config.hidden_size,
          num_v_heads=config.linear_num_value_heads,
          num_k_heads=config.linear_num_key_heads,
          head_k_dim=config.linear_key_head_dim,
          head_v_dim=config.linear_value_head_dim,
          conv_kernel_size=config.linear_conv_kernel_dim,
          rms_norm_eps=config.rms_norm_eps,
          layer_idx=layer_idx,
      )
    elif self.block_type == "full_attention":
      self.self_attn = Qwen3_5Attention(config, layer_idx)
    self.mlp = Qwen3_5MLP(
        config,
        config.intermediate_size,
    )
    self.input_layernorm = Qwen3_5RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    self.post_attention_layernorm = Qwen3_5RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
      positions: Optional[torch.Tensor] = None,
      past_key_values: Optional[Any] = None,
      valid_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.Tensor] = None,
      **kwargs,
  ) -> torch.Tensor:
    if positions is None:
      positions = position_ids
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.block_type == "linear_attention":
      hidden_states = self.linear_attn(
          hidden_states,
          positions,
          past_key_values=past_key_values,
          valid_mask=valid_mask,
          **kwargs,
      )
    elif self.block_type == "full_attention":
      attn_kwargs = dict(kwargs)
      attn_mask = attn_kwargs.pop("attention_mask", None)
      if valid_mask is not None and "valid_mask" not in attn_kwargs:
        attn_kwargs["valid_mask"] = valid_mask
      if positions is not None and "cache_position" not in attn_kwargs:
        attn_kwargs["cache_position"] = positions

      hidden_states, _ = self.self_attn(
          hidden_states,
          position_embeddings,
          attn_mask,
          past_key_values=past_key_values,
          **attn_kwargs,
      )

    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


class Qwen3_5StaticModel(nn.Module):

  def __init__(self, config: Qwen3_5Config):
    super().__init__()
    config = _unwrap_config(config)
    self.config = config
    self.embed_tokens = nn.Embedding(
        config.vocab_size,
        config.hidden_size,
        getattr(config, "pad_token_id", None),
    )
    self.layers = nn.ModuleList([
        Qwen3_5StaticDecoderLayer(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
    ])
    self.norm = Qwen3_5RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

  def forward(
      self,
      input_ids: torch.Tensor,
      positions: torch.Tensor,
      past_key_values: Optional[Any] = None,
      valid_mask: Optional[torch.Tensor] = None,
      **kwargs,
  ) -> Tuple[torch.Tensor, Optional[Any]]:
    hidden_states = self.embed_tokens(input_ids)
    pos_for_rope = positions.unsqueeze(0) if positions.ndim == 1 else positions
    position_embeddings = self.rotary_emb(hidden_states, pos_for_rope)

    for layer in self.layers:
      hidden_states = layer(
          hidden_states,
          position_embeddings,
          positions,
          past_key_values=past_key_values,
          valid_mask=valid_mask,
          **kwargs,
      )

    hidden_states = self.norm(hidden_states)
    return hidden_states, past_key_values


class Qwen3_5StaticForCausalLM(nn.Module):
  _can_compile_fullgraph = True
  _supports_attention_backend = True

  def __init__(self, config: Qwen3_5Config):
    super().__init__()
    config = _unwrap_config(config)
    self.config = config
    self.model = Qwen3_5StaticModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

  @classmethod
  def from_hf_model(cls, hf_model: Any) -> "Qwen3_5StaticForCausalLM":
    config = copy.deepcopy(_unwrap_config(hf_model.config))
    static_model = cls(config)
    static_model.load_state_dict(hf_model.state_dict(), strict=False)
    return static_model

  def get_input_embeddings(self) -> nn.Module:
    return self.model.embed_tokens

  def set_input_embeddings(self, value: nn.Module) -> None:
    self.model.embed_tokens = value  # pyrefly: ignore[bad-assignment]

  def get_output_embeddings(self) -> nn.Module:
    return self.lm_head

  def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
    self.lm_head = new_embeddings  # pyrefly: ignore[bad-assignment]

  def set_attn_implementation(self, implementation: str) -> None:
    self.config._attn_implementation = implementation
    if hasattr(self.model, "config"):
      self.model.config._attn_implementation = implementation
    for layer in getattr(self.model, "layers", []):
      if hasattr(layer, "self_attn"):
        layer.self_attn.config._attn_implementation = implementation

  def forward(
      self,
      input_ids: torch.Tensor,
      positions: torch.Tensor,
      past_key_values: Optional[Any] = None,
      valid_mask: Optional[torch.Tensor] = None,
      **kwargs,
  ) -> Tuple[torch.Tensor, Optional[Any]]:
    hidden_states, past_key_values = self.model(
        input_ids, positions, past_key_values=past_key_values, valid_mask=valid_mask, **kwargs
    )
    logits = self.lm_head(hidden_states)
    return logits, past_key_values


class Qwen3_5PrefillGraph(nn.Module):
  """Static shape graph for prefill (`seq_len = prefill_length`)."""

  def __init__(self, model: Qwen3_5StaticForCausalLM):
    super().__init__()
    self.model = model

  def forward(
      self,
      input_ids: torch.Tensor,
      positions: torch.Tensor,
      past_key_values: Optional[Any] = None,
      valid_mask: Optional[torch.Tensor] = None,
      **kwargs,
  ) -> Optional[Any]:
    _, past_key_values = self.model(
        input_ids, positions, past_key_values=past_key_values, valid_mask=valid_mask, **kwargs
    )
    return past_key_values


class Qwen3_5DecodeGraph(nn.Module):
  """Static shape graph for decode (`seq_len = 1`)."""

  def __init__(self, model: Qwen3_5StaticForCausalLM):
    super().__init__()
    self.model = model

  def forward(
      self,
      input_ids: torch.Tensor,
      positions: torch.Tensor,
      past_key_values: Optional[Any] = None,
      **kwargs,
  ) -> Tuple[torch.Tensor, Optional[Any]]:
    logits, past_key_values = self.model(
        input_ids, positions, past_key_values=past_key_values, valid_mask=None, **kwargs
    )
    return logits, past_key_values
