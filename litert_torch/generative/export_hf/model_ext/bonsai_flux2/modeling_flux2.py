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
"""PyTorch modeling classes for FLUX.2 DiT and VAE (Bonsai-FLUX.2 / FLUX.2-klein).

Matches the Diffusers `Flux2Transformer2DModel` and `AutoencoderKLFlux2` module
hierarchy and parameter names while keeping RoPE frequency calculation in
float32 for LiteRT / torch.export compatibility.
"""

import inspect
import json
import math
import os
import types
from typing import Any

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F


def _load_safetensors_state_dict(directory: str) -> dict[str, torch.Tensor]:
  """Loads single or sharded safetensors weights from a directory."""
  for index_name in (
      "diffusion_pytorch_model.safetensors.index.json",
      "model.safetensors.index.json",
  ):
    index_path = os.path.join(directory, index_name)
    if os.path.exists(index_path):
      with open(index_path, "r") as f:
        index_data = json.load(f)
      shard_files = sorted(set(index_data.get("weight_map", {}).values()))
      state_dict: dict[str, torch.Tensor] = {}
      for shard_file in shard_files:
        shard_path = os.path.join(directory, shard_file)
        shard_tensors = safetensors.torch.load_file(shard_path)
        for k, v in shard_tensors.items():
          state_dict[k] = v.to(torch.float32) if v.is_floating_point() else v
      return state_dict

  for single_name in (
      "diffusion_pytorch_model.safetensors",
      "model.safetensors",
  ):
    single_path = os.path.join(directory, single_name)
    if os.path.exists(single_path):
      tensors = safetensors.torch.load_file(single_path)
      return {
          k: v.to(torch.float32) if v.is_floating_point() else v
          for k, v in tensors.items()
      }
  return {}


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
  """Computes sinusoidal timestep embeddings matching Diffusers."""
  half_dim = embedding_dim // 2
  exponent = -math.log(max_period) * torch.arange(
      start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
  )
  exponent = exponent / (half_dim - downscale_freq_shift)
  emb = torch.exp(exponent)
  emb = timesteps[:, None].float() * emb[None, :]
  emb = scale * emb
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
  if flip_sin_to_cos:
    emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
  if embedding_dim % 2 == 1:
    emb = F.pad(emb, (0, 1, 0, 0))
  return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Computes 1D real rotary positional embeddings in float32."""
  freqs = 1.0 / (
      theta
      ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim)
  )
  freqs = torch.outer(pos.float(), freqs)
  freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
  freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
  return freqs_cos, freqs_sin


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    sequence_dim: int = 1,
) -> torch.Tensor:
  """Applies rotary positional embeddings to query/key tensor."""
  cos, sin = freqs_cis
  if sequence_dim == 2:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
  elif sequence_dim == 1:
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
  else:
    raise ValueError(f"Unsupported sequence_dim={sequence_dim}")
  cos, sin = cos.to(x.device), sin.to(x.device)
  x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
  x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
  return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class Timesteps(nn.Module):
  """Sinusoidal timestep projection module."""

  def __init__(
      self,
      num_channels: int,
      flip_sin_to_cos: bool,
      downscale_freq_shift: float,
      scale: int = 1,
  ):
    super().__init__()
    self.num_channels = num_channels
    self.flip_sin_to_cos = flip_sin_to_cos
    self.downscale_freq_shift = downscale_freq_shift
    self.scale = scale

  def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
    return get_timestep_embedding(
        timesteps,
        self.num_channels,
        flip_sin_to_cos=self.flip_sin_to_cos,
        downscale_freq_shift=self.downscale_freq_shift,
        scale=self.scale,
    )


class TimestepEmbedding(nn.Module):
  """MLP timestep embedder."""

  def __init__(
      self,
      in_channels: int,
      time_embed_dim: int,
      out_dim: int | None = None,
      sample_proj_bias: bool = True,
  ):
    super().__init__()
    self.linear_1 = nn.Linear(
        in_channels, time_embed_dim, bias=sample_proj_bias
    )
    self.act = nn.SiLU()
    time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
    self.linear_2 = nn.Linear(
        time_embed_dim, time_embed_dim_out, bias=sample_proj_bias
    )

  def forward(self, sample: torch.Tensor) -> torch.Tensor:
    sample = self.linear_1(sample)
    sample = self.act(sample)
    sample = self.linear_2(sample)
    return sample


class AdaLayerNormContinuous(nn.Module):
  """Adaptive continuous layer normalization."""

  def __init__(
      self,
      embedding_dim: int,
      conditioning_embedding_dim: int,
      elementwise_affine: bool = True,
      eps: float = 1e-5,
      bias: bool = True,
  ):
    super().__init__()
    self.silu = nn.SiLU()
    self.linear = nn.Linear(
        conditioning_embedding_dim, embedding_dim * 2, bias=bias
    )
    self.norm = nn.LayerNorm(
        embedding_dim, eps=eps, elementwise_affine=elementwise_affine
    )

  def forward(
      self, x: torch.Tensor, conditioning_embedding: torch.Tensor
  ) -> torch.Tensor:
    emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
    scale, shift = torch.chunk(emb, 2, dim=1)
    return self.norm(x) * (scale + 1)[:, None, :] + shift[:, None, :]


class Flux2SwiGLU(nn.Module):
  """SwiGLU activation with fused gate/up projection."""

  def __init__(self):
    super().__init__()
    self.gate_fn = nn.SiLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return self.gate_fn(x[..., :half]) * x[..., half:]


class Flux2FeedForward(nn.Module):
  """Feedforward layer for Flux2 double-stream blocks."""

  def __init__(
      self,
      dim: int,
      dim_out: int | None = None,
      mult: float = 3.0,
      inner_dim: int | None = None,
      bias: bool = False,
  ):
    super().__init__()
    if inner_dim is None:
      inner_dim = int(dim * mult)
    dim_out = dim_out or dim
    self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
    self.act_fn = Flux2SwiGLU()
    self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.linear_in(x)
    x = self.act_fn(x)
    x = self.linear_out(x)
    return x


class Flux2Attention(nn.Module):
  """Joint attention for Flux2 double-stream transformer blocks."""

  def __init__(
      self,
      query_dim: int,
      heads: int = 8,
      dim_head: int = 64,
      dropout: float = 0.0,
      bias: bool = False,
      added_kv_proj_dim: int | None = None,
      added_proj_bias: bool | None = True,
      out_bias: bool = True,
      eps: float = 1e-5,
      out_dim: int | None = None,
      elementwise_affine: bool = True,
  ):
    super().__init__()
    self.head_dim = dim_head
    self.inner_dim = out_dim if out_dim is not None else dim_head * heads
    self.query_dim = query_dim
    self.out_dim = out_dim if out_dim is not None else query_dim
    self.heads = out_dim // dim_head if out_dim is not None else heads
    self.added_kv_proj_dim = added_kv_proj_dim

    self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
    self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
    self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

    self.norm_q = nn.RMSNorm(
        dim_head, eps=eps, elementwise_affine=elementwise_affine
    )
    self.norm_k = nn.RMSNorm(
        dim_head, eps=eps, elementwise_affine=elementwise_affine
    )

    self.to_out = nn.ModuleList([
        nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
        nn.Dropout(dropout),
    ])

    if added_kv_proj_dim is not None:
      self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
      self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)
      self.add_q_proj = nn.Linear(
          added_kv_proj_dim, self.inner_dim, bias=bool(added_proj_bias)
      )
      self.add_k_proj = nn.Linear(
          added_kv_proj_dim, self.inner_dim, bias=bool(added_proj_bias)
      )
      self.add_v_proj = nn.Linear(
          added_kv_proj_dim, self.inner_dim, bias=bool(added_proj_bias)
      )
      self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=out_bias)

  def forward(
      self,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor | None = None,
      attention_mask: torch.Tensor | None = None,
      image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
    key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
    value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

    query = self.norm_q(query)
    key = self.norm_k(key)

    if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
      enc_q = self.add_q_proj(encoder_hidden_states).unflatten(
          -1, (self.heads, -1)
      )
      enc_k = self.add_k_proj(encoder_hidden_states).unflatten(
          -1, (self.heads, -1)
      )
      enc_v = self.add_v_proj(encoder_hidden_states).unflatten(
          -1, (self.heads, -1)
      )
      enc_q = self.norm_added_q(enc_q)
      enc_k = self.norm_added_k(enc_k)
      query = torch.cat([enc_q, query], dim=1)
      key = torch.cat([enc_k, key], dim=1)
      value = torch.cat([enc_v, value], dim=1)

    if image_rotary_emb is not None:
      query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
      key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    q_t = query.permute(0, 2, 1, 3)
    k_t = key.permute(0, 2, 1, 3)
    v_t = value.permute(0, 2, 1, 3)
    attn_out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    hidden_states = attn_out.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)

    if encoder_hidden_states is not None:
      txt_len = encoder_hidden_states.shape[1]
      encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
          [txt_len, hidden_states.shape[1] - txt_len], dim=1
      )
      encoder_hidden_states = self.to_add_out(encoder_hidden_states)

    hidden_states = self.to_out[0](hidden_states)
    hidden_states = self.to_out[1](hidden_states)

    if encoder_hidden_states is not None:
      return hidden_states, encoder_hidden_states
    return hidden_states


class Flux2ParallelSelfAttention(nn.Module):
  """Parallel self-attention + MLP for Flux2 single-stream transformer blocks."""

  def __init__(
      self,
      query_dim: int,
      heads: int = 8,
      dim_head: int = 64,
      dropout: float = 0.0,
      bias: bool = False,
      out_bias: bool = True,
      eps: float = 1e-5,
      out_dim: int | None = None,
      elementwise_affine: bool = True,
      mlp_ratio: float = 4.0,
      mlp_mult_factor: int = 2,
  ):
    super().__init__()
    self.head_dim = dim_head
    self.inner_dim = out_dim if out_dim is not None else dim_head * heads
    self.query_dim = query_dim
    self.out_dim = out_dim if out_dim is not None else query_dim
    self.heads = out_dim // dim_head if out_dim is not None else heads
    self.mlp_ratio = mlp_ratio
    self.mlp_hidden_dim = int(query_dim * self.mlp_ratio)
    self.mlp_mult_factor = mlp_mult_factor

    self.to_qkv_mlp_proj = nn.Linear(
        self.query_dim,
        self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
        bias=bias,
    )
    self.mlp_act_fn = Flux2SwiGLU()
    self.norm_q = nn.RMSNorm(
        dim_head, eps=eps, elementwise_affine=elementwise_affine
    )
    self.norm_k = nn.RMSNorm(
        dim_head, eps=eps, elementwise_affine=elementwise_affine
    )
    self.to_out = nn.Linear(
        self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=out_bias
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: torch.Tensor | None = None,
      image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> torch.Tensor:
    proj = self.to_qkv_mlp_proj(hidden_states)
    qkv, mlp_hidden_states = torch.split(
        proj,
        [3 * self.inner_dim, self.mlp_hidden_dim * self.mlp_mult_factor],
        dim=-1,
    )
    query, key, value = qkv.chunk(3, dim=-1)
    query = self.norm_q(query.unflatten(-1, (self.heads, -1)))
    key = self.norm_k(key.unflatten(-1, (self.heads, -1)))
    value = value.unflatten(-1, (self.heads, -1))

    if image_rotary_emb is not None:
      query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
      key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    q_t = query.permute(0, 2, 1, 3)
    k_t = key.permute(0, 2, 1, 3)
    v_t = value.permute(0, 2, 1, 3)
    attn_out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    attn_out = attn_out.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)
    mlp_hidden_states = self.mlp_act_fn(mlp_hidden_states)
    hidden_states = torch.cat([attn_out, mlp_hidden_states], dim=-1)
    return self.to_out(hidden_states)


class Flux2Modulation(nn.Module):
  """Modulation parameter projection for Flux2."""

  def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
    super().__init__()
    self.mod_param_sets = mod_param_sets
    self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
    self.act_fn = nn.SiLU()

  def forward(self, temb: torch.Tensor) -> torch.Tensor:
    return self.linear(self.act_fn(temb))

  @staticmethod
  def split(
      mod: torch.Tensor, mod_param_sets: int
  ) -> tuple[tuple[torch.Tensor, ...], ...]:
    if mod.ndim == 2:
      mod = mod.unsqueeze(1)
    mod_params = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
    return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))


class Flux2SingleTransformerBlock(nn.Module):
  """Single-stream transformer block for Flux2."""

  def __init__(
      self,
      dim: int,
      num_attention_heads: int,
      attention_head_dim: int,
      mlp_ratio: float = 3.0,
      eps: float = 1e-6,
      bias: bool = False,
  ):
    super().__init__()
    self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
    self.attn = Flux2ParallelSelfAttention(
        query_dim=dim,
        dim_head=attention_head_dim,
        heads=num_attention_heads,
        out_dim=dim,
        bias=bias,
        out_bias=bias,
        eps=eps,
        mlp_ratio=mlp_ratio,
        mlp_mult_factor=2,
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor | None,
      temb_mod: torch.Tensor,
      image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> torch.Tensor:
    if encoder_hidden_states is not None:
      hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]
    norm_hidden_states = (1 + mod_scale) * self.norm(hidden_states) + mod_shift
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )
    return hidden_states + mod_gate * attn_output


class Flux2TransformerBlock(nn.Module):
  """Double-stream transformer block for Flux2."""

  def __init__(
      self,
      dim: int,
      num_attention_heads: int,
      attention_head_dim: int,
      mlp_ratio: float = 3.0,
      eps: float = 1e-6,
      bias: bool = False,
  ):
    super().__init__()
    self.mlp_hidden_dim = int(dim * mlp_ratio)
    self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
    self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
    self.attn = Flux2Attention(
        query_dim=dim,
        added_kv_proj_dim=dim,
        dim_head=attention_head_dim,
        heads=num_attention_heads,
        out_dim=dim,
        bias=bias,
        added_proj_bias=bias,
        out_bias=bias,
        eps=eps,
    )
    self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
    self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)
    self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
    self.ff_context = Flux2FeedForward(
        dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor,
      temb_mod_img: torch.Tensor,
      temb_mod_txt: torch.Tensor,
      image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = (
        Flux2Modulation.split(temb_mod_img, 2)
    )
    (
        (c_shift_msa, c_scale_msa, c_gate_msa),
        (c_shift_mlp, c_scale_mlp, c_gate_mlp),
    ) = Flux2Modulation.split(temb_mod_txt, 2)

    norm_hidden_states = (1 + scale_msa) * self.norm1(hidden_states) + shift_msa
    norm_encoder_hidden_states = (1 + c_scale_msa) * self.norm1_context(
        encoder_hidden_states
    ) + c_shift_msa

    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )

    hidden_states = hidden_states + gate_msa * attn_output
    norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
    hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

    encoder_hidden_states = (
        encoder_hidden_states + c_gate_msa * context_attn_output
    )
    norm_encoder_hidden_states = (
        self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp)
        + c_shift_mlp
    )
    encoder_hidden_states = (
        encoder_hidden_states
        + c_gate_mlp * self.ff_context(norm_encoder_hidden_states)
    )

    return encoder_hidden_states, hidden_states


class Flux2PosEmbed(nn.Module):
  """Multi-axis rotary positional embedding for Flux2 (in float32)."""

  def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
    super().__init__()
    self.theta = theta
    self.axes_dim = list(axes_dim)

  def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos_out = []
    sin_out = []
    pos = ids.float()
    for i in range(len(self.axes_dim)):
      cos, sin = get_1d_rotary_pos_embed(
          self.axes_dim[i],
          pos[..., i],
          theta=self.theta,
      )
      cos_out.append(cos)
      sin_out.append(sin)
    freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
    freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
    return freqs_cos, freqs_sin


class Flux2TimestepGuidanceEmbeddings(nn.Module):
  """Combined timestep and optional guidance embeddings for Flux2."""

  def __init__(
      self,
      in_channels: int = 256,
      embedding_dim: int = 6144,
      bias: bool = False,
      guidance_embeds: bool = True,
  ):
    super().__init__()
    self.time_proj = Timesteps(
        num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0
    )
    self.timestep_embedder = TimestepEmbedding(
        in_channels=in_channels,
        time_embed_dim=embedding_dim,
        sample_proj_bias=bias,
    )
    if guidance_embeds:
      self.guidance_embedder = TimestepEmbedding(
          in_channels=in_channels,
          time_embed_dim=embedding_dim,
          sample_proj_bias=bias,
      )
    else:
      self.guidance_embedder = None

  def forward(
      self, timestep: torch.Tensor, guidance: torch.Tensor | None = None
  ) -> torch.Tensor:
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))
    if guidance is not None and self.guidance_embedder is not None:
      guidance_proj = self.time_proj(guidance)
      guidance_emb = self.guidance_embedder(guidance_proj.to(guidance.dtype))
      return timesteps_emb + guidance_emb
    return timesteps_emb


class Flux2Transformer2DModel(nn.Module):
  """Standalone PyTorch implementation of Flux2Transformer2DModel."""

  def __init__(
      self,
      patch_size: int = 1,
      in_channels: int = 128,
      out_channels: int | None = None,
      num_layers: int = 8,
      num_single_layers: int = 48,
      attention_head_dim: int = 128,
      num_attention_heads: int = 48,
      joint_attention_dim: int = 15360,
      timestep_guidance_channels: int = 256,
      mlp_ratio: float = 3.0,
      axes_dims_rope: tuple[int, ...] | list[int] = (32, 32, 32, 32),
      rope_theta: int = 2000,
      eps: float = 1e-6,
      guidance_embeds: bool = True,
  ):
    super().__init__()
    self.out_channels = out_channels or in_channels
    self.inner_dim = num_attention_heads * attention_head_dim
    self.config = types.SimpleNamespace(
        patch_size=patch_size,
        in_channels=in_channels,
        out_channels=self.out_channels,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        joint_attention_dim=joint_attention_dim,
        timestep_guidance_channels=timestep_guidance_channels,
        mlp_ratio=mlp_ratio,
        axes_dims_rope=list(axes_dims_rope),
        rope_theta=rope_theta,
        eps=eps,
        guidance_embeds=guidance_embeds,
    )

    self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
    self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
        in_channels=timestep_guidance_channels,
        embedding_dim=self.inner_dim,
        bias=False,
        guidance_embeds=guidance_embeds,
    )
    self.double_stream_modulation_img = Flux2Modulation(
        self.inner_dim, mod_param_sets=2, bias=False
    )
    self.double_stream_modulation_txt = Flux2Modulation(
        self.inner_dim, mod_param_sets=2, bias=False
    )
    self.single_stream_modulation = Flux2Modulation(
        self.inner_dim, mod_param_sets=1, bias=False
    )
    self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
    self.context_embedder = nn.Linear(
        joint_attention_dim, self.inner_dim, bias=False
    )

    self.transformer_blocks = nn.ModuleList([
        Flux2TransformerBlock(
            dim=self.inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
            eps=eps,
            bias=False,
        )
        for _ in range(num_layers)
    ])
    self.single_transformer_blocks = nn.ModuleList([
        Flux2SingleTransformerBlock(
            dim=self.inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
            eps=eps,
            bias=False,
        )
        for _ in range(num_single_layers)
    ])

    self.norm_out = AdaLayerNormContinuous(
        self.inner_dim,
        self.inner_dim,
        elementwise_affine=False,
        eps=eps,
        bias=False,
    )
    self.proj_out = nn.Linear(
        self.inner_dim, patch_size * patch_size * self.out_channels, bias=False
    )

  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str,
      subfolder: str | None = None,
      torch_dtype: torch.dtype = torch.float32,
      **kwargs: Any,
  ) -> "Flux2Transformer2DModel":
    """Loads Flux2Transformer2DModel from a local directory."""
    del kwargs
    model_dir = (
        os.path.join(pretrained_model_name_or_path, subfolder)
        if subfolder
        else pretrained_model_name_or_path
    )
    config_path = os.path.join(model_dir, "config.json")
    init_kwargs: dict[str, Any] = {}
    if os.path.exists(config_path):
      with open(config_path, "r") as f:
        cfg = json.load(f)
      valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {
          "self"
      }
      init_kwargs = {k: v for k, v in cfg.items() if k in valid_params}
    model = cls(**init_kwargs)
    state_dict = _load_safetensors_state_dict(model_dir)
    if state_dict:
      model.load_state_dict(state_dict, strict=True)
    return model.to(torch_dtype)

  def forward(
      self,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor,
      timestep: torch.Tensor,
      img_ids: torch.Tensor,
      txt_ids: torch.Tensor,
      guidance: torch.Tensor | None = None,
      return_dict: bool = False,
  ) -> tuple[torch.Tensor, ...] | Any:
    num_txt_tokens = encoder_hidden_states.shape[1]
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
      guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_guidance_embed(timestep, guidance)
    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if img_ids.ndim == 3:
      img_ids = img_ids[0]
    if txt_ids.ndim == 3:
      txt_ids = txt_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    for block in self.transformer_blocks:
      encoder_hidden_states, hidden_states = block(
          hidden_states=hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          temb_mod_img=double_stream_mod_img,
          temb_mod_txt=double_stream_mod_txt,
          image_rotary_emb=concat_rotary_emb,
      )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for block in self.single_transformer_blocks:
      hidden_states = block(
          hidden_states=hidden_states,
          encoder_hidden_states=None,
          temb_mod=single_stream_mod,
          image_rotary_emb=concat_rotary_emb,
      )

    hidden_states = hidden_states[:, num_txt_tokens:, ...]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if not return_dict:
      return (output,)
    return types.SimpleNamespace(sample=output)


class ResnetBlock2D(nn.Module):
  """2D Residual block for VAE Encoder/Decoder."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int | None = None,
      dropout: float = 0.0,
      groups: int = 32,
      eps: float = 1e-6,
  ):
    super().__init__()
    out_channels = in_channels if out_channels is None else out_channels
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.norm1 = nn.GroupNorm(
        num_groups=groups, num_channels=in_channels, eps=eps, affine=True
    )
    self.conv1 = nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1
    )
    self.norm2 = nn.GroupNorm(
        num_groups=groups, num_channels=out_channels, eps=eps, affine=True
    )
    self.dropout = nn.Dropout(dropout)
    self.conv2 = nn.Conv2d(
        out_channels, out_channels, kernel_size=3, stride=1, padding=1
    )
    self.nonlinearity = nn.SiLU()
    self.conv_shortcut = (
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if in_channels != out_channels
        else None
    )

  def forward(
      self, input_tensor: torch.Tensor, temb: torch.Tensor | None = None
  ) -> torch.Tensor:
    del temb
    hidden_states = self.nonlinearity(self.norm1(input_tensor))
    hidden_states = self.conv1(hidden_states)
    hidden_states = self.nonlinearity(self.norm2(hidden_states))
    hidden_states = self.conv2(self.dropout(hidden_states))
    if self.conv_shortcut is not None:
      input_tensor = self.conv_shortcut(input_tensor)
    return input_tensor + hidden_states


class Downsample2D(nn.Module):
  """2D Downsampling layer for VAE Encoder."""

  def __init__(self, channels: int, out_channels: int | None = None):
    super().__init__()
    out_channels = out_channels or channels
    self.conv = nn.Conv2d(
        channels, out_channels, kernel_size=3, stride=2, padding=0
    )

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
    return self.conv(hidden_states)


class Upsample2D(nn.Module):
  """2D Upsampling layer for VAE Decoder."""

  def __init__(self, channels: int, out_channels: int | None = None):
    super().__init__()
    out_channels = out_channels or channels
    self.conv = nn.Conv2d(
        channels, out_channels, kernel_size=3, stride=1, padding=1
    )

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = F.interpolate(
        hidden_states, scale_factor=2.0, mode="nearest"
    )
    return self.conv(hidden_states)


class VaeMidAttention(nn.Module):
  """Spatial self-attention block used in VAE UNetMidBlock2D."""

  def __init__(
      self,
      query_dim: int,
      heads: int = 1,
      dim_head: int = 512,
      norm_num_groups: int = 32,
      eps: float = 1e-6,
  ):
    super().__init__()
    self.heads = heads
    self.head_dim = dim_head
    inner_dim = dim_head * heads
    self.group_norm = nn.GroupNorm(
        num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True
    )
    self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
    self.to_k = nn.Linear(query_dim, inner_dim, bias=True)
    self.to_v = nn.Linear(query_dim, inner_dim, bias=True)
    self.to_out = nn.ModuleList(
        [nn.Linear(inner_dim, query_dim, bias=True), nn.Dropout(0.0)]
    )

  def forward(
      self, hidden_states: torch.Tensor, temb: torch.Tensor | None = None
  ) -> torch.Tensor:
    del temb
    residual = hidden_states
    batch, channel, height, width = hidden_states.shape
    hidden_states = self.group_norm(hidden_states)
    hidden_states = hidden_states.view(
        batch, channel, height * width
    ).transpose(1, 2)

    q = (
        self.to_q(hidden_states)
        .view(batch, -1, self.heads, self.head_dim)
        .transpose(1, 2)
    )
    k = (
        self.to_k(hidden_states)
        .view(batch, -1, self.heads, self.head_dim)
        .transpose(1, 2)
    )
    v = (
        self.to_v(hidden_states)
        .view(batch, -1, self.heads, self.head_dim)
        .transpose(1, 2)
    )

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    out = out.transpose(1, 2).reshape(batch, -1, self.heads * self.head_dim)
    out = self.to_out[1](self.to_out[0](out))
    out = out.transpose(-1, -2).reshape(batch, channel, height, width)
    return out + residual


class UNetMidBlock2D(nn.Module):
  """Mid-block for VAE Encoder and Decoder."""

  def __init__(
      self,
      in_channels: int,
      resnet_eps: float = 1e-6,
      resnet_groups: int = 32,
      attention_head_dim: int | None = None,
      add_attention: bool = True,
  ):
    super().__init__()
    attention_head_dim = attention_head_dim or in_channels
    self.resnets = nn.ModuleList([
        ResnetBlock2D(
            in_channels=in_channels,
            out_channels=in_channels,
            eps=resnet_eps,
            groups=resnet_groups,
        ),
        ResnetBlock2D(
            in_channels=in_channels,
            out_channels=in_channels,
            eps=resnet_eps,
            groups=resnet_groups,
        ),
    ])
    attentions = []
    if add_attention:
      attentions.append(
          VaeMidAttention(
              query_dim=in_channels,
              heads=in_channels // attention_head_dim,
              dim_head=attention_head_dim,
              norm_num_groups=resnet_groups,
              eps=resnet_eps,
          )
      )
    self.attentions = nn.ModuleList(attentions)

  def forward(
      self, hidden_states: torch.Tensor, temb: torch.Tensor | None = None
  ) -> torch.Tensor:
    hidden_states = self.resnets[0](hidden_states, temb)
    for i, resnet in enumerate(self.resnets[1:]):
      if i < len(self.attentions):
        hidden_states = self.attentions[i](hidden_states, temb)
      hidden_states = resnet(hidden_states, temb)
    return hidden_states


class DownEncoderBlock2D(nn.Module):
  """Downsampling block for VAE Encoder."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      num_layers: int = 2,
      resnet_eps: float = 1e-6,
      resnet_groups: int = 32,
      add_downsample: bool = True,
  ):
    super().__init__()
    resnets = []
    for i in range(num_layers):
      in_ch = in_channels if i == 0 else out_channels
      resnets.append(
          ResnetBlock2D(
              in_channels=in_ch,
              out_channels=out_channels,
              eps=resnet_eps,
              groups=resnet_groups,
          )
      )
    self.resnets = nn.ModuleList(resnets)
    self.downsamplers = (
        nn.ModuleList([Downsample2D(out_channels, out_channels=out_channels)])
        if add_downsample
        else None
    )

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    for resnet in self.resnets:
      hidden_states = resnet(hidden_states)
    if self.downsamplers is not None:
      for downsampler in self.downsamplers:
        hidden_states = downsampler(hidden_states)
    return hidden_states


class UpDecoderBlock2D(nn.Module):
  """Upsampling block for VAE Decoder."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      num_layers: int = 3,
      resnet_eps: float = 1e-6,
      resnet_groups: int = 32,
      add_upsample: bool = True,
  ):
    super().__init__()
    resnets = []
    for i in range(num_layers):
      in_ch = in_channels if i == 0 else out_channels
      resnets.append(
          ResnetBlock2D(
              in_channels=in_ch,
              out_channels=out_channels,
              eps=resnet_eps,
              groups=resnet_groups,
          )
      )
    self.resnets = nn.ModuleList(resnets)
    self.upsamplers = (
        nn.ModuleList([Upsample2D(out_channels, out_channels=out_channels)])
        if add_upsample
        else None
    )

  def forward(
      self, hidden_states: torch.Tensor, temb: torch.Tensor | None = None
  ) -> torch.Tensor:
    for resnet in self.resnets:
      hidden_states = resnet(hidden_states, temb=temb)
    if self.upsamplers is not None:
      for upsampler in self.upsamplers:
        hidden_states = upsampler(hidden_states)
    return hidden_states


class Encoder(nn.Module):
  """VAE Encoder."""

  def __init__(
      self,
      in_channels: int = 3,
      out_channels: int = 32,
      down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
      block_out_channels: tuple[int, ...] = (64,),
      layers_per_block: int = 2,
      norm_num_groups: int = 32,
      double_z: bool = True,
      mid_block_add_attention: bool = True,
  ):
    super().__init__()
    self.conv_in = nn.Conv2d(
        in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
    )
    self.down_blocks = nn.ModuleList([])
    output_channel = block_out_channels[0]
    for i, _ in enumerate(down_block_types):
      input_channel = output_channel
      output_channel = block_out_channels[i]
      is_final_block = i == len(block_out_channels) - 1
      self.down_blocks.append(
          DownEncoderBlock2D(
              in_channels=input_channel,
              out_channels=output_channel,
              num_layers=layers_per_block,
              resnet_eps=1e-6,
              resnet_groups=norm_num_groups,
              add_downsample=not is_final_block,
          )
      )
    self.mid_block = UNetMidBlock2D(
        in_channels=block_out_channels[-1],
        resnet_eps=1e-6,
        resnet_groups=norm_num_groups,
        attention_head_dim=block_out_channels[-1],
        add_attention=mid_block_add_attention,
    )
    self.conv_norm_out = nn.GroupNorm(
        num_channels=block_out_channels[-1],
        num_groups=norm_num_groups,
        eps=1e-6,
    )
    self.conv_act = nn.SiLU()
    conv_out_channels = 2 * out_channels if double_z else out_channels
    self.conv_out = nn.Conv2d(
        block_out_channels[-1], conv_out_channels, kernel_size=3, padding=1
    )

  def forward(self, sample: torch.Tensor) -> torch.Tensor:
    sample = self.conv_in(sample)
    for down_block in self.down_blocks:
      sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    return self.conv_out(sample)


class Decoder(nn.Module):
  """VAE Decoder."""

  def __init__(
      self,
      in_channels: int = 32,
      out_channels: int = 3,
      up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
      block_out_channels: tuple[int, ...] = (64,),
      layers_per_block: int = 2,
      norm_num_groups: int = 32,
      mid_block_add_attention: bool = True,
  ):
    super().__init__()
    self.conv_in = nn.Conv2d(
        in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1
    )
    self.mid_block = UNetMidBlock2D(
        in_channels=block_out_channels[-1],
        resnet_eps=1e-6,
        resnet_groups=norm_num_groups,
        attention_head_dim=block_out_channels[-1],
        add_attention=mid_block_add_attention,
    )
    self.up_blocks = nn.ModuleList([])
    reversed_block_out_channels = list(reversed(block_out_channels))
    output_channel = reversed_block_out_channels[0]
    for i, _ in enumerate(up_block_types):
      prev_output_channel = output_channel
      output_channel = reversed_block_out_channels[i]
      is_final_block = i == len(block_out_channels) - 1
      self.up_blocks.append(
          UpDecoderBlock2D(
              in_channels=prev_output_channel,
              out_channels=output_channel,
              num_layers=layers_per_block + 1,
              resnet_eps=1e-6,
              resnet_groups=norm_num_groups,
              add_upsample=not is_final_block,
          )
      )
    self.conv_norm_out = nn.GroupNorm(
        num_channels=block_out_channels[0],
        num_groups=norm_num_groups,
        eps=1e-6,
    )
    self.conv_act = nn.SiLU()
    self.conv_out = nn.Conv2d(
        block_out_channels[0], out_channels, kernel_size=3, padding=1
    )

  def forward(self, sample: torch.Tensor) -> torch.Tensor:
    sample = self.conv_in(sample)
    sample = self.mid_block(sample)
    for up_block in self.up_blocks:
      sample = up_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    return self.conv_out(sample)


class AutoencoderKLFlux2(nn.Module):
  """Standalone PyTorch implementation of AutoencoderKLFlux2."""

  def __init__(
      self,
      in_channels: int = 3,
      out_channels: int = 3,
      down_block_types: tuple[str, ...] | list[str] = (
          "DownEncoderBlock2D",
          "DownEncoderBlock2D",
          "DownEncoderBlock2D",
          "DownEncoderBlock2D",
      ),
      up_block_types: tuple[str, ...] | list[str] = (
          "UpDecoderBlock2D",
          "UpDecoderBlock2D",
          "UpDecoderBlock2D",
          "UpDecoderBlock2D",
      ),
      block_out_channels: tuple[int, ...] | list[int] = (128, 256, 512, 512),
      decoder_block_out_channels: tuple[int, ...] | list[int] | None = None,
      layers_per_block: int = 2,
      act_fn: str = "silu",
      latent_channels: int = 32,
      norm_num_groups: int = 32,
      sample_size: int = 1024,
      force_upcast: bool = True,
      use_quant_conv: bool = True,
      use_post_quant_conv: bool = True,
      mid_block_add_attention: bool = True,
      batch_norm_eps: float = 1e-4,
      batch_norm_momentum: float = 0.1,
      patch_size: tuple[int, int] | list[int] = (2, 2),
  ):
    super().__init__()
    self.config = types.SimpleNamespace(
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        decoder_block_out_channels=(
            tuple(decoder_block_out_channels)
            if decoder_block_out_channels is not None
            else None
        ),
        layers_per_block=layers_per_block,
        act_fn=act_fn,
        latent_channels=latent_channels,
        norm_num_groups=norm_num_groups,
        sample_size=sample_size,
        force_upcast=force_upcast,
        use_quant_conv=use_quant_conv,
        use_post_quant_conv=use_post_quant_conv,
        mid_block_add_attention=mid_block_add_attention,
        batch_norm_eps=batch_norm_eps,
        batch_norm_momentum=batch_norm_momentum,
        patch_size=tuple(patch_size),
    )

    self.encoder = Encoder(
        in_channels=in_channels,
        out_channels=latent_channels,
        down_block_types=tuple(down_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
        double_z=True,
        mid_block_add_attention=mid_block_add_attention,
    )
    self.decoder = Decoder(
        in_channels=latent_channels,
        out_channels=out_channels,
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(
            decoder_block_out_channels or block_out_channels
        ),
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
        mid_block_add_attention=mid_block_add_attention,
    )
    self.quant_conv = (
        nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        if use_quant_conv
        else None
    )
    self.post_quant_conv = (
        nn.Conv2d(latent_channels, latent_channels, 1)
        if use_post_quant_conv
        else None
    )
    self.bn = nn.BatchNorm2d(
        math.prod(patch_size) * latent_channels,
        eps=batch_norm_eps,
        momentum=batch_norm_momentum,
        affine=False,
        track_running_stats=True,
    )

  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str,
      subfolder: str | None = None,
      torch_dtype: torch.dtype = torch.float32,
      **kwargs: Any,
  ) -> "AutoencoderKLFlux2":
    """Loads AutoencoderKLFlux2 from a local directory."""
    del kwargs
    model_dir = (
        os.path.join(pretrained_model_name_or_path, subfolder)
        if subfolder
        else pretrained_model_name_or_path
    )
    config_path = os.path.join(model_dir, "config.json")
    init_kwargs: dict[str, Any] = {}
    if os.path.exists(config_path):
      with open(config_path, "r") as f:
        cfg = json.load(f)
      valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {
          "self"
      }
      init_kwargs = {k: v for k, v in cfg.items() if k in valid_params}
    model = cls(**init_kwargs)
    state_dict = _load_safetensors_state_dict(model_dir)
    if state_dict:
      model.load_state_dict(state_dict, strict=True)
    return model.to(torch_dtype)

  def decode(
      self, z: torch.Tensor, return_dict: bool = True
  ) -> tuple[torch.Tensor, ...] | Any:
    if self.post_quant_conv is not None:
      z = self.post_quant_conv(z)
    dec = self.decoder(z)
    if not return_dict:
      return (dec,)
    return types.SimpleNamespace(sample=dec)
