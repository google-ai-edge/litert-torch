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
"""Self-contained static export model for Inkling-2-Mini (`inkling2_loop`).

Implements the Inkling-2-Mini looped hybrid MoE transformer architecture
strictly in rank <= 4 ops for LiteRT / ML Drift GPU compatibility:
- 24 physical decoder layers with a 3-pass core loop over layers [6, 18)
  (48 logical layer visits per token, each with its own KV + 4 conv states).
- Block Attention Residuals (`AttnRes`) and cross-pass message retrieval +
  rank-64 message gates (`(1 - forget) * h + message_gate * msg`).
- 5:1 hybrid sliding-window / full attention without RoPE, using learned
  hidden-state-conditioned relative position bias (factorized via 1D
  `tfl.gather` on `proj^T` for ML Drift GPU compatibility), per-token attention
  sinks, Q/K L2 clipping (+ log scaling on full layers), unweighted V RMSNorm,
  and sigmoid-gated output projection.
- Static depthwise causal short convolutions (post-attn, post-MLP, sliding K/V)
  and rank-4 unrolled dynamic short convolutions (full K/V).
- Sparse MoE (128 experts, top-8 sigmoid routing with correction bias and
  `route_scale`) lowered through `litert_moe_experts_forward` with Softcapped
  SwiGLU (`mlp_gate_softcap=4.0`, `mlp_up_softcap=25.0`).
"""

import math
from typing import Any, cast

# Registers torch.ops.tfl.* custom ops on import.
from litert_torch.backend import composite as composite_lib
from litert_torch.backend.torch_tfl import _ops as _
from litert_torch.generative.custom_ops import bmm_4d as bmm_lib
from litert_torch.generative.export_hf.experimental.composites import runtime_batched_matmul as runtime_bmm_lib
from litert_torch.generative.layers import moe as moe_lib
from litert_torch.generative.layers import normalization as norm_lib
import torch
from torch import nn
import torch.nn.functional as F
import transformers

SUPPORTED_MODEL_TYPES = ("inkling2_loop", "inkling2_loop_text")


def unwrap_config(config: Any) -> Any:
  """Returns the text sub-config when given a composite `Inkling2LoopConfig`."""
  if hasattr(config, "text_config") and config.text_config is not None:
    return config.text_config
  return config


class Inkling2LoopTextConfig(transformers.PretrainedConfig):
  """Configuration for the Inkling-2-Mini looped text transformer."""

  model_type = "inkling2_loop_text"
  attribute_map = {
      "sliding_window": "sliding_window_size",
      "num_local_experts": "n_routed_experts",
      "sconv_kernel_size": "conv_kernel_size",
      "model_max_length": "max_position_embeddings",
      "intermediate_size": "moe_intermediate_size",
      "routed_scaling_factor": "route_scale",
  }

  def __init__(
      self,
      vocab_size: int = 201024,
      unpadded_vocab_size: int = 200058,
      hidden_size: int = 2304,
      num_hidden_layers: int = 24,
      num_attention_heads: int = 24,
      num_key_value_heads: int = 8,
      head_dim: int = 128,
      swa_num_attention_heads: int = 24,
      swa_num_key_value_heads: int = 8,
      swa_head_dim: int = 128,
      sliding_window_size: int = 1024,
      d_rel: int = 16,
      rel_extent: int = 1024,
      log_scaling_n_floor: int = 8192,
      log_scaling_alpha: float = 0.1,
      layer_types: list[str] | None = None,
      local_layer_ids: list[int] | None = None,
      max_position_embeddings: int = 1048576,
      rms_norm_eps: float = 1e-6,
      conv_kernel_size: int = 4,
      hidden_act: str = "silu",
      moe_intermediate_size: int = 1152,
      n_routed_experts: int = 128,
      num_experts_per_tok: int = 8,
      route_scale: float = 8.0,
      attention_dropout: float = 0.0,
      initializer_range: float = 0.02,
      pad_token_id: int | None = None,
      bos_token_id: int | None = None,
      eos_token_id: int | None = 200006,
      qk_l2_clip_threshold: float = 22.627416997969522,
      rel_l2_clip_threshold: float = 8.0,
      n_layers_per_attn_res_block: int = 6,
      mlp_gate_softcap: float = 4.0,
      mlp_up_softcap: float = 25.0,
      final_logit_softcapping: float = 30.0,
      scoring_func: str = "sigmoid",
      use_cache: bool = True,
      loop_layer_index_start: int = 6,
      loop_layer_index_end: int = 18,
      loop_num_passes: int = 3,
      loop_gate_rank: int = 64,
      loop_config: dict[str, Any] | None = None,
      **kwargs: Any,
  ):
    if loop_config is not None:
      loop_layer_index_start = int(
          loop_config.get("core_start", loop_layer_index_start)
      )
      loop_layer_index_end = int(
          loop_config.get("core_end", loop_layer_index_end)
      )
      loop_num_passes = int(loop_config.get("num_passes", loop_num_passes))
      loop_gate_rank = int(loop_config.get("gate_rank", loop_gate_rank))

    # Handle checkpoint alias keys passed via kwargs.
    if "sconv_kernel_size" in kwargs:
      conv_kernel_size = int(kwargs.pop("sconv_kernel_size"))
    if "intermediate_size" in kwargs:
      moe_intermediate_size = int(kwargs.pop("intermediate_size"))
    if "model_max_length" in kwargs:
      max_position_embeddings = int(kwargs.pop("model_max_length"))

    if layer_types is None:
      if local_layer_ids is not None:
        local_set = set(int(i) for i in local_layer_ids)
        layer_types = [
            "hybrid_sliding" if i in local_set else "hybrid"
            for i in range(num_hidden_layers)
        ]
      else:
        layer_types = [
            "hybrid"
            if (i + 1) % n_layers_per_attn_res_block == 0
            else "hybrid_sliding"
            for i in range(num_hidden_layers)
        ]

    self.vocab_size = int(vocab_size)
    self.unpadded_vocab_size = int(unpadded_vocab_size)
    self.hidden_size = int(hidden_size)
    self.num_hidden_layers = int(num_hidden_layers)
    self.num_attention_heads = int(num_attention_heads)
    self.num_key_value_heads = int(num_key_value_heads)
    self.head_dim = int(head_dim)
    self.swa_num_attention_heads = int(swa_num_attention_heads)
    self.swa_num_key_value_heads = int(swa_num_key_value_heads)
    self.swa_head_dim = int(swa_head_dim)
    self.sliding_window_size = int(sliding_window_size)
    self.d_rel = int(d_rel)
    self.rel_extent = int(rel_extent)
    self.log_scaling_n_floor = int(log_scaling_n_floor)
    self.log_scaling_alpha = float(log_scaling_alpha)
    self.layer_types = list(layer_types)
    self.max_position_embeddings = int(max_position_embeddings)
    self.rms_norm_eps = float(rms_norm_eps)
    self.conv_kernel_size = int(conv_kernel_size)
    self.hidden_act = str(hidden_act)
    self.moe_intermediate_size = int(moe_intermediate_size)
    self.n_routed_experts = int(n_routed_experts)
    self.num_experts_per_tok = int(num_experts_per_tok)
    self.route_scale = float(route_scale)
    self.attention_dropout = float(attention_dropout)
    self.initializer_range = float(initializer_range)
    self.qk_l2_clip_threshold = float(qk_l2_clip_threshold)
    self.rel_l2_clip_threshold = float(rel_l2_clip_threshold)
    self.n_layers_per_attn_res_block = int(n_layers_per_attn_res_block)
    self.mlp_gate_softcap = float(mlp_gate_softcap)
    self.mlp_up_softcap = float(mlp_up_softcap)
    self.final_logit_softcapping = float(final_logit_softcapping)
    self.scoring_func = str(scoring_func)
    self.use_cache = bool(use_cache)
    self.number_of_conv_states = 4
    self.loop_layer_index_start = int(loop_layer_index_start)
    self.loop_layer_index_end = int(loop_layer_index_end)
    self.loop_num_passes = int(loop_num_passes)
    self.loop_gate_rank = int(loop_gate_rank)

    self.pad_token_id = pad_token_id
    self.bos_token_id = bos_token_id
    self.eos_token_id = eos_token_id
    self.tie_word_embeddings = False

    super().__init__(**kwargs)

  @property
  def total_num_hidden_layers(self) -> int:
    """Total logical decoder layer visits across all loop passes."""
    num_loop_layers = self.loop_layer_index_end - self.loop_layer_index_start
    return self.num_hidden_layers + (self.loop_num_passes - 1) * num_loop_layers

  def physical_layer_index_for_cache(self, cache_layer_idx: int) -> int:
    """Maps a logical cache index [0, total_num_hidden_layers) to physical."""
    if cache_layer_idx < self.num_hidden_layers:
      return cache_layer_idx
    num_loop_layers = self.loop_layer_index_end - self.loop_layer_index_start
    extra_idx = cache_layer_idx - self.num_hidden_layers
    return self.loop_layer_index_start + (extra_idx % num_loop_layers)


class Inkling2LoopConfig(transformers.PretrainedConfig):
  """Top-level composite configuration for `Inkling-2-Mini`."""

  model_type = "inkling2_loop"

  def __init__(
      self,
      text_config: Inkling2LoopTextConfig | dict[str, Any] | None = None,
      vision_config: dict[str, Any] | None = None,
      image_token_id: int = 200054,
      eos_token_id: int | None = 200006,
      **kwargs: Any,
  ):
    if isinstance(text_config, dict):
      text_config = Inkling2LoopTextConfig(**text_config)
    elif text_config is None:
      text_config = Inkling2LoopTextConfig()
    if eos_token_id is not None and text_config.eos_token_id is None:
      text_config.eos_token_id = eos_token_id
    self.text_config = text_config
    self.vision_config = vision_config
    self.image_token_id = int(image_token_id)
    self.eos_token_id = eos_token_id
    self.tie_word_embeddings = False
    super().__init__(**kwargs)

  def get_text_config(
      self, decoder: Any = None, encoder: Any = None
  ) -> Inkling2LoopTextConfig:
    del decoder, encoder
    return self.text_config


class Inkling2LoopUnweightedRMSNorm(nn.Module):
  """RMSNorm without learned scale, lowered via fused `odml.rms_norm`."""

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.register_buffer(
        "ones", torch.ones(hidden_size, dtype=torch.float32), persistent=False
    )

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    out = norm_lib.rms_norm_with_hlfb(
        hidden_states.float(),
        self.ones.float(),
        self.eps,
        torch.ones(1, dtype=torch.float32, device=hidden_states.device),
    )
    return out.to(input_dtype)


class Inkling2LoopRMSNorm(nn.Module):
  """Standard T5-style RMSNorm lowered via fused `odml.rms_norm`."""

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    out = norm_lib.rms_norm_with_hlfb(
        hidden_states.float(),
        self.weight.float(),
        self.variance_epsilon,
        torch.ones(1, dtype=torch.float32, device=hidden_states.device),
    )
    return out.to(input_dtype)


class Inkling2LoopSoftcap(nn.Module):
  """Elementwise tanh softcap: `tanh(x / cap) * cap`."""

  def __init__(self, softcap: float):
    super().__init__()
    self.softcap = float(softcap)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.tanh(hidden_states / self.softcap) * self.softcap


class Inkling2LoopMessageGate(nn.Module):
  """Rank-64 cross-pass gate: `(1 - forget) * h + message_gate * message`."""

  def __init__(self, config: Inkling2LoopTextConfig):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.norm = Inkling2LoopUnweightedRMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    )
    self.down_proj = nn.Linear(
        2 * config.hidden_size, config.loop_gate_rank, bias=False
    )
    # Fused `[forget_gate_proj; message_gate_proj]` halves FC and TANH ops.
    self.gate_proj = nn.Linear(
        config.loop_gate_rank, 2 * config.hidden_size, bias=False
    )

  def forward(
      self, hidden_states: torch.Tensor, message: torch.Tensor
  ) -> torch.Tensor:
    h_f32 = hidden_states.float()
    m_f32 = message.float()
    gate_in = torch.cat([self.norm(h_f32), self.norm(m_f32)], dim=-1)
    gate_h = self.down_proj(gate_in.to(self.down_proj.weight.dtype))
    gates = torch.tanh(
        self.gate_proj(gate_h.to(self.gate_proj.weight.dtype)).float()
    )
    forget_gate, message_gate = gates.split(self.hidden_size, dim=-1)
    return (1.0 - forget_gate) * h_f32 + message_gate * m_f32


class Inkling2LoopBlockAttentionResidual(nn.Module):
  """Static rank-4 Block Attention Residual (`AttnRes`) over saved blocks."""

  def __init__(self, config: Inkling2LoopTextConfig, num_queries: int = 1):
    super().__init__()
    self.num_queries = num_queries
    self.hidden_size = config.hidden_size
    shape = (
        (config.hidden_size,)
        if num_queries == 1
        else (num_queries, config.hidden_size)
    )
    self.query = nn.Parameter(torch.zeros(*shape, dtype=torch.float32))
    self.norm = Inkling2LoopUnweightedRMSNorm(config.hidden_size, eps=1e-6)

  def forward(
      self,
      normed_blocks_stack: torch.Tensor | None,
      partial_block_states: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    # Reuse already-normalized immutable block stack `[B, T, K, D]` and only
    # normalize the single active partial state `[B, T, D]`.
    normed_partial = self.norm(partial_block_states.float())
    if normed_blocks_stack is None:
      return tuple(normed_partial for _ in range(self.num_queries))

    normed = torch.cat(
        [normed_blocks_stack, normed_partial.unsqueeze(2)], dim=2
    )  # [B, T, N_src, D]
    scale = self.hidden_size**-0.5

    if self.num_queries == 1:
      logits = (normed * self.query.float()).sum(dim=-1, keepdim=True) * scale
      weights = F.softmax(logits, dim=2)
      out = (weights * normed).sum(dim=2)  # [B, T, D]
      return (out,)

    # F.linear avoids StableHLO 4D-x-2D batch_matmul rank-folding bug.
    logits = F.linear(normed, self.query.float()) * scale  # [B, T, N_src, Q]
    weights = F.softmax(logits, dim=2)  # [B, T, N_src, Q]
    # [B, T, Q, N_src] @ [B, T, N_src, D] -> [B, T, Q, D]
    out = torch.matmul(weights.transpose(-2, -1), normed)
    return tuple(out[:, :, i, :] for i in range(self.num_queries))


class Inkling2LoopMessageAttentionResidual(nn.Module):
  """Retrieves 4 cross-pass messages `[Q, K, V, MLP]` for all loop layers."""

  def __init__(self, config: Inkling2LoopTextConfig):
    super().__init__()
    self.num_loop_layers = (
        config.loop_layer_index_end - config.loop_layer_index_start
    )
    self.hidden_size = config.hidden_size
    # Flattened `[4 * L, D]` queries compute all loop layers' cross-pass
    # messages in a single batched FC + softmax + matmul per pass.
    self.all_queries = nn.Parameter(
        torch.zeros(
            self.num_loop_layers * 4, config.hidden_size, dtype=torch.float32
        )
    )
    self.norm = Inkling2LoopUnweightedRMSNorm(config.hidden_size, eps=1e-6)

  def forward(
      self,
      prev_normed_blocks_stack: torch.Tensor | None,
      previous_hidden_states: torch.Tensor,
  ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    normed_prev = self.norm(previous_hidden_states.float())
    if prev_normed_blocks_stack is None:
      return [
          (normed_prev, normed_prev, normed_prev, normed_prev)
          for _ in range(self.num_loop_layers)
      ]
    normed = torch.cat(
        [prev_normed_blocks_stack, normed_prev.unsqueeze(2)], dim=2
    )  # [B, T, N_src, D]
    logits = F.linear(normed, self.all_queries.float()) * (
        self.hidden_size**-0.5
    )  # [B, T, N_src, 4 * L]
    weights = F.softmax(logits, dim=2)  # [B, T, N_src, 4 * L]
    out = torch.matmul(weights.transpose(-2, -1), normed)  # [B, T, 4 * L, D]
    msgs: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    for i in range(self.num_loop_layers):
      base = i * 4
      msgs.append((
          out[:, :, base, :],
          out[:, :, base + 1, :],
          out[:, :, base + 2, :],
          out[:, :, base + 3, :],
      ))
    return msgs


class Inkling2LoopL2Clip(nn.Module):
  """Clips per-head L2 norm to `threshold` with FP16-safe max-abs scaling."""

  def __init__(self, threshold: float):
    super().__init__()
    self.threshold = float(threshold)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    x = hidden_states.float()
    max_abs = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    x_scaled = x / max_abs
    norm = max_abs * torch.sqrt(x_scaled.square().sum(dim=-1, keepdim=True))
    return (x * (self.threshold / norm.clamp(min=self.threshold))).to(
        input_dtype
    )


class Inkling2LoopScaledL2Clip(nn.Module):
  """Per-head learned scale followed by FP16-safe L2 norm clipping."""

  def __init__(self, num_heads: int, head_dim: int, threshold: float):
    super().__init__()
    self.threshold = float(threshold)
    self.scale = nn.Parameter(torch.ones(num_heads, head_dim))

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    del position_ids
    input_dtype = hidden_states.dtype
    x = hidden_states.float() * self.scale.float()
    max_abs = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    x_scaled = x / max_abs
    norm = max_abs * torch.sqrt(x_scaled.square().sum(dim=-1, keepdim=True))
    return (x * (self.threshold / norm.clamp(min=self.threshold))).to(
        input_dtype
    )


class Inkling2LoopLogScale(nn.Module):
  """Logarithmic position scaling for full-attention Q and relative states."""

  def __init__(self, config: Inkling2LoopTextConfig):
    super().__init__()
    self.log_scaling_alpha = float(config.log_scaling_alpha)
    self.log_scaling_n_floor = float(config.log_scaling_n_floor)

  def forward(
      self, hidden_states: torch.Tensor, position_ids: torch.Tensor
  ) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    pos_ratio = ((position_ids.float() + 1.0) / self.log_scaling_n_floor).clamp(
        min=1.0
    )
    tau: torch.Tensor = torch.log(pos_ratio) * self.log_scaling_alpha + 1.0
    # position_ids is [B, T]; hidden_states is [B, T, H, D].
    return (hidden_states.float() * tau.unsqueeze(-1).unsqueeze(-1)).to(
        input_dtype
    )


class Inkling2LoopLogScaledL2Clip(nn.Module):
  """Learned scale -> L2 clip -> log position scale."""

  def __init__(
      self,
      config: Inkling2LoopTextConfig,
      num_heads: int,
      head_dim: int,
      threshold: float,
  ):
    super().__init__()
    self.scale = nn.Parameter(torch.ones(num_heads, head_dim))
    self.l2_clip = Inkling2LoopL2Clip(threshold)
    self.log_scale = Inkling2LoopLogScale(config)

  def forward(
      self, hidden_states: torch.Tensor, position_ids: torch.Tensor
  ) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    x = self.l2_clip(hidden_states.float() * self.scale.float()).to(input_dtype)
    return self.log_scale(x, position_ids)


class Inkling2LoopRelativePositionBias(nn.Module):
  """Learned relative position bias via `F.linear` + 1D `tfl.gather` for GPU.

  Computes `rel_logits = rel_states @ proj` (`[B, T, H, rel_extent]`, 4x fewer
  FLOPs than multiplying against the full cache length) and gathers with 1D
  `tfl.gather` (zero `GATHER_ND` and zero BF16 constant gathers).
  """

  def __init__(
      self,
      config: Inkling2LoopTextConfig,
      num_heads: int,
      rel_extent: int,
      log_scaling: bool,
      include_linear: bool = True,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.d_rel = config.d_rel
    self.rel_extent = rel_extent
    self.linear: nn.Linear | None = (
        nn.Linear(config.hidden_size, num_heads * config.d_rel, bias=False)
        if include_linear
        else None
    )
    if log_scaling:
      self.norm: nn.Module = Inkling2LoopLogScaledL2Clip(
          config, num_heads, config.d_rel, config.rel_l2_clip_threshold
      )
    else:
      self.norm = Inkling2LoopScaledL2Clip(
          num_heads, config.d_rel, config.rel_l2_clip_threshold
      )
    # Stored pre-transposed as `[rel_extent, d_rel]` for `F.linear`.
    self.proj_t = nn.Parameter(torch.empty(rel_extent, config.d_rel))

  def forward(
      self,
      hidden_states: torch.Tensor,
      relative_distance: torch.Tensor,
      position_ids: torch.Tensor,
      rel_raw: torch.Tensor | None = None,
      precomputed_rel: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> torch.Tensor:
    # hidden_states: [B, T, D]; relative_distance: [T, S] int32.
    batch_size, seq_len, _ = hidden_states.shape
    cache_len = relative_distance.shape[1]

    if rel_raw is None:
      assert self.linear is not None
      rel_raw = self.linear(hidden_states)
    rel_states = rel_raw.view(batch_size, seq_len, self.num_heads, self.d_rel)
    rel_states = self.norm(rel_states, position_ids)  # [B, T, H, d_rel]

    # [B, T, H, d_rel] @ [d_rel, rel_extent] -> [B, T, H, rel_extent].
    rel_logits = F.linear(rel_states.to(self.proj_t.dtype), self.proj_t).float()

    if precomputed_rel is not None:
      flat_idx, valid_mask = precomputed_rel
    else:
      clamped = relative_distance.clamp(0, self.rel_extent - 1).to(torch.int32)
      if seq_len == 1:
        flat_idx = clamped.reshape(-1)
      else:
        row_offsets = (
            torch.arange(
                seq_len, device=clamped.device, dtype=torch.int32
            ).unsqueeze(1)
            * self.rel_extent
        )
        flat_idx = (row_offsets + clamped).reshape(-1)
      valid_dist = (relative_distance >= 0) & (
          relative_distance < self.rel_extent
      )
      valid_mask = valid_dist.view(1, 1, seq_len, cache_len)

    if seq_len == 1:
      # 1D gather along axis 3: [1, 1, H, rel_extent] -> [1, 1, H, S].
      # Keep [1, 1, H, S] without an extra transpose; attention reshapes once
      # into [1, K, G, S].
      position_bias = torch.ops.tfl.gather(rel_logits, flat_idx, 3)
      valid_mask = valid_mask.view(1, 1, 1, cache_len)
    else:
      # Flatten `[T, rel_extent, H]` -> `[T * rel_extent, H]`, 1D gather along
      # axis 0 -> `[T * S, H]` -> `[1, H, T, S]`.
      rel_flat = rel_logits.transpose(2, 3).reshape(
          seq_len * self.rel_extent, self.num_heads
      )
      gathered = torch.ops.tfl.gather(rel_flat, flat_idx, 0)
      position_bias = gathered.view(
          1, seq_len, cache_len, self.num_heads
      ).permute(0, 3, 1, 2)

    return torch.where(
        valid_mask, position_bias, torch.zeros_like(position_bias)
    ).to(rel_states.dtype)


def _short_conv_step_decomp(
    conv_in: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    residual_scale: float,
    dyn_weight: torch.Tensor | None = None,
    num_heads: int = 0,
    num_valid_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """CPU decomposition for static or dynamic `odml.short_conv_step`."""
  batch_size, seq_len, hidden_size = conv_in.shape
  k = conv_weight.shape[1]
  state_len = k - 1
  padded = torch.cat([conv_state, conv_in], dim=1)

  if dyn_weight is not None and num_heads > 0:
    head_dim = hidden_size // num_heads
    dyn_w = dyn_weight.float().view(batch_size, seq_len, num_heads, k)
    acc: torch.Tensor | None = None
    for tap_idx in range(k):
      lag = state_len - tap_idx
      x_tap_flat = padded[:, tap_idx : tap_idx + seq_len, :]
      x_tap = x_tap_flat.view(batch_size, seq_len, num_heads, head_dim)
      dyn_term = (x_tap * dyn_w[:, :, :, lag : lag + 1]).reshape(
          batch_size, seq_len, hidden_size
      )
      term = dyn_term + x_tap_flat * conv_weight[:, tap_idx : tap_idx + 1, :]
      acc = term if acc is None else (acc + term)
    assert acc is not None
    conv_out = acc
  else:
    conv_out = padded[:, 0:seq_len, :] * conv_weight[:, 0:1, :]
    for tap_idx in range(1, k):
      conv_out = (
          conv_out
          + padded[:, tap_idx : tap_idx + seq_len, :]
          * conv_weight[:, tap_idx : tap_idx + 1, :]
      )

  if num_valid_tokens is None:
    next_state = padded[:, -state_len:, :]
  else:
    indices = num_valid_tokens.to(torch.int32) + torch.arange(
        state_len, device=padded.device, dtype=torch.int32
    )
    next_state = torch.ops.tfl.gather(padded, indices, 1)

  out = (conv_out + conv_in) * residual_scale
  return out, next_state


def _apply_short_conv_step_composite(
    conv_in: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    residual_scale: float,
    dyn_weight: torch.Tensor | None = None,
    num_heads: int = 0,
    num_valid_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Emits fused in-place `odml.short_conv_step` for static or dynamic conv."""
  k = int(conv_weight.shape[1])
  attrs: dict[str, int | float | bool] = {
      "conv_L_cache": k,
      "is_gated": False,
      "use_silu": False,
      "residual_scale": float(residual_scale),
  }
  if dyn_weight is not None and num_heads > 0:
    attrs["num_heads"] = int(num_heads)
  builder = composite_lib.StableHLOCompositeBuilder(
      name="odml.short_conv_step", attr=attrs
  )
  if dyn_weight is not None and num_valid_tokens is not None:
    conv_in, conv_state, conv_weight, dyn_weight, num_valid_tokens = (
        builder.mark_inputs(
            conv_in, conv_state, conv_weight, dyn_weight, num_valid_tokens
        )
    )
  elif dyn_weight is not None:
    conv_in, conv_state, conv_weight, dyn_weight = builder.mark_inputs(
        conv_in, conv_state, conv_weight, dyn_weight
    )
  elif num_valid_tokens is not None:
    conv_in, conv_state, conv_weight, num_valid_tokens = builder.mark_inputs(
        conv_in, conv_state, conv_weight, num_valid_tokens
    )
  else:
    conv_in, conv_state, conv_weight = builder.mark_inputs(
        conv_in, conv_state, conv_weight
    )

  out, next_state = _short_conv_step_decomp(
      conv_in,
      conv_state,
      conv_weight,
      residual_scale=residual_scale,
      dyn_weight=dyn_weight,
      num_heads=num_heads,
      num_valid_tokens=num_valid_tokens,
  )
  out, next_state = builder.mark_outputs(out, next_state)
  return out, next_state


class Inkling2LoopShortConvolution(nn.Module):
  """Static depthwise causal conv1d + residual scaled by `1 / sqrt(K + 1)`."""

  def __init__(
      self,
      hidden_size: int,
      conv_kernel_size: int,
      layer_idx: int,
      conv_idx: int,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.layer_idx = layer_idx
    self.conv_idx = conv_idx
    self.conv_kernel_size = conv_kernel_size
    self.inv_scale = 1.0 / math.sqrt(conv_kernel_size + 1)
    # Channels-last `[1, K, C]` (oldest -> newest) for `odml.short_conv_step`.
    self.conv_weight = nn.Parameter(
        torch.zeros(1, conv_kernel_size, hidden_size)
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      dynamic_input: torch.Tensor | None = None,
      cache: Any = None,
      valid_mask: torch.Tensor | None = None,
      num_valid_tokens: torch.Tensor | None = None,
      cache_offset: int = 0,
      apply_gpu_composites: bool = False,
  ) -> torch.Tensor:
    del dynamic_input
    cache_layer_idx = self.layer_idx + cache_offset
    input_dtype = hidden_states.dtype
    x_f32 = hidden_states.float()
    if valid_mask is not None:
      conv_in = x_f32 * valid_mask.unsqueeze(-1).float()
    else:
      conv_in = x_f32

    state_len = self.conv_kernel_size - 1
    if cache is not None:
      conv_state = cache.get_conv_state(cache_layer_idx, self.conv_idx).float()
    else:
      conv_state = torch.zeros(
          (x_f32.shape[0], state_len, self.hidden_size),
          dtype=torch.float32,
          device=x_f32.device,
      )

    if apply_gpu_composites and cache is not None:
      out, next_state = _apply_short_conv_step_composite(
          conv_in,
          conv_state,
          self.conv_weight.float(),
          residual_scale=self.inv_scale,
          num_valid_tokens=num_valid_tokens,
      )
      cache.set_conv_state(cache_layer_idx, self.conv_idx, next_state)
      return out.to(input_dtype)

    out, next_state = _short_conv_step_decomp(
        conv_in,
        conv_state,
        self.conv_weight.float(),
        residual_scale=self.inv_scale,
        num_valid_tokens=num_valid_tokens,
    )
    if cache is not None:
      cache.set_conv_state(cache_layer_idx, self.conv_idx, next_state)
    return out.to(input_dtype)


class Inkling2LoopDynamicShortConvolution(nn.Module):
  """Dynamic causal short conv strictly in rank <= 4 for full-attention K/V."""

  def __init__(
      self,
      hidden_size: int,
      input_size: int,
      num_heads: int,
      conv_kernel_size: int,
      layer_idx: int,
      conv_idx: int,
      include_weight_proj: bool = True,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.conv_kernel_size = conv_kernel_size
    self.layer_idx = layer_idx
    self.conv_idx = conv_idx
    self.inv_scale = 1.0 / math.sqrt(2 * conv_kernel_size + 1)
    # Channels-last `[1, K, C]` (oldest -> newest) for `odml.short_conv_step`.
    self.conv_weight = nn.Parameter(
        torch.zeros(1, conv_kernel_size, hidden_size)
    )
    self.weight_proj: nn.Linear | None = (
        nn.Linear(input_size, num_heads * conv_kernel_size, bias=False)
        if include_weight_proj
        else None
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      dynamic_input: torch.Tensor | None = None,
      cache: Any = None,
      valid_mask: torch.Tensor | None = None,
      num_valid_tokens: torch.Tensor | None = None,
      cache_offset: int = 0,
      dyn_w_raw: torch.Tensor | None = None,
      apply_gpu_composites: bool = False,
  ) -> torch.Tensor:
    cache_layer_idx = self.layer_idx + cache_offset
    input_dtype = hidden_states.dtype
    batch_size = hidden_states.shape[0]
    state_len = self.conv_kernel_size - 1

    if dyn_w_raw is None:
      assert self.weight_proj is not None and dynamic_input is not None
      dyn_w_raw = self.weight_proj(dynamic_input)
    dyn_w_f32 = dyn_w_raw.float()

    x_f32 = hidden_states.float()
    if valid_mask is not None:
      conv_in = x_f32 * valid_mask.unsqueeze(-1).float()
    else:
      conv_in = x_f32

    if cache is not None:
      conv_state = cache.get_conv_state(cache_layer_idx, self.conv_idx).float()
    else:
      conv_state = torch.zeros(
          (batch_size, state_len, self.hidden_size),
          dtype=torch.float32,
          device=x_f32.device,
      )

    if apply_gpu_composites and cache is not None:
      out, next_state = _apply_short_conv_step_composite(
          conv_in,
          conv_state,
          self.conv_weight.float(),
          residual_scale=self.inv_scale,
          dyn_weight=dyn_w_f32,
          num_heads=self.num_heads,
          num_valid_tokens=num_valid_tokens,
      )
      cache.set_conv_state(cache_layer_idx, self.conv_idx, next_state)
      return out.to(input_dtype)

    out, next_state = _short_conv_step_decomp(
        conv_in,
        conv_state,
        self.conv_weight.float(),
        residual_scale=self.inv_scale,
        dyn_weight=dyn_w_f32,
        num_heads=self.num_heads,
        num_valid_tokens=num_valid_tokens,
    )
    if cache is not None:
      cache.set_conv_state(cache_layer_idx, self.conv_idx, next_state)
    return out.to(input_dtype)


class Inkling2LoopQueryProjection(nn.Module):
  """Query norm/L2Clip (+ optional log scaling); linear may be fused upstream."""

  def __init__(
      self,
      config: Inkling2LoopTextConfig,
      num_heads: int,
      head_dim: int,
      log_scaling: bool,
      include_linear: bool = True,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.linear: nn.Linear | None = (
        nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        if include_linear
        else None
    )
    if log_scaling:
      self.norm: nn.Module = Inkling2LoopLogScaledL2Clip(
          config, num_heads, head_dim, config.qk_l2_clip_threshold
      )
    else:
      self.norm = Inkling2LoopScaledL2Clip(
          num_heads, head_dim, config.qk_l2_clip_threshold
      )

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_ids: torch.Tensor,
      q_raw: torch.Tensor | None = None,
  ) -> torch.Tensor:
    b, t, _ = hidden_states.shape
    if q_raw is None:
      assert self.linear is not None
      q_raw = self.linear(hidden_states)
    states = q_raw.view(b, t, self.num_heads, self.head_dim)
    return self.norm(states, position_ids).transpose(1, 2)  # [B, H, T, D]


class Inkling2LoopKeyValueProjection(nn.Module):
  """K/V projection + short conv (static or dynamic) + L2Clip / RMSNorm."""

  def __init__(
      self,
      config: Inkling2LoopTextConfig,
      num_heads: int,
      head_dim: int,
      layer_idx: int,
      conv_idx: int,
      is_key: bool,
      dynamic: bool = False,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.kv_dim = num_heads * head_dim
    self.dynamic = dynamic
    self.dyn_w_dim = num_heads * config.conv_kernel_size if dynamic else 0
    if dynamic:
      # Fuse `k/v linear` and `sconv.weight_proj` into one FC on full layers.
      self.fused_linear = nn.Linear(
          config.hidden_size, self.kv_dim + self.dyn_w_dim, bias=False
      )
      self.sconv: nn.Module = Inkling2LoopDynamicShortConvolution(
          self.kv_dim,
          config.hidden_size,
          num_heads,
          config.conv_kernel_size,
          layer_idx,
          conv_idx=conv_idx,
          include_weight_proj=False,
      )
    else:
      self.linear = nn.Linear(config.hidden_size, self.kv_dim, bias=False)
      self.sconv = Inkling2LoopShortConvolution(
          self.kv_dim,
          config.conv_kernel_size,
          layer_idx,
          conv_idx=conv_idx,
      )
    self.norm: nn.Module = (
        Inkling2LoopL2Clip(config.qk_l2_clip_threshold)
        if is_key
        else Inkling2LoopUnweightedRMSNorm(head_dim, eps=config.rms_norm_eps)
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      cache: Any = None,
      valid_mask: torch.Tensor | None = None,
      num_valid_tokens: torch.Tensor | None = None,
      cache_offset: int = 0,
      apply_gpu_composites: bool = False,
  ) -> torch.Tensor:
    b, t, _ = hidden_states.shape
    if self.dynamic:
      fused = self.fused_linear(hidden_states)
      states, dyn_w_raw = fused.split([self.kv_dim, self.dyn_w_dim], dim=-1)
      states = self.sconv(
          states,
          cache=cache,
          valid_mask=valid_mask,
          num_valid_tokens=num_valid_tokens,
          cache_offset=cache_offset,
          dyn_w_raw=dyn_w_raw,
          apply_gpu_composites=apply_gpu_composites,
      )
    else:
      states = self.linear(hidden_states)
      states = self.sconv(
          states,
          hidden_states,
          cache=cache,
          valid_mask=valid_mask,
          num_valid_tokens=num_valid_tokens,
          cache_offset=cache_offset,
          apply_gpu_composites=apply_gpu_composites,
      )
    states = states.view(b, t, self.num_heads, self.head_dim)
    return self.norm(states).transpose(1, 2)  # [B, H_kv, T, D]


class Inkling2LoopGatedOutputProjection(nn.Module):
  """Sigmoid-gated output projection (gate linear may be fused upstream)."""

  def __init__(
      self,
      config: Inkling2LoopTextConfig,
      num_heads: int,
      head_dim: int,
      include_gate_proj: bool = True,
  ):
    super().__init__()
    self.gate_proj: nn.Linear | None = (
        nn.Linear(config.hidden_size, num_heads * head_dim, bias=False)
        if include_gate_proj
        else None
    )
    self.linear = nn.Linear(
        num_heads * head_dim, config.hidden_size, bias=False
    )

  def forward(
      self,
      attn_output: torch.Tensor,
      hidden_states: torch.Tensor,
      gate_raw: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if gate_raw is None:
      assert self.gate_proj is not None
      gate_raw = self.gate_proj(hidden_states)
    gate = torch.sigmoid(gate_raw.float())
    gated = (attn_output.float() * gate).to(self.linear.weight.dtype)
    return self.linear(gated)


def _inkling2_gqa_attention_with_bias_and_sinks(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_bias: torch.Tensor,
    sinks: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    k_ts_idx: int,
    v_ts_idx: int,
    apply_gpu_composites: bool = False,
    param_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
  """Rank-4 GQA attention with relative bias and per-token attention sinks."""
  b, num_heads, q_len, head_dim = query_states.shape
  num_kv_heads = num_heads // num_key_value_groups
  use_runtime_bmm = (
      apply_gpu_composites
      and param_tensor is not None
      and k_ts_idx == 2
      and v_ts_idx == 3
  )

  # Group Q heads into [B, K, G*T, D] to avoid expanding KV cache.
  # Match cache dtype (`float32` under `model_dtype="bfloat16"`) for `bmm_4d`.
  q_scaled = (
      (query_states * scaling)
      .to(key_cache.dtype)
      .reshape(b, num_kv_heads, num_key_value_groups * q_len, head_dim)
  )
  if use_runtime_bmm:
    assert param_tensor is not None
    attn_weights = runtime_bmm_lib.runtime_bmm(
        q_scaled, key_cache, param_tensor, is_global=True, is_src=False
    )
  elif k_ts_idx == 2:
    # key_cache: [B, K, S, D] -> bmm_4d contracts last dim D -> [B, K, G*T, S].
    attn_weights = bmm_lib.bmm_4d(q_scaled, key_cache)
  else:
    # key_cache: [B, K, D, S].
    attn_weights = torch.matmul(q_scaled, key_cache)

  kv_len = attn_weights.shape[-1]
  neg_fill = torch.full(
      (1,), -1e4, dtype=attn_weights.dtype, device=attn_weights.device
  )
  if q_len == 1:
    # Stay in grouped `[B, K, G, S]` layout throughout decode attention to
    # eliminate 4 reshapes per layer (192 GPU dispatches/token).
    attn_weights = attn_weights + position_bias.reshape(
        b, num_kv_heads, num_key_value_groups, kv_len
    ).to(attn_weights.dtype)
    if attention_mask is not None:
      if use_runtime_bmm:
        # Replace uncomputed tail slots from `runtime_bmm` cleanly with -1e4.
        attn_weights = torch.where(
            attention_mask > -1.0, attn_weights, neg_fill
        )
      else:
        attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
    sinks_grouped = sinks.reshape(b, num_kv_heads, num_key_value_groups, 1)
    combined_logits = torch.cat(
        [attn_weights.float(), sinks_grouped.float()], dim=-1
    )
    probs_grouped = F.softmax(combined_logits, dim=-1)[..., :-1].to(
        value_cache.dtype
    )
  else:
    attn_weights = attn_weights.reshape(b, num_heads, q_len, kv_len)
    attn_weights = attn_weights + position_bias.to(attn_weights.dtype)
    if attention_mask is not None:
      if use_runtime_bmm:
        attn_weights = torch.where(
            attention_mask > -1.0, attn_weights, neg_fill
        )
      else:
        attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)

    # Append per-token sink logit, compute fp32 softmax, drop sink column.
    combined_logits = torch.cat([attn_weights.float(), sinks.float()], dim=-1)
    probs = F.softmax(combined_logits, dim=-1)[..., :-1].to(value_cache.dtype)
    probs_grouped = probs.reshape(
        b, num_kv_heads, num_key_value_groups * q_len, kv_len
    )

  if use_runtime_bmm:
    assert param_tensor is not None
    attn_output = runtime_bmm_lib.runtime_bmm(
        probs_grouped, value_cache, param_tensor, is_global=True, is_src=True
    )
  elif v_ts_idx == 3:
    # value_cache: [B, K, D, S] -> bmm_4d contracts last dim S -> [B, K, G*T, D]
    attn_output = bmm_lib.bmm_4d(probs_grouped, value_cache)
  else:
    # value_cache: [B, K, S, D].
    attn_output = torch.matmul(probs_grouped, value_cache)

  if q_len == 1:
    return attn_output.reshape(b, 1, num_heads * head_dim).to(
        query_states.dtype
    )
  attn_output = attn_output.reshape(b, num_heads, q_len, head_dim)
  return (
      attn_output.transpose(1, 2)
      .reshape(b, q_len, num_heads * head_dim)
      .to(query_states.dtype)
  )


class Inkling2LoopAttention(nn.Module):
  """Unified sliding/full attention module for Inkling-2-Mini."""

  def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int):
    super().__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
    self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
    self.num_heads = (
        config.swa_num_attention_heads
        if self.is_sliding
        else config.num_attention_heads
    )
    self.num_key_value_heads = (
        config.swa_num_key_value_heads
        if self.is_sliding
        else config.num_key_value_heads
    )
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.sliding_window = (
        config.sliding_window_size if self.is_sliding else None
    )
    self.rel_extent = (
        config.sliding_window_size if self.is_sliding else config.rel_extent
    )
    self.scaling = self.head_dim**-0.5

    is_full = not self.is_sliding
    self.q_dim = self.num_heads * self.head_dim
    self.rel_dim = self.num_heads * config.d_rel
    # Fuse `[q_proj, o_gate, rel_linear, sink_proj]` into a single linear op
    # (all 4 consume `q_in`, saving 3 FC dispatches per logical layer).
    self.q_o_rel_sink_proj = nn.Linear(
        config.hidden_size,
        self.q_dim + self.q_dim + self.rel_dim + self.num_heads,
        bias=False,
    )
    self.q_proj = Inkling2LoopQueryProjection(
        config,
        self.num_heads,
        self.head_dim,
        log_scaling=is_full,
        include_linear=False,
    )
    self.k_proj = Inkling2LoopKeyValueProjection(
        config,
        self.num_key_value_heads,
        self.head_dim,
        layer_idx,
        conv_idx=0,
        is_key=True,
        dynamic=is_full,
    )
    self.v_proj = Inkling2LoopKeyValueProjection(
        config,
        self.num_key_value_heads,
        self.head_dim,
        layer_idx,
        conv_idx=1,
        is_key=False,
        dynamic=is_full,
    )
    self.o_proj = Inkling2LoopGatedOutputProjection(
        config, self.num_heads, self.head_dim, include_gate_proj=False
    )
    self.relative_position_bias = Inkling2LoopRelativePositionBias(
        config,
        self.num_heads,
        self.rel_extent,
        log_scaling=is_full,
        include_linear=False,
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: torch.Tensor | None,
      relative_distance: torch.Tensor,
      position_ids: torch.Tensor,
      cache: Any = None,
      key_hidden_states: torch.Tensor | None = None,
      value_hidden_states: torch.Tensor | None = None,
      valid_mask: torch.Tensor | None = None,
      num_valid_tokens: torch.Tensor | None = None,
      cache_offset: int = 0,
      k_ts_idx: int = 2,
      v_ts_idx: int = 3,
      precomputed_rel: tuple[torch.Tensor, torch.Tensor] | None = None,
      **kwargs: Any,
  ) -> torch.Tensor:
    cache_layer_idx = self.layer_idx + cache_offset
    k_in = hidden_states if key_hidden_states is None else key_hidden_states
    v_in = hidden_states if value_hidden_states is None else value_hidden_states

    q_fused = self.q_o_rel_sink_proj(hidden_states)
    q_raw, o_gate_raw, rel_raw, sink_raw = q_fused.split(
        [self.q_dim, self.q_dim, self.rel_dim, self.num_heads], dim=-1
    )
    apply_gpu_composites = bool(kwargs.get("apply_gpu_composites", False))
    param_tensor = kwargs.get("param_tensor", None)
    query_states = self.q_proj(hidden_states, position_ids, q_raw=q_raw)
    key_states = self.k_proj(
        k_in,
        cache=cache,
        valid_mask=valid_mask,
        num_valid_tokens=num_valid_tokens,
        cache_offset=cache_offset,
        apply_gpu_composites=apply_gpu_composites,
    )
    value_states = self.v_proj(
        v_in,
        cache=cache,
        valid_mask=valid_mask,
        num_valid_tokens=num_valid_tokens,
        cache_offset=cache_offset,
        apply_gpu_composites=apply_gpu_composites,
    )

    if cache is not None:
      key_cache, value_cache = cache.update(
          key_states, value_states, cache_layer_idx, cache_kwargs=kwargs
      )
    else:
      key_cache = key_states
      value_cache = (
          value_states.transpose(-2, -1) if v_ts_idx == 3 else value_states
      )

    position_bias = self.relative_position_bias(
        hidden_states,
        relative_distance,
        position_ids,
        rel_raw=rel_raw,
        precomputed_rel=precomputed_rel,
    )
    sinks = (
        sink_raw
        if hidden_states.shape[1] == 1
        else sink_raw.transpose(1, 2).unsqueeze(-1)
    )
    attn_output = _inkling2_gqa_attention_with_bias_and_sinks(
        query_states=query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        attention_mask=attention_mask,
        position_bias=position_bias,
        sinks=sinks,
        scaling=self.scaling,
        num_key_value_groups=self.num_key_value_groups,
        k_ts_idx=k_ts_idx,
        v_ts_idx=v_ts_idx,
        apply_gpu_composites=apply_gpu_composites,
        param_tensor=param_tensor,
    )
    return self.o_proj(attn_output, hidden_states, gate_raw=o_gate_raw)


class Inkling2LoopTopKRouter(nn.Module):
  """Top-k sigmoid MoE router with correction bias and `route_scale`."""

  def __init__(self, config: Inkling2LoopTextConfig):
    super().__init__()
    self.top_k = config.num_experts_per_tok
    self.num_experts = config.n_routed_experts
    self.hidden_dim = config.hidden_size
    self.routed_scaling_factor = float(config.route_scale)
    self.weight = nn.Parameter(
        torch.empty(self.num_experts, self.hidden_dim, dtype=torch.float32)
    )
    self.register_buffer(
        "e_score_correction_bias",
        torch.zeros(self.num_experts, dtype=torch.float32),
        persistent=True,
    )

  def forward(
      self, hidden_states: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    x_f32 = hidden_states.float()
    logits = F.linear(x_f32, self.weight.float())
    scores = torch.sigmoid(logits)
    bias_f32 = self.e_score_correction_bias.float()
    biased_scores = scores + bias_f32
    topk_biased_scores, indices = torch.ops.tfl.topk_v2(
        biased_scores, self.top_k
    )
    bias_selected = torch.ops.tfl.gather(
        bias_f32, indices.reshape(-1), 0
    ).reshape(topk_biased_scores.shape)
    weights = topk_biased_scores - bias_selected
    # Clamp denominator above float16 min normal (~6.1e-5) to prevent GPU FTZ
    # underflow while leaving real-token normalization unbiased.
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-4)
    return weights * self.routed_scaling_factor, indices


class Inkling2LoopStaticExperts(nn.Module):
  """MoE experts lowered through `tfl.custom("moe")` with Softcapped SwiGLU.

  Stores expert weights directly in pre-flattened 4D OHWI delegate layout
  (`[out_channels, num_experts, 1, in_channels]`) so Converter V2 passes them
  straight into `tfl.custom("moe")` without `stablehlo.slice`/`transpose` ops,
  preserving zero-copy `DenseResourceElementsAttr` weight sharing across all
  signatures and loop passes.
  """

  def __init__(self, config: Inkling2LoopTextConfig):
    super().__init__()
    self.config = config
    self.num_experts = config.n_routed_experts
    self.hidden_dim = config.hidden_size
    self.intermediate_dim = config.moe_intermediate_size
    self.flattened_gate_weight = nn.Parameter(
        torch.empty(self.intermediate_dim, self.num_experts, 1, self.hidden_dim)
    )
    self.flattened_ff1_weight = nn.Parameter(
        torch.empty(self.intermediate_dim, self.num_experts, 1, self.hidden_dim)
    )
    self.flattened_linear_weight = nn.Parameter(
        torch.empty(self.hidden_dim, self.num_experts, 1, self.intermediate_dim)
    )
    self.register_buffer(
        "per_expert_scale",
        torch.ones((1, 1, 1, self.num_experts), dtype=torch.float32),
        persistent=False,
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      top_k_index: torch.Tensor,
      top_k_weights: torch.Tensor,
  ) -> torch.Tensor:
    return moe_lib.litert_moe_experts_forward(
        self, hidden_states, top_k_index, top_k_weights
    )


class Inkling2LoopSparseMoeBlock(nn.Module):
  """Sparse MoE block combining TopKRouter and StaticExperts."""

  def __init__(self, config: Inkling2LoopTextConfig):
    super().__init__()
    self.experts = Inkling2LoopStaticExperts(config)
    self.gate = Inkling2LoopTopKRouter(config)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    routing_weights, selected_experts = self.gate(hidden_states)
    return self.experts(hidden_states, selected_experts, routing_weights)


class Inkling2LoopStaticDecoderLayer(nn.Module):
  """Static decoder layer supporting sliding/full and non-loop/loop variants."""

  def __init__(self, config: Inkling2LoopTextConfig, layer_idx: int):
    super().__init__()
    self.layer_idx = layer_idx
    self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
    self.is_loop_layer = (
        config.loop_layer_index_start <= layer_idx < config.loop_layer_index_end
    )
    self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
    self.attention_type = (
        "sliding_attention" if self.is_sliding else "full_attention"
    )

    if layer_idx > 0:
      num_attn_queries = 1 if self.is_sliding else 3
      self.attention_input: Inkling2LoopBlockAttentionResidual | None = (
          Inkling2LoopBlockAttentionResidual(
              config, num_queries=num_attn_queries
          )
      )
    else:
      self.attention_input = None

    self.mlp_input = Inkling2LoopBlockAttentionResidual(config, num_queries=1)
    self.self_attn = Inkling2LoopAttention(config, layer_idx)
    self.mlp = Inkling2LoopSparseMoeBlock(config)

    if self.is_sliding:
      self.input_layernorm = Inkling2LoopRMSNorm(
          config.hidden_size, config.rms_norm_eps
      )
    else:
      self.q_input_layernorm = Inkling2LoopRMSNorm(
          config.hidden_size, config.rms_norm_eps
      )
      self.k_input_layernorm = Inkling2LoopRMSNorm(
          config.hidden_size, config.rms_norm_eps
      )
      self.v_input_layernorm = Inkling2LoopRMSNorm(
          config.hidden_size, config.rms_norm_eps
      )

    self.post_attention_layernorm = Inkling2LoopRMSNorm(
        config.hidden_size, config.rms_norm_eps
    )
    self.post_attention_convolution = Inkling2LoopShortConvolution(
        config.hidden_size,
        config.conv_kernel_size,
        layer_idx=layer_idx,
        conv_idx=2,
    )
    self.post_mlp_convolution = Inkling2LoopShortConvolution(
        config.hidden_size,
        config.conv_kernel_size,
        layer_idx=layer_idx,
        conv_idx=3,
    )

    if self.is_loop_layer:
      self.q_message_gate = Inkling2LoopMessageGate(config)
      self.k_message_gate = Inkling2LoopMessageGate(config)
      self.v_message_gate = Inkling2LoopMessageGate(config)
      self.mlp_message_gate = Inkling2LoopMessageGate(config)

  def forward(
      self,
      hidden_states: torch.Tensor,
      normed_blocks_stack: torch.Tensor | None,
      normed_h_if_boundary: torch.Tensor | None = None,
      messages: (
          tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
      ) = None,
      attention_mask: torch.Tensor | None = None,
      relative_distance: torch.Tensor | None = None,
      position_ids: torch.Tensor | None = None,
      valid_mask: torch.Tensor | None = None,
      num_valid_tokens: torch.Tensor | None = None,
      cache: Any = None,
      cache_offset: int = 0,
      k_ts_idx: int = 2,
      v_ts_idx: int = 3,
      precomputed_rel: tuple[torch.Tensor, torch.Tensor] | None = None,
      **kwargs: Any,
  ) -> torch.Tensor:
    dtype = hidden_states.dtype
    block_residual = hidden_states

    if self.is_sliding:
      attn_in = (
          hidden_states
          if self.attention_input is None
          else self.attention_input(normed_blocks_stack, hidden_states)[0]
      )
      if messages is None:
        q_in = k_in = v_in = self.input_layernorm(attn_in).to(dtype)
        mlp_message = None
      else:
        q_msg, k_msg, v_msg, mlp_message = messages
        q_in = self.input_layernorm(self.q_message_gate(attn_in, q_msg)).to(
            dtype
        )
        k_in = self.input_layernorm(self.k_message_gate(attn_in, k_msg)).to(
            dtype
        )
        v_in = self.input_layernorm(self.v_message_gate(attn_in, v_msg)).to(
            dtype
        )
    else:
      if self.attention_input is None:
        q_h = k_h = v_h = hidden_states
      else:
        q_h, k_h, v_h = self.attention_input(normed_blocks_stack, hidden_states)
      if messages is None:
        mlp_message = None
      else:
        q_msg, k_msg, v_msg, mlp_message = messages
        q_h = self.q_message_gate(q_h, q_msg)
        k_h = self.k_message_gate(k_h, k_msg)
        v_h = self.v_message_gate(v_h, v_msg)
      q_in = self.q_input_layernorm(q_h).to(dtype)
      k_in = self.k_input_layernorm(k_h).to(dtype)
      v_in = self.v_input_layernorm(v_h).to(dtype)

    assert relative_distance is not None and position_ids is not None
    attn_out = self.self_attn(
        hidden_states=q_in,
        key_hidden_states=k_in,
        value_hidden_states=v_in,
        attention_mask=attention_mask,
        relative_distance=relative_distance,
        position_ids=position_ids,
        valid_mask=valid_mask,
        num_valid_tokens=num_valid_tokens,
        cache=cache,
        cache_offset=cache_offset,
        k_ts_idx=k_ts_idx,
        v_ts_idx=v_ts_idx,
        precomputed_rel=precomputed_rel,
        **kwargs,
    )
    apply_gpu_composites = bool(kwargs.get("apply_gpu_composites", False))
    attn_out = self.post_attention_convolution(
        attn_out,
        cache=cache,
        valid_mask=valid_mask,
        num_valid_tokens=num_valid_tokens,
        cache_offset=cache_offset,
        apply_gpu_composites=apply_gpu_composites,
    )

    residual = h_mid = attn_out.float()
    if self.is_block_boundary:
      assert normed_h_if_boundary is not None
      if normed_blocks_stack is None:
        local_normed_stack = normed_h_if_boundary.unsqueeze(2)
      else:
        local_normed_stack = torch.cat(
            [normed_blocks_stack, normed_h_if_boundary.unsqueeze(2)], dim=2
        )
    else:
      local_normed_stack = normed_blocks_stack
      residual = h_mid = block_residual.float() + h_mid

    mlp_in = self.mlp_input(local_normed_stack, h_mid)[0]
    if mlp_message is not None:
      mlp_in = self.mlp_message_gate(mlp_in, mlp_message)
    mlp_out = self.mlp(self.post_attention_layernorm(mlp_in).to(dtype))
    mlp_out = self.post_mlp_convolution(
        mlp_out,
        cache=cache,
        valid_mask=valid_mask,
        num_valid_tokens=num_valid_tokens,
        cache_offset=cache_offset,
        apply_gpu_composites=apply_gpu_composites,
    )
    return (residual + mlp_out.float()).to(dtype)


class Inkling2LoopNormedEmbedding(nn.Embedding):
  """Token embedding followed by `embed_norm` RMSNorm."""

  def __init__(
      self,
      num_embeddings: int,
      embedding_dim: int,
      padding_idx: int | None,
      norm_eps: float,
  ):
    super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
    self.embed_norm = Inkling2LoopRMSNorm(embedding_dim, eps=norm_eps)

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    return self.embed_norm(super().forward(input_ids))


class Inkling2LoopStaticTextModel(nn.Module):
  """Looped 24-physical / 48-logical layer Inkling-2-Mini text backbone."""

  def __init__(self, config: Inkling2LoopTextConfig):
    super().__init__()
    self.config = config
    self.embed_tokens = Inkling2LoopNormedEmbedding(
        config.vocab_size,
        config.hidden_size,
        config.pad_token_id,
        config.rms_norm_eps,
    )
    self.layers = nn.ModuleList([
        Inkling2LoopStaticDecoderLayer(config, i)
        for i in range(config.num_hidden_layers)
    ])
    self.message_input = Inkling2LoopMessageAttentionResidual(config)
    self.final_input = Inkling2LoopBlockAttentionResidual(config, num_queries=1)
    self.block_norm = Inkling2LoopUnweightedRMSNorm(
        config.hidden_size, eps=1e-6
    )
    self.norm = Inkling2LoopRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def _build_masks_and_rel_dist(
      self,
      seq_len: int,
      position_ids: torch.Tensor,
      attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
      cache: Any,
      dtype: torch.dtype,
  ) -> tuple[
      dict[str, torch.Tensor | None],
      torch.Tensor,
      dict[int, tuple[torch.Tensor, torch.Tensor]],
  ]:
    device = position_ids.device
    if cache is not None and getattr(cache, "layers", None):
      layer0 = cache.layers[0]
      k_ts = int(getattr(layer0, "k_ts_idx", 2))
      cache_len = int(layer0.keys.shape[k_ts])
    elif isinstance(attention_mask, torch.Tensor):
      cache_len = int(attention_mask.shape[-1])
    else:
      cache_len = seq_len

    # position_ids is [B, T] (B=1).
    q_pos = position_ids[0].to(torch.int32)  # [T]
    kv_pos = torch.arange(cache_len, device=device, dtype=torch.int32)  # [S]
    rel_dist = q_pos.unsqueeze(1) - kv_pos  # [T, S]

    if isinstance(attention_mask, dict):
      full_mask = attention_mask.get("full_attention")
      sliding_mask = attention_mask.get("sliding_attention", full_mask)
    else:
      full_mask = attention_mask
      sliding_mask = attention_mask

    if full_mask is None:
      causal_invalid = rel_dist < 0
      full_mask = torch.where(
          causal_invalid.view(1, 1, seq_len, cache_len),
          torch.full((1,), -1e4, dtype=dtype, device=device),
          torch.zeros((1,), dtype=dtype, device=device),
      )

    # Always enforce sliding_window_size on sliding layers even if caller only
    # supplied a global causal mask.
    sliding_invalid = (rel_dist < 0) | (
        rel_dist >= self.config.sliding_window_size
    )
    base_sliding = sliding_mask if sliding_mask is not None else full_mask
    sliding_mask_enforced = torch.where(
        sliding_invalid.view(1, 1, seq_len, cache_len),
        torch.full((1,), -1e4, dtype=base_sliding.dtype, device=device),
        base_sliding,
    )

    # Precompute 1D gather indices and valid masks once per unique rel_extent
    # instead of repeating across all 48 logical attention layers.
    rel_precomputed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    row_offsets_base = (
        torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(1)
        if seq_len > 1
        else None
    )
    for ext in {
        int(self.config.sliding_window_size),
        int(self.config.rel_extent),
    }:
      clamped = rel_dist.clamp(0, ext - 1).to(torch.int32)
      if seq_len == 1:
        flat_idx = clamped.reshape(-1)
      else:
        assert row_offsets_base is not None
        flat_idx = (row_offsets_base * ext + clamped).reshape(-1)
      valid_mask_rel = ((rel_dist >= 0) & (rel_dist < ext)).view(
          1, 1, seq_len, cache_len
      )
      rel_precomputed[ext] = (flat_idx, valid_mask_rel)

    return (
        {
            "full_attention": full_mask,
            "sliding_attention": sliding_mask_enforced,
        },
        rel_dist,
        rel_precomputed,
    )

  def forward(
      self,
      input_ids: torch.Tensor | None = None,
      position_ids: torch.Tensor | None = None,
      past_key_values: Any = None,
      attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
      inputs_embeds: torch.Tensor | None = None,
      valid_mask: torch.Tensor | None = None,
      k_ts_idx: int = 2,
      v_ts_idx: int = 3,
      **kwargs: Any,
  ) -> tuple[torch.Tensor, Any]:
    if inputs_embeds is None:
      assert input_ids is not None
      inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_len, _ = inputs_embeds.shape
    del batch_size
    if position_ids is None:
      position_ids = torch.arange(
          seq_len, device=inputs_embeds.device, dtype=torch.int32
      ).unsqueeze(0)
    elif position_ids.ndim == 1:
      position_ids = position_ids.unsqueeze(0)

    # Compute scalar num_valid_tokens [1] for padded prefill conv state update.
    num_valid_tokens: torch.Tensor | None = None
    if valid_mask is not None and seq_len > 1:
      num_valid_tokens = valid_mask[0].to(torch.int32).sum().reshape(1)

    mask_map, rel_dist, rel_precomputed = self._build_masks_and_rel_dist(
        seq_len=seq_len,
        position_ids=position_ids,
        attention_mask=attention_mask,
        cache=past_key_values,
        dtype=inputs_embeds.dtype,
    )

    loop_start = self.config.loop_layer_index_start
    loop_end = self.config.loop_layer_index_end
    num_loop_layers = loop_end - loop_start

    hidden_states = inputs_embeds
    normed_blocks: list[torch.Tensor] = []
    normed_stack: torch.Tensor | None = None

    def _run_layer(
        layer: nn.Module,
        h: torch.Tensor,
        blocks: list[torch.Tensor],
        blocks_stack: torch.Tensor | None,
        msgs: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
        c_offset: int = 0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
      typed_layer = cast(Inkling2LoopStaticDecoderLayer, layer)
      normed_h: torch.Tensor | None = (
          self.block_norm(h.float()) if typed_layer.is_block_boundary else None
      )
      new_h = typed_layer(
          h,
          blocks_stack,
          normed_h_if_boundary=normed_h,
          messages=msgs,
          attention_mask=mask_map[typed_layer.attention_type],
          relative_distance=rel_dist,
          position_ids=position_ids,
          valid_mask=valid_mask,
          num_valid_tokens=num_valid_tokens,
          cache=past_key_values,
          cache_offset=c_offset,
          k_ts_idx=k_ts_idx,
          v_ts_idx=v_ts_idx,
          precomputed_rel=rel_precomputed[typed_layer.self_attn.rel_extent],
          **kwargs,
      )
      if typed_layer.is_block_boundary:
        assert normed_h is not None
        next_blocks = blocks + [normed_h]
        next_stack = (
            normed_h.unsqueeze(2)
            if blocks_stack is None
            else torch.cat([blocks_stack, normed_h.unsqueeze(2)], dim=2)
        )
        return new_h, next_blocks, next_stack
      return new_h, blocks, blocks_stack

    # Prelude: layers [0, loop_start)
    for layer in self.layers[:loop_start]:
      hidden_states, normed_blocks, normed_stack = _run_layer(
          layer, hidden_states, normed_blocks, normed_stack
      )

    pre_loop_hidden_states = hidden_states
    pre_loop_normed_blocks = list(normed_blocks)
    pre_loop_normed_stack = normed_stack
    num_pre_loop_blocks = len(pre_loop_normed_blocks)

    # Pass 1 over core loop layers [loop_start, loop_end)
    for layer in self.layers[loop_start:loop_end]:
      hidden_states, normed_blocks, normed_stack = _run_layer(
          layer, hidden_states, normed_blocks, normed_stack
      )

    # Passes 2..loop_num_passes over core loop layers with cross-pass messages
    for loop_idx in range(1, self.config.loop_num_passes):
      # Skip pre-loop blocks + the entry block appended at loop_start.
      prev_normed_blocks = normed_blocks[num_pre_loop_blocks + 1 :]
      prev_normed_stack = (
          torch.stack(prev_normed_blocks, dim=2) if prev_normed_blocks else None
      )
      # Compute all loop layers' cross-pass messages once per pass.
      pass_messages = self.message_input(prev_normed_stack, hidden_states)
      hidden_states = pre_loop_hidden_states
      normed_blocks = list(pre_loop_normed_blocks)
      normed_stack = pre_loop_normed_stack
      cache_offset = (
          self.config.num_hidden_layers
          + (loop_idx - 1) * num_loop_layers
          - loop_start
      )
      for loop_layer_idx, layer in enumerate(self.layers[loop_start:loop_end]):
        hidden_states, normed_blocks, normed_stack = _run_layer(
            layer,
            hidden_states,
            normed_blocks,
            normed_stack,
            msgs=pass_messages[loop_layer_idx],
            c_offset=cache_offset,
        )

    # Coda: layers [loop_end, num_hidden_layers)
    for layer in self.layers[loop_end:]:
      hidden_states, normed_blocks, normed_stack = _run_layer(
          layer, hidden_states, normed_blocks, normed_stack
      )

    hidden_states = self.final_input(normed_stack, hidden_states)[0]
    hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
    return hidden_states, past_key_values


def aligned_lm_head_vocab_size(config: Inkling2LoopTextConfig) -> int:
  """Rounds `unpadded_vocab_size` up to a multiple of 4 for ML Drift TopK."""
  unpadded = int(config.unpadded_vocab_size)
  aligned = ((unpadded + 3) // 4) * 4
  return min(int(config.vocab_size), aligned)


class Inkling2LoopStaticForCausalLM(nn.Module):
  """Static `ForCausalLM` wrapper with 4-aligned `lm_head` and logit softcap."""

  _can_compile_fullgraph = True

  def __init__(self, config: Inkling2LoopTextConfig | Inkling2LoopConfig):
    super().__init__()
    text_config = unwrap_config(config)
    self.config = text_config
    self.model = Inkling2LoopStaticTextModel(text_config)
    lm_head_vocab = aligned_lm_head_vocab_size(text_config)
    self.lm_head_padded_tokens = lm_head_vocab - int(
        text_config.unpadded_vocab_size
    )
    self.lm_head = nn.Linear(
        text_config.hidden_size,
        lm_head_vocab,
        bias=self.lm_head_padded_tokens > 0,
    )
    if self.lm_head.bias is not None:
      with torch.no_grad():
        self.lm_head.bias.zero_()
        self.lm_head.bias[text_config.unpadded_vocab_size :] = -1e4
    self.logit_softcap = Inkling2LoopSoftcap(
        text_config.final_logit_softcapping
    )

  def get_input_embeddings(self) -> nn.Module:
    return self.model.embed_tokens

  def set_input_embeddings(self, value: Any) -> None:
    self.model.embed_tokens = value

  def get_output_embeddings(self) -> nn.Module:
    return self.lm_head

  def set_output_embeddings(self, new_embeddings: Any) -> None:
    self.lm_head = new_embeddings

  def set_attn_implementation(self, implementation: str) -> None:
    del implementation

  def forward(
      self,
      input_ids: torch.Tensor | None = None,
      position_ids: torch.Tensor | None = None,
      past_key_values: Any = None,
      attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
      inputs_embeds: torch.Tensor | None = None,
      valid_mask: torch.Tensor | None = None,
      **kwargs: Any,
  ) -> tuple[torch.Tensor, Any]:
    hidden_states, past_key_values = self.model(
        input_ids=input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        valid_mask=valid_mask,
        **kwargs,
    )
    logits = self.lm_head(hidden_states).float()
    logits = self.logit_softcap(logits)
    return logits, past_key_values
