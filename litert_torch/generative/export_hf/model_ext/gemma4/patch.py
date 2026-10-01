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
"""Patches for Gemma4 model for on-device deployment."""

import contextlib
from litert_torch.backend import optimization_barrier as optimization_barrier_lib
from litert_torch.backend.experimental.torch_tfl import _ops as _  # Registers torch.ops.tfl.*
from litert_torch.generative.export_hf.experimental.composites import rope as rope_composite
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.layers import normalization
import torch
import transformers


class Gemma4RMSNorm(torch.nn.Module):
  """RMSNorm Layer."""

  def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
    """RMSNorm Layer."""
    super().__init__()
    self.with_scale = with_scale

    if self.with_scale:
      self.weight = torch.nn.Parameter(torch.ones(dim), requires_grad=True)
    else:
      self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    self.variance_epsilon = eps
    self.hidden_size = dim

  def forward(self, hidden_states):
    return normalization.rms_norm_with_hlfb(
        hidden_states,
        self.weight
        if self.with_scale
        else torch.ones((self.hidden_size,), dtype=torch.float32),
        self.variance_epsilon,
        torch.ones((self.hidden_size,), dtype=torch.float32),
    )

  def extra_repr(self):
    return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


try:
  from transformers.models.gemma4 import modeling_gemma4  # pylint: disable=g-import-not-at-top

  Gemma4TextModel = modeling_gemma4.Gemma4TextModel
  Gemma4VisionEncoder = modeling_gemma4.Gemma4VisionEncoder
  Gemma4VisionPatchEmbedder = modeling_gemma4.Gemma4VisionPatchEmbedder
  Gemma4VisionPooler = modeling_gemma4.Gemma4VisionPooler
  Gemma4AudioAttention = modeling_gemma4.Gemma4AudioAttention
  Gemma4AudioModel = modeling_gemma4.Gemma4AudioModel

  class FusedGemma4TextMLP(torch.nn.Module):
    """Fused Gate + Up MLP layer for Gemma4."""

    def __init__(self, original_mlp: modeling_gemma4.Gemma4TextMLP):
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

  class FusedGemma4TextAttention(torch.nn.Module):
    """Fused Attention layer for Gemma4 (Q + K + V)."""

    def __init__(
        self,
        original_attn: modeling_gemma4.Gemma4TextAttention,
        fuse_qkv: bool = False,
        use_rope_composite: bool = False,
    ):
      super().__init__()
      self.o_proj = original_attn.o_proj
      self.q_norm = original_attn.q_norm

      self.config = original_attn.config
      self.layer_idx = original_attn.layer_idx
      self.head_dim = original_attn.head_dim
      self.num_key_value_groups = original_attn.num_key_value_groups
      self.scaling = original_attn.scaling
      self.attention_dropout = original_attn.attention_dropout
      self.is_causal = original_attn.is_causal
      self.sliding_window = original_attn.sliding_window
      self.is_sliding = original_attn.is_sliding
      self.use_alternative_attention = original_attn.use_alternative_attention
      self.is_kv_shared_layer = original_attn.is_kv_shared_layer
      self.store_full_length_kv = original_attn.store_full_length_kv
      self.layer_type = original_attn.layer_type

      self.fuse_qkv = fuse_qkv and not self.is_kv_shared_layer
      self.use_rope_composite = use_rope_composite

      self.q_proj = original_attn.q_proj

      if not self.is_kv_shared_layer:
        self.k_norm = original_attn.k_norm
        self.v_norm = original_attn.v_norm
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj

        if self.fuse_qkv:
          q_out_features = self.q_proj.out_features
          k_out_features = self.k_proj.out_features
          v_out_features = (
              self.v_proj.out_features if self.v_proj is not None else 0
          )

          self.qkv_proj = torch.nn.Linear(
              self.q_proj.in_features,
              q_out_features + k_out_features + v_out_features,
              bias=self.q_proj.bias is not None,
          )

          # Copy weights and biases
          with torch.no_grad():
            tensors_to_cat = [self.q_proj.weight, self.k_proj.weight]
            if self.v_proj is not None:
              tensors_to_cat.append(self.v_proj.weight)
            self.qkv_proj.weight.copy_(torch.cat(tensors_to_cat, dim=0))

            if self.qkv_proj.bias is not None:
              biases_to_cat = [self.q_proj.bias, self.k_proj.bias]
              if self.v_proj is not None:
                biases_to_cat.append(self.v_proj.bias)
              self.qkv_proj.bias.copy_(torch.cat(biases_to_cat, dim=0))

          self.q_size = q_out_features
          self.k_size = k_out_features
          self.v_size = v_out_features

    def _get_rope_base(self) -> float:
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
      return rope_base

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        past_key_values=None,
        **kwargs,
    ):
      input_shape = hidden_states.shape[:-1]
      hidden_shape = (*input_shape, -1, self.head_dim)
      cos, sin = position_embeddings

      if self.is_kv_shared_layer:
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        if kwargs.get("apply_gpu_composites", False) or getattr(
            self, "use_rope_composite", False
        ):
          position_ids = kwargs.get("position_ids", None)
          if position_ids is None:
            seq_len = query_states.shape[1]
            position_ids = torch.arange(
                seq_len, device=query_states.device
            ).unsqueeze(0)

          rope_base = self._get_rope_base()
          query_states = query_states.transpose(1, 2)
          query_states = rope_composite.apply_mldrift_compatible_rope(
              query_states, position_ids, base=rope_base, head_dim=self.head_dim
          )
        else:
          query_states = modeling_gemma4.apply_rotary_pos_emb(
              query_states, cos, sin, unsqueeze_dim=2
          )
          query_states = query_states.transpose(1, 2)

        key_states, value_states = shared_kv_states[self.layer_type]
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
      else:
        if self.fuse_qkv:
          qkv = self.qkv_proj(hidden_states)
          if self.v_proj is not None:
            q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
          else:
            q, k = qkv.split([self.q_size, self.k_size], dim=-1)
            v = k
        else:
          q = self.q_proj(hidden_states)
          k = self.k_proj(hidden_states)
          v = self.v_proj(hidden_states) if self.v_proj is not None else k

        query_states = q.view(hidden_shape)
        query_states = self.q_norm(query_states)

        key_states = k.view(hidden_shape)
        key_states = self.k_norm(key_states)

        if getattr(self, "use_rope_composite", False):
          position_ids = kwargs.get("position_ids", None)
          if position_ids is None:
            seq_len = query_states.shape[1]
            position_ids = torch.arange(
                seq_len, device=query_states.device
            ).unsqueeze(0)

          rope_base = self._get_rope_base()
          query_states = query_states.transpose(1, 2)
          key_states = key_states.transpose(1, 2)
          query_states = rope_composite.apply_mldrift_compatible_rope(
              query_states, position_ids, base=rope_base, head_dim=self.head_dim
          )
          key_states = rope_composite.apply_mldrift_compatible_rope(
              key_states, position_ids, base=rope_base, head_dim=self.head_dim
          )
        else:
          query_states = modeling_gemma4.apply_rotary_pos_emb(
              query_states, cos, sin, unsqueeze_dim=2
          )
          query_states = query_states.transpose(1, 2)
          key_states = modeling_gemma4.apply_rotary_pos_emb(
              key_states, cos, sin, unsqueeze_dim=2
          )
          key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(v.view(hidden_shape))
        value_states = value_states.transpose(1, 2)

      if past_key_values is not None and not self.is_kv_shared_layer:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )
      if self.store_full_length_kv:
        shared_kv_states[self.layer_type] = key_states, value_states

      attention_interface = (
          modeling_gemma4.ALL_ATTENTION_FUNCTIONS.get_interface(
              self.config._attn_implementation,
              modeling_gemma4.eager_attention_forward,
          )
      )
      # Signal SDPA to skip cache update when this layer shares KV cache with a
      # donor layer to avoid redundant or duplicate cache updates.
      kwargs["skip_cache_update"] = self.is_kv_shared_layer
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

  class LiteRTGemma4TextRouter(torch.nn.Module):
    """LiteRT compatible Gemma4 Text Router."""

    def __init__(self, config):
      super().__init__()
      self.config = config
      self.hidden_size = config.hidden_size
      self.scalar_root_size = self.hidden_size**-0.5
      self.eps = config.rms_norm_eps

      self.norm = Gemma4RMSNorm(self.hidden_size, eps=self.eps, with_scale=False)
      self.proj = torch.nn.Linear(
          config.hidden_size, config.num_experts, bias=False
      )
      self.scale = torch.nn.Parameter(torch.ones(self.hidden_size))
      self.per_expert_scale = torch.nn.Parameter(torch.ones(config.num_experts))
      # Cleared by `moe.bind_per_expert_scale` when the experts are lowered to
      # the `moe` custom op, which applies `per_expert_scale` in the delegate.
      self.fold_per_expert_scale = True
      # Set by `moe.bind_per_expert_scale` when the experts are lowered to the
      # `moe` custom op, which consumes int32 indices. `torch.topk` returns
      # int64 indices, which the converter lowers to `tfl.topk_v2` (int32)
      # followed by a CAST to int64, leaving int64 tensors in the graph that
      # GPU delegates cannot represent. Emitting `tfl.topk_v2` directly keeps
      # the routing indices in int32 end to end.
      self.use_int32_indices = False

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      hidden_states = self.norm(hidden_states)
      hidden_states = hidden_states * self.scale * self.scalar_root_size

      expert_scores = self.proj(hidden_states)  # [B*S, E]
      router_probabilities = torch.nn.functional.softmax(expert_scores, dim=-1)

      # topk returns both values (probabilities) and indices directly
      if self.use_int32_indices:
        top_k_weights, top_k_index = torch.ops.tfl.topk_v2(
            router_probabilities, self.config.top_k_experts
        )  # both [B*S, K], indices in int32
      else:
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )  # both [B*S, K]

      # Normalize the top-k weights so they sum to 1 per token
      top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

      if self.fold_per_expert_scale:
        # Dynamic indexing (`per_expert_scale[top_k_index]`) is not exportable,
        # so gather the scales with a one-hot matmul instead.
        expert_ids = torch.arange(
            self.config.num_experts,
            dtype=top_k_index.dtype,
            device=top_k_index.device,
        )

        match_mask = top_k_index.unsqueeze(-1) == expert_ids
        float_mask = match_mask.to(self.per_expert_scale.dtype)
        scales = torch.matmul(float_mask, self.per_expert_scale)
        top_k_weights = top_k_weights * scales

      return router_probabilities, top_k_weights, top_k_index

  @patches_lib.register_model_patch(["gemma4", "gemma4_text"])
  @contextlib.contextmanager
  def patch_gemma4_model(model, export_config):
    """Dynamic model patch for Gemma4 export."""
    fuse_gate_up = export_config.fuse_gate_up
    fuse_qkv = export_config.fuse_qkv
    use_rope = export_config.use_rope_composite
    print(
        "Gemma4 model patch applied. "
        f"fuse_gate_up={fuse_gate_up}, fuse_qkv={fuse_qkv}, "
        f"use_rope_composite={use_rope}"
    )

    replaced_modules = []

    def replace_modules(module):
      for child_name, child in module.named_children():
        if fuse_gate_up and isinstance(child, modeling_gemma4.Gemma4TextMLP):
          print(f"Fusing MLP: {child_name}")
          fused = FusedGemma4TextMLP(child)
          setattr(module, child_name, fused)
          replaced_modules.append((module, child_name, child))
        elif isinstance(child, modeling_gemma4.Gemma4TextAttention):
          if fuse_qkv or use_rope:
            print(
                f"Replacing Attention: {child_name} "
                f"(fuse_qkv={fuse_qkv}, use_rope={use_rope})"
            )
            fused = FusedGemma4TextAttention(
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


except ImportError:
  Gemma4VisionEncoder = torch.nn.Module
  Gemma4VisionPatchEmbedder = torch.nn.Module
  Gemma4VisionPooler = torch.nn.Module
  Gemma4TextModel = torch.nn.Module
  Gemma4AudioAttention = torch.nn.Module
  Gemma4AudioModel = torch.nn.Module

  class FusedGemma4TextMLP(torch.nn.Module):
    pass

  class FusedGemma4TextAttention(torch.nn.Module):
    pass

  @patches_lib.register_model_patch(["gemma4", "gemma4_text"])
  @contextlib.contextmanager
  def patch_gemma4_model(model, export_config):
    yield


class LiteRTGemma4VisionPatchEmbedder(Gemma4VisionPatchEmbedder):
  """LiteRT Gemma4 Vision Patch Embedder."""

  def _position_embeddings(
      self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
  ) -> torch.Tensor:
    clamped_positions = pixel_position_ids.clamp(min=0)

    classes = torch.arange(
        self.position_embedding_size,
        device=clamped_positions.device,
        dtype=torch.int32,
    )
    clamped_positions_x = clamped_positions[..., 0]
    clamped_positions_y = clamped_positions[..., 1]
    one_hot_x = clamped_positions_x.unsqueeze(-1) == classes
    one_hot_y = clamped_positions_y.unsqueeze(-1) == classes
    one_hot_x = one_hot_x.to(self.position_embedding_table.dtype)
    one_hot_y = one_hot_y.to(self.position_embedding_table.dtype)

    # AI Edge Quantizer crashes on BMM weight quant.
    table_x = self.position_embedding_table[0]
    table_y = self.position_embedding_table[1]
    x_embeddings = one_hot_x @ table_x
    y_embeddings = one_hot_y @ table_y
    position_embeddings = x_embeddings + y_embeddings
    position_embeddings = torch.where(
        padding_positions.unsqueeze(-1), 0.0, position_embeddings
    )
    return position_embeddings


class LiteRTGemma4TextModel(Gemma4TextModel):
  """Gemma4 text model."""

  def project_per_layer_inputs(
      self,
      inputs_embeds: torch.Tensor,
      per_layer_inputs: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if not self.hidden_size_per_layer_input:
      raise RuntimeError(
          "Attempting to call project_per_layer_inputs() from a model"
          " initialized with a config that does not support per-layer"
          f" embeddings. {self.config}"
      )

    per_layer_projection = self.per_layer_model_projection(inputs_embeds)
    per_layer_projection, _ = optimization_barrier_lib.optimization_barrier(
        (per_layer_projection, inputs_embeds)
    )
    per_layer_projection *=  self.per_layer_model_projection_scale

    per_layer_projection = per_layer_projection.reshape(
        *inputs_embeds.shape[:-1],
        self.config.num_hidden_layers,
        self.hidden_size_per_layer_input,
    )
    per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

    if per_layer_inputs is None:
      return per_layer_projection

    return (
        per_layer_projection + per_layer_inputs
    ) * self.per_layer_input_scale


class LiteRTGemma4VisionPooler(Gemma4VisionPooler):
  """LiteRT Gemma4 Vision Pooler."""

  def _avg_pool_by_positions(
      self,
      hidden_states: torch.Tensor,
      pixel_position_ids: torch.Tensor,
      length: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """2D spatial pooling according to patch positions.

    Pools the input tokens by averaging patches within a `k^2` grid, where `k`
    is determined by the ratio between input and output lengths

    Args:
      hidden_states: The input hidden states.
      pixel_position_ids: The pixel position ids.
      length: The output length.

    Returns:
      The pooled hidden states and the mask.
    """
    input_seq_len = hidden_states.shape[1]
    k = int((input_seq_len // length) ** 0.5)
    k_squared = k**2
    if k_squared * length != input_seq_len:
      raise ValueError(
          f"Cannot pool {hidden_states.shape} to {length}: {k=}^2 times"
          f" {length=} must be {input_seq_len}."
      )

    clamped_positions = pixel_position_ids.clamp(min=0)
    max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
    kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]

    classes = torch.arange(length, device=kernel_idxs.device, dtype=torch.int32)
    weights = (kernel_idxs.int().unsqueeze(-1) == classes).float() / k_squared

    output = weights.transpose(1, 2) @ hidden_states.float()
    mask = torch.logical_not((weights == 0).all(dim=1))
    return output.to(hidden_states.dtype), mask


class LiteRTGemma4VisionEncoder(Gemma4VisionEncoder):
  """LiteRT Gemma4 Vision Encoder."""

  def forward(
      self,
      inputs_embeds: torch.Tensor,
      attention_mask: torch.Tensor,
      pixel_position_ids: torch.LongTensor | None = None,
      **kwargs,
  ) -> transformers.modeling_outputs.BaseModelOutputWithPast:
    num_seq = attention_mask.shape[1]
    attention_mask = torch.zeros((1, 1, num_seq, num_seq), dtype=torch.float32)

    # embed positions
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)

    # decoder layers
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
      hidden_states = decoder_layer(
          hidden_states,
          attention_mask=attention_mask,
          position_embeddings=position_embeddings,
          position_ids=pixel_position_ids,
          **kwargs,
      )

    return transformers.modeling_outputs.BaseModelOutputWithPast(
        last_hidden_state=hidden_states
    )


class LiteRTGemma4AudioSubSampleConvProjectionLayer(
    modeling_gemma4.Gemma4AudioSubSampleConvProjectionLayer
):
  """LiteRT Gemma4 Audio SubSample Conv Projection Layer for GPU export."""

  def forward(
      self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None
  ):
    if mask is not None:
      mask_f = mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
      hidden_states = hidden_states * mask_f[:, None, :, None]

    hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
    hidden_states = self.act(
        self.norm(hidden_states.permute(0, 2, 3, 1))
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    if mask is not None:
      mask_f = mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
      batch_size, seq_len = mask_f.shape
      if seq_len % 2 != 0:
        pad_one = torch.zeros(
            (batch_size, 1), dtype=mask_f.dtype, device=mask_f.device
        )
        mask_f = torch.cat([mask_f, pad_one], dim=1)
      half_len = (seq_len + 1) // 2
      mask = mask_f.reshape(batch_size, half_len, 2)[:, :, :1].reshape(
          batch_size, half_len
      )

    return hidden_states, mask


class LiteRTGemma4AudioAttention(Gemma4AudioAttention):
  """LiteRT Gemma4 Audio Attention using <=4D tensors for GPU compatibility."""

  def _convert_to_block_4d(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Converts `(B, L, H, D)` to 4D `(B, H * num_blocks, chunk_size, D)`."""
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
    pad = num_blocks * self.chunk_size - seq_len
    if pad > 0:
      pad_states = torch.zeros(
          (batch_size, pad, num_heads, head_dim),
          dtype=hidden_states.dtype,
          device=hidden_states.device,
      )
      hidden_states = torch.cat([hidden_states, pad_states], dim=1)
    # 4D transpose: (B, num_blocks * chunk_size, H, D) ->
    # (B, H, num_blocks * chunk_size, D)
    hidden_states = hidden_states.transpose(1, 2).contiguous()
    return hidden_states.reshape(
        batch_size, num_heads * num_blocks, self.chunk_size, head_dim
    )

  def _extract_block_context_4d(
      self, hidden_states: torch.Tensor
  ) -> torch.Tensor:
    """Extracts context windows into 4D `(B, H*num_blocks, context_size, D)`."""
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
    num_strips = (self.context_size + self.chunk_size - 1) // self.chunk_size
    required_len = (num_chunks + num_strips - 1) * self.chunk_size
    future_len = max(
        self.max_future_horizon + self.chunk_size - 1,
        required_len - seq_len - self.max_past_horizon,
    )

    past_pad = torch.zeros(
        (batch_size, self.max_past_horizon, num_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    future_pad = torch.zeros(
        (batch_size, future_len, num_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    # 4D transpose first: (B, L_pad, H, D) -> (B, H, L_pad, D)
    padded_4d = (
        torch.cat([past_pad, hidden_states, future_pad], dim=1)
        .transpose(1, 2)
        .contiguous()
    )

    strips = []
    for k in range(num_strips):
      start = k * self.chunk_size
      end = (k + num_chunks) * self.chunk_size
      strip = padded_4d[:, :, start:end, :].reshape(
          batch_size * num_heads, num_chunks, self.chunk_size, head_dim
      )
      strips.append(strip)

    windows = torch.cat(strips, dim=2)[:, :, : self.context_size, :]
    return windows.reshape(
        batch_size, num_heads * num_chunks, self.context_size, head_dim
    )

  def _rel_shift_4d(
      self, x: torch.Tensor, num_blocks: int, block_size: int
  ) -> torch.Tensor:
    """Relative position shift in 4D `(B, H*num_blocks, block_size, pos_len)`."""
    batch_size, heads_times_blocks, _, position_length = x.shape
    context_size = self.context_size
    pad_len = context_size + 1 - position_length
    pad_tensor = torch.zeros(
        (batch_size, heads_times_blocks, block_size, pad_len),
        dtype=x.dtype,
        device=x.device,
    )
    x = torch.cat([x, pad_tensor], dim=-1)
    x = x.view(
        batch_size, heads_times_blocks, block_size * (context_size + 1)
    )
    x = x[:, :, : block_size * context_size]
    return x.view(
        batch_size, heads_times_blocks, block_size, context_size
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      position_embeddings: torch.Tensor,
      attention_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

    query_states = self.q_proj(hidden_states).float().view(hidden_shape)
    key_states = self.k_proj(hidden_states).float().view(hidden_shape)
    value_states = self.v_proj(hidden_states).float().view(hidden_shape)

    query_states = (
        query_states
        * self.q_scale
        * torch.nn.functional.softplus(self.per_dim_scale)
    )
    key_states = key_states * self.k_scale

    num_blocks = (seq_length + self.chunk_size - 1) // self.chunk_size
    queries_4d = self._convert_to_block_4d(query_states)
    keys_4d = self._extract_block_context_4d(key_states)
    values_4d = self._extract_block_context_4d(value_states)

    relative_key_states = self.relative_k_proj(position_embeddings)
    relative_key_states = relative_key_states.view(
        -1, self.num_heads, self.head_dim
    ).to(dtype=queries_4d.dtype)
    # (1, H, head_dim, pos_len)
    rel_k_4d = relative_key_states.permute(1, 2, 0).unsqueeze(0)

    # queries_flat: (B, H, num_blocks * chunk_size, head_dim)
    queries_flat = queries_4d.view(
        batch_size, self.num_heads, num_blocks * self.chunk_size, self.head_dim
    )
    matrix_bd = queries_flat @ rel_k_4d
    matrix_bd = matrix_bd.reshape(
        batch_size, self.num_heads * num_blocks, self.chunk_size, -1
    )
    matrix_bd = self._rel_shift_4d(matrix_bd, num_blocks, self.chunk_size)

    matrix_ac = queries_4d @ keys_4d.transpose(-1, -2)

    attn_weights = matrix_ac + matrix_bd
    attn_weights = attn_weights / self.softcap
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * self.softcap

    if attention_mask is not None:
      attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(values_4d.dtype)
    attn_output = attn_weights @ values_4d
    # (B, H * num_blocks, chunk_size, D) -> (B, H, num_blocks * chunk_size, D)
    attn_output = attn_output.view(
        batch_size, self.num_heads, num_blocks * self.chunk_size, self.head_dim
    )
    # 4D transpose -> (B, num_blocks * chunk_size, H, D)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(
        batch_size, num_blocks * self.chunk_size, -1
    )
    attn_output = attn_output[:, :seq_length].contiguous()
    attn_output = self.post(attn_output.to(hidden_states.dtype))

    return attn_output, attn_weights


class LiteRTGemma4AudioModel(Gemma4AudioModel):
  """LiteRT Gemma4 Audio Model with 4D additive attention mask for GPU/CPU."""

  def _build_blocked_4d_attention_mask(
      self, output_mask: torch.Tensor, dtype: torch.dtype
  ) -> torch.Tensor:
    batch_size, seq_len = output_mask.shape
    device = output_mask.device

    chunk_size = self.config.attention_chunk_size
    max_past_horizon = self.config.attention_context_left - 1
    max_future_horizon = self.config.attention_context_right
    context_size = chunk_size + max_past_horizon + max_future_horizon

    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    num_strips = (context_size + chunk_size - 1) // chunk_size
    required_len = (num_blocks + num_strips - 1) * chunk_size
    future_len = max(
        (num_blocks * chunk_size - seq_len)
        + max_future_horizon
        + chunk_size
        - 1,
        required_len - seq_len - max_past_horizon,
    )

    past_pad = torch.zeros(
        (batch_size, max_past_horizon), dtype=dtype, device=device
    )
    future_pad = torch.zeros(
        (batch_size, future_len), dtype=dtype, device=device
    )
    padded_key_mask = torch.cat(
        [past_pad, output_mask.to(dtype=dtype), future_pad], dim=1
    )

    mask_strips = []
    for k in range(num_strips):
      start = k * chunk_size
      end = (k + num_blocks) * chunk_size
      mask_strips.append(
          padded_key_mask[:, start:end].reshape(
              batch_size, num_blocks, 1, chunk_size
          )
      )
    key_mask_4d = torch.cat(mask_strips, dim=3)[:, :, :, :context_size]

    block_starts = torch.arange(num_blocks, device=device) * chunk_size
    q_idx = torch.arange(chunk_size, device=device)[:, None]
    c_idx = torch.arange(context_size, device=device)[None, :]
    dist = max_past_horizon + q_idx - c_idx
    left_mask = (dist >= 0) & (dist < max_past_horizon)
    right_mask = (dist < 0) & (-dist < max_future_horizon)
    window_mask = (left_mask | right_mask).to(dtype=dtype)[None, None, :, :]

    q_global_idx = (
        block_starts[:, None]
        + torch.arange(chunk_size, device=device)[None, :]
    )
    q_valid_4d = (q_global_idx < seq_len).to(dtype=dtype)[None, :, :, None]

    static_mask_4d = window_mask * q_valid_4d
    valid_mask_4d = key_mask_4d * static_mask_4d
    valid_mask_4d = torch.cat(
        [valid_mask_4d] * self.config.num_attention_heads, dim=1
    )
    return (1.0 - valid_mask_4d) * self.config.attention_invalid_logits_value

  def forward(
      self,
      input_features: torch.Tensor,
      attention_mask: torch.Tensor | None = None,
      **kwargs,
  ):
    hidden_states, output_mask = self.subsample_conv_projection(
        input_features, attention_mask
    )
    position_embeddings = self.rel_pos_enc(hidden_states)

    if output_mask is None:
      output_mask = torch.ones(
          hidden_states.shape[:2],
          dtype=hidden_states.dtype,
          device=hidden_states.device,
      )
    attention_mask_4d = self._build_blocked_4d_attention_mask(
        output_mask, dtype=hidden_states.dtype
    )

    for encoder_layer in self.layers[: self.config.num_hidden_layers]:
      hidden_states = encoder_layer(
          hidden_states,
          attention_mask=attention_mask_4d,
          position_embeddings=position_embeddings,
          **kwargs,
      )

    hidden_states = self.output_proj(hidden_states)
    return modeling_gemma4.Gemma4AudioModelOutput(
        last_hidden_state=hidden_states,
        attention_mask=output_mask > 0.5,
    )


# pytype: disable=import-error
@patches_lib.register_patch(["gemma4", "gemma4_text"])
@contextlib.contextmanager
def gemma4_litert_patch():
  """Gemma4 patch."""
  print("Gemma4 patch applied.")
  from transformers.models.gemma4 import modeling_gemma4  # pylint: disable=g-import-not-at-top

  original_norm = modeling_gemma4.Gemma4RMSNorm
  modeling_gemma4.Gemma4RMSNorm = Gemma4RMSNorm

  original_vision_encoder = modeling_gemma4.Gemma4VisionEncoder
  modeling_gemma4.Gemma4VisionEncoder = LiteRTGemma4VisionEncoder

  original_patch_embedder = modeling_gemma4.Gemma4VisionPatchEmbedder
  modeling_gemma4.Gemma4VisionPatchEmbedder = LiteRTGemma4VisionPatchEmbedder

  original_pooler = modeling_gemma4.Gemma4VisionPooler
  modeling_gemma4.Gemma4VisionPooler = LiteRTGemma4VisionPooler

  original_text_model = modeling_gemma4.Gemma4TextModel
  modeling_gemma4.Gemma4TextModel = LiteRTGemma4TextModel

  original_text_router = modeling_gemma4.Gemma4TextRouter
  modeling_gemma4.Gemma4TextRouter = LiteRTGemma4TextRouter

  original_audio_subsample_layer = (
      modeling_gemma4.Gemma4AudioSubSampleConvProjectionLayer
  )
  modeling_gemma4.Gemma4AudioSubSampleConvProjectionLayer = (
      LiteRTGemma4AudioSubSampleConvProjectionLayer
  )

  original_audio_attention = modeling_gemma4.Gemma4AudioAttention
  modeling_gemma4.Gemma4AudioAttention = LiteRTGemma4AudioAttention

  original_audio_model = modeling_gemma4.Gemma4AudioModel
  modeling_gemma4.Gemma4AudioModel = LiteRTGemma4AudioModel

  try:
    yield
  finally:
    modeling_gemma4.Gemma4RMSNorm = original_norm
    modeling_gemma4.Gemma4VisionEncoder = original_vision_encoder
    modeling_gemma4.Gemma4VisionPatchEmbedder = original_patch_embedder
    modeling_gemma4.Gemma4VisionPooler = original_pooler
    modeling_gemma4.Gemma4TextModel = original_text_model
    modeling_gemma4.Gemma4TextRouter = original_text_router
    modeling_gemma4.Gemma4AudioSubSampleConvProjectionLayer = (
        original_audio_subsample_layer
    )
    modeling_gemma4.Gemma4AudioAttention = original_audio_attention
    modeling_gemma4.Gemma4AudioModel = original_audio_model


# pytype: enable=import-error
