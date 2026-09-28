# Copyright 2025 The LiteRT Torch Authors.
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
"""Optimized Cache class for HuggingFace integration.

Shape annotations used here:
  B: batch size
  K: num_key_value_heads
  G: number of KV groups
  N: number of attention heads. N // K = G
  T: target / input length
  S: sequence / context length
  H: head dimension
"""

import copy
from typing import Any, List, Tuple

import jaxtyping as jt
import litert_torch.generative.custom_ops.dynamic_update_slice as tfl_dus
from litert_torch.generative.export_hf.core import exportable_module_config
import litert_torch.generative.export_hf.core.cache_base as cache_base_lib
from litert_torch.generative.export_hf.experimental.composites import cache_update as gpu_cache_update
import torch
import torch.utils._pytree as pytree
from transformers import cache_utils

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig


# Shape annotations for the cache entries.
KeyCache = (
    jt.Shaped[torch.Tensor, "1 BK S H"] | jt.Shaped[torch.Tensor, "1 BK H S"]
)
KeySlice = (
    jt.Shaped[torch.Tensor, "1 BK T H"] | jt.Shaped[torch.Tensor, "1 BK H T"]
)
ValueCache = (
    jt.Shaped[torch.Tensor, "1 BK H S"] | jt.Shaped[torch.Tensor, "1 BK S H"]
)
ValueSlice = (
    jt.Shaped[torch.Tensor, "1 BK H T"] | jt.Shaped[torch.Tensor, "1 BK T H"]
)


def _get_slice_indices(
    positions: jt.Int32[torch.Tensor, "1"], cache_dim: int, ts_idx: int
) -> jt.Int32[torch.Tensor, "cache_dim"]:
  """Returns the slice indices.

  Args:
    positions: The positions tensor.
    cache_dim: Rank of the cache tensor..
    ts_idx: The index of the sequence dimension in cache.

  Returns:
    The indices tensor for tfl.dynamic_update_slice.
  """

  assert ts_idx < cache_dim, "ts_idx must be less than cache_dim."
  assert ts_idx >= 0, "ts_idx must be greater than or equal to 0."

  zeros = torch.zeros((1,), dtype=positions.dtype)
  indices = []
  for i in range(cache_dim):
    if i == ts_idx:
      indices.append(
          positions.reshape(
              1,
          )
      )
    else:
      indices.append(zeros)
  slice_indices = torch.cat(indices, dim=0)
  return slice_indices


def _update_kv_impl(
    key_state: KeyCache,
    value_state: ValueCache,
    k_slice: KeySlice,
    v_slice: ValueSlice,
    cache_position: jt.Int32[torch.Tensor, "T"],
    k_ts_idx: int,
    v_ts_idx: int,
    **kwargs,
):
  """Updates the cache buffer using tfl.dynamic_update_slice."""
  cache_dim = 4

  apply_gpu_composites = kwargs.get("apply_gpu_composites", False)
  if apply_gpu_composites:
    param_tensor = kwargs.get("param_tensor", None)
    if param_tensor is not None:
      positions = cache_position[0]
      k_slice_indices = _get_slice_indices(
          positions.clone(), cache_dim, k_ts_idx
      )
      v_slice_indices = _get_slice_indices(
          positions.clone(), cache_dim, v_ts_idx
      )
      # Extract values to pass into cache_update_composite
      _, bk_size, v_dim2, v_dim3 = value_state.shape
      head_size = v_dim2 if v_ts_idx == 3 else v_dim3
      cache_size = v_dim3 if v_ts_idx == 3 else v_dim2
      k, v = gpu_cache_update.cache_update(
          k_slice,
          v_slice.transpose(-2, -1),
          param_tensor,
          key_state,
          value_state,
          indices_k=k_slice_indices,
          indices_v=v_slice_indices,
          kv_heads=bk_size,
          kv_batch_size=1,
          cache_len=cache_size,
          head_size=head_size,
          is_ring_buffer=False,
          k_ts_idx=k_ts_idx,
          v_ts_idx=v_ts_idx,
      )
      return k, v

  positions = cache_position[0]  # The position of the first input token.
  k_slice_indices = _get_slice_indices(positions.clone(), cache_dim, k_ts_idx)
  v_slice_indices = _get_slice_indices(positions.clone(), cache_dim, v_ts_idx)
  k = tfl_dus.dynamic_update_slice(
      key_state, k_slice, [x for x in k_slice_indices]
  )
  v = tfl_dus.dynamic_update_slice(
      value_state, v_slice, [x for x in v_slice_indices]
  )
  return k, v


int32_one_hot = gpu_cache_update.int32_one_hot
update_kv_cache_with_sliding = gpu_cache_update.update_kv_cache_with_sliding


def _update_kv_sliding_impl(
    key_state: KeyCache,
    value_state: ValueCache,
    k_slice: KeySlice,
    v_slice: ValueSlice,
    cache_position: jt.Int32[torch.Tensor, "T"],
    valid_mask: jt.Bool[torch.Tensor, "T"],
    k_ts_idx: int,
    v_ts_idx: int,
    **unused_kwargs,
):
  """Updates the cache buffer using tfl.dynamic_update_slice."""
  new_k = update_kv_cache_with_sliding(
      key_state, k_slice, cache_position, valid_mask, k_ts_idx
  )
  new_v = update_kv_cache_with_sliding(
      value_state, v_slice, cache_position, valid_mask, v_ts_idx
  )
  # (Updated cache, (cache_past, cache_slice))
  # Former stores in the cache layer (to be gathered after transformer stack),
  # latter are used for the current transformer stack dot product attention.
  return (new_k, (key_state, k_slice)), (new_v, (value_state, v_slice))


class LiteRTLMCacheLayer(cache_base_lib.LiteRTLMCacheLayerMixin):
  """Optimized Cache layer class for HuggingFace integration."""

  is_compileable = True
  is_sliding = False

  def __init__(
      self,
      key_cache: KeyCache,
      value_cache: ValueCache,
      layer_type: str,
      batch_size: int = 1,
      k_ts_idx: int = 2,
      v_ts_idx: int = 3,
      **kwargs,
  ):
    super().__init__()
    self.keys = key_cache
    self.values = value_cache
    self.k_ts_idx = k_ts_idx  # The index of the sequence dimension in K cache.
    self.v_ts_idx = v_ts_idx  # The index of the sequence dimension in V cache.
    assert k_ts_idx in [2, 3]
    assert v_ts_idx in [2, 3]
    self.is_initialized = True
    self.layer_type = layer_type
    assert self.layer_type in ["full_attention", "sliding_attention"]

    self.k_cache_shape = self.keys.shape
    self.v_cache_shape = self.values.shape
    self.max_cache_len = self.v_cache_shape[self.v_ts_idx]
    self.batch_size = batch_size
    v_head_dim_idx = 3 if self.v_ts_idx == 2 else 2
    self.head_dim = self.v_cache_shape[v_head_dim_idx]

    self.additional_states = kwargs.get("additional_states", None)

    self.cumulative_length = 0

    self.layer_type = layer_type
    if layer_type == "sliding_attention":
      self.is_sliding = True

  def get_batch_size(self) -> int:
    return self.batch_size

  def get_k_ts_idx(self) -> int:
    return self.k_ts_idx

  def get_v_ts_idx(self) -> int:
    return self.v_ts_idx

  def lazy_initialization(self, key_states: torch.Tensor):  # pyrefly: ignore[bad-override]
    # Since we don't support real lazy initialization, this function could only
    # be called by Cache.early_initialization, where uses a standard cache
    # layout [batch_size, num_heads, ?, head_dim].
    # TODO(weiyiw): Implement this function.
    raise NotImplementedError(
        "Lazy initialization is not supported in LiteRTLMCacheLayer."
    )

  def update(
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      *args,
      **kwargs,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    cache_kwargs = self.get_cache_runtime_args()
    seq_len = key_states.shape[2]
    self.cumulative_length += seq_len
    key_states = key_states.to(self.keys.dtype)  # pyrefly: ignore[missing-attribute]

    value_states = value_states.to(self.values.dtype)  # pyrefly: ignore[missing-attribute]

    if not cache_kwargs.get("kv_slice_preprocessed", False):
      if self.k_ts_idx == 3:
        key_target_shape = (1, -1, self.head_dim, seq_len)
        key_states = key_states.permute(0, 1, 3, 2).reshape(*key_target_shape)
      elif self.k_ts_idx == 2:
        key_target_shape = (1, -1, seq_len, self.head_dim)
        key_states = key_states.reshape(*key_target_shape)
      else:
        raise ValueError(f"Unsupported k_ts_idx: {self.k_ts_idx}")
      if self.v_ts_idx == 3:
        value_target_shape = (1, -1, self.head_dim, seq_len)
        value_states = value_states.permute(0, 1, 3, 2).reshape(
            *value_target_shape
        )
      elif self.v_ts_idx == 2:
        value_target_shape = (1, -1, seq_len, self.head_dim)
        value_states = value_states.reshape(*value_target_shape)
      else:
        raise ValueError(f"Unsupported v_ts_idx: {self.v_ts_idx}")

    cache_position: jt.Int32[torch.Tensor, "T"] = cache_kwargs.get(
        "cache_position"
    )
    assert (
        cache_position is not None
    ), "For export, cache position should always be set."
    if cache_position.dim() == 0:
      cache_position = cache_position.unsqueeze(0)
    merged_kwargs = {**kwargs, **cache_kwargs}
    merged_kwargs.pop("cache_position", None)
    apply_gpu_composites = merged_kwargs.get("apply_gpu_composites", False)
    enable_ring_buffer = merged_kwargs.get("enable_ring_buffer", False)
    if apply_gpu_composites and enable_ring_buffer and self.is_sliding:
      # In the case where we use actual ring buffer for local layers and
      # gpu composite op, we pass key and value as tuples so attention
      # can handle the cache update before or after BMM as needed.
      param_tensor = merged_kwargs.get("param_tensor", None)
      return (
          (self.keys, key_states, self, cache_position, param_tensor),
          (self.values, value_states, self),
      )
    valid_mask = merged_kwargs.pop("valid_mask", None)
    if valid_mask is None or not self.is_sliding:
      # Sliding window with full context or Full attention.
      self.keys, self.values = _update_kv_impl(
          self.keys,
          self.values,
          key_states,
          value_states,
          cache_position,
          self.k_ts_idx,
          self.v_ts_idx,
          **merged_kwargs,
      )
      return self.keys, self.values
    else:
      assert valid_mask is not None, (
          "valid_mask should not be None for sliding window ring buffer."
      )
      valid_mask = valid_mask.squeeze(0)
      kk, vv = _update_kv_sliding_impl(
          self.keys,
          self.values,
          key_states,
          value_states,
          cache_position,
          valid_mask,
          self.k_ts_idx,
          self.v_ts_idx,
          **merged_kwargs,
      )
      self.keys = kk[0]
      self.values = vv[0]
    return kk[1], vv[1]

  def get_mask_sizes(self, cache_position: torch.Tensor):
    """Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for."""
    kv_offset = 0
    kv_length = self.max_cache_len
    return kv_length, kv_offset

  def get_seq_length(self) -> int:
    return (self.keys[0, 0].any(dim=-1)).sum() if self.is_initialized else 0  # pyrefly: ignore[unsupported-operation]

  def get_max_cache_shape(self) -> int:
    return self.max_cache_len

  def get_max_length(self) -> int:
    return self.max_cache_len

  @classmethod
  def _infer_cache_shape_from_config(
      cls,
      model_config,
      layer_index,
      export_config: ExportableModuleConfig,
  ):
    """Infers the KV cache shape from the model config."""
    cache_length = export_config.cache_length
    batch_size = export_config.batch_size
    k_ts_idx = export_config.k_ts_idx
    v_ts_idx = export_config.v_ts_idx
    num_kv_heads = model_config.num_key_value_heads
    layer_type = "full_attention"
    if hasattr(model_config, "layer_types"):
      layer_type = model_config.layer_types[layer_index]
    if layer_type == "sliding_attention":
      cache_length = (
          export_config.sliding_window_ring_buffer_size or cache_length
      )
    if hasattr(model_config, "num_global_key_value_heads"):
      if layer_type == "full_attention":
        num_kv_heads = model_config.num_global_key_value_heads or num_kv_heads

    # --- HETEROGENEOUS BACKEND PATCH ---
    head_dim = None
    per_layer_config = getattr(model_config, "per_layer_config", None)
    if (
        per_layer_config is not None
        and layer_index is not None
        and layer_index < len(per_layer_config)
    ):
      layer_cfg = per_layer_config[layer_index]
      if isinstance(layer_cfg, dict):
        head_dim = layer_cfg.get("head_dim")
        per_layer_num_kv_heads = layer_cfg.get("num_key_value_heads")
      else:
        head_dim = getattr(layer_cfg, "head_dim", None)
        per_layer_num_kv_heads = getattr(layer_cfg, "num_key_value_heads", None)
      if per_layer_num_kv_heads is not None:
        num_kv_heads = per_layer_num_kv_heads

    if head_dim is None:
      # Try accessing globally, catching custom AmbiguousGlobalPerLayerAttributeError
      try:
        head_dim = getattr(model_config, "head_dim", None)
      except (AttributeError, RuntimeError):
        # Opt-in to global access for per-layer config attributes to avoid blocking heterogeneous pipelines
        if hasattr(model_config, "allow_global_per_layer_attribute_access"):
          setattr(model_config, "allow_global_per_layer_attribute_access", True)
        try:
          head_dim = getattr(model_config, "head_dim", None)
        except (AttributeError, RuntimeError):
          head_dim = None

    embed_size_per_head = (
        head_dim or model_config.hidden_size // model_config.num_attention_heads
    )
    # -----------------------------------

    if hasattr(model_config, "global_head_dim"):
      if layer_type == "full_attention":
        embed_size_per_head = (
            model_config.global_head_dim or embed_size_per_head
        )

    if k_ts_idx == 2:
      k_cache_shape = (
          1,
          batch_size * num_kv_heads,
          cache_length,
          embed_size_per_head,
      )
    elif k_ts_idx == 3:
      k_cache_shape = (
          1,
          batch_size * num_kv_heads,
          embed_size_per_head,
          cache_length,
      )
    else:
      raise ValueError(f"Unsupported k_ts_idx: {k_ts_idx}")
    if v_ts_idx == 2:
      v_cache_shape = (
          1,
          batch_size * num_kv_heads,
          cache_length,
          embed_size_per_head,
      )
    elif v_ts_idx == 3:
      v_cache_shape = (
          1,
          batch_size * num_kv_heads,
          embed_size_per_head,
          cache_length,
      )
    else:
      raise ValueError(f"Unsupported v_ts_idx: {v_ts_idx}")
    return k_cache_shape, v_cache_shape, layer_type

  @classmethod
  def create_from_config(
      cls,
      model_config,
      layer_index,
      export_config: ExportableModuleConfig,
      **kwargs,
  ) -> "LiteRTLMCacheLayer":
    """Creates a KV cache from the model config."""
    k_cache_shape, v_cache_shape, layer_type = (
        cls._infer_cache_shape_from_config(
            model_config, layer_index, export_config
        )
    )
    if hasattr(export_config, "get_torch_dtype"):
      cache_dtype = export_config.get_torch_dtype()
    else:
      cache_dtype = (
          torch.float16
          if getattr(export_config, "experimental_use_fp16", False)
          else torch.float32
      )
    keys = torch.zeros(k_cache_shape, dtype=cache_dtype)
    values = torch.zeros(v_cache_shape, dtype=cache_dtype)
    return cls(
        keys,
        values,
        # pytype: disable=bad-argument-type
        k_ts_idx=export_config.k_ts_idx,
        v_ts_idx=export_config.v_ts_idx,
        layer_type=layer_type,
        # pytype: enable=bad-argument-type
        **kwargs,
    )


class LiteRTLMConvCacheLayer(
    cache_base_lib.LiteRTLMCacheLayerMixin,
    cache_utils.LinearAttentionCacheLayerMixin,
):
  """Optimized Conv Cache layer class for HuggingFace integration."""

  is_compileable = True
  is_sliding = False

  def __init__(
      self,
      conv_states: torch.Tensor,
      recurrent_states: torch.Tensor | None = None,
      layer_type: str | None = None,
      batch_size: int = 1,
      **kwargs,
  ):
    cache_utils.LinearAttentionCacheLayerMixin.__init__(self)
    self.conv_states: Any = conv_states
    self.recurrent_states: Any = recurrent_states
    self.is_conv_states_initialized: Any = True
    self.is_recurrent_states_initialized: Any = recurrent_states is not None
    self.batch_size = batch_size
    self.is_initialized = True
    self.conv_kernel_size: int = conv_states.shape[-1]
    self.keys = torch.zeros((1, 1, 1, 1))
    self.values = torch.zeros((1, 1, 1, 1))
    self.layer_type = layer_type
    assert self.layer_type in ("conv", "linear_attention")

  def get_batch_size(self) -> int:
    return self.batch_size

  def get_k_ts_idx(self) -> int:
    # Not used.
    return 2

  def get_v_ts_idx(self) -> int:
    # Not used.
    return 3

  def lazy_initialization(self, *args, **kwargs):
    raise NotImplementedError("Lazy initialization is not supported.")

  def update(self, key_states, value_states, *args, **kwargs):
    raise ValueError("Cannot call update on ConvCacheLayer.")

  def update_conv_state(
      self,
      conv_states: torch.Tensor,
      state_idx: int = 0,
      **kwargs) -> torch.Tensor:
    seq_len = conv_states.shape[-1]
    cache_kwargs = self.get_cache_runtime_args()
    cache_position = cache_kwargs.get("cache_position", None)
    valid_mask = kwargs.get("valid_mask", None)

    if valid_mask is None and cache_position is not None:
      if seq_len > 1:
        valid_mask = torch.ones_like(cache_position, dtype=torch.bool)
        valid_mask[1:] = cache_position[1:] > cache_position[:-1]

    if valid_mask is not None:
      if valid_mask.dim() == 1:
        mask = valid_mask.unsqueeze(0).unsqueeze(0)
      else:
        mask = valid_mask.unsqueeze(1)
      conv_states = conv_states * mask

    padded_input = torch.cat([self.conv_states, conv_states], dim=-1)

    if seq_len > 1:
      if valid_mask is not None:
        l_state = self.conv_kernel_size
        total_len = l_state + seq_len
        num_real = valid_mask.to(padded_input.dtype).sum()
        row_idx = torch.arange(
            total_len, device=padded_input.device, dtype=padded_input.dtype
        ).unsqueeze(1)
        col_target = num_real + torch.arange(
            l_state, device=padded_input.device, dtype=padded_input.dtype
        ).unsqueeze(0)
        selector = (row_idx == col_target).to(padded_input.dtype)
        next_state = padded_input @ selector
      else:
        next_state = padded_input[:, :, -self.conv_kernel_size:]
    else:
      next_state = padded_input[:, :, -self.conv_kernel_size:]

    self.conv_states.copy_(next_state)
    return self.conv_states

  def update_recurrent_state(
      self, recurrent_states: torch.Tensor, state_idx: int = 0, **kwargs
  ) -> torch.Tensor:
    if self.recurrent_states is None:
      raise ValueError(
          "recurrent_states is not initialized in LiteRTLMConvCacheLayer."
      )
    self.recurrent_states.copy_(recurrent_states)
    return self.recurrent_states

  def get_mask_sizes(self, cache_position: torch.Tensor):
    return self.conv_kernel_size, 0

  def get_seq_length(self) -> int:
    return 0

  def get_max_cache_shape(self) -> int:
    return self.conv_kernel_size

  def get_max_length(self) -> int:
    return self.conv_kernel_size

  @classmethod
  def create_from_config(
      cls,
      model_config,
      layer_index,
      export_config: ExportableModuleConfig,
      **kwargs,
  ) -> "LiteRTLMConvCacheLayer":
    assert hasattr(model_config, "layer_types"), (
        "model_config must have layer_types attribute for ConvCacheLayer."
    )
    layer_type = model_config.layer_types[layer_index]
    assert layer_type in (
        "conv",
        "linear_attention",
    ), f"Unsupported layer type: {layer_type}"
    batch_size = kwargs.pop("batch_size", export_config.batch_size)
    if hasattr(export_config, "get_torch_dtype"):
      cache_dtype = export_config.get_torch_dtype()
    else:
      cache_dtype = (
          torch.float16
          if getattr(export_config, "experimental_use_fp16", False)
          else torch.float32
      )
    if layer_type == "conv":
      c_state_shape = (
          batch_size,
          model_config.hidden_size,
          model_config.conv_L_cache - 1,
      )
      c_state = torch.zeros(c_state_shape, dtype=cache_dtype)
      return cls(
          c_state,
          batch_size=batch_size,
          layer_type=layer_type,
          **kwargs,
      )
    else:
      key_dim = (
          model_config.linear_key_head_dim * model_config.linear_num_key_heads
      )
      value_dim = (
          model_config.linear_value_head_dim
          * model_config.linear_num_value_heads
      )
      conv_dim = key_dim * 2 + value_dim
      conv_cache_len = model_config.linear_conv_kernel_dim - 1
      c_state_shape = (batch_size, conv_dim, conv_cache_len)
      r_state_shape = (
          batch_size,
          model_config.linear_num_value_heads,
          model_config.linear_key_head_dim,
          model_config.linear_value_head_dim,
      )
      c_state = torch.zeros(c_state_shape, dtype=cache_dtype)
      r_state = torch.zeros(r_state_shape, dtype=torch.float32)
      return cls(
          conv_states=c_state,
          recurrent_states=r_state,
          batch_size=batch_size,
          layer_type=layer_type,
          **kwargs,
      )

  def to(self, *args, **kwargs) -> "LiteRTLMConvCacheLayer":
    if hasattr(self, "c_state") and isinstance(self.c_state, torch.Tensor):
      self.c_state = self.c_state.to(*args, **kwargs)
    if hasattr(self, "conv_states") and isinstance(
        self.conv_states, torch.Tensor
    ):
      self.conv_states = self.conv_states.to(*args, **kwargs)
    if hasattr(self, "recurrent_states") and isinstance(
        self.recurrent_states, torch.Tensor
    ):
      device = kwargs.get("device", None)
      if not device and args and isinstance(args[0], (torch.device, str)):
        device = args[0]
      if device is not None:
        self.recurrent_states = self.recurrent_states.to(
            device=device, dtype=torch.float32
        )
      else:
        self.recurrent_states = self.recurrent_states.to(dtype=torch.float32)
    return self

LAYER_TYPE_TO_CLASS = {
    "full_attention": LiteRTLMCacheLayer,
    "sliding_attention": LiteRTLMCacheLayer,
    "conv": LiteRTLMConvCacheLayer,
    "linear_attention": LiteRTLMConvCacheLayer,
}


@cache_base_lib.register_cache_implementation
class LiteRTLMCache(cache_base_lib.LiteRTLMCacheMixin):
  """Optimized Cache class for HuggingFace integration."""

  @classmethod
  def create_from_config(
      cls,
      model_config,
      export_config: ExportableModuleConfig,
      **kwargs,
  ) -> "LiteRTLMCache":
    """Creates a KV cache from the model config."""
    num_layers = model_config.num_hidden_layers
    num_shared_layers = getattr(model_config, "num_kv_shared_layers", 0)
    layers = []
    for layer_index in range(num_layers - num_shared_layers):
      layer_type = "full_attention"
      if hasattr(model_config, "layer_types"):
        layer_type = model_config.layer_types[layer_index]
      layer_class = LAYER_TYPE_TO_CLASS.get(layer_type, LiteRTLMCacheLayer)
      layers.append(
          layer_class.create_from_config(
              model_config,
              layer_index,
              export_config,
              **kwargs,
          )
      )
    return cls(layers)

  def insert_dummy_cache_layers(self, model_config):
    num_layers = model_config.num_hidden_layers
    num_shared_layers = getattr(model_config, "num_kv_shared_layers", 0)
    num_unshared_layers = num_layers - num_shared_layers
    assert len(self.layers) == num_unshared_layers
    for i in range(num_shared_layers):
      self.layers.append(copy.copy(self.layers[i % num_unshared_layers]))
    return self

  def remove_dummy_cache_layers(self, model_config):
    num_layers = model_config.num_hidden_layers
    num_shared_layers = getattr(model_config, "num_kv_shared_layers", 0)
    num_unshared_layers = num_layers - num_shared_layers
    assert len(self.layers) == num_layers
    self.layers = self.layers[:num_unshared_layers]
    return self


def _flatten_kvc_t(
    kvc: LiteRTLMCache,
) -> Tuple[
    List[torch.Tensor], Tuple[List[str], Tuple[int, int, int, int, List[str]]]
]:
  """Flattens the cache into a list of tensors."""
  flattened = []
  flat_names = []
  num_layers = len(kvc.layers)
  attention_layer = None
  for layer in kvc.layers:
    if isinstance(layer, LiteRTLMCacheLayer):
      attention_layer = layer
      break

  if attention_layer is not None:
    batch_size = attention_layer.get_batch_size()
    k_ts_idx = attention_layer.get_k_ts_idx()
    v_ts_idx = attention_layer.get_v_ts_idx()
  else:
    layer_0 = kvc.layers[0]
    assert isinstance(layer_0, cache_base_lib.LiteRTLMCacheLayerMixin)
    batch_size = layer_0.get_batch_size()
    k_ts_idx = layer_0.get_k_ts_idx()
    v_ts_idx = layer_0.get_v_ts_idx()
  layer_types = [
      getattr(layer, "layer_type", "full_attention") for layer in kvc.layers
  ]
  for i, layer in enumerate(kvc.layers):
    if layer_types[i] == "conv":
      assert hasattr(layer, "conv_states")
      flattened.append(layer.conv_states)
      flat_names.append(f"c_{i}")
    elif layer_types[i] == "linear_attention":
      assert hasattr(layer, "conv_states")
      assert hasattr(layer, "recurrent_states")
      flattened.append(layer.conv_states)
      flat_names.append(f"c_{i}")
      flattened.append(layer.recurrent_states)
      flat_names.append(f"r_{i}")
    elif layer_types[i] in ["full_attention", "sliding_attention"]:
      assert hasattr(layer, "keys")
      assert hasattr(layer, "values")
      flattened.append(layer.keys)
      flat_names.append(f"k_{i}")
      flattened.append(layer.values)
      flat_names.append(f"v_{i}")
    else:
      raise ValueError(f"Unsupported layer type: {type(layer)}")
  return flattened, (
      flat_names,
      (batch_size, num_layers, k_ts_idx, v_ts_idx, layer_types),
  )


def _unflatten_kvc_t(
    values: List[torch.Tensor],
    context: Tuple[List[str], Tuple[int, int, int, int, List[str]]],
) -> LiteRTLMCache:
  """Unflattens the cache from a list of tensors."""
  flat_names = context[0]
  batch_size, num_layers, k_ts_idx, v_ts_idx, layer_types = context[1]
  layers = []
  for i in range(num_layers):
    layer_type = layer_types[i]
    if layer_type == "conv":
      c_cache_idx = flat_names.index(f"c_{i}")
      layers.append(
          LiteRTLMConvCacheLayer(
              conv_states=values[c_cache_idx],
              batch_size=batch_size,
              layer_type=layer_types[i],
          )
      )
    elif layer_type == "linear_attention":
      c_cache_idx = flat_names.index(f"c_{i}")
      r_cache_idx = flat_names.index(f"r_{i}")
      layers.append(
          LiteRTLMConvCacheLayer(
              conv_states=values[c_cache_idx],
              recurrent_states=values[r_cache_idx],
              batch_size=batch_size,
              layer_type=layer_types[i],
          )
      )
    elif layer_type in ["full_attention", "sliding_attention"]:
      k_cache_idx = flat_names.index(f"k_{i}")
      v_cache_idx = flat_names.index(f"v_{i}")
      layers.append(
          LiteRTLMCacheLayer(
              key_cache=values[k_cache_idx],
              value_cache=values[v_cache_idx],
              batch_size=batch_size,
              k_ts_idx=k_ts_idx,
              v_ts_idx=v_ts_idx,
              layer_type=layer_types[i],
          )
      )
    else:
      raise ValueError(f"Unsupported layer type: {layer_type}")
  obj = LiteRTLMCache(layers)
  return obj


def _flatten_kvc_t_with_keys(
    kvc: LiteRTLMCache,
):
  flattened, (flat_names, _) = _flatten_kvc_t(kvc)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


pytree.register_pytree_node(
    LiteRTLMCache,
    _flatten_kvc_t,
    _unflatten_kvc_t,  # pyrefly: ignore[bad-argument-type]
    flatten_with_keys_fn=_flatten_kvc_t_with_keys,
    serialized_type_name="",
)
