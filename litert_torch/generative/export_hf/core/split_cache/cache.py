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

from collections.abc import Sequence
import copy
from typing import List, Tuple
import jaxtyping as jt
from litert_torch.generative.export_hf.core import exportable_module_config
import litert_torch.generative.export_hf.core.cache_base as cache_base_lib
import torch
import torch.utils._pytree as pytree

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig

KeyCache = (
    jt.Shaped[torch.Tensor, "1 BK H S"] | jt.Shaped[torch.Tensor, "1 BK S H"]
)
KeySlice = (
    jt.Shaped[torch.Tensor, "1 BK H T"] | jt.Shaped[torch.Tensor, "1 BK T H"]
)
ValueCache = (
    jt.Shaped[torch.Tensor, "1 BK S H"] | jt.Shaped[torch.Tensor, "1 BK H S"]
)
ValueSlice = (
    jt.Shaped[torch.Tensor, "1 BK T H"] | jt.Shaped[torch.Tensor, "1 BK H T"]
)

KeyCacheEntry = Tuple[KeyCache, KeySlice | None]
ValueCacheEntry = Tuple[ValueCache, ValueSlice | None]


class SplitCacheTuple(Sequence):
  """A tuple that holds the split KV cache."""

  def __init__(self, iterable=()):
    self._data = tuple(iterable)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return type(self)(self._data[index])
    return self._data[index]

  def __len__(self):
    return len(self._data)

  def __repr__(self):
    return repr(self._data)

  def to(self, dtype):
    return type(self)(item.to(dtype) for item in self._data)


def split_cache_tuple_flatten(tup: SplitCacheTuple):
  return list(tup._data), None  # pylint: disable=protected-access


def split_cache_tuple_unflatten(values, context):
  del context
  return SplitCacheTuple(values)


pytree.register_pytree_node(
    SplitCacheTuple, split_cache_tuple_flatten, split_cache_tuple_unflatten
)


class LiteRTLMSplitCacheLayer(cache_base_lib.LiteRTLMCacheLayerMixin):
  """Optimized Cache layer class for HuggingFace integration."""

  is_compileable = True
  is_sliding = False

  def __init__(
      self,
      key_cache: KeyCacheEntry,
      value_cache: ValueCacheEntry,
      batch_size: int = 1,
      k_ts_idx: int = 2,
      v_ts_idx: int = 3,
      **kwargs,
  ):
    super().__init__()
    if not isinstance(key_cache, tuple):
      key_cache = (key_cache, None)
    if not isinstance(value_cache, tuple):
      value_cache = (value_cache, None)
    self.keys = key_cache
    self.values = value_cache
    self.is_initialized = True

    self.k_ts_idx = k_ts_idx
    self.v_ts_idx = v_ts_idx

    self.k_cache_shape = self.keys[0].shape
    self.v_cache_shape = self.values[0].shape
    self.max_cache_len = self.k_cache_shape[self.k_ts_idx]
    self.batch_size = batch_size
    self.head_dim = (
        self.k_cache_shape[2] if self.k_ts_idx == 3 else self.k_cache_shape[3]
    )
    self.additional_states = kwargs.get("additional_states", None)

    self.cumulative_length = 0

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

  def update(  # pyrefly: ignore[bad-override]
      self,
      key_states: torch.Tensor,
      value_states: torch.Tensor,
      *args,
      **kwargs,
  ) -> SplitCacheTuple:
    seq_len = key_states.shape[2]
    self.cumulative_length += seq_len

    key_states = key_states.to(self.keys[0].dtype)  # pyrefly: ignore[unsupported-operation]

    value_states = value_states.to(self.values[0].dtype)  # pyrefly: ignore[unsupported-operation]

    if self.k_ts_idx == 2:
      key_states = key_states.reshape(
          1, -1, seq_len, self.head_dim
      )  # 1, bk, s, h
    else:
      assert self.k_ts_idx == 3, "k_ts_idx must be 2 or 3."
      key_states = key_states.permute(0, 1, 3, 2).reshape(
          1, -1, self.head_dim, seq_len
      )  # 1, bk, h, s

    if self.v_ts_idx == 2:
      value_states = value_states.reshape(
          1, -1, seq_len, self.head_dim
      )  # 1, bk, s, h
    else:
      assert self.v_ts_idx == 3, "v_ts_idx must be 2 or 3."
      value_states = value_states.permute(0, 1, 3, 2).reshape(
          1, -1, self.head_dim, seq_len
      )  # 1, bk, h, s

    self.keys = (self.keys[0], key_states)  # pyrefly: ignore[unsupported-operation]
    self.values = (self.values[0], value_states)  # pyrefly: ignore[unsupported-operation]

    return SplitCacheTuple([self.keys, self.values])

  def get_mask_sizes(self, cache_position: torch.Tensor):
    """Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for."""
    kv_offset = 0
    kv_length = self.max_cache_len
    return kv_length, kv_offset

  def get_seq_length(self) -> int:
    return (self.keys[0][0, 0].any(dim=-1)).sum() if self.is_initialized else 0  # pyrefly: ignore[unsupported-operation]

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
      **kwargs,
  ):
    """Infers the KV cache shape from the model config."""
    del kwargs  # Unused.
    cache_length = export_config.cache_length
    batch_size = export_config.batch_size
    k_ts_idx = export_config.k_ts_idx
    v_ts_idx = export_config.v_ts_idx
    num_kv_heads = model_config.num_key_value_heads
    if hasattr(model_config, "num_global_key_value_heads") and hasattr(
        model_config, "layer_types"
    ):
      layer_type = model_config.layer_types[layer_index]
      if layer_type == "full_attention":
        num_kv_heads = model_config.num_global_key_value_heads or num_kv_heads
    embed_size_per_head = (
        getattr(model_config, "head_dim", None)
        or model_config.hidden_size // model_config.num_attention_heads
    )
    if hasattr(model_config, "global_head_dim") and hasattr(
        model_config, "layer_types"
    ):
      layer_type = model_config.layer_types[layer_index]
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
    else:
      assert k_ts_idx == 3, "k_ts_idx must be 2 or 3."
      k_cache_shape = (
          1,
          batch_size * num_kv_heads,
          embed_size_per_head,
          cache_length,
      )
    if v_ts_idx == 2:
      v_cache_shape = (
          1,
          batch_size * num_kv_heads,
          cache_length,
          embed_size_per_head,
      )
    else:
      assert v_ts_idx == 3, "v_ts_idx must be 2 or 3."
      v_cache_shape = (
          1,
          batch_size * num_kv_heads,
          embed_size_per_head,
          cache_length,
      )
    return k_cache_shape, v_cache_shape

  @classmethod
  def create_from_config(
      cls,
      model_config,
      layer_index,
      export_config: ExportableModuleConfig,
      **kwargs,
  ) -> "LiteRTLMSplitCacheLayer":
    """Creates a KV cache from the model config."""
    k_cache_shape, v_cache_shape = cls._infer_cache_shape_from_config(
        model_config, layer_index, export_config, **kwargs
    )
    keys = torch.zeros(k_cache_shape, dtype=torch.float32)
    values = torch.zeros(v_cache_shape, dtype=torch.float32)
    return cls((keys, None), (values, None), **kwargs)


@cache_base_lib.register_cache_implementation
class LiteRTLMSplitCache(cache_base_lib.LiteRTLMCacheMixin):
  """Optimized Cache class for HuggingFace integration."""

  @classmethod
  def create_from_config(
      cls,
      model_config,
      export_config: ExportableModuleConfig,
      **kwargs,
  ) -> "LiteRTLMSplitCache":
    """Creates a KV cache from the model config."""
    num_layers = model_config.num_hidden_layers
    num_shared_layers = getattr(model_config, "num_kv_shared_layers", 0)
    layers = []
    for layer_index in range(num_layers - num_shared_layers):
      layers.append(
          LiteRTLMSplitCacheLayer.create_from_config(
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
    kvc: LiteRTLMSplitCache,
) -> Tuple[List[torch.Tensor], Tuple[List[str], Tuple[int, int, int, int]]]:
  """Flattens the KV cache to a list of tensors."""
  flattened = []
  flat_names = []
  num_layers = len(kvc.layers)
  layer_0 = kvc.layers[0]
  assert isinstance(layer_0, cache_base_lib.LiteRTLMCacheLayerMixin)
  batch_size = layer_0.get_batch_size()
  k_ts_idx = layer_0.get_k_ts_idx()
  v_ts_idx = layer_0.get_v_ts_idx()
  for i, cache_layer in enumerate(kvc.layers):
    flattened.append(cache_layer.keys[0])  # pyrefly: ignore[missing-attribute]
    flat_names.append(f"k_{i}")
    flattened.append(cache_layer.values[0])  # pyrefly: ignore[missing-attribute]
    flat_names.append(f"v_{i}")
    if cache_layer.keys[1] is not None:  # pyrefly: ignore[missing-attribute]
      assert cache_layer.values[1] is not None  # pyrefly: ignore[missing-attribute]
      flattened.append(cache_layer.keys[1])  # pyrefly: ignore[missing-attribute]
      flat_names.append(f"k_{i}_slice")
      flattened.append(cache_layer.values[1])  # pyrefly: ignore[missing-attribute]
      flat_names.append(f"v_{i}_slice")
  return flattened, (flat_names, (batch_size, num_layers, k_ts_idx, v_ts_idx))


def _unflatten_kvc_t(
    values: List[torch.Tensor],
    context: Tuple[List[str], Tuple[int, int, int, int]],
) -> LiteRTLMSplitCache:
  """Unflattens the KV cache from a list of tensors."""
  flat_names = context[0]
  batch_size = context[1][0]
  k_ts_idx = context[1][2]
  v_ts_idx = context[1][3]
  num_layers = context[1][1]
  kv_entries = []
  for i in range(num_layers):
    k_cache_idx = flat_names.index(f"k_{i}")
    k_cache = values[k_cache_idx]
    try:
      k_slice_idx = flat_names.index(f"k_{i}_slice")
      k_cache_update = values[k_slice_idx]
    except ValueError:
      k_cache_update = None
    v_cache_idx = flat_names.index(f"v_{i}")
    v_cache = values[v_cache_idx]
    try:
      v_slice_idx = flat_names.index(f"v_{i}_slice")
      v_cache_update = values[v_slice_idx]
    except ValueError:
      v_cache_update = None
    kv_entries.append(
        LiteRTLMSplitCacheLayer(
            key_cache=(k_cache, k_cache_update),
            value_cache=(v_cache, v_cache_update),
            batch_size=batch_size,
            k_ts_idx=k_ts_idx,
            v_ts_idx=v_ts_idx,
        )
    )
  obj = LiteRTLMSplitCache(kv_entries)
  return obj


def _flatten_kvc_t_with_keys(
    kvc: LiteRTLMSplitCache,
):
  flattened, (flat_names, _) = _flatten_kvc_t(kvc)
  return [
      (pytree.MappingKey(k), v) for k, v in zip(flat_names, flattened)
  ], flat_names


pytree.register_pytree_node(
    LiteRTLMSplitCache,
    _flatten_kvc_t,
    _unflatten_kvc_t,  # pyrefly: ignore[bad-argument-type]
    flatten_with_keys_fn=_flatten_kvc_t_with_keys,
    serialized_type_name="",
)
