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
"""LiteRT-LM cache implementation for Inkling-2-Mini looped transformer.

Allocates `total_num_hidden_layers` (48 for Inkling-2-Mini) logical cache
layers: 24 for pass 1 + 12 for pass 2 + 12 for pass 3. Every logical layer
holds transposed K/V buffers (`kv_cache_k_{i}`, `kv_cache_v_{i}`) and 4
channels-last float32 short-convolution states (`kv_cache_c_{i}_0..3` for
K, V, post-attention, and post-MLP convolutions).
"""

from collections.abc import Iterable
from typing import Any, cast

from litert_torch.generative.export_hf.core import cache as cache_lib
from litert_torch.generative.export_hf.core import cache_base as cache_base_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext.inkling2_loop import modeling_inkling2_loop_static
import torch
from torch.utils import _pytree as pytree


def _resolve_runtime_cache_dtype(
    export_config: exportable_module_config.ExportableModuleConfig,
) -> torch.dtype:
  cache_dtype = export_config.get_cache_dtype()
  if cache_dtype == torch.bfloat16:
    return torch.float32
  return cache_dtype


class Inkling2LoopCacheLayer(cache_lib.LiteRTLMCacheLayer):
  """Cache layer holding K/V cache plus 4 float32 short-conv state tensors."""

  def __init__(
      self,
      key_cache: torch.Tensor,
      value_cache: torch.Tensor,
      conv_states: list[torch.Tensor],
      layer_type: str = "full_attention",
      batch_size: int = 1,
      k_ts_idx: int = 2,
      v_ts_idx: int = 3,
      **kwargs: Any,
  ):
    super().__init__(
        key_cache=key_cache,
        value_cache=value_cache,
        layer_type=layer_type,
        batch_size=batch_size,
        k_ts_idx=k_ts_idx,
        v_ts_idx=v_ts_idx,
        **kwargs,
    )
    self.conv_states = list(conv_states)

  def get_conv_state(self, conv_idx: int) -> torch.Tensor:
    return self.conv_states[conv_idx]

  def set_conv_state(self, conv_idx: int, state: torch.Tensor) -> None:
    self.conv_states[conv_idx] = state

  def to(self, *args: Any, **kwargs: Any) -> "Inkling2LoopCacheLayer":
    super().to(*args, **kwargs)
    device = kwargs.get("device")
    if device is None and args and isinstance(args[0], (torch.device, str)):
      device = args[0]
    self.conv_states = [
        (
            t.to(device=device, dtype=torch.float32)
            if device is not None
            else t.to(dtype=torch.float32)
        )
        for t in self.conv_states
    ]
    return self

  @classmethod
  def create_from_config(
      cls,
      model_config: Any,
      layer_index: int,
      export_config: exportable_module_config.ExportableModuleConfig,
      **kwargs: Any,
  ) -> "Inkling2LoopCacheLayer":
    text_cfg = modeling_inkling2_loop_static.unwrap_config(model_config)
    phys_idx = text_cfg.physical_layer_index_for_cache(layer_index)
    is_sliding = text_cfg.layer_types[phys_idx] == "hybrid_sliding"
    kv_heads = (
        text_cfg.swa_num_key_value_heads
        if is_sliding
        else text_cfg.num_key_value_heads
    )
    head_dim = text_cfg.swa_head_dim if is_sliding else text_cfg.head_dim
    batch_size = int(kwargs.get("batch_size", export_config.batch_size))
    cache_length = int(kwargs.get("cache_length", export_config.cache_length))
    k_ts = (
        int(export_config.k_ts_idx) if export_config.k_ts_idx is not None else 2
    )
    v_ts = (
        int(export_config.v_ts_idx) if export_config.v_ts_idx is not None else 3
    )
    cache_dtype = _resolve_runtime_cache_dtype(export_config)

    k_shape = (
        (1, batch_size * kv_heads, cache_length, head_dim)
        if k_ts == 2
        else (1, batch_size * kv_heads, head_dim, cache_length)
    )
    v_shape = (
        (1, batch_size * kv_heads, head_dim, cache_length)
        if v_ts == 3
        else (1, batch_size * kv_heads, cache_length, head_dim)
    )
    keys = torch.zeros(k_shape, dtype=cache_dtype)
    values = torch.zeros(v_shape, dtype=cache_dtype)

    state_len = int(text_cfg.conv_kernel_size) - 1
    kv_dim = kv_heads * head_dim
    hidden_size = int(text_cfg.hidden_size)
    conv_states = [
        torch.zeros((batch_size, state_len, kv_dim), dtype=torch.float32),
        torch.zeros((batch_size, state_len, kv_dim), dtype=torch.float32),
        torch.zeros((batch_size, state_len, hidden_size), dtype=torch.float32),
        torch.zeros((batch_size, state_len, hidden_size), dtype=torch.float32),
    ]
    # Keep full_attention layer_type in LiteRTLMCacheLayer so linear static
    # cache update (`tfl.dynamic_update_slice`) is used for all layers while
    # sliding window masking is applied directly in attention.
    return cls(
        key_cache=keys,
        value_cache=values,
        conv_states=conv_states,
        layer_type="full_attention",
        batch_size=batch_size,
        k_ts_idx=k_ts,
        v_ts_idx=v_ts,
    )


@cache_base_lib.register_cache_implementation
class Inkling2LoopLiteRTLMCache(cache_lib.LiteRTLMCache):
  """Cache container with `total_num_hidden_layers` KV + 4x conv layers."""

  @classmethod
  def create_from_config(
      cls,
      model_config: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
      **kwargs: Any,
  ) -> "Inkling2LoopLiteRTLMCache":
    text_cfg = modeling_inkling2_loop_static.unwrap_config(model_config)
    total_layers = int(text_cfg.total_num_hidden_layers)
    layers = [
        Inkling2LoopCacheLayer.create_from_config(
            text_cfg, i, export_config, **kwargs
        )
        for i in range(total_layers)
    ]
    return cls(cast(Any, layers))

  def get_conv_state(self, layer_idx: int, state_idx: int = 0) -> torch.Tensor:
    layer = cast(Inkling2LoopCacheLayer, self.layers[layer_idx])
    return layer.get_conv_state(state_idx)

  def set_conv_state(
      self, layer_idx: int, state_idx: int, state: torch.Tensor
  ) -> None:
    layer = cast(Inkling2LoopCacheLayer, self.layers[layer_idx])
    layer.set_conv_state(state_idx, state)

  def to(self, *args: Any, **kwargs: Any) -> "Inkling2LoopLiteRTLMCache":
    for layer in self.layers:
      cast(Inkling2LoopCacheLayer, layer).to(*args, **kwargs)
    return self


def _flatten_inkling2_kvc(
    kvc: Inkling2LoopLiteRTLMCache,
) -> tuple[list[torch.Tensor], tuple[list[str], tuple[Any, ...]]]:
  """Flattens an `Inkling2LoopLiteRTLMCache` for `torch.utils._pytree`."""
  flattened: list[torch.Tensor] = []
  flat_names: list[str] = []
  typed_layers = [cast(Inkling2LoopCacheLayer, l) for l in kvc.layers]
  batch_size = int(typed_layers[0].batch_size) if typed_layers else 1
  num_layers = len(typed_layers)
  k_ts_idx = int(typed_layers[0].k_ts_idx) if typed_layers else 2
  v_ts_idx = int(typed_layers[0].v_ts_idx) if typed_layers else 3
  num_convs = len(typed_layers[0].conv_states) if typed_layers else 4

  for i, layer in enumerate(typed_layers):
    flattened.append(layer.keys)
    flat_names.append(f"k_{i}")
    flattened.append(layer.values)
    flat_names.append(f"v_{i}")
    for c_idx, c_state in enumerate(layer.conv_states):
      flattened.append(c_state)
      flat_names.append(f"c_{i}_{c_idx}")

  meta = (batch_size, num_layers, k_ts_idx, v_ts_idx, num_convs)
  return flattened, (flat_names, meta)


def _unflatten_inkling2_kvc(
    values: Iterable[Any],
    context: Any,
) -> Inkling2LoopLiteRTLMCache:
  """Unflattens tensors back into an `Inkling2LoopLiteRTLMCache`."""
  flat_names, meta = context
  batch_size, num_layers, k_ts_idx, v_ts_idx, num_convs = meta
  name_to_val = dict(zip(flat_names, list(values), strict=True))
  layers = []
  for i in range(num_layers):
    conv_states = [name_to_val[f"c_{i}_{c_idx}"] for c_idx in range(num_convs)]
    layers.append(
        Inkling2LoopCacheLayer(
            key_cache=name_to_val[f"k_{i}"],
            value_cache=name_to_val[f"v_{i}"],
            conv_states=conv_states,
            layer_type="full_attention",
            batch_size=batch_size,
            k_ts_idx=k_ts_idx,
            v_ts_idx=v_ts_idx,
        )
    )
  return Inkling2LoopLiteRTLMCache(cast(Any, layers))


def _flatten_inkling2_kvc_with_keys(kvc: Inkling2LoopLiteRTLMCache) -> Any:
  flattened, (flat_names, _) = _flatten_inkling2_kvc(kvc)
  return [
      (pytree.MappingKey(k), v)
      for k, v in zip(flat_names, flattened, strict=True)
  ], flat_names


pytree.register_pytree_node(
    Inkling2LoopLiteRTLMCache,
    _flatten_inkling2_kvc,
    _unflatten_inkling2_kvc,
    flatten_with_keys_fn=_flatten_inkling2_kvc_with_keys,
    serialized_type_name="",
)
