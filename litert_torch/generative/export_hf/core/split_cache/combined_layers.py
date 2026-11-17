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
"""Combined KV cache layout for split-cache exports.

Experimental (`split_cache_combined_layers`): instead of one K and one V cache
tensor per layer, the KV caches of all layers with the same cache layout are
passed as one K and one V tensor per layer group. The existing split-cache
exportable modules are not changed: `with_combined_layers` adds the layout to
whichever prefill / decode classes the export selected (generic or
model-specific), and `CombinedCacheUpdate` replaces `CacheUpdate` in the
auxiliary model.
"""

from typing import Any

from litert_torch.generative.export_hf.core import cache as base_cache_lib
from litert_torch.generative.export_hf.core import exportable_module as base_exportable_module
from litert_torch.generative.export_hf.core.split_cache import cache as kv_cache_lib
from litert_torch.generative.export_hf.core.split_cache import exportable_module as split_cache_module
import torch

# Combined KV cache layout (split_cache_combined_layers): the per-layer caches
# are grouped by (layer type, K shape, V shape) and every group is stacked into
# one K and one V tensor with the layer as the leading dimension. Groups are
# numbered in order of their first layer, giving signature names such as
# `kv_cache_k_0` / `kv_slice_k_0`, which keep the name prefixes that LiteRT-LM
# uses to discover cache tensors. Models whose layers all have the same cache
# shape (e.g. Qwen3) get a single group; Gemma 4 gets one group for its sliding
# and one for its global layers.
#
# LiteRT-LM's runtime KV cache update pairs `kv_cache_{k,v}_<n>` with
# `kv_slice_{k,v}_<n>` by the suffix only, reads the sequence axis and the
# local (ring buffer) / global type of each cache from the executor metadata,
# and treats all dimensions except the last two as independent rows. A group
# [L_g, H, S, D] is therefore updated like L_g * H per-layer rows without
# runtime changes, as long as the slice output stacks its layers in the same
# order as the cache input (both follow the same group layout).


def _k_key(group: int) -> str:
  return f'k_{group}'


def _v_key(group: int) -> str:
  return f'v_{group}'


def kv_cache_group_layout(layer_types, k_shapes, v_shapes) -> list[list[int]]:
  """Returns the layer indices of each combined cache group, in group order."""
  groups: dict[tuple[str, tuple[int, ...], tuple[int, ...]], list[int]] = {}
  for i, (layer_type, k_shape, v_shape) in enumerate(
      zip(layer_types, k_shapes, v_shapes)
  ):
    if layer_type not in ('full_attention', 'sliding_attention'):
      raise ValueError(
          'split_cache_combined_layers only supports attention layers, got'
          f' layer type {layer_type!r} for layer {i}.'
      )
    for name, shape in (('K', k_shape), ('V', v_shape)):
      if shape[0] != 1:
        raise ValueError(
            'split_cache_combined_layers requires a leading dimension of 1 in'
            f' the per-layer {name} cache, got shape {tuple(shape)}.'
        )
    key = (layer_type, tuple(k_shape), tuple(v_shape))
    groups.setdefault(key, []).append(i)
  return list(groups.values())


def model_kv_cache_group_layout(
    model_config, export_config: base_exportable_module.ExportableModuleConfig
) -> tuple[list[list[int]], list[str]]:
  """Returns the combined cache group layout and per-layer types of a model."""
  num_layers = model_config.num_hidden_layers - (
      getattr(model_config, 'num_kv_shared_layers', 0) or 0
  )
  layer_types = list(
      getattr(model_config, 'layer_types', None)
      or ['full_attention'] * num_layers
  )[:num_layers]
  k_shapes, v_shapes = [], []
  for i in range(num_layers):
    k_shape, v_shape = (
        kv_cache_lib.LiteRTLMSplitCacheLayer._infer_cache_shape_from_config(  # pylint: disable=protected-access
            model_config, i, export_config
        )
    )
    k_shapes.append(k_shape)
    v_shapes.append(v_shape)
  return kv_cache_group_layout(layer_types, k_shapes, v_shapes), layer_types


def combine_kv_layers(
    k_tensors, v_tensors, layer_types
) -> dict[str, torch.Tensor]:
  """Stacks per-layer [1, ...] K/V tensors into one tensor per group."""
  layout = kv_cache_group_layout(
      layer_types,
      [t.shape for t in k_tensors],
      [t.shape for t in v_tensors],
  )
  combined = {}
  for g, layers in enumerate(layout):
    combined[_k_key(g)] = torch.cat([k_tensors[i] for i in layers], dim=0)
    combined[_v_key(g)] = torch.cat([v_tensors[i] for i in layers], dim=0)
  return combined


def combine_split_cache(
    kv_cache: kv_cache_lib.LiteRTLMSplitCache,
) -> dict[str, torch.Tensor]:
  """Converts a per-layer split cache (no slices) into the combined layout."""
  return combine_kv_layers(
      [layer.keys[0] for layer in kv_cache.layers],  # pyrefly: ignore[missing-attribute]
      [layer.values[0] for layer in kv_cache.layers],  # pyrefly: ignore[missing-attribute]
      [getattr(l, 'layer_type', 'full_attention') for l in kv_cache.layers],
  )


def uncombine_split_cache(
    kv_cache: dict[str, torch.Tensor],
    layout: list[list[int]],
    layer_types: list[str],
    export_config: base_exportable_module.ExportableModuleConfig,
) -> kv_cache_lib.LiteRTLMSplitCache:
  """Builds a per-layer split cache that views slices of the combined cache.

  Layer i of group g reads `kv_cache_k_<g>[j:j+1]` (j = its position in the
  group), a static slice on the leading (layer) dimension, so the attention
  math is unchanged.

  Args:
    kv_cache: The combined cache, {'k_<g>': [L_g, ...], 'v_<g>': [L_g, ...]}.
    layout: The layer indices of each group (see kv_cache_group_layout).
    layer_types: The layer type of each cache layer.
    export_config: The export config.

  Returns:
    The per-layer split cache, in layer order.
  """
  assert len(kv_cache) == 2 * len(layout), (sorted(kv_cache), layout)
  per_layer = {}
  for g, layers in enumerate(layout):
    keys = kv_cache[_k_key(g)]
    values = kv_cache[_v_key(g)]
    assert keys.shape[0] == values.shape[0] == len(layers)
    for j, i in enumerate(layers):
      per_layer[i] = kv_cache_lib.LiteRTLMSplitCacheLayer(
          key_cache=(keys[j : j + 1], None),
          value_cache=(values[j : j + 1], None),
          batch_size=export_config.batch_size,
          k_ts_idx=export_config.k_ts_idx,  # pyrefly: ignore[bad-argument-type]
          v_ts_idx=export_config.v_ts_idx,  # pyrefly: ignore[bad-argument-type]
          layer_type=layer_types[i],
      )
  return kv_cache_lib.LiteRTLMSplitCache(
      [per_layer[i] for i in range(len(per_layer))]
  )


def combine_slices(
    k_slices, v_slices, layout: list[list[int]]
) -> dict[str, list[torch.Tensor]]:
  """Stacks per-layer slice outputs into one output per group."""
  return {
      'kv_slice_k': [
          torch.cat([k_slices[i] for i in layers], dim=0) for layers in layout
      ],
      'kv_slice_v': [
          torch.cat([v_slices[i] for i in layers], dim=0) for layers in layout
      ],
  }


class CombinedLayersMixin:
  """Adds the combined KV cache layout to a split-cache prefill / decode module.

  Must come before a split-cache exportable module class in the bases (see
  `with_combined_layers`). It only hooks the three places where the cache
  crosses the module boundary and defers everything else to that class:
  sample inputs are combined into groups, the combined cache input is
  presented to the model as per-layer views, and the per-layer slice outputs
  are stacked into one output per group.
  """

  def kv_cache_group_layout(self) -> tuple[list[list[int]], list[str]]:
    model_config = self.model.model.config  # pyrefly: ignore[missing-attribute]
    if hasattr(model_config, 'text_config'):
      model_config = model_config.text_config
    return model_kv_cache_group_layout(
        model_config, self.export_config  # pyrefly: ignore[missing-attribute]
    )

  def get_sample_kv_cache(self, model_config):
    inputs, _ = super().get_sample_kv_cache(model_config)  # pyrefly: ignore[missing-attribute]
    inputs['kv_cache'] = combine_split_cache(inputs['kv_cache'])
    return inputs, {'kv_cache': {k: None for k in inputs['kv_cache']}}

  def adapt_inputs(self, *args, **kwargs):
    """Presents the combined cache input to the model as per-layer views.

    The KV cache is the last argument of every split-cache `adapt_inputs`
    (model-specific ones add arguments before it).

    Args:
      *args: The arguments of the wrapped class's `adapt_inputs`.
      **kwargs: The keyword arguments of the wrapped class's `adapt_inputs`.

    Returns:
      The wrapped class's model inputs.
    """
    layout, layer_types = self.kv_cache_group_layout()
    export_config = self.export_config  # pyrefly: ignore[missing-attribute]
    if 'kv_cache' in kwargs:
      kwargs['kv_cache'] = uncombine_split_cache(
          kwargs['kv_cache'], layout, layer_types, export_config
      )
    else:
      *rest, kv_cache = args
      args = (
          *rest,
          uncombine_split_cache(kv_cache, layout, layer_types, export_config),
      )
    return super().adapt_inputs(*args, **kwargs)  # pyrefly: ignore[missing-attribute]

  def post_process_kv_cache(self, output_cache):
    slices = super().post_process_kv_cache(output_cache)  # pyrefly: ignore[missing-attribute]
    layout, _ = self.kv_cache_group_layout()
    # One [layers_in_group, ...] output per group, named kv_slice_k_<g> /
    # kv_slice_v_<g>.
    return combine_slices(slices['kv_slice_k'], slices['kv_slice_v'], layout)


_COMBINED_CLASSES: dict[type[Any], type[Any]] = {}


def with_combined_layers(module_cls: type[Any]) -> type[Any]:
  """Returns `module_cls` (a split-cache exportable module) with the layout.

  A mixin rather than fixed subclasses: model-specific split-cache classes
  (e.g. Gemma 4's) are selected before the generic ones, so the combined layout
  is added to whichever class was selected.

  Args:
    module_cls: A split-cache prefill or decode exportable module class.

  Returns:
    A subclass of `module_cls` with `CombinedLayersMixin` first in its bases.
  """
  if module_cls not in _COMBINED_CLASSES:
    _COMBINED_CLASSES[module_cls] = type(
        f'Combined{module_cls.__name__}',
        (CombinedLayersMixin, module_cls),
        {'__module__': __name__},
    )
  return _COMBINED_CLASSES[module_cls]


def ring_buffer_groups_for(
    model_config,
    export_config: base_exportable_module.ExportableModuleConfig,
) -> tuple[bool, ...]:
  """Returns whether each combined cache group is a ring buffer."""
  layout, layer_types = model_kv_cache_group_layout(model_config, export_config)
  ring = export_config.sliding_window_ring_buffer_size is not None
  return tuple(
      ring and layer_types[layers[0]] == 'sliding_attention'
      for layers in layout
  )


class CombinedCacheUpdate(split_cache_module.CacheUpdate):
  """Auxiliary cache-update module for the combined KV cache layout.

  Writes the new entries (`kv_slice`, {'k_<g>': ..., 'v_<g>': ...}) into the
  combined cache group by group: a ring-buffer write (positions modulo the
  buffer size, padding masked out) for sliding-window ring buffer groups, and
  one dynamic_update_slice at `input_pos` for the other groups.
  """

  def __init__(
      self,
      k_ts_idx: int = 2,
      v_ts_idx: int = 3,
      ring_buffer_groups: tuple[bool, ...] = (),
  ):
    super().__init__()
    # The sequence axis of the combined K / V tensors.
    self.k_ts_idx = k_ts_idx
    self.v_ts_idx = v_ts_idx
    # Whether each combined group is a sliding-window ring buffer.
    self.ring_buffer_groups = tuple(ring_buffer_groups)

  @classmethod
  def from_config(
      cls,
      model_config,
      export_config: base_exportable_module.ExportableModuleConfig,
  ) -> 'CombinedCacheUpdate':
    return cls(
        k_ts_idx=export_config.k_ts_idx,  # pyrefly: ignore[bad-argument-type]
        v_ts_idx=export_config.v_ts_idx,  # pyrefly: ignore[bad-argument-type]
        ring_buffer_groups=ring_buffer_groups_for(model_config, export_config),
    )

  def forward(  # pyrefly: ignore[bad-override]
      self,
      kv_slice: dict[str, torch.Tensor],
      kv_cache: dict[str, torch.Tensor],
      input_pos: torch.Tensor,
      valid_mask: torch.Tensor | None = None,
  ):
    """Updates every combined group with one write per K/V tensor.

    Args:
      kv_slice: {'k_<g>': [L_g, ...T...], 'v_<g>': [L_g, ...T...]}.
      kv_cache: {'k_<g>': [L_g, ...S_g...], 'v_<g>': [L_g, ...S_g...]}.
      input_pos: [T] positions of the slice entries.
      valid_mask: [1, T] valid (non-padding) slice entries.

    Returns:
      {'kv_cache': {'k_<g>': ..., 'v_<g>': ...}} with the updated caches.
    """
    num_groups = len(kv_cache) // 2
    assert len(kv_slice) == len(kv_cache) == 2 * num_groups
    ring = self.ring_buffer_groups or (False,) * num_groups
    assert len(ring) == num_groups, (ring, sorted(kv_cache))
    if input_pos.dim() == 0:
      input_pos = input_pos.unsqueeze(0)
    new_cache = {}
    for g in range(num_groups):
      k_cache = kv_cache[_k_key(g)]
      v_cache = kv_cache[_v_key(g)]
      k_slice = kv_slice[_k_key(g)].to(k_cache.dtype)
      v_slice = kv_slice[_v_key(g)].to(v_cache.dtype)
      if ring[g]:
        assert valid_mask is not None
        new_k = base_cache_lib.update_kv_cache_with_sliding(
            k_cache, k_slice, input_pos, valid_mask.squeeze(0), self.k_ts_idx
        )
        new_v = base_cache_lib.update_kv_cache_with_sliding(
            v_cache, v_slice, input_pos, valid_mask.squeeze(0), self.v_ts_idx
        )
      else:
        # Written contiguously from input_pos[0], like the per-layer update of
        # full-attention layers; padding entries land after the valid ones and
        # are overwritten by later steps.
        new_k, new_v = base_cache_lib._update_kv_impl(  # pylint: disable=protected-access
            k_cache,
            v_cache,
            k_slice,
            v_slice,
            input_pos,
            self.k_ts_idx,
            self.v_ts_idx,
        )
      new_cache[_k_key(g)] = new_k
      new_cache[_v_key(g)] = new_v
    return {'kv_cache': new_cache}

  @classmethod
  def _get_input(
      cls,
      model_config,
      input_length,
      export_config: base_exportable_module.ExportableModuleConfig,
  ):
    """Gets sample inputs: the per-layer ones, combined into groups."""
    inputs = super()._get_input(model_config, input_length, export_config)
    for name in ('kv_cache', 'kv_slice'):
      cache = inputs[name]
      inputs[name] = combine_kv_layers(
          [layer.keys for layer in cache.layers],  # pyrefly: ignore[missing-attribute]
          [layer.values for layer in cache.layers],  # pyrefly: ignore[missing-attribute]
          [getattr(l, 'layer_type', 'full_attention') for l in cache.layers],
      )
    return inputs
