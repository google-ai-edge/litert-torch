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
"""Tests for the combined KV cache layout of split-cache exports."""

import types

from absl.testing import parameterized
from litert_torch.generative.export_hf.core.split_cache import cache as split_cache_lib
from litert_torch.generative.export_hf.core.split_cache import combined_layers
import torch

from absl.testing import absltest as googletest

_L, _BK, _S, _H = 3, 2, 16, 8


def _per_layer_split_cache(dtype=torch.float16):
  layers = []
  for _ in range(_L):
    layers.append(
        split_cache_lib.LiteRTLMSplitCacheLayer(
            (torch.randn(1, _BK, _S, _H).to(dtype), None),
            (torch.randn(1, _BK, _H, _S).to(dtype), None),
        )
    )
  return split_cache_lib.LiteRTLMSplitCache(layers)


class CombinedLayersTest(parameterized.TestCase):

  def test_combine_uncombine_round_trip(self):
    kv_cache = _per_layer_split_cache()
    combined = combined_layers.combine_split_cache(kv_cache)
    self.assertEqual(set(combined), {'k_0', 'v_0'})
    self.assertEqual(combined['k_0'].shape, (_L, _BK, _S, _H))
    self.assertEqual(combined['v_0'].shape, (_L, _BK, _H, _S))

    config = types.SimpleNamespace(batch_size=1, k_ts_idx=2, v_ts_idx=3)
    restored = combined_layers.uncombine_split_cache(
        combined, [list(range(_L))], ['full_attention'] * _L, config
    )
    self.assertLen(restored.layers, _L)
    for orig, new in zip(kv_cache.layers, restored.layers):
      torch.testing.assert_close(new.keys[0], orig.keys[0], rtol=0, atol=0)
      torch.testing.assert_close(new.values[0], orig.values[0], rtol=0, atol=0)
      self.assertEqual(new.max_cache_len, _S)
      self.assertEqual(new.head_dim, _H)

  def test_group_layout_by_layer_type_and_shape(self):
    # Gemma 4 style: interleaved sliding (small) and global (large) layers.
    types_ = ['sliding_attention'] * 2 + ['full_attention']
    types_ += ['sliding_attention'] * 2 + ['full_attention']
    small_k, small_v = (1, _BK, _S, _H), (1, _BK, _H, _S)
    big_k, big_v = (1, 1, 2 * _S, 2 * _H), (1, 1, 2 * _H, 2 * _S)
    k_shapes = [big_k if t == 'full_attention' else small_k for t in types_]
    v_shapes = [big_v if t == 'full_attention' else small_v for t in types_]
    layout = combined_layers.kv_cache_group_layout(types_, k_shapes, v_shapes)
    self.assertEqual(layout, [[0, 1, 3, 4], [2, 5]])

    k = [torch.randn(s).to(torch.float16) for s in k_shapes]
    v = [torch.randn(s).to(torch.float16) for s in v_shapes]
    combined = combined_layers.combine_kv_layers(k, v, types_)
    self.assertEqual(combined['k_0'].shape, (4, _BK, _S, _H))
    self.assertEqual(combined['k_1'].shape, (2, 1, 2 * _S, 2 * _H))
    self.assertEqual(combined['v_1'].shape, (2, 1, 2 * _H, 2 * _S))

    config = types.SimpleNamespace(batch_size=1, k_ts_idx=2, v_ts_idx=3)
    restored = combined_layers.uncombine_split_cache(
        combined, layout, types_, config
    )
    self.assertLen(restored.layers, len(types_))
    for i, layer in enumerate(restored.layers):
      torch.testing.assert_close(layer.keys[0], k[i], rtol=0, atol=0)
      torch.testing.assert_close(layer.values[0], v[i], rtol=0, atol=0)
      self.assertEqual(layer.is_sliding, types_[i] == 'sliding_attention')

    slices = combined_layers.combine_slices(k, v, layout)
    self.assertLen(slices['kv_slice_k'], 2)
    self.assertEqual(slices['kv_slice_k'][1].shape, (2, 1, 2 * _S, 2 * _H))

  def test_group_layout_rejects_non_attention_layers(self):
    with self.assertRaisesRegex(ValueError, 'only supports attention layers'):
      combined_layers.kv_cache_group_layout(
          ['full_attention', 'conv'],
          [(1, _BK, _S, _H)] * 2,
          [(1, _BK, _H, _S)] * 2,
      )

  def test_combined_cache_update_ring_buffer_group(self):
    # Group 0: ring buffer (sliding); group 1: linear (global).
    s_ring = 8
    k0 = torch.randn(_L, _BK, s_ring, _H).to(torch.float16)
    v0 = torch.randn(_L, _BK, _H, s_ring).to(torch.float16)
    k1 = torch.randn(2, _BK, _S, _H).to(torch.float16)
    v1 = torch.randn(2, _BK, _H, _S).to(torch.float16)
    t = 3
    start = 7  # Wraps around the 8-slot ring buffer: slots 7, 0, 1.
    input_pos = torch.arange(start, start + t, dtype=torch.int32)
    valid_mask = torch.tensor([[True, True, False]])
    ks0 = torch.randn(_L, _BK, t, _H).to(torch.float16)
    vs0 = torch.randn(_L, _BK, _H, t).to(torch.float16)
    ks1 = torch.randn(2, _BK, t, _H).to(torch.float16)
    vs1 = torch.randn(2, _BK, _H, t).to(torch.float16)

    module = combined_layers.CombinedCacheUpdate(
        k_ts_idx=2, v_ts_idx=3, ring_buffer_groups=(True, False)
    )
    out = module(
        {'k_0': ks0, 'v_0': vs0, 'k_1': ks1, 'v_1': vs1},
        {
            'k_0': k0.clone(),
            'v_0': v0.clone(),
            'k_1': k1.clone(),
            'v_1': v1.clone(),
        },
        input_pos,
        valid_mask,
    )['kv_cache']

    exp_k0, exp_v0 = k0.clone(), v0.clone()
    for j, slot in enumerate([7, 0]):  # The third entry is padding.
      exp_k0[:, :, slot] = ks0[:, :, j]
      exp_v0[:, :, :, slot] = vs0[:, :, :, j]
    torch.testing.assert_close(out['k_0'], exp_k0, rtol=0, atol=0)
    torch.testing.assert_close(out['v_0'], exp_v0, rtol=0, atol=0)

    exp_k1, exp_v1 = k1.clone(), v1.clone()
    exp_k1[:, :, start : start + t] = ks1
    exp_v1[:, :, :, start : start + t] = vs1
    torch.testing.assert_close(out['k_1'], exp_k1, rtol=0, atol=0)
    torch.testing.assert_close(out['v_1'], exp_v1, rtol=0, atol=0)

  @parameterized.parameters(1, 4)
  def test_combined_cache_update(self, t):
    k_cache = torch.randn(_L, _BK, _S, _H).to(torch.float16)
    v_cache = torch.randn(_L, _BK, _H, _S).to(torch.float16)
    k_slice = torch.randn(_L, _BK, t, _H).to(torch.float16)
    v_slice = torch.randn(_L, _BK, _H, t).to(torch.float16)
    start = 5
    input_pos = torch.arange(start, start + t, dtype=torch.int32)
    valid_mask = torch.ones((1, t), dtype=torch.bool)

    module = combined_layers.CombinedCacheUpdate(k_ts_idx=2, v_ts_idx=3)
    out = module(
        {'k_0': k_slice, 'v_0': v_slice},
        {'k_0': k_cache.clone(), 'v_0': v_cache.clone()},
        input_pos,
        valid_mask,
    )['kv_cache']

    expected_k = k_cache.clone()
    expected_k[:, :, start : start + t, :] = k_slice
    expected_v = v_cache.clone()
    expected_v[:, :, :, start : start + t] = v_slice
    torch.testing.assert_close(out['k_0'], expected_k, rtol=0, atol=0)
    torch.testing.assert_close(out['v_0'], expected_v, rtol=0, atol=0)

    exported = torch.export.export(
        module,
        (
            {'k_0': k_slice, 'v_0': v_slice},
            {'k_0': k_cache, 'v_0': v_cache},
            input_pos,
            valid_mask,
        ),
    )
    nodes = [n for n in exported.graph.nodes if n.op == 'call_function']
    targets = [str(n.target) for n in nodes]
    # One in-place-able update per combined tensor.
    self.assertLen(
        [x for x in targets if 'dynamic_update_slice' in x], 2, targets
    )
    # No other op materializes a cache-sized tensor (e.g. a full-cache copy).
    cache_numel = k_cache.numel()
    big = [
        str(n.target)
        for n in nodes
        if 'dynamic_update_slice' not in str(n.target)
        and isinstance(n.meta.get('val'), torch.Tensor)
        and n.meta['val'].numel() >= cache_numel
    ]
    self.assertEmpty(big, targets)


if __name__ == '__main__':
  googletest.main()
