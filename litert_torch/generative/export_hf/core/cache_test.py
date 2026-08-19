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
"""Tests for cache layers."""

from typing import List, Tuple

from litert_torch.generative.export_hf.core import cache as cache_lib
import torch

from absl.testing import absltest as googletest


def build_cache_data(
    batch_size: int,
    num_layers: int,
    context_len: int,
    head_dim: int,
    all_ones: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
  cache_data = []
  for _ in range(num_layers):
    if all_ones:
      key_cache = torch.ones(
          1, batch_size, context_len, head_dim, dtype=torch.float32
      )
      value_cache = torch.ones(
          1, batch_size, head_dim, context_len, dtype=torch.float32
      )
    else:
      key_cache = torch.randn(
          1, batch_size, context_len, head_dim, dtype=torch.float32
      )
      value_cache = torch.randn(
          1, batch_size, head_dim, context_len, dtype=torch.float32
      )
    cache_data.append(
        cache_lib.LiteRTLMCacheLayer(
            key_cache, value_cache, layer_type="full_attention"
        )
    )
  return cache_data


def update_cache(slices, kv_cache):
  """Updates the cache with the given slices."""
  for i in range(len(slices)):
    kv_cache.update(slices[i][0], slices[i][1], i)
  return kv_cache


class CacheTest(googletest.TestCase):

  def assert_cache_equals(
      self,
      kv_cache: cache_lib.LiteRTLMCache,
      expected_kv_cache: cache_lib.LiteRTLMCache,
      num_layers,
  ):
    """Asserts that the cache is equal to the expected cache."""
    for i in range(num_layers):
      layer_cache = kv_cache.layers[i]
      expected_layer_cache = expected_kv_cache.layers[i]
      self.assertTrue(
          torch.allclose(layer_cache.keys, expected_layer_cache.keys)
      )
      self.assertTrue(
          torch.allclose(layer_cache.values, expected_layer_cache.values)
      )

  def assert_cache_slice_equals(
      self,
      slices: List[Tuple[torch.Tensor, torch.Tensor]],
      kv_cache: cache_lib.LiteRTLMCache,
      num_layers,
      time_step,
      input_seq,
  ):
    """Asserts that the cache slices are equal to the expected slices."""
    for i in range(num_layers):
      k_slice = kv_cache.layers[i].keys[
          :, :, time_step : time_step + input_seq, :
      ]
      v_slice = kv_cache.layers[i].values[
          :, :, :, time_step : time_step + input_seq
      ]
      expected_k_slice = slices[i][0]
      expected_v_slice = slices[i][1]
      self.assertTrue(torch.allclose(k_slice, expected_k_slice))
      self.assertTrue(torch.allclose(v_slice, expected_v_slice))

  def test_accessors(self):
    batch_head_size = 2
    num_layers = 5
    context_len = 1024
    head_dim = 64

    kv_cache = cache_lib.LiteRTLMCache(
        build_cache_data(batch_head_size, num_layers, context_len, head_dim)
    )

    # Cache entries shape.
    self.assertEqual(
        kv_cache.layers[0].keys.shape,
        (1, batch_head_size, context_len, head_dim),
    )
    self.assertEqual(
        kv_cache.layers[0].values.shape,
        (1, batch_head_size, head_dim, context_len),
    )
    self.assertLen(kv_cache.layers, num_layers)
    # Cache attributes
    self.assertTrue(kv_cache.is_compileable)
    self.assertTrue([not x for x in kv_cache.is_sliding])
    self.assertEqual(kv_cache.max_cache_len, context_len)

  def test_update(self):
    batch_size = 2
    batch_head_size = 8
    kv_head_size = batch_head_size // batch_size
    num_layers = 5
    context_len = 1024
    head_dim = 64
    input_seq = 10
    time_step = 33

    kv_cache = cache_lib.LiteRTLMCache(
        build_cache_data(
            batch_head_size, num_layers, context_len, head_dim, all_ones=True
        )
    )
    cache_kwargs = {
        "cache_position": torch.tensor([time_step], dtype=torch.int32)
    }
    kv_cache.set_cache_runtime_args(cache_kwargs)
    k_slice = torch.zeros(batch_size, kv_head_size, input_seq, head_dim)
    v_slice = torch.zeros(batch_size, kv_head_size, input_seq, head_dim)
    slices = [(k_slice, v_slice)] * num_layers
    expected_k_slice = torch.zeros(
        1, batch_size * kv_head_size, input_seq, head_dim
    )
    expected_v_slice = torch.zeros(
        1, batch_size * kv_head_size, head_dim, input_seq
    )
    expected_slices = [(expected_k_slice, expected_v_slice)] * num_layers

    kv_cache = update_cache(slices, kv_cache)

    self.assert_cache_slice_equals(
        expected_slices, kv_cache, num_layers, time_step, input_seq
    )

  def test_flatten_round_trip(self):
    batch_head_size = 2
    num_layers = 5
    context_len = 1024
    head_dim = 64
    kv_cache = cache_lib.LiteRTLMCache(
        build_cache_data(batch_head_size, num_layers, context_len, head_dim)
    )

    flattened, context = cache_lib._flatten_kvc_t(kv_cache)
    unflattened = cache_lib._unflatten_kvc_t(flattened, context)

    self.assertLen(flattened, num_layers * 2)
    self.assertEqual(
        kv_cache.layers[0].keys.shape, unflattened.layers[0].keys.shape
    )
    self.assertEqual(
        kv_cache.layers[0].values.shape, unflattened.layers[0].values.shape
    )
    self.assert_cache_equals(kv_cache, unflattened, num_layers)

  def test_flatten_with_keys(self):
    batch_head_size = 2
    num_layers = 5
    context_len = 1024
    head_dim = 64
    kv_cache = cache_lib.LiteRTLMCache(
        build_cache_data(batch_head_size, num_layers, context_len, head_dim)
    )

    flattened_list, flattend_names = cache_lib._flatten_kvc_t_with_keys(
        kv_cache
    )
    self.assertLen(flattened_list, num_layers * 2)
    self.assertLen(flattend_names, num_layers * 2)

  def test_gemma4_cache(self):
    class MockGemma4Config:

      def __init__(self):
        self.num_hidden_layers = 4
        self.num_key_value_heads = 2
        self.num_global_key_value_heads = 4
        self.global_head_dim = 128
        self.head_dim = 64
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.layer_types = [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]
        self.num_kv_shared_layers = 1

    model_config = MockGemma4Config()
    export_config = cache_lib.ExportableModuleConfig(
        model="dummy_model",
        cache_length=1024,
        batch_size=1,
        k_ts_idx=2,
        v_ts_idx=2,
    )

    # Create cache
    kv_cache = cache_lib.LiteRTLMCache.create_from_config(
        model_config, export_config
    )

    # Verify that only 3 layers are created (num_layers - num_shared_layers)
    self.assertLen(kv_cache.layers, 3)

    # Verify shapes of created layers
    # Layer 0: local_attention (uses default num_kv_heads=2, head_dim=64)
    self.assertEqual(kv_cache.layers[0].keys.shape, (1, 2, 1024, 64))
    self.assertEqual(kv_cache.layers[0].values.shape, (1, 2, 1024, 64))

    # Layer 1: full_attention (uses global_num_kv_heads=4, global_head_dim=128)
    self.assertEqual(kv_cache.layers[1].keys.shape, (1, 4, 1024, 128))
    self.assertEqual(kv_cache.layers[1].values.shape, (1, 4, 1024, 128))

    # Layer 2: local_attention (uses default num_kv_heads=2, head_dim=64)
    self.assertEqual(kv_cache.layers[2].keys.shape, (1, 2, 1024, 64))
    self.assertEqual(kv_cache.layers[2].values.shape, (1, 2, 1024, 64))

    # Test insert_dummy_cache_layers
    kv_cache.insert_dummy_cache_layers(model_config)
    self.assertLen(kv_cache.layers, 4)
    self.assertTrue(
        torch.allclose(kv_cache.layers[3].keys, kv_cache.layers[0].keys)
    )

    # Test remove_dummy_cache_layers
    kv_cache.remove_dummy_cache_layers(model_config)
    self.assertLen(kv_cache.layers, 3)

  def test_linear_attention_conv_cache(self):
    class MockQwenConfig:

      def __init__(self):
        self.num_hidden_layers = 2
        self.layer_types = ["full_attention", "linear_attention"]
        self.hidden_size = 64
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.head_dim = 16
        self.linear_conv_kernel_dim = 4
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 2
        self.linear_num_value_heads = 4

    class MockExportConfig:

      def __init__(self):
        self.batch_size = 1
        self.cache_length = 128
        self.k_ts_idx = 2
        self.v_ts_idx = 2
        self.experimental_use_fp16 = False

    model_config = MockQwenConfig()
    export_config = MockExportConfig()

    kv_cache = cache_lib.LiteRTLMCache.create_from_config(
        model_config, export_config
    )
    self.assertLen(kv_cache.layers, 2)
    self.assertIsInstance(kv_cache.layers[0], cache_lib.LiteRTLMCacheLayer)
    self.assertIsInstance(kv_cache.layers[1], cache_lib.LiteRTLMConvCacheLayer)

    conv_layer = kv_cache.layers[1]
    self.assertIsNotNone(conv_layer.recurrent_states)
    self.assertEqual(conv_layer.conv_states.shape, (1, 2 * 16 * 2 + 4 * 16, 3))
    self.assertEqual(conv_layer.recurrent_states.shape, (1, 4, 16, 16))

    # Test update_recurrent_state
    new_r = torch.ones_like(conv_layer.recurrent_states)
    conv_layer.update_recurrent_state(new_r)
    self.assertTrue(torch.allclose(conv_layer.recurrent_states, new_r))

    # Test flatten / unflatten roundtrip
    flattened, context = cache_lib._flatten_kvc_t(kv_cache)
    self.assertLen(flattened, 4)  # k_0, v_0, c_1, r_1
    flat_names = context[0]
    self.assertEqual(flat_names, ["k_0", "v_0", "c_1", "r_1"])

    unflattened = cache_lib._unflatten_kvc_t(flattened, context)
    self.assertIsInstance(
        unflattened.layers[1], cache_lib.LiteRTLMConvCacheLayer
    )
    self.assertTrue(
        torch.allclose(
            unflattened.layers[1].conv_states, conv_layer.conv_states
        )
    )
    self.assertTrue(
        torch.allclose(
            unflattened.layers[1].recurrent_states, conv_layer.recurrent_states
        )
    )


if __name__ == "__main__":
  googletest.main()

