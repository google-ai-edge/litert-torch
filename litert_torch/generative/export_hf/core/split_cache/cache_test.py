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
"""Tests for split cache layers."""

from typing import List, Tuple

from litert_torch.generative.export_hf.core.split_cache import cache as split_cache_lib
import torch
import torch.utils._pytree as pytree

from absl.testing import absltest as googletest


def build_cache_data(
    batch_size: int,
    num_layers: int,
    context_len: int,
    head_dim: int,
    all_ones: bool = False,
) -> list[split_cache_lib.LiteRTLMSplitCacheLayer]:
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
        split_cache_lib.LiteRTLMSplitCacheLayer((key_cache, None), (value_cache, None))
    )
  return cache_data


class SplitCacheTest(googletest.TestCase):

  def test_accessors(self):
    batch_head_size = 2
    num_layers = 5
    context_len = 1024
    head_dim = 64

    kv_cache = split_cache_lib.LiteRTLMSplitCache(
        build_cache_data(batch_head_size, num_layers, context_len, head_dim)
    )

    # Cache entries shape.
    self.assertEqual(
        kv_cache.layers[0].keys[0].shape,
        (1, batch_head_size, context_len, head_dim),
    )
    self.assertEqual(
        kv_cache.layers[0].values[0].shape,
        (1, batch_head_size, head_dim, context_len),
    )
    self.assertLen(kv_cache.layers, num_layers)
    # Cache attributes
    self.assertTrue(kv_cache.is_compileable)
    self.assertTrue([not x for x in kv_cache.is_sliding])
    self.assertEqual(kv_cache.get_max_length(), context_len)
    self.assertEqual(kv_cache.max_cache_len, context_len)

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
            "local_attention",
            "full_attention",
            "local_attention",
            "full_attention",
        ]
        self.num_kv_shared_layers = 1

    model_config = MockGemma4Config()
    export_config = split_cache_lib.ExportableModuleConfig(
        model="dummy_model",
        cache_length=1024,
        batch_size=1,
        k_ts_idx=2,
        v_ts_idx=2,
    )

    # Create cache
    kv_cache = split_cache_lib.LiteRTLMSplitCache.create_from_config(
        model_config, export_config
    )

    # Verify that only 3 layers are created (num_layers - num_shared_layers)
    self.assertLen(kv_cache.layers, 3)

    # Verify shapes of created layers
    # Layer 0: local_attention (uses default num_kv_heads=2, head_dim=64)
    self.assertEqual(kv_cache.layers[0].keys[0].shape, (1, 2, 1024, 64))
    self.assertEqual(kv_cache.layers[0].values[0].shape, (1, 2, 1024, 64))

    # Layer 1: full_attention (uses global_num_kv_heads=4, global_head_dim=128)
    self.assertEqual(kv_cache.layers[1].keys[0].shape, (1, 4, 1024, 128))
    self.assertEqual(kv_cache.layers[1].values[0].shape, (1, 4, 1024, 128))

    # Layer 2: local_attention (uses default num_kv_heads=2, head_dim=64)
    self.assertEqual(kv_cache.layers[2].keys[0].shape, (1, 2, 1024, 64))
    self.assertEqual(kv_cache.layers[2].values[0].shape, (1, 2, 1024, 64))

    # Test insert_dummy_cache_layers
    kv_cache.insert_dummy_cache_layers(model_config)
    self.assertLen(kv_cache.layers, 4)
    self.assertTrue(
        torch.allclose(kv_cache.layers[3].keys[0], kv_cache.layers[0].keys[0])
    )

    # Test remove_dummy_cache_layers
    kv_cache.remove_dummy_cache_layers(model_config)
    self.assertLen(kv_cache.layers, 3)

  def test_to_moves_cache_and_slices(self):
    kv_cache = split_cache_lib.LiteRTLMSplitCache(build_cache_data(2, 2, 16, 8))
    layer = kv_cache.layers[1]
    layer.update(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8))

    self.assertIs(kv_cache.to("meta"), kv_cache)

    self.assertTrue(kv_cache.layers[0].keys[0].is_meta)
    self.assertTrue(kv_cache.layers[0].values[0].is_meta)
    self.assertIsNone(kv_cache.layers[0].keys[1])
    self.assertIsNone(kv_cache.layers[0].values[1])
    self.assertTrue(all(t.is_meta for t in layer.keys + layer.values))
    self.assertEqual(layer.keys[0].shape, (1, 2, 16, 8))
    self.assertEqual(layer.keys[1].shape, (1, 2, 4, 8))

  def test_create_from_config_uses_cache_dtype(self):
    class MockConfig:
      num_hidden_layers = 2
      num_key_value_heads = 2
      head_dim = 8
      hidden_size = 32
      num_attention_heads = 4

    for use_fp16, expected_dtype in (
        (False, torch.float32),
        (True, torch.float16),
    ):
      with self.subTest(use_fp16=use_fp16):
        export_config = split_cache_lib.ExportableModuleConfig(
            model="dummy_model",
            cache_length=16,
            experimental_use_fp16=use_fp16,
        )
        kv_cache = split_cache_lib.LiteRTLMSplitCache.create_from_config(
            MockConfig(), export_config
        )
        layer = kv_cache.layers[0]
        self.assertEqual(layer.keys[0].dtype, expected_dtype)
        self.assertEqual(layer.values[0].dtype, expected_dtype)

        # Slices are stored in the cache dtype too.
        layer.update(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8))
        self.assertEqual(layer.keys[1].dtype, expected_dtype)
        self.assertEqual(layer.values[1].dtype, expected_dtype)

  def test_pytree_round_trip_with_conv_layer(self):
    class MockLfm2Config:
      num_hidden_layers = 2
      layer_types = ["conv", "full_attention"]
      conv_L_cache = 3
      hidden_size = 32
      num_attention_heads = 4
      num_key_value_heads = 2

    for k_ts_idx, v_ts_idx in ((2, 3), (3, 2)):
      with self.subTest(k_ts_idx=k_ts_idx, v_ts_idx=v_ts_idx):
        export_config = split_cache_lib.ExportableModuleConfig(
            model="dummy_model",
            cache_length=16,
            k_ts_idx=k_ts_idx,
            v_ts_idx=v_ts_idx,
        )
        kv_cache = split_cache_lib.LiteRTLMSplitCache.create_from_config(
            MockLfm2Config(), export_config
        )

        leaves, spec = pytree.tree_flatten(kv_cache)
        restored = pytree.tree_unflatten(leaves, spec)

        conv_layer, attention_layer = restored.layers
        self.assertEqual(conv_layer.layer_type, "conv")
        self.assertIs(conv_layer.conv_states, kv_cache.layers[0].conv_states)
        # The K/V axes come from the attention layer, not from conv layer 0.
        self.assertEqual(attention_layer.k_ts_idx, k_ts_idx)
        self.assertEqual(attention_layer.v_ts_idx, v_ts_idx)
        self.assertEqual(attention_layer.max_cache_len, 16)
        self.assertEqual(attention_layer.head_dim, 8)


if __name__ == "__main__":
  googletest.main()
