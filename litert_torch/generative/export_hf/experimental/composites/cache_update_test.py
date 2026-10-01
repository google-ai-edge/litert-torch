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
"""Tests for cache_update composite and ring buffer wrapping."""

from absl.testing import parameterized
from litert_torch.generative.export_hf.experimental.composites import cache_update
import torch
from absl.testing import absltest as googletest


class CacheUpdateTest(parameterized.TestCase):

  def test_update_kv_cache_with_sliding_wrapping(self):
    # Test ts_idx=2 (e.g. K cache: [1, 2, S, D])
    cache_k = torch.zeros((1, 2, 8, 16), dtype=torch.float32)
    k_update = torch.ones((1, 2, 3, 16), dtype=torch.float32)
    positions = torch.tensor([6, 7, 8], dtype=torch.int32)
    valid_mask = torch.tensor([True, True, True], dtype=torch.bool)

    new_k = cache_update.update_kv_cache_with_sliding(
        cache_k, k_update, positions, valid_mask, ts_idx=2
    )
    self.assertEqual(new_k.shape, (1, 2, 8, 16))
    self.assertTrue((new_k[:, :, 6, :] == 1).all())
    self.assertTrue((new_k[:, :, 7, :] == 1).all())
    self.assertTrue((new_k[:, :, 0, :] == 1).all())
    self.assertTrue((new_k[:, :, 1:6, :] == 0).all())

    # Test ts_idx=3 (e.g. V cache: [1, 2, D, S])
    cache_v = torch.zeros((1, 2, 16, 8), dtype=torch.float32)
    v_update = torch.ones((1, 2, 16, 3), dtype=torch.float32)

    new_v = cache_update.update_kv_cache_with_sliding(
        cache_v, v_update, positions, valid_mask, ts_idx=3
    )
    self.assertEqual(new_v.shape, (1, 2, 16, 8))
    self.assertTrue((new_v[:, :, :, 6] == 1).all())
    self.assertTrue((new_v[:, :, :, 7] == 1).all())
    self.assertTrue((new_v[:, :, :, 0] == 1).all())
    self.assertTrue((new_v[:, :, :, 1:6] == 0).all())

  @parameterized.parameters(2, 3)
  def test_update_kv_cache_with_sliding_fp16_cache_fp32_compute(self, ts_idx):
    s, t, d = 8, 3, 16
    shape_cache = (1, 2, s, d) if ts_idx == 2 else (1, 2, d, s)
    shape_update = (1, 2, t, d) if ts_idx == 2 else (1, 2, d, t)
    cache = torch.zeros(shape_cache, dtype=torch.float16)
    update = torch.full(shape_update, 1.5, dtype=torch.float16)
    positions = torch.tensor([6, 7, 8], dtype=torch.int32)
    valid_mask = torch.tensor([True, True, False], dtype=torch.bool)

    new_cache = cache_update.update_kv_cache_with_sliding(
        cache,
        update,
        positions,
        valid_mask,
        ts_idx=ts_idx,
        compute_dtype=torch.float32,
    )
    self.assertEqual(new_cache.dtype, torch.float16)
    self.assertEqual(new_cache.shape, shape_cache)
    new_cache = new_cache if ts_idx == 2 else new_cache.transpose(2, 3)
    self.assertTrue((new_cache[:, :, 6:8, :] == 1.5).all())
    # Slot 0 (position 8) is masked out by valid_mask.
    self.assertTrue((new_cache[:, :, 0:6, :] == 0).all())

    class UpdateModule(torch.nn.Module):

      def forward(self, c, u, p, m):
        return cache_update.update_kv_cache_with_sliding(
            c, u, p, m, ts_idx=ts_idx, compute_dtype=torch.float32
        )

    exported = torch.export.export(
        UpdateModule(), (cache, update, positions, valid_mask)
    )
    matmul_dtypes = [
        node.meta["val"].dtype
        for node in exported.graph.nodes
        if node.op == "call_function"
        and ("matmul" in str(node.target) or "mm" in str(node.target))
    ]
    self.assertNotEmpty(matmul_dtypes)
    self.assertTrue(all(dt == torch.float32 for dt in matmul_dtypes))

  def test_cache_update_composite_with_ring_buffer(self):
    cache_k = torch.zeros((1, 2, 8, 16), dtype=torch.float32)
    cache_v = torch.zeros((1, 2, 16, 8), dtype=torch.float32)
    key_proj = torch.ones((1, 2, 3, 16), dtype=torch.float32)
    # value_proj is transposed when passed into cache_update
    value_proj = torch.ones((1, 2, 3, 16), dtype=torch.float32)

    param_tensor = torch.zeros((1, 1, 1, 7), dtype=torch.int32)
    param_tensor[..., 0] = 6  # offset
    param_tensor[..., 3] = 3  # update_len

    indices_k = torch.zeros(4, dtype=torch.int32)
    indices_k[2] = 6
    indices_v = torch.zeros(4, dtype=torch.int32)
    indices_v[3] = 6

    new_k, new_v = cache_update.cache_update(
        key_proj=key_proj,
        value_proj=value_proj,
        runtime_param_tensor=param_tensor,
        cache_k=cache_k,
        cache_v=cache_v,
        indices_k=indices_k,
        indices_v=indices_v,
        kv_heads=2,
        kv_batch_size=1,
        cache_len=8,
        head_size=16,
        is_ring_buffer=True,
        k_ts_idx=2,
        v_ts_idx=3,
    )

    self.assertTrue((new_k[:, :, 6, :] == 1).all())
    self.assertTrue((new_k[:, :, 7, :] == 1).all())
    self.assertTrue((new_k[:, :, 0, :] == 1).all())
    self.assertTrue((new_k[:, :, 1:6, :] == 0).all())

    self.assertTrue((new_v[:, :, :, 6] == 1).all())
    self.assertTrue((new_v[:, :, :, 7] == 1).all())
    self.assertTrue((new_v[:, :, :, 0] == 1).all())
    self.assertTrue((new_v[:, :, :, 1:6] == 0).all())

  def test_sequential_multi_step_ring_buffer_updates(self):
    # Simulates:
    # Turn 1: 6 tokens prefill (fill slots 0..5)
    # Turn 2: 4 tokens prefill (fill slots 6, 7, 0, 1 - wrap around)
    # Turn 2 decode: 1 token (fill slot 2)
    # Turn 2 decode: 1 token (fill slot 3)
    cache_k = torch.zeros((1, 2, 8, 16), dtype=torch.float32)
    cache_v = torch.zeros((1, 2, 16, 8), dtype=torch.float32)

    # Step 1: Turn 1 prefill 6 tokens
    key_proj_1 = torch.full((1, 2, 6, 16), 1.0, dtype=torch.float32)
    val_proj_1 = torch.full((1, 2, 6, 16), 1.0, dtype=torch.float32)
    param_1 = torch.zeros((1, 1, 1, 7), dtype=torch.int32)
    param_1[..., 0] = 0
    param_1[..., 3] = 6
    cache_k, cache_v = cache_update.cache_update(
        key_proj=key_proj_1,
        value_proj=val_proj_1,
        runtime_param_tensor=param_1,
        cache_k=cache_k,
        cache_v=cache_v,
        indices_k=torch.zeros(4, dtype=torch.int32),
        indices_v=torch.zeros(4, dtype=torch.int32),
        kv_heads=2,
        kv_batch_size=1,
        cache_len=8,
        head_size=16,
        is_ring_buffer=True,
        k_ts_idx=2,
        v_ts_idx=3,
    )
    self.assertTrue((cache_k[:, :, 0:6, :] == 1.0).all())
    self.assertTrue((cache_k[:, :, 6:8, :] == 0.0).all())

    # Step 2: Turn 2 prefill 4 tokens (offset=6, wraps to 6, 7, 0, 1)
    key_proj_2 = torch.full((1, 2, 4, 16), 2.0, dtype=torch.float32)
    val_proj_2 = torch.full((1, 2, 4, 16), 2.0, dtype=torch.float32)
    param_2 = torch.zeros((1, 1, 1, 7), dtype=torch.int32)
    param_2[..., 0] = 6
    param_2[..., 3] = 4
    cache_k, cache_v = cache_update.cache_update(
        key_proj=key_proj_2,
        value_proj=val_proj_2,
        runtime_param_tensor=param_2,
        cache_k=cache_k,
        cache_v=cache_v,
        indices_k=torch.zeros(4, dtype=torch.int32),
        indices_v=torch.zeros(4, dtype=torch.int32),
        kv_heads=2,
        kv_batch_size=1,
        cache_len=8,
        head_size=16,
        is_ring_buffer=True,
        k_ts_idx=2,
        v_ts_idx=3,
    )
    self.assertTrue((cache_k[:, :, [6, 7, 0, 1], :] == 2.0).all())
    self.assertTrue((cache_k[:, :, 2:6, :] == 1.0).all())
    self.assertTrue((cache_v[:, :, :, [6, 7, 0, 1]] == 2.0).all())
    self.assertTrue((cache_v[:, :, :, 2:6] == 1.0).all())

    # Step 3: Turn 2 decode 1 token (offset=2, writes to slot 2)
    key_proj_3 = torch.full((1, 2, 1, 16), 3.0, dtype=torch.float32)
    val_proj_3 = torch.full((1, 2, 1, 16), 3.0, dtype=torch.float32)
    param_3 = torch.zeros((1, 1, 1, 7), dtype=torch.int32)
    param_3[..., 0] = 2
    param_3[..., 3] = 1
    cache_k, cache_v = cache_update.cache_update(
        key_proj=key_proj_3,
        value_proj=val_proj_3,
        runtime_param_tensor=param_3,
        cache_k=cache_k,
        cache_v=cache_v,
        indices_k=torch.zeros(4, dtype=torch.int32),
        indices_v=torch.zeros(4, dtype=torch.int32),
        kv_heads=2,
        kv_batch_size=1,
        cache_len=8,
        head_size=16,
        is_ring_buffer=True,
        k_ts_idx=2,
        v_ts_idx=3,
    )
    self.assertTrue((cache_k[:, :, 2, :] == 3.0).all())
    self.assertTrue((cache_k[:, :, 3:6, :] == 1.0).all())
    self.assertTrue((cache_k[:, :, [6, 7, 0, 1], :] == 2.0).all())

  def test_export_contains_cache_update_composite(self):
    class UpdateModule(torch.nn.Module):

      def forward(self, kp, vp, param, ck, cv, ik, iv):
        return cache_update.cache_update(
            key_proj=kp,
            value_proj=vp,
            runtime_param_tensor=param,
            cache_k=ck,
            cache_v=cv,
            indices_k=ik,
            indices_v=iv,
            kv_heads=2,
            kv_batch_size=1,
            cache_len=8,
            head_size=16,
            is_ring_buffer=True,
            k_ts_idx=2,
            v_ts_idx=3,
        )

    mod = UpdateModule()
    example_args = (
        torch.randn((1, 2, 3, 16)),
        torch.randn((1, 2, 3, 16)),
        torch.zeros((1, 1, 1, 7), dtype=torch.int32),
        torch.randn((1, 2, 8, 16)),
        torch.randn((1, 2, 16, 8)),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
    )
    exported = torch.export.export(mod, example_args)
    graph_str = str(exported.graph)
    self.assertIn("odml.cache_update", graph_str)


if __name__ == "__main__":
  googletest.main()
