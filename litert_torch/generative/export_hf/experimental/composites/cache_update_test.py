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
  def test_update_kv_cache_with_sliding_fp16_cache(self, ts_idx):
    s, t, d = 8, 3, 16
    shape_cache = (1, 2, s, d) if ts_idx == 2 else (1, 2, d, s)
    shape_update = (1, 2, t, d) if ts_idx == 2 else (1, 2, d, t)
    cache = torch.zeros(shape_cache, dtype=torch.float16)
    update = torch.full(shape_update, 1.5, dtype=torch.float16)
    positions = torch.tensor([6, 7, 8], dtype=torch.int32)
    valid_mask = torch.tensor([True, True, False], dtype=torch.bool)

    new_cache = cache_update.update_kv_cache_with_sliding(
        cache, update, positions, valid_mask, ts_idx=ts_idx
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
            c, u, p, m, ts_idx=ts_idx
        )

    exported = torch.export.export(
        UpdateModule(), (cache, update, positions, valid_mask)
    )
    # The cache must not be upcast: the routing matmul runs in fp16, and fp32
    # only appears on the small [T, S] routing mask.
    tensor_vals = [
        node.meta["val"]
        for node in exported.graph.nodes
        if node.op == "call_function"
        and isinstance(node.meta.get("val"), torch.Tensor)
    ]
    matmul_dtypes = [
        node.meta["val"].dtype
        for node in exported.graph.nodes
        if node.op == "call_function"
        and ("matmul" in str(node.target) or "mm" in str(node.target))
    ]
    self.assertNotEmpty(matmul_dtypes)
    self.assertTrue(all(dt == torch.float16 for dt in matmul_dtypes))
    fp32_numels = [v.numel() for v in tensor_vals if v.dtype == torch.float32]
    self.assertTrue(all(n <= t * s for n in fp32_numels))

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

  @parameterized.product(
      is_ring_buffer=(True, False),
      k_update_ts_idx=(2, 3),
      v_update_ts_idx=(2, 3),
  )
  def test_update_layouts_write_the_same_cache(
      self, is_ring_buffer, k_update_ts_idx, v_update_ts_idx
  ):
    heads, cache_len, head_size, new_len, start = 2, 8, 4, 3, 6
    if not is_ring_buffer:
      start = 2  # A linear cache does not wrap.
    cache_k = torch.randn((1, heads, cache_len, head_size))
    cache_v = torch.randn((1, heads, head_size, cache_len))
    # New tokens in the cache layout, i.e. K [1, H, T, D] and V [1, H, D, T].
    k_new = torch.randn((1, heads, new_len, head_size))
    v_new = torch.randn((1, heads, head_size, new_len))
    param = torch.zeros((1, 1, 1, 7), dtype=torch.int32)
    param[..., 0] = start
    param[..., 3] = new_len
    indices_k = torch.zeros(4, dtype=torch.int32)
    indices_k[2] = start
    indices_v = torch.zeros(4, dtype=torch.int32)
    indices_v[3] = start

    def run(k_ts, v_ts):
      return cache_update.cache_update(
          key_proj=k_new if k_ts == 2 else k_new.transpose(-2, -1),
          value_proj=v_new if v_ts == 3 else v_new.transpose(-2, -1),
          runtime_param_tensor=param,
          cache_k=cache_k,
          cache_v=cache_v,
          indices_k=indices_k,
          indices_v=indices_v,
          kv_heads=heads,
          kv_batch_size=1,
          cache_len=cache_len,
          head_size=head_size,
          is_ring_buffer=is_ring_buffer,
          k_update_ts_idx=k_ts,
          v_update_ts_idx=v_ts,
      )

    new_k, new_v = run(k_update_ts_idx, v_update_ts_idx)
    # Every layout must write what the historical K [T, D], V [T, D] one does.
    want_k, want_v = run(2, 2)
    torch.testing.assert_close(new_k, want_k)
    torch.testing.assert_close(new_v, want_v)
    slots = [(start + i) % cache_len for i in range(new_len)]
    torch.testing.assert_close(new_k[:, :, slots, :], k_new)
    torch.testing.assert_close(new_v[:, :, :, slots], v_new)

  def test_defaults_match_historical_layout(self):
    cache_k = torch.zeros((1, 1, 8, 4))
    cache_v = torch.zeros((1, 1, 4, 8))
    k_new = torch.randn((1, 1, 2, 4))
    v_new_td = torch.randn((1, 1, 2, 4))  # [T, D], the historical V layout.
    param = torch.zeros((1, 1, 1, 7), dtype=torch.int32)
    param[..., 3] = 2
    args = dict(
        runtime_param_tensor=param,
        cache_k=cache_k,
        cache_v=cache_v,
        indices_k=torch.zeros(4, dtype=torch.int32),
        indices_v=torch.zeros(4, dtype=torch.int32),
        kv_heads=1,
        kv_batch_size=1,
        cache_len=8,
        head_size=4,
        is_ring_buffer=True,
    )
    _, new_v = cache_update.cache_update(
        key_proj=k_new, value_proj=v_new_td, **args
    )
    torch.testing.assert_close(new_v[:, :, :, 0:2], v_new_td.transpose(-2, -1))

  @parameterized.named_parameters(
      ("default", None, None),
      ("v_cache_layout", 3, {"k_update_ts_idx": 2, "v_update_ts_idx": 3}),
  )
  def test_export_emits_layout_attributes(self, v_update_ts_idx, expected):
    """Layout attributes appear only for non-historical update layouts."""
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
            v_update_ts_idx=v_update_ts_idx,
        )

    value_shape = (1, 2, 3, 16) if v_update_ts_idx != 3 else (1, 2, 16, 3)
    exported = torch.export.export(
        UpdateModule(),
        (
            torch.randn((1, 2, 3, 16)),
            torch.randn(value_shape),
            torch.zeros((1, 1, 1, 7), dtype=torch.int32),
            torch.randn((1, 2, 8, 16)),
            torch.randn((1, 2, 16, 8)),
            torch.zeros(4, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
        ),
    )
    marks = "".join(
        str(n.args) + str(n.kwargs)
        for n in exported.graph.nodes
        if n.op == "call_function"
        and "mark_tensor" in str(n.target)
        and "odml.cache_update" in (str(n.args) + str(n.kwargs))
    )
    self.assertNotEmpty(marks)
    if expected is None:
      # Existing exports keep their exact attributes.
      self.assertNotIn("ts_idx", marks)
      return
    expected = {"k_cache_ts_idx": 2, "v_cache_ts_idx": 3, **expected}
    for name, value in expected.items():
      self.assertIn(f"('{name}', {value})", marks)

  @parameterized.product(
      offset=[0, 3, 7, 8, 13],
      v_update_ts_idx=[2, 3],
      write_single_token_in_place=[False, True],
  )
  def test_ring_buffer_decode_matches_one_hot(
      self, offset, v_update_ts_idx, write_single_token_in_place
  ):
    torch.manual_seed(offset)
    cache_k = torch.randn((1, 2, 8, 16))
    cache_v = torch.randn((1, 2, 16, 8))
    key_proj = torch.randn((1, 2, 1, 16))
    value_proj = torch.randn(
        (1, 2, 1, 16) if v_update_ts_idx == 2 else (1, 2, 16, 1)
    )
    param_tensor = torch.zeros((1, 1, 1, 7), dtype=torch.int32)
    param_tensor[..., 0] = offset
    param_tensor[..., 3] = 1

    new_k, new_v = cache_update.cache_update(
        key_proj=key_proj,
        value_proj=value_proj,
        runtime_param_tensor=param_tensor,
        cache_k=cache_k,
        cache_v=cache_v,
        indices_k=torch.zeros(4, dtype=torch.int32),
        indices_v=torch.zeros(4, dtype=torch.int32),
        kv_heads=2,
        kv_batch_size=1,
        cache_len=8,
        head_size=16,
        is_ring_buffer=True,
        v_update_ts_idx=v_update_ts_idx,
        write_single_token_in_place=write_single_token_in_place,
    )

    positions = torch.tensor([offset], dtype=torch.int32)
    valid = torch.tensor([True])
    v_update = value_proj if v_update_ts_idx == 3 else value_proj.transpose(
        -2, -1
    )
    ref_k = cache_update.update_kv_cache_with_sliding(
        cache_k, key_proj, positions, valid, ts_idx=2
    )
    ref_v = cache_update.update_kv_cache_with_sliding(
        cache_v, v_update, positions, valid, ts_idx=3
    )
    torch.testing.assert_close(new_k, ref_k, rtol=0, atol=0)
    torch.testing.assert_close(new_v, ref_v, rtol=0, atol=0)


if __name__ == "__main__":
  googletest.main()
