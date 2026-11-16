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
"""Tests for split cache attention (plain vs blocked)."""

from unittest import mock

from absl.testing import parameterized

# Imported first for its transformers / torchao version workaround.
from litert_torch.generative import export_hf  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core.split_cache import attention
import torch
from torch.fx.experimental import proxy_tensor

from absl.testing import absltest as googletest


def _inputs(
    cache_len,
    t,
    g,
    kv_heads,
    h,
    fill,
    mask_value,
    dtype,
    seed=0,
    k_ts_idx=2,
    v_ts_idx=3,
):
  """Random q / caches / chunk K, V and a causal split mask.

  `fill` cache slots hold past tokens; the rest of the static cache is masked
  out (like an unfilled bucket). The chunk's own keys are causal.

  Args:
    cache_len: Number of cache slots.
    t: Number of new tokens (the chunk length).
    g: Number of query heads per KV head.
    kv_heads: Number of KV heads.
    h: Head dimension.
    fill: Number of filled (unmasked) cache slots.
    mask_value: Value of masked mask entries.
    dtype: Dtype of all returned tensors.
    seed: Random seed.
    k_ts_idx: Time axis of the keys (2: [.., s, h], 3: [.., h, s]).
    v_ts_idx: Time axis of the values (2: [.., s, h], 3: [.., h, s]).

  Returns:
    (query, (key_past, key), (value_past, value), mask).
  """
  gen = torch.Generator().manual_seed(seed)
  r = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32).to(dtype)
  query = r(1, kv_heads, g * t, h)

  def time_major(length, ts_idx):
    return (
        r(1, kv_heads, length, h) if ts_idx == 2 else r(1, kv_heads, h, length)
    )

  key_past = time_major(cache_len, k_ts_idx)
  value_past = time_major(cache_len, v_ts_idx)
  key = time_major(t, k_ts_idx)
  value = time_major(t, v_ts_idx)
  allowed = torch.zeros(t, cache_len + t, dtype=torch.bool)
  allowed[:, :fill] = True
  allowed[:, cache_len:] = torch.tril(torch.ones(t, t, dtype=torch.bool))
  mask = torch.where(allowed, 0.0, mask_value).reshape(1, 1, t, cache_len + t)
  mask = mask.to(dtype)
  return query, (key_past, key), (value_past, value), mask


class AttentionTest(parameterized.TestCase):

  @parameterized.product(
      cache_len_block=[(32, 8), (64, 16), (64, 64), (128, 32)],
      t=[1, 5, 16],
      g=[1, 2],
      fill=[0, 7, 30],
      softcap=[None, 3.0],
      # The finite values used for FP32 masks: the auxiliary mask model (-100)
      # and LiteRT-LM's built-in mask (-1e9).
      mask_value=[-100.0, -1e9],
      # (k_ts_idx, v_ts_idx): all four K / V layouts.
      layout=[(2, 3), (2, 2), (3, 3), (3, 2)],
  )
  def test_blocked_equals_plain(
      self, cache_len_block, t, g, fill, softcap, mask_value, layout
  ):
    cache_len, block = cache_len_block
    k_ts_idx, v_ts_idx = layout
    fill = min(fill, cache_len)
    q, k, v, mask = _inputs(
        cache_len,
        t,
        g,
        2,
        8,
        fill,
        mask_value,
        torch.float32,
        k_ts_idx=k_ts_idx,
        v_ts_idx=v_ts_idx,
    )
    kwargs = dict(
        head_size=8,
        k_ts_idx=k_ts_idx,
        v_ts_idx=v_ts_idx,
        mask=mask,
        softcap=softcap,
    )
    plain = attention._scaled_dot_product_attention(q, k, v, **kwargs)
    blocked = attention._scaled_dot_product_attention(
        q, k, v, kv_block_size=block, **kwargs
    )
    self.assertFalse(torch.isnan(blocked).any())
    torch.testing.assert_close(blocked, plain, rtol=1e-5, atol=1e-5)

  @parameterized.parameters(0, 7, 20)
  def test_fp16_inf_mask_is_nan_free(self, fill):
    # LiteRT-LM fills FP16 masks with -inf; fully masked blocks must not give
    # NaN (the mask is floored before the blocks).
    q, k, v, mask = _inputs(64, 8, 2, 2, 8, fill, float("-inf"), torch.float16)
    kwargs = dict(head_size=8, k_ts_idx=2, v_ts_idx=3, mask=mask)
    ref_kwargs = dict(kwargs, mask=mask.float())
    ref = attention._scaled_dot_product_attention(
        q.float(),
        tuple(x.float() for x in k),
        tuple(x.float() for x in v),
        **ref_kwargs,
    )
    blocked = attention._scaled_dot_product_attention(
        q, k, v, kv_block_size=16, **kwargs
    )
    self.assertFalse(torch.isnan(blocked).any())
    torch.testing.assert_close(blocked.float(), ref, rtol=2e-2, atol=2e-3)

  def test_no_clamp_on_row_max(self):
    # Some NPU compilers reject MAXIMUM(REDUCE_MAX(s), const): the row max must
    # feed exp / the combine directly. Check via the traced FX graph.
    q, k, v, mask = _inputs(64, 8, 2, 2, 8, 20, -100.0, torch.float32)

    def f(q, kp, kc, vp, vc, m):
      return attention._scaled_dot_product_attention(
          q,
          (kp, kc),
          (vp, vc),
          head_size=8,
          k_ts_idx=2,
          v_ts_idx=3,
          mask=m,
          kv_block_size=16,
      )

    gm = proxy_tensor.make_fx(f)(q, *k, *v, mask)
    targets = [n.target for n in gm.graph.nodes]
    self.assertIn(torch.ops.aten.amax.default, targets)

    # MAXIMUM of row maxes across blocks (the combine) is fine; MAXIMUM of a
    # row max with a constant is not.
    def is_const(a):
      return (
          not isinstance(a, torch.fx.Node)
          or a.op == "get_attr"
          or (
              a.target in (
                  torch.ops.aten.full.default,
                  torch.ops.aten.scalar_tensor.default,
                  torch.ops.aten.lift_fresh_copy.default,
              )
          )
      )

    for node in gm.graph.nodes:
      if node.target == torch.ops.aten.maximum.default:
        a, b = node.args[:2]
        for x, y in ((a, b), (b, a)):
          if (
              isinstance(x, torch.fx.Node)
              and x.target == torch.ops.aten.amax.default
          ):
            self.assertFalse(is_const(y), str(node))

  def test_blocked_fp16_close_to_fp32_plain(self):
    q, k, v, mask = _inputs(4096, 64, 2, 2, 64, 3000, -100.0, torch.float16)
    kwargs = dict(head_size=64, k_ts_idx=2, v_ts_idx=3, mask=mask)
    ref = attention._scaled_dot_product_attention(
        q.float(),
        tuple(x.float() for x in k),
        tuple(x.float() for x in v),
        **dict(kwargs, mask=mask.float()),
    )
    blocked = attention._scaled_dot_product_attention(
        q, k, v, kv_block_size=1024, **kwargs
    )
    self.assertEqual(blocked.dtype, torch.float16)
    torch.testing.assert_close(blocked.float(), ref, rtol=2e-2, atol=2e-3)

  def test_non_dividing_block_raises(self):
    q, k, v, mask = _inputs(48, 4, 1, 1, 8, 20, -100.0, torch.float32)
    kwargs = dict(head_size=8, k_ts_idx=2, v_ts_idx=3, mask=mask)
    with self.assertRaisesRegex(ValueError, "doesn't divide"):
      attention._scaled_dot_product_attention(
          q, k, v, kv_block_size=32, **kwargs
      )

  def test_block_not_smaller_than_cache_is_plain(self):
    q, k, v, mask = _inputs(32, 4, 1, 1, 8, 20, -100.0, torch.float32)
    kwargs = dict(head_size=8, k_ts_idx=2, v_ts_idx=3, mask=mask)
    plain = attention._scaled_dot_product_attention(q, k, v, **kwargs)
    same = attention._scaled_dot_product_attention(
        q, k, v, kv_block_size=32, **kwargs
    )
    torch.testing.assert_close(same, plain, rtol=0, atol=0)

  @parameterized.parameters(0, 7, 20)
  def test_bf16_inf_mask_is_nan_free(self, fill):
    q, k, v, mask = _inputs(64, 8, 2, 2, 8, fill, float("-inf"), torch.bfloat16)
    kwargs = dict(head_size=8, k_ts_idx=2, v_ts_idx=3, mask=mask)
    ref = attention._scaled_dot_product_attention(
        q.float(),
        tuple(x.float() for x in k),
        tuple(x.float() for x in v),
        **dict(kwargs, mask=mask.float()),
    )
    blocked = attention._scaled_dot_product_attention(
        q, k, v, kv_block_size=16, **kwargs
    )
    self.assertEqual(blocked.dtype, torch.bfloat16)
    self.assertFalse(torch.isnan(blocked).any())
    torch.testing.assert_close(blocked.float(), ref, rtol=5e-2, atol=2e-2)

  def test_fully_masked_row_is_finite(self):
    # Row 0 is masked in every block, including the chunk's own keys.
    q, k, v, mask = _inputs(64, 8, 2, 2, 8, 20, -100.0, torch.float32)
    mask[..., 0, :] = -100.0
    kwargs = dict(head_size=8, k_ts_idx=2, v_ts_idx=3, mask=mask)
    plain = attention._scaled_dot_product_attention(q, k, v, **kwargs)
    blocked = attention._scaled_dot_product_attention(
        q, k, v, kv_block_size=16, **kwargs
    )
    self.assertTrue(torch.isfinite(blocked).all())
    torch.testing.assert_close(blocked, plain, rtol=1e-5, atol=1e-5)

  def test_fp16_long_cache_does_not_overflow(self):
    # 65,536 unmasked keys with equal scores: every block has l_j = 4096, so
    # the unscaled sum of block weights would be 65,536 > 65,504 (FP16 max).
    cache_len, block, h = 65536, 4096, 8
    q, k, v, mask = _inputs(
        cache_len, 1, 1, 1, h, cache_len, -100.0, torch.float16
    )
    key_past = torch.ones(1, 1, cache_len, h, dtype=torch.float16)
    k = (key_past, torch.ones(1, 1, 1, h, dtype=torch.float16))
    q = torch.ones_like(q)
    kwargs = dict(head_size=h, k_ts_idx=2, v_ts_idx=3, mask=mask)
    blocked = attention._scaled_dot_product_attention(
        q, k, v, kv_block_size=block, **kwargs
    )
    self.assertTrue(torch.isfinite(blocked).all())
    ref = attention._scaled_dot_product_attention(
        q.float(),
        tuple(x.float() for x in k),
        tuple(x.float() for x in v),
        **dict(kwargs, mask=mask.float()),
    )
    torch.testing.assert_close(blocked.float(), ref, rtol=2e-2, atol=2e-3)

  @parameterized.named_parameters(
      ("full_cache_length", 64, 1),
      ("ring_buffer", 32, 0),
  )
  def test_blocks_only_layers_with_full_cache_length(self, cache_len, calls):
    # split_cache_attention blocks a layer iff its cache has the full
    # cache_length (here 64); smaller caches (ring buffers) stay plain.
    q, k, v, mask = _inputs(cache_len, 4, 2, 1, 8, 10, -100.0, torch.float32)
    query = q.reshape(1, 2, 4, 8)  # b, n = kv_heads * g, seq, h
    module = torch.nn.Module()
    module.num_key_value_groups = 2
    with mock.patch.object(
        attention, "_blocked_attention", wraps=attention._blocked_attention
    ) as spy:
      attention.split_cache_attention(
          module,
          query,
          k,
          v,
          mask,
          k_ts_idx=2,
          v_ts_idx=3,
          attention_kv_block_size=16,
          attention_kv_block_cache_length=64,
      )
    self.assertEqual(spy.call_count, calls)

  def test_missing_cache_length_raises(self):
    # Without the cache length no layer could be blocked; fail loudly instead
    # of silently using plain attention everywhere.
    q, k, v, mask = _inputs(64, 4, 2, 1, 8, 10, -100.0, torch.float32)
    module = torch.nn.Module()
    module.num_key_value_groups = 2
    with self.assertRaisesRegex(ValueError, "attention_kv_block_cache_length"):
      attention.split_cache_attention(
          module,
          q.reshape(1, 2, 4, 8),
          k,
          v,
          mask,
          k_ts_idx=2,
          v_ts_idx=3,
          attention_kv_block_size=16,
      )


if __name__ == "__main__":
  googletest.main()
