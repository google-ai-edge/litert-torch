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
"""Custom Split Cache attention implementation for ODML."""

import math
from typing import Optional

from litert_torch.generative.custom_ops import bmm_4d as bmm_lib
from litert_torch.generative.export_hf.core.split_cache import cache as kv_cache_lib
import torch
import torch.nn.functional as F
import transformers


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key_cache: kv_cache_lib.KeyCacheEntry,
    value_cache: kv_cache_lib.ValueCacheEntry,
    head_size: int,
    k_ts_idx: int,
    v_ts_idx: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    softcap: Optional[float] = None,
    kv_block_size: Optional[int] = None,
):
  """Scaled dot product attention with transposed key and value.

  Args:
    query: Query tensor, 1(bk)(gt)h.
    key_cache: A tuple of Key tensor. 1(bk)hs
    value_cache: A tuple of Value tensor. 1(bk)sh
    head_size (int): head dimension.
    k_ts_idx (int): the timestamp index of the key tensor.
    v_ts_idx (int): the timestamp index of the value tensor.
    mask (torch.Tensor): the optional mask tensor.
    scale (float): the optional scale factor.
    softcap (float): the optional softcap for the logits.
    kv_block_size (int): if set and smaller than the cache length, attend over
      the cache in blocks of this many slots (see `_blocked_attention`); it
      must divide the cache length.

  Returns:
    The output tensor of scaled_dot_product_attention_transposed.
  """
  # The KV cache may be stored in a narrower dtype than the graph (e.g. float16
  # with `experimental_use_fp16`); attention is computed in the query dtype.
  key_past, key = key_cache
  value_past, value = value_cache
  assert key is not None and value is not None
  compute_dtype = query.dtype
  key_past = key_past.to(compute_dtype)
  key = key.to(compute_dtype)
  value_past = value_past.to(compute_dtype)
  value = value.to(compute_dtype)

  if scale is None:
    scale = 1.0 / math.sqrt(head_size)

  query = query * scale

  assert mask is not None, "Mask should not be None!"
  t = mask.shape[2]

  if k_ts_idx == 2:
    bmm_fn = bmm_lib.bmm_4d
  else:
    assert k_ts_idx == 3, "k_ts_idx must be 2 or 3."
    bmm_fn = lambda x, y: torch.einsum("abth,abhs->abts", x, y)
  if v_ts_idx == 3:
    bmm_v_fn = bmm_lib.bmm_4d
  else:
    assert v_ts_idx == 2, "v_ts_idx must be 2 or 3."
    bmm_v_fn = lambda x, y: torch.einsum("abts,absh->abth", x, y)

  cache_len = key_past.shape[k_ts_idx]
  if kv_block_size is not None and cache_len > kv_block_size:
    if cache_len % kv_block_size:
      raise ValueError(
          f"attention_kv_block_size={kv_block_size} doesn't divide the cache"
          f" length {cache_len}."
      )
    return _blocked_attention(
        query,
        key_past,
        key,
        value_past,
        value,
        mask,
        t,
        bmm_fn,
        bmm_v_fn,
        k_ts_idx,
        v_ts_idx,
        softcap,
        kv_block_size,
    )

  logits0 = bmm_fn(query, key_past)
  logits1 = bmm_fn(query, key)
  logits = torch.cat([logits0, logits1], dim=-1)

  _, _, gt, _ = logits.shape
  g = gt // t
  if softcap is not None:
    logits = torch.tanh(logits / softcap)
    logits = logits * softcap

  if g != 1:
    mask_to_bc = []
    for _ in range(g):
      mask_to_bc.append(mask)
    mask = torch.cat(mask_to_bc, dim=-2)  # 1, 1, gt, s

  padded_logits = logits + mask
  probs = F.softmax(padded_logits, dim=-1).type_as(key)
  probs0, probs1 = probs[..., :-t], probs[..., -t:]

  encoded0 = bmm_v_fn(probs0, value_past)
  encoded1 = bmm_v_fn(probs1, value)
  encoded = encoded0 + encoded1

  return encoded  # 1, bk, gt, h


# Floor applied to FP16 / BF16 masks before blocked attention. Mask producers
# use finite values for FP32 masks (the auxiliary mask model: -100; LiteRT-LM's
# built-in mask: -1e9) but -inf for FP16 / BF16 masks, and a fully masked block
# with -inf logits would give exp(-inf - (-inf)) = NaN. -3e4 stays finite in
# FP16 after adding the logits.
_LOW_PRECISION_MASK_FLOOR = -3e4


def _blocked_attention(
    query,
    key_past,
    key,
    value_past,
    value,
    mask,
    t,
    bmm_k_fn,
    bmm_v_fn,
    k_ts_idx,
    v_ts_idx,
    softcap,
    block_size,
):
  """Attention over the cache in blocks of `block_size` slots.

  Each block (the cache blocks, then the current chunk's own keys) is
  normalized with only its own row max; the blocks are combined at the end on
  [.., rows, 1] tensors. Nothing feeds from one block into another block's big
  [rows, block] tensors (unlike a sequential online softmax), so each block's
  computation is independent:

    s_j = q k_j^T (+ softcap) + mask_j
    m_j = rowmax(s_j);  p_j = exp(s_j - m_j);  l_j = rowsum(p_j)
    o_j = (p_j v_j) / l_j
    m = max_j m_j;  w_j = exp(m_j - m) * l_j / B
    out = sum_j w_j o_j / sum_j w_j

  This equals softmax attention over all keys. l_j >= 1 (the row max
  contributes exp(0) = 1), so the division needs no guard. The common factor
  1 / B (B = block size) cancels in out; it bounds sum_j w_j by the number of
  blocks instead of the number of keys, which would overflow FP16 (max 65504)
  for long caches. Fully masked blocks (unfilled cache slots) get w_j -> 0 as
  long as the mask is finite; FP16 / BF16 masks, which may hold -inf, are
  floored first. The row max m_j itself is not clamped, because some NPU
  compilers (e.g. Core AI's compiler for the Apple Neural Engine) fail on
  MAXIMUM(REDUCE_MAX(s), const).

  FP16 bound: o_j is normalized after the matmul (one division on the small
  [rows, h] tensor instead of one on the [rows, B] tensor), so the
  intermediate p_j v_j is bounded by l_j * max|v| <= B * max|v| rather than
  by max|v| as in plain attention. It can only overflow FP16 if
  B * max|v| > 65504 (max|v| > 16 for B = 4096) and a row attends nearly
  uniformly over a block whose values agree in sign and magnitude. Gemma 4's
  v_norm (RMSNorm without scale) keeps |v| <= sqrt(head_dim) and typically
  ~1.

  Args:
    query: [1, bk, gt, h] queries.
    key_past: The cache keys (time axis k_ts_idx).
    key: The current chunk's keys (time axis k_ts_idx).
    value_past: The cache values (time axis v_ts_idx).
    value: The current chunk's values (time axis v_ts_idx).
    mask: [1, 1, t, cache_len + t] additive mask.
    t: Number of new tokens (the chunk length).
    bmm_k_fn: Computes q k^T for the key layout.
    bmm_v_fn: Computes p v for the value layout.
    k_ts_idx: Time axis of the keys (2 or 3).
    v_ts_idx: Time axis of the values (2 or 3).
    softcap: Optional logit softcap.
    block_size: Number of cache slots per block; divides the cache length.

  Returns:
    [1, bk, gt, h] attention output, in the dtype of `value`.
  """
  _, _, gt, _ = query.shape
  g = gt // t
  if mask.dtype in (torch.float16, torch.bfloat16):
    mask = torch.maximum(
        mask, torch.full((), _LOW_PRECISION_MASK_FLOOR, dtype=mask.dtype)
    )
  if g != 1:
    mask = torch.cat([mask] * g, dim=-2)  # 1, 1, gt, s + t
  cache_len = key_past.shape[k_ts_idx]
  blocks = []
  for start in range(0, cache_len, block_size):
    blocks.append((
        key_past.narrow(k_ts_idx, start, block_size),
        value_past.narrow(v_ts_idx, start, block_size),
        mask[..., start : start + block_size],
    ))
  blocks.append((key, value, mask[..., cache_len:]))

  maxes, sums, outs = [], [], []
  for k_j, v_j, mask_j in blocks:
    s_j = bmm_k_fn(query, k_j)
    if softcap is not None:
      s_j = torch.tanh(s_j / softcap) * softcap
    s_j = s_j + mask_j
    m_j = torch.amax(s_j, dim=-1, keepdim=True)
    p_j = torch.exp(s_j - m_j)
    l_j = torch.sum(p_j, dim=-1, keepdim=True)
    o_j = bmm_v_fn(p_j.type_as(v_j), v_j) / l_j
    maxes.append(m_j)
    sums.append(l_j)
    outs.append(o_j)

  m = maxes[0]
  for m_j in maxes[1:]:
    m = torch.maximum(m, m_j)
  inv_block = 1.0 / block_size
  weights = [
      torch.exp(m_j - m) * (l_j * inv_block) for m_j, l_j in zip(maxes, sums)
  ]
  num = weights[0] * outs[0]
  den = weights[0]
  for w_j, o_j in zip(weights[1:], outs[1:]):
    num = num + w_j * o_j
    den = den + w_j
  return (num / den).type_as(value)  # 1, bk, gt, h


def split_cache_attention(
    module: torch.nn.Module,  # required arg
    query: torch.Tensor,  # required arg
    key: kv_cache_lib.KeyCacheEntry,  # required arg
    value: kv_cache_lib.ValueCacheEntry,  # required arg
    attention_mask: Optional[torch.Tensor],  # required arg
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,  # You need to accept **kwargs as models will pass other args
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
  """ODML transposed attention implementation for NPU."""

  b, n, seq_len, h = query.shape
  if hasattr(module, "num_key_value_groups"):
    g = module.num_key_value_groups
  else:
    g = 1
  num_query_groups = n // g
  k_ts_idx: int | None = kwargs.get("k_ts_idx", None)
  v_ts_idx: int | None = kwargs.get("v_ts_idx", None)
  kv_block_size: int | None = kwargs.get("attention_kv_block_size", None)
  if k_ts_idx is None or v_ts_idx is None:
    raise ValueError(
        "Timestamp indices not passed to attention module. The model is not"
        " passing the kwargs correctly."
    )
  if kv_block_size is not None:
    # Blocking applies to the layers whose physical cache length equals the
    # signature's (bucket's) cache_length (global layers, and sliding layers
    # without a ring buffer); other caches (ring buffers) use plain attention.
    full_cache_length = kwargs.get("attention_kv_block_cache_length", None)
    if full_cache_length is None:
      raise ValueError(
          "attention_kv_block_size is set but attention_kv_block_cache_length"
          " is not; pass both (see ExportableModuleBase.attention_kwargs)."
      )
    if key[0].shape[k_ts_idx] != full_cache_length:
      kv_block_size = None
  # bnth -> b(kg)th -> 1(bk)(gt)h
  query = query.reshape(1, b * num_query_groups, g * seq_len, h)

  sdpa_out = _scaled_dot_product_attention(
      query,
      key,
      value,
      h,
      mask=attention_mask,
      scale=scaling,
      softcap=softcap,
      k_ts_idx=k_ts_idx,
      v_ts_idx=v_ts_idx,
      kv_block_size=kv_block_size,
  )  # 1, bk, gt, h
  sdpa_out = sdpa_out.reshape(b, -1, seq_len, h).permute(0, 2, 1, 3)
  return sdpa_out, None


transformers.AttentionInterface.register(
    "lrt_split_cache_attention", split_cache_attention
)
