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
"""Optimized GPU cache update operation for attention."""

from litert_torch.backend import composite
import litert_torch.generative.custom_ops.dynamic_update_slice as tfl_dus
import torch


def int32_one_hot(
    indices: torch.Tensor, num_classes: int, dtype: torch.dtype
) -> torch.Tensor:
  """A LiteRT-friendly one-hot encoder that stays entirely in int32."""
  classes = torch.arange(num_classes, dtype=torch.int32, device=indices.device)
  return (indices.unsqueeze(-1) == classes).to(dtype)


def update_kv_cache_with_sliding(
    cache: torch.Tensor,
    update: torch.Tensor,
    positions: torch.Tensor,
    valid_mask: torch.Tensor,
    ts_idx: int = 2,
) -> torch.Tensor:
  """Updates the ring buffer KV cache.

  Args:
      cache: [B, H, S, D] (if ts_idx=2) or [B, H, D, S] (if ts_idx=3)
      update: [B, H, T, D] (if ts_idx=2) or [B, H, D, T] (if ts_idx=3)
      positions: [T] - Global token positions
      valid_mask: [T] - 1 for valid tokens, 0 for padding
      ts_idx: 2 or 3, indicating the time sequence dimension

  Returns:
      Updated cache tensor.
  """
  cache_size = cache.size(ts_idx)
  seq_len = positions.size(0)

  # 1. Calculate modulo indices
  indices = positions % cache_size

  # 2. One-Hot routing matrix: [T, S]
  # Built in fp32 because the mask ops below (SUM, GREATER) have no fp16
  # kernels in LiteRT. This tensor is only [T, S]; the cache itself and the
  # cache-sized routed update stay in the cache dtype.
  one_hot = int32_one_hot(indices, num_classes=cache_size, dtype=torch.float32)

  # 3. Apply the valid_mask (Zero out padding rows)
  valid_mask_float = valid_mask.to(torch.float32).unsqueeze(1)
  one_hot = one_hot * valid_mask_float
  routing = one_hot.to(cache.dtype)  # No-op for fp32 caches.

  # 4. Project and Route based on the sequence dimension
  if ts_idx == 2:
    # cache: [B, H, S, D] | update: [B, H, T, D]
    # Matmul: [1, 1, S, T] @ [B, H, T, D] -> [B, H, S, D]
    routing_matrix = routing.transpose(0, 1).view(1, 1, cache_size, seq_len)
    update_expanded = torch.matmul(routing_matrix, update)

    # Blend mask broadcasts across the D dimension
    update_mask = (one_hot.sum(dim=0) > 0).view(1, 1, cache_size, 1)

  elif ts_idx == 3:
    # cache: [B, H, D, S] | update: [B, H, D, T]
    # Matmul: [B, H, D, T] @ [1, 1, T, S] -> [B, H, D, S]
    routing_matrix = routing.view(1, 1, seq_len, cache_size)
    update_expanded = torch.matmul(update, routing_matrix)

    # Blend mask broadcasts across the D dimension
    update_mask = (one_hot.sum(dim=0) > 0).view(1, 1, 1, cache_size)

  else:
    raise ValueError("ts_idx must be 2 or 3")

  # 5. BLEND: Combine projected updates with the old cache
  updated_cache = torch.where(update_mask, update_expanded, cache)

  return updated_cache


def _ring_write_one_token(
    cache: torch.Tensor,
    update: torch.Tensor,
    offset: torch.Tensor,
    ts_idx: int,
) -> torch.Tensor:
  """Writes a single token into slot `offset % S` of a ring buffer.

  Equivalent to `update_kv_cache_with_sliding` for one valid token, but touches
  one slot instead of the whole cache: the one-hot routing matmul and the
  select both produce a full [.., S, ..] tensor, which dominates CPU decode
  time. Like the linear-cache path, the write is unconditional; it must not
  read the old cache either, since such reads depend only on graph inputs and
  get hoisted into one costly delegate partition.

  Args:
    cache: [B, H, S, D] (if ts_idx=2) or [B, H, D, S] (if ts_idx=3).
    update: The token, size 1 along `ts_idx`.
    offset: Scalar int32 write position (reduced modulo S here).
    ts_idx: 2 or 3, the time axis.

  Returns:
    Updated cache tensor.
  """
  slot = offset % cache.size(ts_idx)
  zero = torch.zeros((), dtype=torch.int32, device=cache.device)
  start = [zero] * cache.dim()
  start[ts_idx] = slot
  return tfl_dus.dynamic_update_slice(cache, update, start)


def _to_cache_layout(
    update: torch.Tensor, update_ts_idx: int, cache_ts_idx: int
) -> torch.Tensor:
  """Transposes a [B, H, *, *] update so its time axis matches the cache's."""
  if update_ts_idx == cache_ts_idx:
    return update
  return update.transpose(-2, -1)


def _default_v_update_ts_idx(v_cache_ts_idx: int) -> int:
  """Returns the V update layout historically paired with a V cache layout.

  The V update has always been passed as the transpose of the V cache layout,
  e.g. [B, H, T, D] for a [B, H, D, S] cache.

  Args:
    v_cache_ts_idx: The time axis of the V cache, 2 or 3.
  """
  return 5 - v_cache_ts_idx


def cache_update(
    key_proj: torch.Tensor,
    value_proj: torch.Tensor,
    runtime_param_tensor: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    indices_k: torch.Tensor,
    indices_v: torch.Tensor,
    kv_heads: int,
    kv_batch_size: int,
    cache_len: int,
    head_size: int,
    is_ring_buffer: bool = False,
    k_ts_idx: int = 2,
    v_ts_idx: int = 3,
    k_update_ts_idx: int | None = None,
    v_update_ts_idx: int | None = None,
    write_single_token_in_place: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Cache update composite op for float cache.

  Args:
    key_proj: New keys; time axis `k_update_ts_idx`.
    value_proj: New values; time axis `v_update_ts_idx`.
    runtime_param_tensor: Runtime param tensor ([0] = write offset, [3] = number
      of valid new tokens for ring buffers).
    cache_k: Key cache; time axis `k_ts_idx`.
    cache_v: Value cache; time axis `v_ts_idx`.
    indices_k: dynamic_update_slice start indices for the key cache.
    indices_v: dynamic_update_slice start indices for the value cache.
    kv_heads: Number of KV heads.
    kv_batch_size: KV cache batch size.
    cache_len: Cache length (ring buffer size for ring buffers).
    head_size: Head dimension.
    is_ring_buffer: Whether the caches are ring buffers.
    k_ts_idx: Time axis of the key cache, 2 or 3.
    v_ts_idx: Time axis of the value cache, 2 or 3.
    k_update_ts_idx: Time axis of `key_proj`; defaults to `k_ts_idx`.
    v_update_ts_idx: Time axis of `value_proj`; defaults to the transpose of
      the value cache layout (the op's historical contract).
    write_single_token_in_place: For a ring buffer update of one token, write
      only its slot instead of rebuilding the whole cache through the one-hot
      routing (the decomposition backends fall back to). The op and its
      attributes are unchanged.

  Returns:
    The updated (key cache, value cache).
  """
  if k_update_ts_idx is None:
    k_update_ts_idx = k_ts_idx
  if v_update_ts_idx is None:
    v_update_ts_idx = _default_v_update_ts_idx(v_ts_idx)
  for ts_idx in (k_ts_idx, v_ts_idx, k_update_ts_idx, v_update_ts_idx):
    assert ts_idx in (2, 3), f"time axis must be 2 or 3, got {ts_idx}"

  attrs = {
      "kv_cache_batch_size": kv_heads * kv_batch_size,
      "cache_size": cache_len,
      "head_size": head_size,
      "is_ring_buffer": is_ring_buffer,
  }
  if (k_update_ts_idx, v_update_ts_idx) != (
      k_ts_idx,
      _default_v_update_ts_idx(v_ts_idx),
  ):
    # Explicit layouts, only for updates that deviate from the historical
    # contract: backends that predate these attributes assume
    # k_cache_ts_idx=2, v_cache_ts_idx=3, k_update_ts_idx=2 and
    # v_update_ts_idx=2, and existing exports keep their exact attributes.
    attrs.update(
        k_cache_ts_idx=k_ts_idx,
        v_cache_ts_idx=v_ts_idx,
        k_update_ts_idx=k_update_ts_idx,
        v_update_ts_idx=v_update_ts_idx,
    )
  builder = composite.StableHLOCompositeBuilder(
      name="odml.cache_update", attr=attrs
  )
  (
      key_proj,
      value_proj,
      runtime_param_tensor,
      cache_k,
      cache_v,
      indices_k,
      indices_v,
  ) = builder.mark_inputs(
      key_proj,
      value_proj,
      runtime_param_tensor,
      cache_k,
      cache_v,
      indices_k,
      indices_v,
  )

  dummy = (
      (runtime_param_tensor.sum() * 0)
      + (indices_k.sum() * 0)
      + (indices_v.sum() * 0)
  )
  # The updates are built where they are used, K before V, so that existing
  # exports trace the same op order (and hence the same tensor numbering).
  def make_k_update():
    return _to_cache_layout(key_proj, k_update_ts_idx, k_ts_idx) + dummy.to(
        key_proj.dtype
    )

  def make_v_update():
    return _to_cache_layout(
        value_proj, v_update_ts_idx, v_ts_idx
    ) + dummy.to(value_proj.dtype)

  if is_ring_buffer:
    k_update = make_k_update()
    v_update = make_v_update()
    if (
        write_single_token_in_place
        and k_update.size(k_ts_idx) == 1
        and v_update.size(v_ts_idx) == 1
    ):
      # Decode: write the single token in place rather than rebuilding the
      # whole ring buffer through the one-hot routing below. Decode always
      # carries one valid token (param[3] == 1).
      offset = runtime_param_tensor.reshape(-1)[0]
      out_k = _ring_write_one_token(cache_k, k_update, offset, ts_idx=k_ts_idx)
      out_v = _ring_write_one_token(cache_v, v_update, offset, ts_idx=v_ts_idx)
    else:
      t_k = k_update.size(k_ts_idx)
      offset = runtime_param_tensor.reshape(-1)[0]
      positions_k = offset + torch.arange(
          t_k, dtype=torch.int32, device=k_update.device
      )
      update_len = runtime_param_tensor.reshape(-1)[3]
      valid_mask_k = (
          torch.arange(t_k, dtype=torch.int32, device=k_update.device)
          < update_len
      )
      out_k = update_kv_cache_with_sliding(
          cache_k, k_update, positions_k, valid_mask_k, ts_idx=k_ts_idx
      )

      t_v = v_update.size(v_ts_idx)
      positions_v = offset + torch.arange(
          t_v, dtype=torch.int32, device=v_update.device
      )
      valid_mask_v = (
          torch.arange(t_v, dtype=torch.int32, device=v_update.device)
          < update_len
      )
      out_v = update_kv_cache_with_sliding(
          cache_v, v_update, positions_v, valid_mask_v, ts_idx=v_ts_idx
      )
  else:
    out_k = tfl_dus.dynamic_update_slice(
        cache_k,
        make_k_update(),
        [x for x in indices_k],
    )
    out_v = tfl_dus.dynamic_update_slice(
        cache_v,
        make_v_update(),
        [x for x in indices_v],
    )
  return builder.mark_outputs(out_k, out_v)
