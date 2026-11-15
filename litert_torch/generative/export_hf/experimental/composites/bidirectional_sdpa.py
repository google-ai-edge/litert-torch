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
"""Non-autoregressive bidirectional SDPA as an `odml.scaled_dot_product_attention` composite."""

from litert_torch.backend import composite
import torch
import torch.nn.functional as F

COMPOSITE_NAME = "odml.scaled_dot_product_attention"
# The additive mask value of masked keys (e.g. padded vision patches). Finite
# in float16 (with headroom for logits of up to ~3e4) and in ML Drift's base-2
# flash SDPA kernel (`mask * log2(e)` ~ -4.3e4); masked keys still get exactly
# zero weight. A small finite value also quantizes well on NPUs.
PADDING_MASK_VALUE = -3.0e4


def _sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """SDPA with 4D BMMs, used as the decomposition of the composite.

  Args:
    query: Query tensor, `[B, T, N, H]`.
    key: Key tensor, `[B, S, K, H]`.
    value: Value tensor, `[B, S, K, H]`.
    scale: Scale applied to the query before the QK^T product.
    mask: None, or an additive float mask `[B_m, 1, T_m, S]`.

  Returns:
    Attention result, `[B, T, N, H]`.
  """
  b, t, n, h = query.shape
  _, s, k, _ = key.shape
  g = n // k
  key = key.to(query.dtype)
  value = value.to(query.dtype)
  query = query * scale

  # Query heads are grouped by KV head (head `i` uses KV head `i // g`), the
  # same convention as `transformers`' `repeat_kv`.
  query = query.permute(0, 2, 1, 3).reshape(b, k, g * t, h)  # B, K, G*T, H
  key = key.permute(0, 2, 3, 1)  # B, K, H, S
  value = value.permute(0, 2, 1, 3)  # B, K, S, H

  logits = torch.matmul(query, key)  # B, K, G*T, S
  if mask is not None:
    logits = logits.reshape(b, n, t, s) + mask.to(logits.dtype)
    logits = logits.reshape(b, k, g * t, s)
  probs = F.softmax(logits.float(), dim=-1).to(query.dtype)
  out = torch.matmul(probs, value)  # B, K, G*T, H
  return out.reshape(b, n, t, h).permute(0, 2, 1, 3)  # B, T, N, H


def bidirectional_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """Non-autoregressive bidirectional SDPA as an ODML SDPA composite.

  Emits an `odml.scaled_dot_product_attention` StableHLO composite, unmasked or
  with an additive float mask as its 4th input:

    output = softmax(scale * Q @ K^T + mask) @ V

  Runtimes that support the composite (e.g. ML Drift GPU with the 2D-tiled
  flash attention kernel, YNNPACK CPU) replace it with a fused kernel that
  avoids materializing the `[B, N, T, S]` attention logits; other runtimes
  (e.g. XNNPACK CPU) execute the decomposition (`_sdpa_reference`).

  Requirements for callers:
  - Non-autoregressive only: no KV cache. K/V are the keys/values of the
    current input only (typically `S == T`). Autoregressive decoders use
    (transposed) KV caches and need a different composite (e.g.
    `odml.sdpa_transposed`, see `sdpa.py`).
  - Bidirectional attention: with `mask` None, every query attends to every
    key. To mask keys (e.g. padded vision patches), pass an additive float
    `mask` of shape `[B_m, 1, T_m, S]` (`B_m` in `{1, B}`, `T_m` in `{1, T}`; 0
    keeps a key, a large negative value such as `PADDING_MASK_VALUE` masks
    it). Float masks run on every path: ML Drift's flash kernel (which
    broadcasts it over heads and queries) and its fallback (which adds it),
    XNNPACK (float32 only) and NPU compilers. Bool masks are rejected: ML
    Drift's fallback would add them as 0/1. Not meant for causal attention.
  - Layout: `query` is `[B, T, N, H]` and `key`/`value` are `[B, S, K, H]`
    (BTNH, the layout right after the Q/K/V projections), with `N % K == 0`
    (MHA, GQA or MQA). Do NOT repeat K/V heads for GQA; runtimes handle it.
    Callers holding BNTH tensors (e.g. `transformers` attention interfaces)
    must transpose to BTNH; the transposes usually cancel out with the
    model's own BTNH -> BNTH transposes.
  - `scale` is always emitted as an explicit attribute, since runtimes disagree
    on its default (ML Drift: 1.0, XNNPACK: 1/sqrt(H)).
  - No attention logits soft-cap or logits bias: ML Drift ignores the
    composite's `logit_cap` attribute.
  - `query` must be a float tensor. `key`/`value` are passed to the composite
    as-is (and only cast to `query.dtype` in the decomposition) so that a
    quantized K/V can be handled by a quantized SDPA kernel in the runtime.

  Args:
    query: Query tensor, `[B, T, N, H]`.
    key: Key tensor, `[B, S, K, H]`.
    value: Value tensor, `[B, S, K, H]`.
    scale: Scale applied to the query before the QK^T product.
    mask: None (full attention), or an additive float mask `[B_m, 1, T_m, S]`.

  Returns:
    Attention result, `[B, T, N, H]`.
  """
  if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
    raise ValueError(
        "query/key/value must be 4D [B, T, N, H] / [B, S, K, H], got"
        f" {tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
    )
  if key.shape != value.shape:
    raise ValueError(
        f"key and value shapes must match: {tuple(key.shape)} vs"
        f" {tuple(value.shape)}."
    )
  if query.shape[0] != key.shape[0] or query.shape[3] != key.shape[3]:
    raise ValueError(
        "query and key must have the same batch size and head dim:"
        f" {tuple(query.shape)} vs {tuple(key.shape)}."
    )
  if query.shape[2] % key.shape[2] != 0:
    raise ValueError(
        f"Number of query heads ({query.shape[2]}) must be a multiple of the"
        f" number of KV heads ({key.shape[2]})."
    )
  if not query.is_floating_point():
    raise ValueError(f"query must be a float tensor, got {query.dtype}.")

  if mask is not None:
    b, t, _, _ = query.shape
    s = key.shape[1]
    if not mask.is_floating_point():
      raise ValueError(
          f"mask must be an additive float mask, got {mask.dtype}; use"
          " torch.where(mask, 0.0, PADDING_MASK_VALUE)."
      )
    if (
        mask.dim() != 4
        or mask.shape[0] not in (1, b)
        or mask.shape[1] != 1
        or mask.shape[2] not in (1, t)
        or mask.shape[3] != s
    ):
      raise ValueError(
          f"mask must be [B_m, 1, T_m, S] with B_m in (1, {b}), T_m in"
          f" (1, {t}) and S={s}, got {tuple(mask.shape)}."
      )

  builder = composite.StableHLOCompositeBuilder(
      name=COMPOSITE_NAME, attr={"scale": float(scale)}
  )
  if mask is None:
    query, key, value = builder.mark_inputs(query, key, value)
  else:
    query, key, value, mask = builder.mark_inputs(query, key, value, mask)
  out = _sdpa_reference(query, key, value, float(scale), mask)
  return builder.mark_outputs(out)
