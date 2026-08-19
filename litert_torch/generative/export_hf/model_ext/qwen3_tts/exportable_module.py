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
"""Exportable modules for Qwen3-TTS Talker and MTP Drafter."""

from litert_torch.generative.custom_ops import dynamic_update_slice as dus_utils
from litert_torch.generative.layers import normalization
from litert_torch.generative.layers import scaled_dot_product_attention as sdpa
import torch
from torch import nn
import torch.nn.functional as F

VOCAB = 2048
CACHE = 32
LAYERS = 5
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
EPS = 1e-6
THETA = 1e6


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
  """Applies RMSNorm using LiteRT High-Level Function Binding."""
  return normalization.rms_norm_with_hlfb(
      x,
      weight,
      EPS,
      torch.ones((weight.shape[-1],), dtype=torch.float32, device=x.device),
  )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
  """Rotates the two halves of the last dimension for RoPE."""
  a, b = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
  return torch.cat((-b, a), dim=-1)


class MtpStep(nn.Module):
  """One MTP decode step (all tensors rank <= 4)."""

  def __init__(self, weights: dict[str, torch.Tensor]):
    super().__init__()
    for key, tensor in weights.items():
      if key == "heads":
        self.register_buffer(
            "heads_2d", tensor.reshape(-1, 1024).contiguous(), persistent=False
        )
      else:
        self.register_buffer(key.replace(".", "_"), tensor, persistent=False)
    inv_freq = 1.0 / (
        THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  def forward(
      self,
      embeddings: torch.Tensor,  # [1, 1, 1024]
      input_ids: torch.Tensor,  # [1] int32
      # Note it should be [1, 1, 1, 17] originally, but to ensure static shapes
      # and broad delegate compatibility, the dimension is padded to a power of
      # 2 ([1, 1, 1, 32]).
      mask: torch.Tensor,  # [1, 1, 1, 32]
      # Same as above, the dimension should be [1, 17, 8, 128] originally, but
      # we make it [1, 32, 8, 128] here.
      kv_cache_k_0: torch.Tensor,  # [1, 32, 8, 128]
      kv_cache_k_1: torch.Tensor,
      kv_cache_k_2: torch.Tensor,
      kv_cache_k_3: torch.Tensor,
      kv_cache_k_4: torch.Tensor,
      kv_cache_v_0: torch.Tensor,  # [1, 32, 8, 128]
      kv_cache_v_1: torch.Tensor,
      kv_cache_v_2: torch.Tensor,
      kv_cache_v_3: torch.Tensor,
      kv_cache_v_4: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    x = embeddings
    angles = input_ids.float().reshape(1, 1) * self.inv_freq.reshape(1, -1)
    angles = torch.cat((angles, angles), dim=-1)
    cos = angles.cos().reshape(1, 1, 1, HEAD_DIM)
    sin = angles.sin().reshape(1, 1, 1, HEAD_DIM)

    zero = torch.zeros([]).int()
    pos_idx = input_ids.int()[0].reshape([])
    slice_indices = [zero, pos_idx, zero, zero]

    k_in_list = [
        kv_cache_k_0,
        kv_cache_k_1,
        kv_cache_k_2,
        kv_cache_k_3,
        kv_cache_k_4,
    ]
    v_in_list = [
        kv_cache_v_0,
        kv_cache_v_1,
        kv_cache_v_2,
        kv_cache_v_3,
        kv_cache_v_4,
    ]
    k_new_list, v_new_list = [], []
    for i in range(LAYERS):
      w_norm = getattr(self, f"layers_{i}_input_layernorm_weight")
      w_q = getattr(self, f"layers_{i}_self_attn_q_proj_weight")
      w_k = getattr(self, f"layers_{i}_self_attn_k_proj_weight")
      w_v = getattr(self, f"layers_{i}_self_attn_v_proj_weight")
      w_q_norm = getattr(self, f"layers_{i}_self_attn_q_norm_weight")
      w_k_norm = getattr(self, f"layers_{i}_self_attn_k_norm_weight")

      h = _rms_norm(x, w_norm)
      q = F.linear(h, w_q).view(1, 1, HEADS, HEAD_DIM)
      k = F.linear(h, w_k).view(1, 1, KV_HEADS, HEAD_DIM)
      v = F.linear(h, w_v).view(1, 1, KV_HEADS, HEAD_DIM)

      q = _rms_norm(q, w_q_norm).transpose(1, 2)  # [1, 16, 1, 128]
      k = _rms_norm(k, w_k_norm).transpose(1, 2)  # [1, 8, 1, 128]
      v = v.transpose(1, 2)  # [1, 8, 1, 128]

      q = q * cos + _rotate_half(q) * sin
      k_rot = (k * cos) + (_rotate_half(k) * sin)

      q_btnh = q.transpose(1, 2)  # [1, 1, 16, 128]
      k_btnh = k_rot.transpose(1, 2)  # [1, 1, 8, 128]
      v_btnh = v.transpose(1, 2)  # [1, 1, 8, 128]

      # In-place slice update on BTNH cache [1, 17, 8, 128]
      # using Dynamic Update Slice HLFB
      k_cache = dus_utils.dynamic_update_slice(
          k_in_list[i], k_btnh, slice_indices
      )
      v_cache = dus_utils.dynamic_update_slice(
          v_in_list[i], v_btnh, slice_indices
      )
      k_new_list.append(k_cache)
      v_new_list.append(v_cache)

      # Explicitly expand KV cache heads from 8 to 16 using Rank-4 operations
      # (cat + reshape) to ensure consistent 1:1 Head matching without
      # 5D tensors
      k_attn = torch.cat([k_cache, k_cache], dim=-1).reshape(
          1, CACHE, HEADS, HEAD_DIM
      )
      v_attn = torch.cat([v_cache, v_cache], dim=-1).reshape(
          1, CACHE, HEADS, HEAD_DIM
      )

      # Fused SDPA evaluation with HLFB annotations for acceleration
      out_attn = sdpa.scaled_dot_product_attention_with_hlfb(
          q_btnh,  # [1, 1, 16, 128]
          k_attn,  # [1, 17, 16, 128]
          v_attn,  # [1, 17, 16, 128]
          HEAD_DIM,
          mask=mask,
      )  # [1, 1, 16, 128]
      out = out_attn.reshape(1, 1, HEADS * HEAD_DIM)

      w_o = getattr(self, f"layers_{i}_self_attn_o_proj_weight")
      x = x + F.linear(out, w_o)

      w_post_norm = getattr(self, f"layers_{i}_post_attention_layernorm_weight")
      h2 = _rms_norm(x, w_post_norm)
      w_gate = getattr(self, f"layers_{i}_mlp_gate_proj_weight")
      w_up = getattr(self, f"layers_{i}_mlp_up_proj_weight")
      w_down = getattr(self, f"layers_{i}_mlp_down_proj_weight")

      ff = F.linear(F.silu(F.linear(h2, w_gate)) * F.linear(h2, w_up), w_down)
      x = x + ff

    w_final_norm = getattr(self, "norm_weight")
    x = _rms_norm(x, w_final_norm)
    logits_all = F.linear(
        x.reshape(1, 1024), getattr(self, "heads_2d")
    ).reshape(15, VOCAB)
    return {
        "logits": logits_all,
        **{f"kv_cache_k_{i}": k_new_list[i] for i in range(LAYERS)},
        **{f"kv_cache_v_{i}": v_new_list[i] for i in range(LAYERS)},
    }


# Backwards compatibility alias
MtpStepGpu = MtpStep
