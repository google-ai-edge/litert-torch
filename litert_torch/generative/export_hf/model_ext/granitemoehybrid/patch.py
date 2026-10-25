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
"""Patch for Granite-4.0-H (GraniteMoeHybrid: Mamba2 and attention layers).

The Mamba2 mixer layers are exported with `mamba2_mixer_forward`, which
replaces the transformers mixer forward:

1. Every tensor stays at rank <= 4 and no op lowers to BROADCAST_TO, so the
   layer can be fully delegated to mobile GPU backends. The transformers SSD
   scan contracts rank-5 and rank-6 tensors with einsums and replicates groups
   with `expand` / `repeat_interleave`. Here the same contractions are batched
   matmuls with the (batch, head, chunk) axes folded into one, groups are
   replicated with a concatenation (prefill) or a matmul with a ones row
   (decode), and sequences are padded with a concatenation of zeros.
2. The conv and SSM states live in `LiteRTLMConvCacheLayer` (`kv_cache_c_{i}`
   and `kv_cache_r_{i}`). The conv state holds the last `conv_kernel - 1`
   inputs of the depthwise convolution; the recurrent state is float32 with
   shape (batch, heads, head_dim, state_size).
3. Pad positions of a prefill chunk do not change the states. The runtime pads
   the last prefill chunk with the pad token, and `valid_mask` marks the real
   tokens. A pad position gets a zero input and a zero time step, which is an
   identity SSM step, and the stored conv state ends at the last real input.
   The mask only applies to prefill: a generated token is never a pad.

`check_mamba2_mixer_forward` compares the rewrite with the installed
transformers mixer when the patch is applied, so a change of the transformers
scan fails the export instead of exporting different math.
"""

import contextlib
import math
import types
from typing import Any

from litert_torch.generative.export_hf.model_ext import patches as patches_lib
import torch
import torch.nn.functional as F
from transformers.models.granitemoehybrid import configuration_granitemoehybrid
from transformers.models.granitemoehybrid import modeling_granitemoehybrid


def _repeat_groups(t: torch.Tensor, rep: int) -> torch.Tensor:
  """(b, s, g, n) -> (b, s, g * rep, n), each group repeated `rep` times.

  The same tensor as `repeat_interleave(rep, dim=2)`, which lowers to
  BROADCAST_TO.

  Args:
    t: The tensor to repeat.
    rep: The number of heads per group.

  Returns:
    The repeated tensor.
  """
  return torch.cat(
      [t[:, :, i : i + 1] for i in range(t.shape[2]) for _ in range(rep)],
      dim=2,
  )


def _repeat_groups_step(
    t: torch.Tensor, n_groups: int, rep: int, state_size: int
) -> torch.Tensor:
  """(b, g * n) -> (b, g * rep, n) for the single-token step.

  An outer product with a ones row instead of `expand`, which lowers to
  BROADCAST_TO.

  Args:
    t: The tensor to repeat.
    n_groups: The number of groups.
    rep: The number of heads per group.
    state_size: The SSM state size.

  Returns:
    The repeated tensor.
  """
  b = t.shape[0]
  ones = torch.ones(1, rep, device=t.device, dtype=t.dtype)
  t = torch.matmul(t.reshape(b * n_groups, state_size, 1), ones)
  return t.transpose(1, 2).reshape(b, n_groups * rep, state_size)


def _zeros_cat(t: torch.Tensor, pad_size: int, dim: int) -> torch.Tensor:
  """Pads `t` with `pad_size` trailing zeros along `dim`, as a concatenation."""
  if pad_size == 0:
    return t
  shape = list(t.shape)
  shape[dim] = pad_size
  zeros = torch.zeros(shape, device=t.device, dtype=t.dtype)
  return torch.cat([t, zeros], dim=dim)


def _exp_segment_sum(cs: torch.Tensor) -> torch.Tensor:
  """exp(segment_sum(x)) from cs = cumsum(x, -1): (n, l) -> (n, l, l).

  segment_sum(x)[i, j] = sum_{k = j + 1 .. i} x[k] = cs[i] - cs[j] for i >= j.
  The difference is masked to 0 above the diagonal before the exponential and
  the result after it, which is the transformers matrix (masked to -inf before
  the exponential) without an infinite value. On and below the diagonal the
  exponent is <= 0, since A < 0 and dt >= 0.

  Args:
    cs: The cumulative sum along the last axis.

  Returns:
    The lower-triangular decay matrix.
  """
  l = cs.shape[-1]
  diff = cs[..., :, None] - cs[..., None, :]
  keep = torch.tril(torch.ones(l, l, device=cs.device, dtype=cs.dtype))
  return torch.exp(diff * keep) * keep


def _next_conv_state(
    window: torch.Tensor, width: int, valid: torch.Tensor | None
) -> torch.Tensor:
  """Returns the last `width` real inputs of the convolution window.

  Args:
    window: The convolution input [conv state, chunk], (b, conv_dim, w).
    width: The conv state width, `conv_kernel - 1`.
    valid: The (b, seq) mask of the real positions of the chunk, or None if all
      positions are real.

  Returns:
    The next conv state, (b, conv_dim, width).
  """
  if valid is None:
    return window[..., -width:]
  # The chunk is right-padded, so with n real tokens the state is the window
  # columns [n, n + width). They are selected with a one-hot matmul built with
  # float arithmetic on a (b, 1) count: an int64 index adds CAST / ADD ops that
  # GPU delegates reject, and adding a rank-0 count to an `arange` was measured
  # to come back wrong on a GPU delegate while the (b, 1) form was exact.
  total = window.shape[-1]
  num_valid = valid.to(torch.float32).sum(dim=-1, keepdim=True)
  target = num_valid + torch.arange(
      width, device=window.device, dtype=torch.float32
  )
  cols = torch.arange(total, device=window.device, dtype=torch.float32)
  # selector[b, c, j] = 1 if c == target[b, j] else 0, without a bool EQUAL.
  selector = torch.relu(
      1.0 - torch.abs(cols[None, :, None] - target[:, None, :])
  )
  return torch.matmul(window, selector.to(window.dtype))


def _ssd_step(
    mixer: Any,
    x: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """The single-token SSM recurrence.

  Args:
    mixer: The Mamba2 mixer module.
    x: (b, 1, heads * head_dim).
    b: (b, 1, n_groups * state_size).
    c: (b, 1, n_groups * state_size).
    dt: (b, 1, heads), before the softplus.
    a: (heads,), A = -exp(A_log).
    state: (b, heads, head_dim, state_size), the previous SSM state.

  Returns:
    The output (b, 1, heads * head_dim) and the next SSM state.
  """
  batch_size = x.shape[0]
  num_heads, head_dim = mixer.num_heads, mixer.head_dim
  state_size, n_groups = mixer.ssm_state_size, mixer.n_groups
  rep = num_heads // n_groups
  # The transformers recurrent form applies no time step limit.
  dt = F.softplus(dt[:, 0, :] + mixer.dt_bias)  # (b, heads)
  da = torch.exp(dt * a)  # (b, heads)
  x = x.reshape(batch_size, num_heads, head_dim)
  b = b.reshape(batch_size, n_groups * state_size)
  c = c.reshape(batch_size, n_groups * state_size)
  if n_groups == 1:
    # A head axis of 1 broadcasts in the rank-4 arithmetic below.
    b = b.reshape(batch_size, 1, state_size)
    c = c.reshape(batch_size, 1, state_size)
  elif rep > 1:
    b = _repeat_groups_step(b, n_groups, rep, state_size)
    c = _repeat_groups_step(c, n_groups, rep, state_size)
  else:
    b = b.reshape(batch_size, num_heads, state_size)
    c = c.reshape(batch_size, num_heads, state_size)
  # Implicit broadcasting instead of `expand`, which lowers to BROADCAST_TO.
  dbx = dt[:, :, None, None] * b[:, :, None, :] * x[:, :, :, None]
  # The state stays float32 whatever the activation dtype.
  state = state.to(torch.float32) * da[:, :, None, None] + dbx
  y = torch.sum(state * c[:, :, None, :], dim=-1)  # (b, heads, head_dim)
  y = y + x * mixer.D[None, :, None]
  return y.reshape(batch_size, 1, num_heads * head_dim), state


def _ssd_chunked(
    mixer: Any,
    x: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor | None,
    valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """The chunked SSD scan, with every tensor at rank <= 4.

  The five contractions of the transformers scan, written as batched matmuls
  with the (batch, head, chunk) axes folded into the leading axis.

  Args:
    mixer: The Mamba2 mixer module.
    x: (b, seq, heads * head_dim).
    b: (b, seq, n_groups * state_size).
    c: (b, seq, n_groups * state_size).
    dt: (b, seq, heads), before the softplus.
    a: (heads,), A = -exp(A_log).
    state: (b, heads, head_dim, state_size), the initial SSM state, or None
      for a zero state.
    valid: The (b, seq) mask of the real positions, or None.

  Returns:
    The output (b, seq, heads * head_dim) and the final SSM state.
  """
  batch_size, seq_len, _ = x.shape
  num_heads, head_dim = mixer.num_heads, mixer.head_dim
  state_size, n_groups = mixer.ssm_state_size, mixer.n_groups
  rep = num_heads // n_groups
  chunk = mixer.chunk_size
  pad_size = (chunk - seq_len % chunk) % chunk
  seq_p = seq_len + pad_size
  n_chunks = seq_p // chunk
  bh = batch_size * num_heads

  dt = F.softplus(dt + mixer.dt_bias)  # (b, seq, heads)
  dt_min, dt_max = mixer.time_step_limit
  # softplus is >= 0, and an infinite bound is no bound.
  if dt_min > 0.0:
    dt = torch.clamp(dt, min=dt_min)
  if not math.isinf(dt_max):
    dt = torch.clamp(dt, max=dt_max)
  if valid is not None:
    # After the clamp: a zero time step is an identity step (no decay, no
    # input), a clamped minimum time step would still decay the state.
    dt = dt * valid[..., None]

  x = x.reshape(batch_size, seq_len, num_heads, head_dim).float()
  b = b.reshape(batch_size, seq_len, n_groups, state_size).float()
  c = c.reshape(batch_size, seq_len, n_groups, state_size).float()
  if rep > 1:
    b = _repeat_groups(b, rep)
    c = _repeat_groups(c, rep)

  d_residual = mixer.D[:, None] * x  # (b, seq, heads, head_dim)
  x = x * dt[..., None]
  a = a.to(x.dtype) * dt  # (b, seq, heads)

  # Zero x adds nothing and zero A decays nothing, so the chunk padding leaves
  # the real outputs and the final state unchanged.
  x = _zeros_cat(x, pad_size, 1)
  b = _zeros_cat(b, pad_size, 1)
  c = _zeros_cat(c, pad_size, 1)
  a = _zeros_cat(a, pad_size, 1)

  def fold(t, last):  # (b, seq_p, heads, last) -> (b * heads * chunks, l, last)
    return t.permute(0, 2, 1, 3).reshape(bh * n_chunks, chunk, last)

  xf = fold(x, head_dim)
  bf = fold(b, state_size)
  cf = fold(c, state_size)
  af = a.permute(0, 2, 1).reshape(bh * n_chunks, chunk)

  a_cumsum = torch.cumsum(af, dim=-1)  # (bhc, l)

  # 1. Intra-chunk outputs (diagonal blocks).
  scores = torch.matmul(cf, bf.transpose(1, 2)) * _exp_segment_sum(a_cumsum)
  y_diag = torch.matmul(scores, xf)  # (bhc, l, head_dim)

  # 2. State at the end of each chunk, as (state_size, head_dim) so that the
  # contraction is a plain matmul.
  decay_states = torch.exp(a_cumsum[:, -1:] - a_cumsum)  # (bhc, l)
  states = torch.matmul((bf * decay_states[..., None]).transpose(1, 2), xf)

  # 3. Inter-chunk recurrence, a contraction over the chunk axis.
  states = states.reshape(bh, n_chunks, state_size * head_dim)
  if state is not None:
    prev = state.to(states.dtype).permute(0, 1, 3, 2)
    prev = prev.reshape(bh, 1, state_size * head_dim)
  else:
    prev = torch.zeros_like(states[:, :1])
  states = torch.cat([prev, states], dim=1)  # (bh, chunks + 1, n * p)
  chunk_decay = a_cumsum[:, -1].reshape(bh, n_chunks)
  chunk_decay = torch.cat(
      [torch.zeros_like(chunk_decay[:, :1]), chunk_decay], dim=1
  )
  decay_chunk = _exp_segment_sum(torch.cumsum(chunk_decay, dim=-1))
  new_states = torch.matmul(decay_chunk, states)  # (bh, chunks + 1, n * p)
  chunk_states, final_state = new_states[:, :-1], new_states[:, -1]

  # 4. Outputs from the chunk start states (off-diagonal blocks).
  chunk_states = chunk_states.reshape(bh * n_chunks, state_size, head_dim)
  y_off = torch.matmul(cf, chunk_states) * torch.exp(a_cumsum)[..., None]

  y = (y_diag + y_off).reshape(batch_size, num_heads, seq_p, head_dim)
  y = y.permute(0, 2, 1, 3)[:, :seq_len]  # (b, seq, heads, head_dim)
  y = (y + d_residual).reshape(batch_size, seq_len, num_heads * head_dim)
  final_state = final_state.reshape(
      batch_size, num_heads, state_size, head_dim
  ).permute(0, 1, 3, 2)
  return y, final_state


def mamba2_mixer_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Any = None,
    attention_mask: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
  """Exportable forward of a transformers Mamba2 mixer.

  Bound to `GraniteMoeHybridMambaLayer` and `NemotronHMamba2Mixer` instances,
  which share these attribute names.

  Args:
    self: The Mamba2 mixer module.
    hidden_states: (batch, seq, hidden).
    cache_params: A `LiteRTLMCache`, or None.
    attention_mask: Unused; pad positions are given by `valid_mask`.
    valid_mask: The (batch, seq) mask of the real positions of a prefill
      chunk, or None.
    **kwargs: Unused.

  Returns:
    The mixer output, (batch, seq, hidden).
  """
  del attention_mask, kwargs
  batch_size, seq_len, _ = hidden_states.shape
  dtype = hidden_states.dtype
  state_width = self.conv_kernel_size - 1
  layer_cache = None
  if cache_params is not None:
    layer_cache = cache_params.layers[self.layer_idx]

  valid = None
  if valid_mask is not None and seq_len > 1:
    valid = valid_mask.reshape(batch_size, seq_len).to(dtype)
    hidden_states = hidden_states * valid[..., None]

  projected_states = self.in_proj(hidden_states)
  gate, xbc, dt = projected_states.split(
      [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
  )

  # Causal depthwise convolution over [conv state, chunk].
  if layer_cache is not None:
    conv_state = layer_cache.conv_states.to(xbc.dtype)
  else:
    conv_state = torch.zeros(
        (batch_size, self.conv_dim, state_width),
        device=xbc.device,
        dtype=xbc.dtype,
    )
  window = torch.cat([conv_state, xbc.transpose(1, 2)], dim=-1)
  if seq_len == 1:
    xbc = torch.sum(window * self.conv1d.weight[:, 0, :], dim=-1, keepdim=True)
    if self.conv1d.bias is not None:
      xbc = xbc + self.conv1d.bias[:, None]
  else:
    xbc = F.conv1d(
        window, self.conv1d.weight, self.conv1d.bias, groups=self.conv_dim
    )
  xbc = self.act(xbc).transpose(1, 2)  # (batch, seq, conv_dim)
  if layer_cache is not None:
    layer_cache.conv_states = _next_conv_state(window, state_width, valid).to(
        layer_cache.conv_states.dtype
    )
  if valid is not None:
    xbc = xbc * valid[..., None]
  groups_size = self.n_groups * self.ssm_state_size
  x, b, c = torch.split(
      xbc, [self.intermediate_size, groups_size, groups_size], dim=-1
  )

  # Selective state space model.
  a = -torch.exp(self.A_log.float())
  if layer_cache is not None and seq_len == 1:
    y, ssm_state = _ssd_step(self, x, b, c, dt, a, layer_cache.recurrent_states)
  else:
    initial_state = None
    if layer_cache is not None:
      initial_state = layer_cache.recurrent_states
    y, ssm_state = _ssd_chunked(self, x, b, c, dt, a, initial_state, valid)
  if layer_cache is not None:
    layer_cache.recurrent_states = ssm_state.to(torch.float32)

  y = self.norm(y, gate)
  return self.out_proj(y.to(dtype))


def check_mamba2_mixer_forward(
    mixer_cls: type[torch.nn.Module], config: Any, tol: float = 2e-4
) -> float:
  """Compares `mamba2_mixer_forward` with the installed transformers mixer.

  Runs the cache-free chunked path at a chunk multiple and a ragged length.

  Args:
    mixer_cls: The transformers Mamba2 mixer class.
    config: A small config for the mixer, with a chunk size that divides 16.
    tol: The largest allowed max |difference| relative to max |reference|.

  Returns:
    The relative max |difference|.

  Raises:
    AssertionError: If the difference is larger than `tol`.
  """
  worst = 0.0
  with torch.random.fork_rng(devices=[]):
    torch.manual_seed(0)
    mixer = mixer_cls(config, layer_idx=0).eval().float()
    for seq_len in (16, 13):
      x = torch.randn(1, seq_len, config.hidden_size)
      with torch.no_grad():
        ref = mixer_cls.forward(mixer, x)
        new = mamba2_mixer_forward(mixer, x)
      rel = ((ref - new).abs().max() / ref.abs().max()).item()
      worst = max(worst, rel)
  if worst > tol:
    raise AssertionError(
        f"The exported {mixer_cls.__name__} forward disagrees with the"
        " installed transformers implementation (relative max |difference|"
        f" {worst:.2e} > {tol:.0e}); the transformers Mamba2 scan changed."
    )
  return worst


@contextlib.contextmanager
def patch_mamba2_mixers(model: torch.nn.Module, mixer_cls: type[Any]):
  """Binds `mamba2_mixer_forward` to every `mixer_cls` instance of `model`.

  The prefill `valid_mask` is a keyword argument of the model call, and the
  causal LM passes it on to its base model. Not every transformers layer passes
  it on to its mixer (the NemotronH block of transformers 5.16.1 does not), so
  a forward pre-hook on the base model keeps the mask of the call, and a
  forward pre-hook on each mixer adds it when the mixer is called without one.

  Args:
    model: The model to patch.
    mixer_cls: The transformers Mamba2 mixer class.

  Yields:
    The patched mixers.

  Raises:
    ValueError: If the number of mixers differs from the number of
      "linear_attention" layers of the config.
  """
  config = getattr(model.config, "text_config", model.config)
  layer_types = getattr(config, "layer_types", None) or []
  num_mamba_layers = sum(t == "linear_attention" for t in layer_types)
  mixers = [m for m in model.modules() if isinstance(m, mixer_cls)]
  if len(mixers) != num_mamba_layers:
    raise ValueError(
        f"Found {len(mixers)} {mixer_cls.__name__} modules for"
        f" {num_mamba_layers} linear_attention layers."
    )
  # The valid_mask of the base model call in progress. It is cleared after
  # every call, so no traced tensor outlives the call that made it.
  call_kwargs: dict[str, Any] = {}

  def keep_valid_mask(module, args, kwargs):
    del module, args
    call_kwargs["valid_mask"] = kwargs.get("valid_mask")

  def clear_valid_mask(module, args, output):
    del module, args, output
    call_kwargs.clear()

  def add_valid_mask(module, args, kwargs):
    del module
    valid_mask = call_kwargs.get("valid_mask")
    if valid_mask is None or kwargs.get("valid_mask") is not None:
      return None
    return args, {**kwargs, "valid_mask": valid_mask}

  base_model = getattr(model, "base_model", model)
  handles = [
      base_model.register_forward_pre_hook(keep_valid_mask, with_kwargs=True),
      base_model.register_forward_hook(clear_valid_mask, always_call=True),
  ]
  for mixer in mixers:
    mixer.forward = types.MethodType(mamba2_mixer_forward, mixer)
    handles.append(
        mixer.register_forward_pre_hook(add_valid_mask, with_kwargs=True)
    )
  try:
    yield mixers
  finally:
    for handle in handles:
      handle.remove()
    call_kwargs.clear()
    for mixer in mixers:
      del mixer.forward


def _check_config() -> configuration_granitemoehybrid.GraniteMoeHybridConfig:
  return configuration_granitemoehybrid.GraniteMoeHybridConfig(
      hidden_size=64,
      num_hidden_layers=1,
      mamba_n_heads=8,
      mamba_d_head=16,
      mamba_d_state=16,
      mamba_expand=2,
      mamba_chunk_size=8,
      mamba_n_groups=2,
  )


@patches_lib.register_model_patch(["granitemoehybrid"])
@contextlib.contextmanager
def patch_granitemoehybrid_model(model: torch.nn.Module, export_config: Any):
  """Model patch for Granite-4.0-H export."""
  del export_config
  rel = check_mamba2_mixer_forward(
      modeling_granitemoehybrid.GraniteMoeHybridMambaLayer, _check_config()
  )
  with patch_mamba2_mixers(
      model, modeling_granitemoehybrid.GraniteMoeHybridMambaLayer
  ) as mixers:
    print(
        f"GraniteMoeHybrid model patch applied to {len(mixers)} Mamba2 layers"
        f" (check vs. transformers: relative max |difference| {rel:.1e})."
    )
    yield
