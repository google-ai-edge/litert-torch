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
"""Fuses parallel linear projections (e.g. q/k/v, gate/up) for export.

Model patches fuse projections that read the same input into one wider
projection along the output dimension. Float `torch.nn.Linear` layers are
fused here. Other layer types (e.g. pre-quantized layers) opt in by defining a
`fuse_linear_modules` classmethod (`FusibleLinear`). Layers that can't be
fused raise `FusionNotSupportedError`, which names the flags that disable the
fusion.
"""

from collections.abc import Mapping
from typing import Protocol

import torch

# Flags that disable each fusion, quoted in `FusionNotSupportedError`.
FUSE_QKV_FLAGS = "--fuse_qkv=False --use_qkv_norm_rope_composite=False"
FUSE_GATE_UP_FLAGS = "--fuse_gate_up=False --use_swiglu_composite=False"


class FusionNotSupportedError(ValueError):
  """Raised when projections can't be fused into one projection."""


class FusibleLinear(Protocol):
  """A linear layer type that fuses instances of itself.

  `fuse_linear_modules(parts)` receives the ordered `{attribute name: module}`
  parts (e.g. `{"q_proj": ..., "k_proj": ..., "v_proj": ...}`) and returns one
  module whose output is the concatenation of the parts' outputs along the
  last dimension. It raises `FusionNotSupportedError` if the parts can't be
  fused (e.g. different quantization schemes).
  """

  @classmethod
  def fuse_linear_modules(
      cls, parts: Mapping[str, torch.nn.Module]
  ) -> torch.nn.Module:
    ...


def _fusion_error(
    parts: Mapping[str, torch.nn.Module],
    fused_name: str,
    disable_flags: str,
    reason: Exception | str,
) -> FusionNotSupportedError:
  types = sorted({type(m).__qualname__ for m in parts.values()})
  return FusionNotSupportedError(
      f"Cannot fuse {list(parts)} ({', '.join(types)}) into `{fused_name}`:"
      f" {reason}. The weights' quantization scheme doesn't support this"
      f" fusion; disable it with {disable_flags}."
  )


def _float_linear_problem(
    parts: Mapping[str, torch.nn.Module],
) -> str | None:
  """Returns why `parts` aren't fusible float `nn.Linear` layers, or None."""
  first = None
  for name, module in parts.items():
    if not isinstance(module, torch.nn.Linear):
      return (
          f"`{name}` is a {type(module).__qualname__}, but only float"
          " torch.nn.Linear layers are supported"
      )
    if not torch.is_floating_point(module.weight):
      return (
          f"`{name}.weight` has non-floating-point dtype {module.weight.dtype}"
      )
    if first is None:
      first = module
      continue
    if module.in_features != first.in_features:
      return (
          f"`{name}` has in_features={module.in_features}, expected"
          f" {first.in_features}"
      )
    if (module.bias is None) != (first.bias is None):
      return "the projections don't all have (or all lack) a bias"
  return None


def check_float_linears(
    parts: Mapping[str, torch.nn.Module],
    *,
    fused_name: str,
    disable_flags: str,
) -> None:
  """Checks that `parts` are fusible float `nn.Linear` layers.

  For model patches whose fused layout rearranges weight rows, which only
  float layers support.

  Args:
    parts: Ordered `{attribute name: module}` projections to fuse.
    fused_name: Attribute name of the fused projection, for error messages.
    disable_flags: Export flags that disable this fusion, for error messages.

  Raises:
    FusionNotSupportedError: If the parts aren't fusible float `nn.Linear`s.
  """
  problem = _float_linear_problem(parts)
  if problem is not None:
    raise _fusion_error(parts, fused_name, disable_flags, problem)


def _fuse_float_linears(
    parts: Mapping[str, torch.nn.Module],
) -> torch.nn.Linear:
  """Fuses float `nn.Linear` layers by concatenating weights and biases."""
  modules = [m for m in parts.values() if isinstance(m, torch.nn.Linear)]
  first = modules[0]
  fused = torch.nn.Linear(
      first.in_features,
      sum(m.out_features for m in modules),
      bias=first.bias is not None,
      device=first.weight.device,
      dtype=first.weight.dtype,
  )
  # Converter V2 exports on the meta device and reconstructs the fused weight
  # from the checkpoint (`HFCheckpointWeightsLoader` fused specs).
  if not first.weight.is_meta:
    with torch.no_grad():
      fused.weight.copy_(torch.cat([m.weight for m in modules], dim=0))
      if fused.bias is not None:
        fused.bias.copy_(torch.cat([m.bias for m in modules], dim=0))
  return fused


def fuse_linear_modules(
    parts: Mapping[str, torch.nn.Module],
    *,
    fused_name: str,
    disable_flags: str,
) -> torch.nn.Module:
  """Fuses `parts` into one projection along the output dimension.

  Args:
    parts: Ordered `{attribute name: module}` projections reading the same
      input, e.g. `{"q_proj": ..., "k_proj": ..., "v_proj": ...}`.
    fused_name: Attribute name of the fused projection (e.g. `qkv_proj`), for
      error messages.
    disable_flags: Export flags that disable this fusion, for error messages
      (e.g. `FUSE_QKV_FLAGS`).

  Returns:
    The fused projection.

  Raises:
    FusionNotSupportedError: If the projections can't be fused.
  """
  if not parts:
    raise ValueError("No projections to fuse.")
  modules = list(parts.values())
  if all(isinstance(m, torch.nn.Linear) for m in modules):
    check_float_linears(
        parts, fused_name=fused_name, disable_flags=disable_flags
    )
    return _fuse_float_linears(parts)
  cls = type(modules[0])
  fuse = getattr(cls, "fuse_linear_modules", None)
  # Exact types: a subclass may change the layer's semantics.
  same_type = all(type(m) is cls for m in modules)  # pylint: disable=unidiomatic-typecheck
  if not same_type or not callable(fuse):
    types = sorted({type(m).__qualname__ for m in modules})
    raise _fusion_error(
        parts,
        fused_name,
        disable_flags,
        f"layer types {types} don't implement `fuse_linear_modules`",
    )
  try:
    return fuse(parts)
  except FusionNotSupportedError as e:
    raise _fusion_error(parts, fused_name, disable_flags, e) from e
