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
"""LoRA utilities and module wrappers for export_hf."""

import contextlib
from typing import Any, Callable, Tuple

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.layers import lora as lora_utils
import torch
from torch import nn


# Attribute names under which bound LoRA weights are stored on projection
# modules. Using forward hooks instead of wrapper modules keeps parameter FQNs
# (and thus checkpoint name resolution, e.g. Converter V2 weights loaders)
# unchanged.
_LORA_ATTR = "_litert_lora_weight"
_FUSED_LORA_ATTR = "_litert_fused_qkv_lora"
_FUSED_DIMS_ATTR = "_litert_fused_qkv_dims"


def _linear_lora_hook(
    module: nn.Module, args: tuple[Any, ...], out: torch.Tensor
) -> torch.Tensor:
  """Adds `(x @ a_prime) @ b_prime` to a projection output when bound."""
  weight = module.__dict__.get(_LORA_ATTR)
  if weight is None:
    return out
  return out + lora_utils.apply_lora(args[0], weight, shape=out.shape)


def _fused_qkv_lora_hook(
    module: nn.Module, args: tuple[Any, ...], out: torch.Tensor
) -> torch.Tensor:
  """Adds concatenated Q/K/V LoRA outputs to a fused QKV projection output."""
  weights = module.__dict__.get(_FUSED_LORA_ATTR)
  if weights is None:
    return out
  x = args[0]
  prefix_shape = tuple(out.shape[:-1])
  deltas = [
      lora_utils.apply_lora(x, w, shape=(*prefix_shape, dim))
      for w, dim in zip(weights, module.__dict__[_FUSED_DIMS_ATTR])
  ]
  return out + torch.cat(deltas, dim=-1)


def _has_separate_qkv(attn: Any) -> bool:
  return all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj"))


def _get_qkv_out_dims_from_fused_attn(
    attn: Any,
) -> Tuple[int, int, int]:
  """Extracts (q_out_dim, k_out_dim, v_out_dim) from a fused QKV attention module."""
  if hasattr(attn, "q_size") and hasattr(attn, "kv_size"):
    return int(attn.q_size), int(attn.kv_size), int(attn.kv_size)
  o_in = int(attn.o_proj.in_features)
  num_heads = int(
      getattr(attn, "num_heads", getattr(attn, "num_attention_heads", 1))
  )
  num_kv_heads = int(getattr(attn, "num_key_value_heads", num_heads))
  head_dim = int(getattr(attn, "head_dim", o_in // num_heads))
  q_out = int(num_heads * head_dim)
  kv_out = int(num_kv_heads * head_dim)
  return q_out, kv_out, kv_out


def get_attention_modules(model: nn.Module) -> list[Any]:
  """Returns the ordered list of self-attention modules across decoder layers."""
  candidates = [
      model,
      getattr(model, "model", None),
      getattr(model, "language_model", None),
      getattr(getattr(model, "language_model", None), "model", None),
      getattr(getattr(model, "model", None), "language_model", None),
  ]
  for candidate in candidates:
    if candidate is None:
      continue
    layers = getattr(candidate, "layers", None)
    if isinstance(layers, nn.ModuleList) and len(layers) > 0:
      attn_modules = []
      for layer in layers:
        attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
        if attn is not None and hasattr(attn, "o_proj"):
          attn_modules.append(attn)
      if len(attn_modules) == len(layers):
        return attn_modules

  # Fallback to module traversal in registration order.
  attn_modules = []
  for module in model.modules():
    if hasattr(module, "o_proj") and (
        hasattr(module, "q_proj") or hasattr(module, "qkv_proj")
    ):
      attn_modules.append(module)
  if not attn_modules:
    raise ValueError("Could not find any attention modules in the model.")
  return attn_modules


def create_lora_from_model(
    model: nn.Module,
    rank: int,
    tensor_generator: Callable[[Tuple[int, ...], torch.dtype], torch.Tensor],
    dtype: torch.dtype | None = None,
) -> lora_utils.LoRA:
  """Creates a LoRA PyTree matching the attention projection dimensions of `model`."""
  first_param = next(model.parameters(), None)
  if dtype is None:
    dtype = first_param.dtype if first_param is not None else torch.float32
  device = first_param.device if first_param is not None else None

  def _gen(shape: Tuple[int, ...], dt: torch.dtype) -> torch.Tensor:
    return tensor_generator(shape, dt).to(device)

  attn_modules = get_attention_modules(model)
  adapters = []
  for attn in attn_modules:
    o_in = int(attn.o_proj.in_features)
    o_out = int(attn.o_proj.out_features)
    if (
        hasattr(attn, "q_proj")
        and hasattr(attn, "k_proj")
        and hasattr(attn, "v_proj")
    ):
      q_in, q_out = int(attn.q_proj.in_features), int(attn.q_proj.out_features)
      k_in, k_out = int(attn.k_proj.in_features), int(attn.k_proj.out_features)
      v_in, v_out = int(attn.v_proj.in_features), int(attn.v_proj.out_features)
    elif hasattr(attn, "qkv_proj"):
      q_in = k_in = v_in = int(attn.qkv_proj.in_features)
      q_out, k_out, v_out = _get_qkv_out_dims_from_fused_attn(attn)
    else:
      raise ValueError(
          f"Unsupported attention module structure for LoRA: {type(attn)}"
      )

    attention_lora = lora_utils.AttentionLoRA(
        query=lora_utils.LoRAWeight(
            a_prime=_gen((q_in, rank), dtype),
            b_prime=_gen((rank, q_out), dtype),
        ),
        key=lora_utils.LoRAWeight(
            a_prime=_gen((k_in, rank), dtype),
            b_prime=_gen((rank, k_out), dtype),
        ),
        value=lora_utils.LoRAWeight(
            a_prime=_gen((v_in, rank), dtype),
            b_prime=_gen((rank, v_out), dtype),
        ),
        output=lora_utils.LoRAWeight(
            a_prime=_gen((o_in, rank), dtype),
            b_prime=_gen((rank, o_out), dtype),
        ),
    )
    adapters.append(lora_utils.LoRAEntry(attention=attention_lora))

  return lora_utils.LoRA(adapters=tuple(adapters))


def create_zeros_lora(
    rank: int,
    model: nn.Module,
    dtype: torch.dtype | None = None,
) -> lora_utils.LoRA:
  """Creates zero-initialized LoRA weights matching `model`."""
  return create_lora_from_model(
      model=model,
      rank=rank,
      tensor_generator=lambda shape, dt: torch.zeros(shape, dtype=dt),
      dtype=dtype,
  )


def create_random_lora(
    rank: int,
    model: nn.Module,
    dtype: torch.dtype | None = None,
) -> lora_utils.LoRA:
  """Creates random-initialized LoRA weights matching `model`."""
  return create_lora_from_model(
      model=model,
      rank=rank,
      tensor_generator=lambda shape, dt: torch.randn(shape, dtype=dt),
      dtype=dtype,
  )


def set_lora_weights(
    model: nn.Module,
    lora: lora_utils.LoRA | None,
) -> None:
  """Binds `lora` weights to the hooked projections, or clears them if None."""
  if lora is None:
    clear_lora_weights(model)
    return

  attn_modules = get_attention_modules(model)
  if len(attn_modules) != len(lora.adapters):
    raise ValueError(
        "Mismatch between number of model attention layers"
        f" ({len(attn_modules)}) and LoRA adapters ({len(lora.adapters)})."
    )

  for attn, entry in zip(attn_modules, lora.adapters):
    attn_lora = entry.attention
    qkv_proj = getattr(attn, "qkv_proj", None)
    if qkv_proj is not None and _FUSED_DIMS_ATTR in qkv_proj.__dict__:
      qkv_proj.__dict__[_FUSED_LORA_ATTR] = (
          attn_lora.query,
          attn_lora.key,
          attn_lora.value,
      )
    elif _has_separate_qkv(attn) and _LORA_ATTR in attn.q_proj.__dict__:
      attn.q_proj.__dict__[_LORA_ATTR] = attn_lora.query
      attn.k_proj.__dict__[_LORA_ATTR] = attn_lora.key
      attn.v_proj.__dict__[_LORA_ATTR] = attn_lora.value
    else:
      raise ValueError(
          "Model attention projections are not hooked for LoRA. "
          "Ensure the `patch_model_for_lora` context is active."
      )
    attn.o_proj.__dict__[_LORA_ATTR] = attn_lora.output


def clear_lora_weights(model: nn.Module) -> None:
  """Clears bound LoRA weights from all hooked projections in `model`."""
  for module in model.modules():
    if _LORA_ATTR in module.__dict__:
      module.__dict__[_LORA_ATTR] = None
    if _FUSED_LORA_ATTR in module.__dict__:
      module.__dict__[_FUSED_LORA_ATTR] = None


@contextlib.contextmanager
def patch_model_for_lora(
    model: nn.Module,
    export_config: exportable_module_config.ExportableModuleConfig,
):
  """Temporarily hooks attention projections to add LoRA outputs when bound.

  Forward hooks are used instead of wrapper modules so that the module tree,
  parameter FQNs and state dict are unchanged.

  Args:
    model: The model to patch.
    export_config: The export config; a no-op unless `lora_ranks` is set.

  Yields:
    None.
  """
  if not export_config.lora_ranks:
    yield
    return

  handles = []
  hooked: list[nn.Module] = []

  def _hook(module: nn.Module, hook_fn, attrs: dict[str, Any]) -> None:
    module.__dict__.update(attrs)
    handles.append(module.register_forward_hook(hook_fn))
    hooked.append(module)

  for attn in get_attention_modules(model):
    if _has_separate_qkv(attn):
      for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
        _hook(proj, _linear_lora_hook, {_LORA_ATTR: None})
    elif hasattr(attn, "qkv_proj"):
      _hook(
          attn.qkv_proj,
          _fused_qkv_lora_hook,
          {
              _FUSED_LORA_ATTR: None,
              _FUSED_DIMS_ATTR: _get_qkv_out_dims_from_fused_attn(attn),
          },
      )
    else:
      raise ValueError(
          f"Unsupported attention module structure for LoRA: {type(attn)}"
      )
    _hook(attn.o_proj, _linear_lora_hook, {_LORA_ATTR: None})

  # Ensure weights bound for one forward never leak into the next.
  handles.append(
      model.register_forward_hook(
          lambda module, args, out: clear_lora_weights(module)
      )
  )
  try:
    yield
  finally:
    for handle in handles:
      handle.remove()
    for module in hooked:
      for attr in (_LORA_ATTR, _FUSED_LORA_ATTR, _FUSED_DIMS_ATTR):
        module.__dict__.pop(attr, None)


@contextlib.contextmanager
def bind_lora_weights(
    model: nn.Module,
    lora: lora_utils.LoRA | None,
):
  """Binds `lora` weights for the duration of a forward pass."""
  set_lora_weights(model, lora)
  try:
    yield
  finally:
    clear_lora_weights(model)
