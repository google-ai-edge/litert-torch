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
"""Weight loading, verification, and quantization helpers for MiniCPM-o 4.5."""

from __future__ import annotations

import gc
import os
from typing import Any

from litert_torch.generative.export_hf.core import quantization
import safetensors
import torch
from torch import nn

LARGE_TENSOR_NUMEL_THRESHOLD = 1_000_000


def make_linear(
    in_features: int, out_features: int, bias: bool = True
) -> nn.Linear:
  """Creates an `nn.Linear`, skipping slow CPU RNG init for large matrices.

  Args:
    in_features: Size of each input sample.
    out_features: Size of each output sample.
    bias: If True, adds a learnable bias to the output.

  Returns:
    An initialized `nn.Linear` module.
  """
  if in_features * out_features <= LARGE_TENSOR_NUMEL_THRESHOLD:
    return nn.Linear(in_features, out_features, bias=bias)
  linear = nn.Linear(in_features, out_features, bias=bias, device="meta")
  linear.weight = nn.Parameter(
      torch.zeros(out_features, in_features, dtype=torch.float32)
  )
  if bias:
    linear.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
  return linear


def make_embedding(num_embeddings: int, embedding_dim: int) -> nn.Embedding:
  """Creates an `nn.Embedding`, skipping slow CPU RNG init for large tables.

  Args:
    num_embeddings: Size of the dictionary of embeddings.
    embedding_dim: The size of each embedding vector.

  Returns:
    An initialized `nn.Embedding` module.
  """
  if num_embeddings * embedding_dim <= LARGE_TENSOR_NUMEL_THRESHOLD:
    return nn.Embedding(num_embeddings, embedding_dim)
  return nn.Embedding(
      num_embeddings,
      embedding_dim,
      _weight=torch.zeros(num_embeddings, embedding_dim, dtype=torch.float32),
  )


def get_weight(
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    key: str,
) -> torch.Tensor | None:
  """Retrieves a weight tensor by key from memory or sharded safetensors.

  Args:
    state_dict: In-memory mapping of tensor keys to tensors.
    weight_map: Mapping of tensor keys to safetensors shard file paths.
    key: Checkpoint tensor key to look up.

  Returns:
    The loaded `torch.Tensor` if found, or `None`.
  """
  if key in state_dict:
    return state_dict[key]
  shard_path = weight_map.get(key)
  if shard_path and os.path.exists(shard_path):
    with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
      if key in f.keys():
        return f.get_tensor(key)
  return None


def copy_weight(
    target: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    *keys: str,
    strict: bool = True,
    populated: set[int] | None = None,
) -> None:
  """Copies the first matching checkpoint tensor in `keys` into `target`.

  Args:
    target: Destination parameter tensor.
    state_dict: In-memory mapping of tensor keys to tensors.
    weight_map: Mapping of tensor keys to safetensors shard file paths.
    *keys: Candidate checkpoint tensor keys in priority order.
    strict: When True, raises KeyError if none of `keys` exists in the
      checkpoint and ValueError if a matching key's shape does not match
      `target.shape`.
    populated: Optional set of `data_ptr()` integers recording parameters that
      were populated from the checkpoint.

  Raises:
    KeyError: If `strict` is True and none of `keys` exists in the checkpoint.
    ValueError: If `strict` is True and a matching tensor's shape does not
      match `target.shape`.
  """
  for key in keys:
    raw = get_weight(state_dict, weight_map, key)
    if raw is None:
      continue
    if raw.shape != target.shape:
      if strict:
        raise ValueError(
            f"Weight shape mismatch for {key!r}: checkpoint has"
            f" {tuple(raw.shape)}, target expects {tuple(target.shape)}."
        )
      continue
    src = raw.to(dtype=target.dtype, device=target.device)
    with torch.no_grad():
      target.copy_(src)
    if populated is not None:
      populated.add(target.data_ptr())
    return
  if strict:
    raise KeyError(f"Missing checkpoint weight; tried keys: {keys}.")


def copy_qkv_in_proj(
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    v_proj: nn.Linear,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    weight_key: str,
    bias_key: str,
    strict: bool,
    populated: set[int],
) -> None:
  """Copies fused QKV `in_proj_weight` and `in_proj_bias` into Q/K/V layers.

  Args:
    q_proj: Destination query `nn.Linear` layer.
    k_proj: Destination key `nn.Linear` layer.
    v_proj: Destination value `nn.Linear` layer.
    state_dict: In-memory mapping of tensor keys to tensors.
    weight_map: Mapping of tensor keys to safetensors shard file paths.
    weight_key: Checkpoint key for fused `(3 * hidden, hidden)` weight.
    bias_key: Checkpoint key for fused `(3 * hidden,)` bias.
    strict: Whether missing or shape-mismatched tensors raise errors.
    populated: Set of `data_ptr()` integers tracking populated parameters.

  Raises:
    KeyError: If `strict` is True and `weight_key` or `bias_key` is missing.
    ValueError: If `strict` is True and tensor shapes do not match.
  """
  hidden = q_proj.weight.shape[1]
  in_proj_w = get_weight(state_dict, weight_map, weight_key)
  if in_proj_w is None:
    if strict:
      raise KeyError(f"Missing checkpoint weight: {weight_key!r}.")
  else:
    expected_w_shape = (3 * hidden, hidden)
    if in_proj_w.shape != expected_w_shape:
      if strict:
        raise ValueError(
            f"Weight shape mismatch for {weight_key!r}: checkpoint has"
            f" {tuple(in_proj_w.shape)}, target expects {expected_w_shape}."
        )
    else:
      qw, kw, vw = in_proj_w.to(dtype=q_proj.weight.dtype).chunk(3, dim=0)
      with torch.no_grad():
        q_proj.weight.copy_(qw)
        k_proj.weight.copy_(kw)
        v_proj.weight.copy_(vw)
      populated.update((
          q_proj.weight.data_ptr(),
          k_proj.weight.data_ptr(),
          v_proj.weight.data_ptr(),
      ))

  in_proj_b = get_weight(state_dict, weight_map, bias_key)
  if in_proj_b is None:
    if strict:
      raise KeyError(f"Missing checkpoint weight: {bias_key!r}.")
  else:
    expected_b_shape = (3 * hidden,)
    if in_proj_b.shape != expected_b_shape:
      if strict:
        raise ValueError(
            f"Weight shape mismatch for {bias_key!r}: checkpoint has"
            f" {tuple(in_proj_b.shape)}, target expects {expected_b_shape}."
        )
    else:
      qb, kb, vb = in_proj_b.to(dtype=q_proj.bias.dtype).chunk(3, dim=0)
      with torch.no_grad():
        q_proj.bias.copy_(qb)
        k_proj.bias.copy_(kb)
        v_proj.bias.copy_(vb)
      populated.update((
          q_proj.bias.data_ptr(),
          k_proj.bias.data_ptr(),
          v_proj.bias.data_ptr(),
      ))


def _compute_weight_norm(
    g: torch.Tensor,
    v: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
  """Computes `g * v / ||v||` for PyTorch `weight_norm` parameters."""
  gf = g.to(dtype=torch.float32)
  vf = v.to(dtype=torch.float32)
  norm = vf.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
  while norm.ndim < vf.ndim:
    norm = norm.unsqueeze(-1)
  while gf.ndim < vf.ndim:
    gf = gf.unsqueeze(-1)
  return (gf * vf / norm).to(dtype=dtype, device=device)


def copy_weight_norm(
    target: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    weight_key: str,
    orig0_key: str,
    orig1_key: str,
    strict: bool,
    populated: set[int],
) -> None:
  """Copies a plain `weight_key` or PyTorch `weight_norm` `(g, v)` pair.

  Args:
    target: Destination parameter tensor.
    state_dict: In-memory mapping of tensor keys to tensors.
    weight_map: Mapping of tensor keys to safetensors shard file paths.
    weight_key: Direct weight tensor key.
    orig0_key: `parametrizations.weight.original0` magnitude key.
    orig1_key: `parametrizations.weight.original1` direction key.
    strict: Whether missing or shape-mismatched tensors raise errors.
    populated: Set of `data_ptr()` integers tracking populated parameters.

  Raises:
    KeyError: If `strict` is True and neither `weight_key` nor `(orig0_key,
      orig1_key)` exists.
    ValueError: If `strict` is True and tensor shapes do not match `target`.
  """
  if get_weight(state_dict, weight_map, weight_key) is not None:
    copy_weight(
        target,
        state_dict,
        weight_map,
        weight_key,
        strict=strict,
        populated=populated,
    )
    return
  g = get_weight(state_dict, weight_map, orig0_key)
  v = get_weight(state_dict, weight_map, orig1_key)
  if g is not None and v is not None:
    if v.shape != target.shape or g.shape[0] != target.shape[0]:
      if strict:
        raise ValueError(
            f"Weight shape mismatch for ({orig0_key!r}, {orig1_key!r}):"
            f" got g={tuple(g.shape)}, v={tuple(v.shape)}, expected"
            f" {tuple(target.shape)}."
        )
      return
    w = _compute_weight_norm(g, v, target.dtype, target.device)
    with torch.no_grad():
      target.copy_(w)
    populated.add(target.data_ptr())
    return
  if strict:
    raise KeyError(
        "Missing checkpoint weight; tried keys:"
        f" {(weight_key, orig0_key, orig1_key)}."
    )


def copy_module_parameters(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    prefix: str,
    param_map: dict[str, str] | None = None,
    strict: bool = True,
    populated: set[int] | None = None,
) -> None:
  """Copies checkpoint weights into `module`'s named parameters.

  Args:
    module: Module whose parameters will be populated.
    state_dict: In-memory mapping of tensor keys to tensors.
    weight_map: Mapping of tensor keys to safetensors shard file paths.
    prefix: Checkpoint prefix prepended to parameter names.
    param_map: Optional mapping from local parameter name to checkpoint suffix.
      If None or identity, local name is appended directly to `prefix`.
    strict: Whether missing or shape-mismatched tensors raise errors.
    populated: Set of `data_ptr()` integers tracking populated parameters.
  """
  for name, param in module.named_parameters():
    suffix = param_map.get(name, name) if param_map is not None else name
    ckpt_key = f"{prefix}.{suffix}" if prefix else suffix
    copy_weight(
        param,
        state_dict,
        weight_map,
        ckpt_key,
        strict=strict,
        populated=populated,
    )


def copy_parameter_dict(
    mapping: dict[torch.Tensor, str | tuple[str, ...]],
    state_dict: dict[str, torch.Tensor],
    weight_map: dict[str, str],
    strict: bool = True,
    populated: set[int] | None = None,
) -> None:
  """Copies checkpoint keys to destination parameter tensors in `mapping`.

  Args:
    mapping: Dict mapping destination tensors to checkpoint key(s).
    state_dict: In-memory mapping of tensor keys to tensors.
    weight_map: Mapping of tensor keys to safetensors shard file paths.
    strict: Whether missing or shape-mismatched tensors raise errors.
    populated: Set of `data_ptr()` integers tracking populated parameters.
  """
  for target, keys in mapping.items():
    key_tuple = (keys,) if isinstance(keys, str) else keys
    copy_weight(
        target,
        state_dict,
        weight_map,
        *key_tuple,
        strict=strict,
        populated=populated,
    )


def copy_from_dict(
    target: torch.Tensor,
    sd: dict[str, torch.Tensor],
    key: str,
    strict: bool,
    populated: set[int],
) -> None:
  """Copies `sd[key]` into `target` if shapes match.

  Args:
    target: Destination parameter tensor.
    sd: Source state dictionary.
    key: Key to look up in `sd`.
    strict: When True, raises KeyError on missing key and ValueError on shape
      mismatch. When False, leaves `target` unchanged on missing or mismatched
      tensors without marking it populated.
    populated: Set of `data_ptr()` integers tracking populated parameters.

  Raises:
    KeyError: If `strict` is True and `key` is not in `sd`.
    ValueError: If `strict` is True and `sd[key].shape != target.shape`.
  """
  raw = sd.get(key)
  if raw is None:
    if strict:
      raise KeyError(f"Missing token2wav weight: {key!r}.")
    return
  if raw.shape != target.shape:
    if strict:
      raise ValueError(
          f"Weight shape mismatch for {key!r}: checkpoint has"
          f" {tuple(raw.shape)}, target expects {tuple(target.shape)}."
      )
    return
  with torch.no_grad():
    target.copy_(raw.to(dtype=target.dtype, device=target.device))
  populated.add(target.data_ptr())


def copy_linear_from_dict(
    layer: Any,
    sd: dict[str, torch.Tensor],
    prefix: str,
    strict: bool,
    populated: set[int],
    has_bias: bool = True,
) -> None:
  """Copies `<prefix>.weight` (and `<prefix>.bias`) from `sd` into `layer`.

  Args:
    layer: Module with `weight` and optional `bias` attributes.
    sd: Source state dictionary.
    prefix: Key prefix in `sd`.
    strict: Whether missing or shape-mismatched tensors raise errors.
    populated: Set of `data_ptr()` integers tracking populated parameters.
    has_bias: Whether to also copy `<prefix>.bias` when `layer.bias` exists.
  """
  copy_from_dict(layer.weight, sd, f"{prefix}.weight", strict, populated)
  if has_bias and layer.bias is not None:
    copy_from_dict(layer.bias, sd, f"{prefix}.bias", strict, populated)


def copy_wn_from_dict(
    layer: Any,
    sd: dict[str, torch.Tensor],
    prefix: str,
    strict: bool,
    populated: set[int],
) -> None:
  """Copies a weight-normed convolution or linear layer from `sd`.

  Args:
    layer: Module with `weight` and optional `bias` attributes.
    sd: Source state dictionary.
    prefix: Key prefix in `sd`.
    strict: Whether missing or shape-mismatched tensors raise errors.
    populated: Set of `data_ptr()` integers tracking populated parameters.

  Raises:
    KeyError: If `strict` is True and weight-norm keys are missing.
    ValueError: If `strict` is True and weight-norm tensor shapes mismatch.
  """
  w_key = f"{prefix}.weight"
  g_key = f"{prefix}.parametrizations.weight.original0"
  v_key = f"{prefix}.parametrizations.weight.original1"
  if w_key in sd:
    copy_from_dict(layer.weight, sd, w_key, strict, populated)
  else:
    g = sd.get(g_key)
    v = sd.get(v_key)
    if g is None or v is None:
      if strict:
        raise KeyError(
            f"Missing weight_norm keys for {prefix!r}: tried"
            f" {(w_key, g_key, v_key)}."
        )
    elif v.shape != layer.weight.shape or g.shape[0] != layer.weight.shape[0]:
      if strict:
        raise ValueError(
            f"Weight shape mismatch for ({g_key!r}, {v_key!r}):"
            f" got g={tuple(g.shape)}, v={tuple(v.shape)}, expected"
            f" {tuple(layer.weight.shape)}."
        )
    else:
      w = _compute_weight_norm(g, v, layer.weight.dtype, layer.weight.device)
      with torch.no_grad():
        layer.weight.copy_(w)
      populated.add(layer.weight.data_ptr())
  if layer.bias is not None:
    copy_from_dict(layer.bias, sd, f"{prefix}.bias", strict, populated)


def verify_all_parameters_populated(
    mod: nn.Module, populated: set[int], module_name: str
) -> None:
  """Raises ValueError if any parameter in `mod` was not populated.

  Args:
    mod: Module whose `named_parameters()` are checked.
    populated: Set of `data_ptr()` integers for populated parameters.
    module_name: Human-readable module name for error reporting.

  Raises:
    ValueError: If any parameter in `mod` has a `data_ptr()` not in `populated`.
  """
  unpopulated = [
      name
      for name, param in mod.named_parameters()
      if param.data_ptr() not in populated
  ]
  if unpopulated:
    raise ValueError(f"Unpopulated parameters in {module_name}: {unpopulated}")


def maybe_quantize(out_path: str, quant_recipe: str | None) -> str:
  """Applies comma-separated quantization recipes to `out_path` if specified.

  Args:
    out_path: Path to the exported FP32 `.tflite` file.
    quant_recipe: Optional quantization recipe string (or comma-separated list).

  Returns:
    Path to the final (possibly quantized) `.tflite` file.
  """
  quant_list = quant_recipe.split(",") if quant_recipe else [None]
  final_path = out_path
  for recipe in quant_list:
    prev_path = final_path
    final_path = quantization.maybe_quantize_model(final_path, recipe)
    if final_path != prev_path and os.path.exists(prev_path):
      os.remove(prev_path)
    gc.collect()
  return final_path
