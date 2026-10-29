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
"""Streaming weights loading for Converter V2 (meta-device) export."""

from collections.abc import Callable, Iterator, Sequence
import contextlib
import glob
import json
import os
from typing import Any

from absl import logging
import huggingface_hub
from huggingface_hub import errors as hf_errors
import safetensors
import torch
from torch import nn

WeightsLoader = Callable[[str], torch.Tensor]

_SNAPSHOT_PATTERNS = ("*.safetensors", "*.json")
_INDEX_FILE = "model.safetensors.index.json"
# Wrapper prefixes that export-time modules add in front of HF parameter names.
_WRAPPER_PREFIXES = (
    "model.",
    "module.",
    "static_model.",
    "original_hf_model.",
    "language_model.",
)
_EMBEDDING_KEYS = (
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "transformer.wte.weight",
)
# Built-in fused linear specs `(fused_name, part_names, dim)`. Gemma-4 global
# attention layers with `attention_k_eq_v=True` omit `v_proj`, so `qkv_proj` has
# both 3-way and 2-way entries.
_DEFAULT_FUSED_SPECS: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("qkv_proj", ("q_proj", "k_proj", "v_proj"), 0),
    ("qkv_proj", ("q_proj", "k_proj"), 0),
    ("gate_up_proj", ("gate_proj", "up_proj"), 0),
)


@contextlib.contextmanager
def init_params_on_meta() -> Iterator[None]:
  """Creates `nn.Parameter`s on the meta device; buffers stay on CPU.

  Unlike `with torch.device("meta")`, non-persistent buffers (RoPE `inv_freq`,
  embedding scales, ...) keep their real values. They are not stored in
  checkpoints, so a weights loader could not recover them otherwise.

  Yields:
    None.
  """
  original_register_parameter = nn.Module.register_parameter

  def register_meta_parameter(
      self: nn.Module, name: str, param: nn.Parameter | None
  ) -> None:
    original_register_parameter(self, name, param)
    if param is not None and param.device.type != "meta":
      self._parameters[name] = nn.Parameter(  # pylint: disable=protected-access
          param.to("meta"), requires_grad=param.requires_grad
      )

  nn.Module.register_parameter = register_meta_parameter
  try:
    yield
  finally:
    nn.Module.register_parameter = original_register_parameter


def cast_params_(model: nn.Module, dtype: torch.dtype) -> nn.Module:
  """Casts floating-point parameters in place, leaving buffers untouched.

  `nn.Module.to(dtype)` would also cast buffers such as RoPE `inv_freq`, whose
  precision matters (bf16 phase error grows linearly with position). Shared
  (tied) parameters stay shared.

  Args:
    model: The model to cast.
    dtype: The target floating-point dtype.

  Returns:
    The same model.
  """
  cast: dict[int, nn.Parameter] = {}
  for module in model.modules():
    for name, param in list(module._parameters.items()):  # pylint: disable=protected-access
      if param is None or not param.is_floating_point() or param.dtype == dtype:
        continue
      if id(param) not in cast:
        cast[id(param)] = nn.Parameter(
            param.to(dtype), requires_grad=param.requires_grad
        )
      module._parameters[name] = cast[id(param)]  # pylint: disable=protected-access
  return model


def load_persistent_buffers_(
    model: nn.Module, weights_loader: WeightsLoader
) -> nn.Module:
  """Loads persistent buffers stored in the checkpoint, in place.

  `init_params_on_meta` keeps buffers on CPU with their init values. That is
  right for non-persistent buffers (e.g. RoPE `inv_freq`), but persistent
  buffers saved in the checkpoint (e.g. Gemma-4 `layer_scalar`) must be loaded,
  as `from_pretrained` does. Buffers missing from the checkpoint keep their
  init values.

  Args:
    model: The model whose buffers to load.
    weights_loader: Resolves FQNs relative to `model`, as for its parameters.

  Returns:
    The same model.

  Raises:
    ValueError: If a checkpoint tensor's shape does not match its buffer.
  """
  missing = []
  with torch.no_grad():
    for prefix, module in model.named_modules():
      for name, buf in module._buffers.items():  # pylint: disable=protected-access
        if (
            buf is None
            or buf.is_meta
            or name in module._non_persistent_buffers_set  # pylint: disable=protected-access
        ):
          continue
        fqn = f"{prefix}.{name}" if prefix else name
        try:
          value = weights_loader(fqn)
        except KeyError:
          missing.append(fqn)
          continue
        if value.shape != buf.shape:
          raise ValueError(
              f"Checkpoint tensor for buffer '{fqn}' has shape"
              f" {tuple(value.shape)}, expected {tuple(buf.shape)}."
          )
        buf.copy_(value)
  if missing:
    logging.warning(
        "Persistent buffers not found in the checkpoint keep their init"
        " values: %s",
        missing,
    )
  model._weights_loader = weights_loader  # pylint: disable=protected-access
  return model


def embed_tokens_on_cpu(
    model: nn.Module,
    token_ids: list[int],
    weights_loader: WeightsLoader | None = None,
) -> torch.Tensor:
  """Evaluates `model.get_input_embeddings()` on `token_ids` on CPU.

  Used for constant special-token embeddings (e.g. EOI / EOA) so export does
  not trace a gather over the full vocabulary embedding table. When the model's
  parameters are on the meta device, the requested rows are read via
  `weights_loader` without materializing the full table on the module.

  Args:
    model: The source model.
    token_ids: Token IDs to embed (batch size 1).
    weights_loader: Optional loader; defaults to the one attached by
      `load_persistent_buffers_`.

  Returns:
    A CPU tensor of shape `[1, len(token_ids), embedding_dim]`.

  Raises:
    RuntimeError: If the embedding weights are on the meta device and no
      `weights_loader` is available.
  """
  emb = model.get_input_embeddings()
  ids = torch.tensor([token_ids], dtype=torch.int32)
  with torch.no_grad():
    if not emb.weight.is_meta:
      return emb(ids.to(emb.weight.device)).detach().cpu().clone()
    if weights_loader is None:
      weights_loader = getattr(model, "_weights_loader", None)
    if weights_loader is None:
      raise RuntimeError(
          "Input embedding weights are on the meta device, but no"
          " `weights_loader` is available."
      )
    fqn = next(
        (
            f"{n}.weight" if n else "weight"
            for n, m in model.named_modules()
            if m is emb
        ),
        "embed_tokens.weight",
    )
    rows = weights_loader(fqn)[ids.long()].to(emb.weight.dtype).clone()
    scale = getattr(emb, "embed_scale", None)
    if scale is not None:
      rows = rows * scale.detach().cpu().to(rows.dtype)
    else:
      scalar_scale = getattr(emb, "scalar_embed_scale", None)
      if scalar_scale is not None:
        rows = rows * scalar_scale
    return rows


@contextlib.contextmanager
def buffers_on_meta(
    *roots: nn.Module,
) -> Iterator[dict[str, torch.Tensor]]:
  """Temporarily moves real buffers of `roots` to meta, yielding their values.

  With meta parameters, export runs entirely on the meta device, so real
  (CPU) buffers would cause device mismatches when traced. This swaps them to
  meta for the duration of the context and yields `{fqn: real_tensor}` keyed
  by FQN relative to each root (the names Converter V2 passes to its weights
  loader), so their values can be served via `with_tensors`. Buffers shared
  between roots are swapped once. Originals are restored on exit.

  Args:
    *roots: The modules to be exported.

  Yields:
    Mapping from buffer FQN (relative to its root) to its real value.
  """
  values: dict[str, torch.Tensor] = {}
  meta_of: dict[int, torch.Tensor] = {}  # id(real) -> meta replacement.
  real_of: dict[int, torch.Tensor] = {}  # id(meta replacement) -> real.
  originals: list[tuple[nn.Module, str, torch.Tensor]] = []
  for root in roots:
    for prefix, module in root.named_modules(remove_duplicate=False):
      for name, buf in list(module._buffers.items()):  # pylint: disable=protected-access
        if buf is None:
          continue
        if not buf.is_meta:
          if id(buf) not in meta_of:
            meta_of[id(buf)] = buf.to("meta")
            real_of[id(meta_of[id(buf)])] = buf
          module._buffers[name] = meta_of[id(buf)]  # pylint: disable=protected-access
          originals.append((module, name, buf))
        real = real_of.get(id(module._buffers[name]))  # pylint: disable=protected-access
        if real is not None:
          values[f"{prefix}.{name}" if prefix else name] = real
  try:
    yield values
  finally:
    for module, name, buf in originals:
      module._buffers[name] = buf  # pylint: disable=protected-access


def with_tensors(
    weights_loader: WeightsLoader | None, tensors: dict[str, torch.Tensor]
) -> WeightsLoader | None:
  """Returns a loader that serves `tensors` first, then `weights_loader`."""
  if not tensors or weights_loader is None:
    return weights_loader

  def load(name: str) -> torch.Tensor:
    if name in tensors:
      return tensors[name]
    return weights_loader(name)

  return load


def submodule_weights_loader(
    weights_loader: WeightsLoader,
    root: nn.Module,
    submodule: nn.Module,
    wrapper_attr: str = "model",
) -> WeightsLoader:
  """Resolves names of a module wrapping `submodule` to FQNs within `root`.

  E.g. an exported embedder `Wrapper(model=root.model.embed_tokens)` has a
  parameter named `model.weight`; this maps it to
  `model.embed_tokens.weight`, which the checkpoint loader understands.

  Args:
    weights_loader: Loader that resolves FQNs relative to `root`.
    root: The full model.
    submodule: A module inside `root`.
    wrapper_attr: Attribute under which the export wrapper holds `submodule`.

  Returns:
    A loader for parameter names relative to the export wrapper.
  """
  fqn = next((n for n, m in root.named_modules() if m is submodule), None)
  if fqn is None:
    raise ValueError(f"{type(submodule).__name__} is not a submodule of root.")
  wrapper_prefix = f"{wrapper_attr}."

  def load(name: str) -> torch.Tensor:
    if not name.startswith(wrapper_prefix):
      raise KeyError(name)
    return weights_loader(f"{fqn}.{name.removeprefix(wrapper_prefix)}")

  return load


def _has_all_weights(checkpoint_dir: str) -> bool:
  """Returns whether `checkpoint_dir` holds all of its safetensors files."""
  index_path = os.path.join(checkpoint_dir, _INDEX_FILE)
  if os.path.exists(index_path):
    with open(index_path, "r") as f:
      files = set(json.load(f).get("weight_map", {}).values())
    return bool(files) and all(
        os.path.exists(os.path.join(checkpoint_dir, name)) for name in files
    )
  return bool(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))


class HFCheckpointWeightsLoader:
  """Streams weights on demand from a HuggingFace safetensors checkpoint.

  Used by Converter V2 (`use_v2=True` with `delete_in_memory_params=True`) so
  large checkpoints can be traced on the `meta` device and materialized one
  tensor at a time during serialization. Parameter names of the export-time
  module are matched against checkpoint keys after stripping standard wrapper
  prefixes (`model.`, `language_model.`, ...) plus any `extra_prefixes` from an
  active `ModelExtension`.

  Resolution order in `__call__(param_name)`:
  1. Direct safetensors lookup across `_candidates(param_name)`.
  2. Custom `tensor_resolver(param_name, try_load)` hook from
     `ModelExtension.resolve_checkpoint_tensor` (if provided).
  3. Extension-declared `extra_fused_specs`
     (`ModelExtension.fused_weight_specs`) followed by built-in `qkv_proj` and
     `gate_up_proj` fusions.
  4. Tied `lm_head.weight` -> input embedding weight fallback when
     `tie_word_embeddings=True`.
  """

  def __init__(
      self,
      model_path: str,
      tie_word_embeddings: bool = False,
      extra_prefixes: Sequence[str] = (),
      extra_fused_specs: Sequence[
          tuple[str, Sequence[str]] | tuple[str, Sequence[str], int] | Any
      ] = (),
      tensor_resolver: (
          Callable[
              [str, Callable[[str], torch.Tensor | None]], torch.Tensor | None
          ]
          | None
      ) = None,
      config: Any = None,
      export_config: Any = None,
  ):
    """Initializes the safetensors checkpoint weight loader.

    Args:
      model_path: Local checkpoint directory or HuggingFace repo id.
      tie_word_embeddings: Whether `lm_head` shares the embedding table. Only
        then is a missing `lm_head.weight` served from the embedding table;
        the two have the same shape, so a wrong fallback would go unnoticed.
      extra_prefixes: Additional checkpoint/module prefixes (e.g.
        `('backbone.', 'decoder.')`) from
        `ModelExtension.checkpoint_key_prefixes`.
      extra_fused_specs: Additional `FusedWeightSpec` entries or
        `(fused_name, part_names[, dim])` tuples from
        `ModelExtension.fused_weight_specs` for reconstructing fused projections
        from unfused checkpoint tensors.
      tensor_resolver: Optional `(param_name, try_load) -> Tensor | None`
        callback from `ModelExtension.resolve_checkpoint_tensor` for custom
        slicing, transposition, or key remapping.
      config: Optional HuggingFace model config used for architecture-specific
        weight layout adjustments.
      export_config: Optional export configuration controlling composite weight
        transformations.
    """
    self._tie_word_embeddings = tie_word_embeddings
    self._extra_prefixes = tuple(
        p if p.endswith(".") else f"{p}." for p in extra_prefixes if p
    )
    normalized_extra = tuple(
        (spec[0], tuple(spec[1]), int(spec[2]) if len(spec) > 2 else 0)
        for spec in extra_fused_specs
    )
    self._fused_specs = (*normalized_extra, *_DEFAULT_FUSED_SPECS)
    self._tensor_resolver = tensor_resolver
    self.config = config
    self.export_config = export_config
    self._handles: dict[str, Any] = {}
    self._checkpoint_dir = self._resolve_checkpoint_dir(model_path)
    self._weight_map = self._load_weight_map()
    if not self._weight_map:
      raise ValueError(
          f"No safetensors weights found for {model_path!r} in"
          f" {self._checkpoint_dir}."
      )
    logging.info(
        "HFCheckpointWeightsLoader: %d tensors from %s.",
        len(self._weight_map),
        self._checkpoint_dir,
    )

  @staticmethod
  def _resolve_checkpoint_dir(model_path: str) -> str:
    """Returns a local directory holding the checkpoint."""
    if os.path.isdir(model_path):
      return model_path
    try:
      local_dir = huggingface_hub.snapshot_download(
          model_path,
          allow_patterns=list(_SNAPSHOT_PATTERNS),
          local_files_only=True,
      )
      # The cached snapshot may hold only config / tokenizer files (e.g. after
      # loading the model `from_config`); download the weights in that case.
      if _has_all_weights(local_dir):
        return local_dir
    except hf_errors.LocalEntryNotFoundError:
      pass
    return huggingface_hub.snapshot_download(
        model_path, allow_patterns=list(_SNAPSHOT_PATTERNS)
    )

  def _load_weight_map(self) -> dict[str, str]:
    """Maps tensor names to the safetensors file (relative) that holds them."""
    index_path = os.path.join(self._checkpoint_dir, _INDEX_FILE)
    if os.path.exists(index_path):
      with open(index_path, "r") as f:
        return json.load(f).get("weight_map", {})
    weight_map = {}
    for path in glob.glob(os.path.join(self._checkpoint_dir, "*.safetensors")):
      for key in self._handle(path).keys():
        weight_map[key] = os.path.basename(path)
    return weight_map

  def _handle(self, path: str) -> Any:
    if path not in self._handles:
      self._handles[path] = safetensors.safe_open(
          path, framework="pt", device="cpu"
      )
    return self._handles[path]

  def _get(self, key: str) -> torch.Tensor:
    path = os.path.join(self._checkpoint_dir, self._weight_map[key])
    return self._handle(path).get_tensor(key)

  @staticmethod
  def _strip_prefixes(param_name: str, prefixes: Sequence[str]) -> str:
    """Repeatedly strips leading `prefixes` from `param_name`."""
    stripped = param_name
    while True:
      for prefix in prefixes:
        if stripped.startswith(prefix):
          stripped = stripped.removeprefix(prefix)
          break
      else:
        break
    return stripped

  @staticmethod
  def _candidates(param_name: str) -> list[str]:
    """Returns checkpoint keys `param_name` may correspond to, in order."""
    stripped = HFCheckpointWeightsLoader._strip_prefixes(
        param_name, _WRAPPER_PREFIXES
    )
    candidates = [
        param_name,
        stripped,
        f"model.{stripped}",
        f"model.language_model.{stripped}",
        f"language_model.{stripped}",
        f"transformer.{stripped}",
    ]
    return list(dict.fromkeys(candidates))  # Dedupe, preserving order.

  def _is_qwen3_5(self) -> bool:
    if self.config is None:
      return False
    model_types = {
        getattr(self.config, "model_type", None),
        getattr(getattr(self.config, "text_config", None), "model_type", None),
    }
    return bool(
        model_types
        & {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
    )

  def _get_qwen3_5_partial_rope_perm(self) -> torch.Tensor | None:
    """Returns the Qwen3.5 partial-rotary channel permutation for a head."""
    if not self._is_qwen3_5():
      return None
    cfg = getattr(self.config, "text_config", None) or self.config
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    rope_params = getattr(cfg, "rope_parameters", None)
    partial_factor = 0.25
    if isinstance(rope_params, dict):
      partial_factor = float(
          rope_params.get("partial_rotary_factor", partial_factor)
      )
    elif rope_params is not None:
      partial_factor = float(
          getattr(rope_params, "partial_rotary_factor", partial_factor)
      )
    else:
      partial_factor = float(
          getattr(cfg, "partial_rotary_factor", partial_factor)
      )
    rotary_dim = int(head_dim * partial_factor)
    if not (0 < rotary_dim < head_dim):
      return None
    half_rot = rotary_dim // 2
    half_dim = head_dim // 2
    first_pass_len = half_dim - half_rot
    pass_indices = torch.arange(rotary_dim, head_dim)
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[:half_rot] = torch.arange(half_rot)
    perm[half_rot:half_dim] = pass_indices[:first_pass_len]
    perm[half_dim : half_dim + half_rot] = torch.arange(half_rot, rotary_dim)
    perm[half_dim + half_rot :] = pass_indices[first_pass_len:]
    return perm

  def _try_load(self, param_name: str) -> torch.Tensor | None:
    """Returns the checkpoint tensor matching `param_name`, or `None`.

    For Qwen3.5 with `use_qkv_norm_rope_composite`, `q_norm` / `k_norm` weights
    get the partial-rotary channel permutation used by `odml.qkv_norm_rope`.

    Args:
      param_name: Export-time parameter name to look up.

    Returns:
      The (possibly permuted) checkpoint tensor, or `None` if no candidate key
      is in the checkpoint.
    """
    candidates = self._candidates(param_name)
    if self._extra_prefixes:
      stripped = self._strip_prefixes(
          param_name, (*_WRAPPER_PREFIXES, *self._extra_prefixes)
      )
      candidates = list(
          dict.fromkeys([
              param_name,
              stripped,
              *(f"{prefix}{stripped}" for prefix in self._extra_prefixes),
              *self._candidates(stripped),
          ])
      )
    for candidate in candidates:
      if candidate in self._weight_map:
        tensor = self._get(candidate)
        if (
            self._is_qwen3_5()
            and getattr(
                self.export_config, "use_qkv_norm_rope_composite", False
            )
            and self._strip_prefixes(param_name, _WRAPPER_PREFIXES).endswith(
                (".self_attn.q_norm.weight", ".self_attn.k_norm.weight")
            )
        ):
          perm = self._get_qwen3_5_partial_rope_perm()
          if perm is not None:
            tensor = tensor[perm]
        return tensor
    return None

  def _load_fused(
      self,
      param_name: str,
      fused: str,
      parts: tuple[str, ...],
      dim: int = 0,
  ) -> torch.Tensor | None:
    """Concatenates `parts` along `dim` if `param_name` is a fused projection."""
    head, sep, tail = param_name.rpartition(f"{fused}.")
    if not sep or tail not in ("weight", "bias"):
      return None
    tensors: list[torch.Tensor] = []
    for part in parts:
      tensor = self._try_load(f"{head}{part}.{tail}")
      if tensor is None:
        return None
      tensors.append(tensor)
    if (
        fused == "qkv_proj"
        and len(parts) == 3
        and dim == 0
        and self._is_qwen3_5()
        and getattr(self.export_config, "use_qkv_norm_rope_composite", False)
    ):
      cfg = getattr(self.config, "text_config", None) or self.config
      num_heads = cfg.num_attention_heads
      num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
      head_dim = getattr(cfg, "head_dim", None) or (
          cfg.hidden_size // num_heads
      )
      q_only_size = num_heads * head_dim
      k_size = num_kv_heads * head_dim
      perm = self._get_qwen3_5_partial_rope_perm()
      q_t, k_t, v_t = tensors[0], tensors[1], tensors[2]
      if tail == "weight":
        q_raw_w = q_t.view(num_heads, 2, head_dim, -1)
        q_only_w = q_raw_w[:, 0, :, :].clone()
        gate_w = q_raw_w[:, 1, :, :].reshape(q_only_size, -1)
        k_w = k_t.view(num_kv_heads, head_dim, -1).clone()
        if perm is not None:
          q_only_w = q_only_w[:, perm, :]
          k_w = k_w[:, perm, :]
        return torch.cat(
            [
                q_only_w.reshape(q_only_size, -1),
                k_w.reshape(k_size, -1),
                v_t,
                gate_w,
            ],
            dim=0,
        )
      else:
        q_raw_b = q_t.view(num_heads, 2, head_dim)
        q_only_b = q_raw_b[:, 0, :].clone()
        gate_b = q_raw_b[:, 1, :].reshape(q_only_size)
        k_b = k_t.view(num_kv_heads, head_dim).clone()
        if perm is not None:
          q_only_b = q_only_b[:, perm]
          k_b = k_b[:, perm]
        return torch.cat(
            [
                q_only_b.reshape(q_only_size),
                k_b.reshape(k_size),
                v_t,
                gate_b,
            ],
            dim=0,
        )
    return torch.cat(tensors, dim=dim)

  def __call__(self, param_name: str) -> torch.Tensor:
    tensor = self._try_load(param_name)
    if tensor is None and self._tensor_resolver is not None:
      tensor = self._tensor_resolver(param_name, self._try_load)
    if tensor is None:
      for fused_name, part_names, dim in self._fused_specs:
        tensor = self._load_fused(param_name, fused_name, part_names, dim=dim)
        if tensor is not None:
          break
    if (
        tensor is None
        and self._tie_word_embeddings
        and param_name.endswith("lm_head.weight")
    ):
      # Tied word embeddings: the checkpoint only stores the embedding table.
      for emb_key in (*_EMBEDDING_KEYS, "wte.weight"):
        tensor = self._try_load(emb_key)
        if tensor is not None:
          break
    if tensor is None:
      raise KeyError(
          f"'{param_name}' not found in checkpoint {self._checkpoint_dir}."
      )
    return tensor

  def close(self) -> None:
    """Releases open safetensors handles; they are reopened on demand."""
    self._handles.clear()

  def __enter__(self) -> "HFCheckpointWeightsLoader":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    del exc_type, exc_val, exc_tb
    self.close()
