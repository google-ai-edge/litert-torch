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

from collections.abc import Callable, Iterator
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


@contextlib.contextmanager
def materialize_params_on_cpu(
    *roots: nn.Module,
    weights_loader: WeightsLoader,
) -> Iterator[None]:
  """Temporarily loads meta parameters of `roots` onto CPU from `weights_loader`.

  Used when exporting a submodule with Converter V1 while the parent model's
  parameters live on the meta device (Converter V2). Shared parameters are
  materialized once, and all parameters are restored to meta on exit so CPU
  memory is released immediately after conversion.

  Args:
    *roots: The modules whose meta parameters should be materialized on CPU.
    weights_loader: Loader resolving parameter FQNs relative to each root.

  Yields:
    None.
  """
  materialized: dict[int, nn.Parameter] = {}
  originals: list[tuple[nn.Module, str, nn.Parameter]] = []
  with torch.no_grad():
    for root in roots:
      for prefix, module in root.named_modules(remove_duplicate=False):
        for name, param in list(module._parameters.items()):  # pylint: disable=protected-access
          if param is None or not param.is_meta:
            continue
          if id(param) not in materialized:
            fqn = f"{prefix}.{name}" if prefix else name
            value = None
            for candidate in (fqn, f"model.{fqn}", f"model.model.{fqn}"):
              try:
                value = weights_loader(candidate)
                break
              except KeyError:
                continue
            if value is None:
              value = weights_loader(fqn)
            materialized[id(param)] = nn.Parameter(
                value.to(dtype=param.dtype, device="cpu"),
                requires_grad=param.requires_grad,
            )
          originals.append((module, name, param))
          module._parameters[name] = materialized[id(param)]  # pylint: disable=protected-access
  try:
    yield
  finally:
    for module, name, param in originals:
      module._parameters[name] = param  # pylint: disable=protected-access


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

  if hasattr(weights_loader, "close"):
    load.close = weights_loader.close  # type: ignore[attr-defined]
  if hasattr(weights_loader, "checkpoint_dir"):
    load.checkpoint_dir = weights_loader.checkpoint_dir  # type: ignore[attr-defined]
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

  if hasattr(weights_loader, "close"):
    load.close = weights_loader.close  # type: ignore[attr-defined]
  if hasattr(weights_loader, "checkpoint_dir"):
    load.checkpoint_dir = weights_loader.checkpoint_dir  # type: ignore[attr-defined]
  return load


def _flatten_expert_weight(weight: torch.Tensor) -> torch.Tensor:
  """Converts [experts, out_channels, in_channels] to OHWI delegate layout."""
  num_experts, out_channels, in_channels = weight.shape
  return (
      weight.permute(1, 0, 2)
      .contiguous()
      .reshape(out_channels, num_experts, 1, in_channels)
  )


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

  Parameter names of the export-time module are matched against checkpoint
  keys after stripping common wrapper prefixes. Fused QKV / gate-up
  projections created by `fuse_qkv` / `fuse_gate_up` patches and tied
  `lm_head` weights are reconstructed from their checkpoint components.
  """

  def __init__(
      self,
      model_path: str,
      tie_word_embeddings: bool = False,
      config: Any = None,
  ):
    """Initializes the loader.

    Args:
      model_path: Local checkpoint directory or HuggingFace repo id.
      tie_word_embeddings: Whether `lm_head` shares the embedding table. Only
        then is a missing `lm_head.weight` served from the embedding table; the
        two have the same shape, so a wrong fallback would go unnoticed.
      config: Optional model config used for architecture-specific parameter
        adjustments.
    """
    self._tie_word_embeddings = tie_word_embeddings
    self.config = config
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

  @property
  def checkpoint_dir(self) -> str:
    return self._checkpoint_dir

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
  def _candidates(param_name: str) -> list[str]:
    """Returns checkpoint keys `param_name` may correspond to, in order."""
    stripped = param_name
    while True:
      for prefix in _WRAPPER_PREFIXES:
        if stripped.startswith(prefix):
          stripped = stripped.removeprefix(prefix)
          break
      else:
        break
    candidates = [
        param_name,
        stripped,
        f"model.{stripped}",
        f"model.language_model.{stripped}",
        f"language_model.{stripped}",
        f"transformer.{stripped}",
    ]
    return list(dict.fromkeys(candidates))  # Dedupe, preserving order.

  def _has_candidate(self, param_name: str) -> bool:
    return any(c in self._weight_map for c in self._candidates(param_name))

  def _try_load(self, param_name: str) -> torch.Tensor | None:
    for candidate in self._candidates(param_name):
      if candidate in self._weight_map:
        return self._get(candidate)
    if param_name.endswith(".self_attn.sinks"):
      return self._try_load(
          param_name.removesuffix(".sinks") + ".softmax_offset"
      )
    return None

  def _stack_per_expert_weights(
      self, head: str, proj: str
  ) -> torch.Tensor | None:
    """Stacks `{head}{i}.{proj}.weight` across experts `i = 0, 1, ...`."""
    first = self._try_load(f"{head}0.{proj}.weight")
    if first is None:
      return None
    tensors = [first]
    idx = 1
    while True:
      w = self._try_load(f"{head}{idx}.{proj}.weight")
      if w is None:
        break
      tensors.append(w)
      idx += 1
    return torch.stack(tensors, dim=0)

  def _load_fused(
      self, param_name: str, fused: str, parts: tuple[str, ...]
  ) -> torch.Tensor | None:
    """Concatenates `parts` along dim 0 if `param_name` is a fused projection."""
    head, sep, tail = param_name.rpartition(f"{fused}.")
    if not sep or tail not in ("weight", "bias"):
      return None
    tensors = [self._try_load(f"{head}{part}.{tail}") for part in parts]
    if any(t is None for t in tensors):
      return None
    return torch.cat(tensors, dim=0)

  def _load_moe(self, param_name: str) -> torch.Tensor | None:
    """Reconstructs flattened (`litert_moe`) or split (`litert_moe_sequential`) MoE expert weights."""
    if param_name.endswith("flattened_gate_weight"):
      head = param_name.removesuffix("flattened_gate_weight")
      gate_up = self._try_load(f"{head}gate_up_proj")
      if gate_up is not None:
        gate_w, _ = gate_up.chunk(2, dim=1)
        return _flatten_expert_weight(gate_w)
      gate_w = self._try_load(f"{head}gate_proj")
      if gate_w is not None:
        return _flatten_expert_weight(gate_w)
      gate_w = self._stack_per_expert_weights(head, "gate_proj")
      if gate_w is not None:
        return _flatten_expert_weight(gate_w)
      return None

    if param_name.endswith("flattened_ff1_weight"):
      head = param_name.removesuffix("flattened_ff1_weight")
      gate_up = self._try_load(f"{head}gate_up_proj")
      if gate_up is not None:
        _, ff1_w = gate_up.chunk(2, dim=1)
        return _flatten_expert_weight(ff1_w)
      ff1_w = self._try_load(f"{head}up_proj")
      if ff1_w is not None:
        return _flatten_expert_weight(ff1_w)
      ff1_w = self._stack_per_expert_weights(head, "up_proj")
      if ff1_w is not None:
        return _flatten_expert_weight(ff1_w)
      return None

    if param_name.endswith("flattened_linear_weight"):
      head = param_name.removesuffix("flattened_linear_weight")
      down_w = self._try_load(f"{head}down_proj")
      if down_w is not None:
        return _flatten_expert_weight(down_w)
      down_w = self._stack_per_expert_weights(head, "down_proj")
      if down_w is not None:
        return _flatten_expert_weight(down_w)
      return None

    if param_name.endswith("per_expert_scale"):
      head = param_name.removesuffix("per_expert_scale")
      for attr in ("down_proj", "gate_up_proj", "gate_proj"):
        expert_w = self._try_load(f"{head}{attr}")
        if expert_w is not None:
          return torch.ones((1, 1, 1, expert_w.shape[0]), dtype=torch.float32)
      if self._has_candidate(f"{head}0.down_proj.weight"):
        num_experts = 1
        while self._has_candidate(f"{head}{num_experts}.down_proj.weight"):
          num_experts += 1
        return torch.ones((1, 1, 1, num_experts), dtype=torch.float32)
      return None

    for split_attr, source_attr in (
        ("split_gate_up_proj", "gate_up_proj"),
        ("split_down_proj", "down_proj"),
        ("split_gate_up_bias", "gate_up_bias"),
        ("split_down_bias", "down_bias"),
    ):
      head, sep, tail = param_name.rpartition(f"{split_attr}.")
      if sep and tail.isdigit():
        source = self._try_load(f"{head}{source_attr}")
        if source is not None:
          return source[int(tail)].contiguous()
        if split_attr == "split_gate_up_proj":
          gate = self._try_load(f"{head}{tail}.gate_proj.weight")
          up = self._try_load(f"{head}{tail}.up_proj.weight")
          if gate is not None and up is not None:
            return torch.cat([gate, up], dim=0).contiguous()
        elif split_attr == "split_down_proj":
          down = self._try_load(f"{head}{tail}.down_proj.weight")
          if down is not None:
            return down.contiguous()
        return None

    return None

  def __call__(self, param_name: str) -> torch.Tensor:
    tensor = self._try_load(param_name)
    if tensor is None:
      tensor = self._load_fused(
          param_name, "qkv_proj", ("q_proj", "k_proj", "v_proj")
      )
    if tensor is None:
      # Gemma-4 global attention layers with `attention_k_eq_v=True` have no
      # `v_proj`, so `qkv_proj` only fuses `q_proj` and `k_proj`.
      tensor = self._load_fused(param_name, "qkv_proj", ("q_proj", "k_proj"))
    if tensor is None:
      tensor = self._load_fused(
          param_name, "gate_up_proj", ("gate_proj", "up_proj")
      )
    if tensor is None:
      tensor = self._load_moe(param_name)
    if (
        tensor is None
        and self._tie_word_embeddings
        and param_name.endswith("lm_head.weight")
    ):
      # Tied word embeddings: the checkpoint only stores the embedding table.
      key = next((k for k in _EMBEDDING_KEYS if k in self._weight_map), None)
      if key is not None:
        tensor = self._get(key)
    if (
        tensor is None
        and param_name.endswith("inv_freq")
        and self.config is not None
    ):
      cfg = getattr(self.config, "text_config", None) or self.config
      if (
          cfg is not None
          and hasattr(cfg, "hidden_size")
          and hasattr(cfg, "num_attention_heads")
      ):
        rope_params = getattr(cfg, "rope_parameters", None) or {}
        if not isinstance(rope_params, dict):
          rope_params = {}
        partial_rotary_factor = float(
            rope_params.get(
                "partial_rotary_factor",
                getattr(cfg, "partial_rotary_factor", 1.0),
            )
        )
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        base = float(
            rope_params.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
        )
        tensor = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
    if tensor is None:
      raise KeyError(
          f"'{param_name}' not found in checkpoint {self._checkpoint_dir}."
      )
    return tensor

  def close(self) -> None:
    """Releases open safetensors handles; they are reopened on demand."""
    self._handles.clear()
