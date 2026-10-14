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
  return model


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
        then is a missing `lm_head.weight` served from the embedding table;
        the two have the same shape, so a wrong fallback would go unnoticed.
      config: Optional model config used for architecture-specific parameter
        adjustments (e.g. Qwen3.5 zero-centered RMSNorm weights).
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
  def _strip_wrapper_prefixes(param_name: str) -> str:
    stripped = param_name
    while True:
      for prefix in _WRAPPER_PREFIXES:
        if stripped.startswith(prefix):
          stripped = stripped.removeprefix(prefix)
          break
      else:
        break
    return stripped

  @classmethod
  def _candidates(cls, param_name: str) -> list[str]:
    """Returns checkpoint keys `param_name` may correspond to, in order."""
    stripped = cls._strip_wrapper_prefixes(param_name)
    candidates = [
        param_name,
        stripped,
        f"model.{stripped}",
        f"model.language_model.{stripped}",
        f"language_model.{stripped}",
        f"transformer.{stripped}",
    ]
    return list(dict.fromkeys(candidates))  # Dedupe, preserving order.

  def _is_qwen3_5_zero_centered_rmsnorm(self, param_name: str) -> bool:
    if self.config is None:
      return False
    cleaned_name = self._strip_wrapper_prefixes(param_name)
    model_types = {
        getattr(self.config, "model_type", None),
        getattr(getattr(self.config, "text_config", None), "model_type", None),
    }
    if not (
        model_types
        & {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
    ):
      return False
    return cleaned_name == "norm.weight" or cleaned_name.endswith((
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".self_attn.q_norm.weight",
        ".self_attn.k_norm.weight",
    ))

  def _try_load(self, param_name: str) -> torch.Tensor | None:
    for candidate in self._candidates(param_name):
      if candidate in self._weight_map:
        tensor = self._get(candidate)
        if self._is_qwen3_5_zero_centered_rmsnorm(param_name):
          return tensor.to(torch.float32) + 1.0
        return tensor
    return None

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

  def __call__(self, param_name: str) -> torch.Tensor:
    tensor = self._try_load(param_name)
    if tensor is None:
      tensor = self._load_fused(
          param_name, "qkv_proj", ("q_proj", "k_proj", "v_proj")
      )
    if tensor is None:
      tensor = self._load_fused(
          param_name, "gate_up_proj", ("gate_proj", "up_proj")
      )
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
