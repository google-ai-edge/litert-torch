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
"""Quantization layer wrappers and checkpoint helpers for HuggingFace export."""

from collections.abc import Callable, Mapping
from typing import Any

from litert_torch.generative.export_hf.core.quantization_layer import gemma_quant
import torch

WeightsLoader = gemma_quant.WeightsLoader

_QUANT_PARAM_SUFFIXES: dict[str, tuple[str, ...]] = {
    "gemma": gemma_quant.QUANT_PARAM_SUFFIXES,
}
_EXPORT_QUANT_METHODS: dict[str, Callable[..., WeightsLoader]] = {}


def register_export_quant_method(
    quant_method: str,
    prepare_for_export: Callable[..., WeightsLoader],
    param_suffixes: tuple[str, ...] = (),
) -> None:
  """Registers a pre-quantized checkpoint `quant_method` for Converter V2 export."""
  _EXPORT_QUANT_METHODS[quant_method] = prepare_for_export
  if param_suffixes:
    _QUANT_PARAM_SUFFIXES[quant_method] = tuple(param_suffixes)


def _quant_config_to_dict(quantization_config: Any) -> dict[str, Any] | None:
  """Converts a `PretrainedConfig.quantization_config` to a plain dict."""
  if not quantization_config:
    return None
  if isinstance(quantization_config, Mapping):
    return dict(quantization_config)
  to_dict = getattr(quantization_config, "to_dict", None)
  if callable(to_dict):
    return dict(to_dict())
  return None


def get_quantization_config(model_config: Any) -> Any:
  """Extracts `quantization_config` from `model_config` or its `text_config`."""
  if model_config is None:
    return None
  if isinstance(model_config, Mapping):
    qcfg = model_config.get("quantization_config")
    if qcfg:
      return qcfg
    text_cfg = model_config.get("text_config")
    if isinstance(text_cfg, Mapping):
      return text_cfg.get("quantization_config")
    return None
  qcfg = getattr(model_config, "quantization_config", None)
  if qcfg:
    return qcfg
  text_cfg = getattr(model_config, "text_config", None)
  if text_cfg is not None:
    qcfg = getattr(text_cfg, "quantization_config", None)
    if qcfg:
      return qcfg
  get_text_config = getattr(model_config, "get_text_config", None)
  if callable(get_text_config):
    try:
      text_cfg = get_text_config(decoder=True)
    except TypeError:
      text_cfg = get_text_config()
    return getattr(text_cfg, "quantization_config", None)
  return None


def is_supported_quantization_config(quantization_config: Any) -> bool:
  """Returns whether `quantization_config` is supported by `export_hf`."""
  cfg = _quant_config_to_dict(quantization_config)
  if not cfg:
    return False
  quant_method = cfg.get("quant_method")
  return isinstance(quant_method, str) and quant_method in _EXPORT_QUANT_METHODS


def prepare_quantized_model_for_export(
    model: torch.nn.Module,
    quantization_config: Any,
    weights_loader: WeightsLoader,
) -> WeightsLoader:
  """Prepares a meta-device model with `quantization_config` for V2 export."""
  cfg = _quant_config_to_dict(quantization_config)
  if not cfg:
    raise ValueError("Missing quantization_config.")
  quant_method = cfg.get("quant_method")
  prepare_fn = (
      _EXPORT_QUANT_METHODS.get(quant_method)
      if isinstance(quant_method, str)
      else None
  )
  if prepare_fn is None:
    raise NotImplementedError(
        f"Quantization method {quant_method!r} is not supported yet."
    )
  return prepare_fn(model, cfg, weights_loader)


def get_quant_param_suffixes(
    quant_method: str | None = None,
) -> tuple[str, ...]:
  """Returns parameter name suffixes holding quantization scales/zero-points."""
  if quant_method is None:
    return ()
  return _QUANT_PARAM_SUFFIXES.get(quant_method, ())


def load_quant_scales_(
    model: torch.nn.Module,
    weights_loader: WeightsLoader,
    quant_method: str | None = "gemma",
) -> torch.nn.Module:
  """Loads quantization scale/metadata parameters for `quant_method` onto CPU."""
  if quant_method == "gemma":
    return gemma_quant.load_gemma_quant_scales_(model, weights_loader)
  return model
