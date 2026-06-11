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
"""Monkey patches for extended modules."""

import contextlib
from typing import Any, Callable, Dict

# A map of model_type -> context manager factory
_CONTEXT_REGISTRY: Dict[str, Callable[[], Any]] = {}

# A map of model_type -> context manager factory that takes
# (model, export_config)
_MODEL_PATCH_REGISTRY: Dict[str, Callable[[Any, Any], Any]] = {}


def register_patch(model_types):
  """Decorator that registers a context manager for one or more architectures.

  Args:
    model_types: A single string or a list/tuple of strings representing the
      model types to register the patch for.

  Returns:
    A decorator that registers the context manager for the given model types.

  Usage: @register_patch(["gemma3", "gemma3_text"])
  """

  def wrapper(func: Callable[[], Any]):
    types = [model_types] if isinstance(model_types, str) else model_types
    for mt in types:
      _CONTEXT_REGISTRY[mt] = func
    return func

  return wrapper


def register_model_patch(model_types):
  """Decorator that registers a patch context manager for a model instance.

  Args:
    model_types: A single string or a list/tuple of strings representing the
      model types to register the model patch for.

  Returns:
    A decorator that registers the context manager for the given model types.
  """

  def wrapper(func: Callable[[Any, Any], Any]):
    types = [model_types] if isinstance(model_types, str) else model_types
    for mt in types:
      _MODEL_PATCH_REGISTRY[mt] = func
    return func

  return wrapper


def get_patch_context(model_type: str):
  """Retrieves the registered context manager or a no-op."""
  # Logic to ensure the model-specific submodule is loaded
  if model_type not in _CONTEXT_REGISTRY:
    try:
      __import__(
          f"litert_torch.generative.export_hf.model_ext.{model_type}.patch"
      )
    except ImportError:
      return contextlib.nullcontext()

  patch_cm = _CONTEXT_REGISTRY.get(model_type)
  return patch_cm() if patch_cm else contextlib.nullcontext()


def patch_model(model: Any, model_type: str, export_config: Any):
  """Retrieves the registered model patch context manager or a no-op."""
  if model_type not in _MODEL_PATCH_REGISTRY:
    try:
      __import__(
          f"litert_torch.generative.export_hf.model_ext.{model_type}.patch"
      )
    except ImportError:
      return contextlib.nullcontext()

  patch_cm = _MODEL_PATCH_REGISTRY.get(model_type)
  if patch_cm:
    return patch_cm(model, export_config)
  return contextlib.nullcontext()
