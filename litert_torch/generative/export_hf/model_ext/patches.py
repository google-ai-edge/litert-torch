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
"""Runtime patch registries and context dispatchers for `export_hf` models.

`export_hf` applies architecture-specific patches at two distinct phases:

1. **Load-time class patches** (`get_patch_context` / `@register_patch` /
   `ModelExtension.patch_context`): Entered around `from_pretrained` or
   `from_config` so module classes on a `transformers.models.<arch>` module
   (such as `RMSNorm` or attention functions) are swapped before the model tree
   is instantiated.
2. **Trace-time model patches** (`patch_model` / `@register_model_patch` /
   `ModelExtension.patch_model`): Entered around `torch.export` tracing on an
   already-instantiated `(model, export_config)` pair so child modules can be
   replaced with fused QKV/MLP or HLFB composite wrappers and restored cleanly
   afterward.

Both decorator-registered built-in patches (e.g. `model_ext/gemma3/patch.py`,
`model_ext/qwen3/patch.py`, `model_ext/lfm2/patch.py`) and `ModelExtension`
hook methods are entered together via `contextlib.ExitStack`. New out-of-tree
extensions can implement `ModelExtension.patch_context` and
`ModelExtension.patch_model` directly on their `ModelExtension` subclass.
"""

import contextlib
from typing import Any, Callable, Dict

import torch

from litert_torch.generative.export_hf.model_ext import extension as extension_lib

# A map of model_type -> context manager factory
_CONTEXT_REGISTRY: Dict[str, Callable[[], Any]] = {}

# A map of model_type -> context manager factory that takes
# (model, export_config)
_MODEL_PATCH_REGISTRY: Dict[str, Callable[[Any, Any], Any]] = {}


def register_patch(model_types):
  """Decorator that registers a load-time class patch context manager.

  See `gemma3_litert_patch` in `model_ext/gemma3/patch.py` and
  `qwen3_litert_patch` in `model_ext/qwen3/patch.py` for built-in examples.

  Args:
    model_types: A single `model_type` string or sequence of `model_type`
      strings to register the patch context manager for.

  Returns:
    A decorator that registers the zero-arg context manager factory for
    `model_types`.
  """

  def wrapper(func: Callable[[], Any]):
    types = [model_types] if isinstance(model_types, str) else model_types
    for mt in types:
      _CONTEXT_REGISTRY[mt] = func
    return func

  return wrapper


def register_model_patch(model_types):
  """Decorator that registers a trace-time `(model, export_config)` patch.

  See `patch_gemma3_model` in `model_ext/gemma3/patch.py` and
  `patch_qwen3_model` in `model_ext/qwen3/patch.py` for built-in examples.

  Args:
    model_types: A single `model_type` string or sequence of `model_type`
      strings to register the model-instance patch context manager for.

  Returns:
    A decorator that registers the `(model, export_config)` context manager
    factory for `model_types`.
  """

  def wrapper(func: Callable[[Any, Any], Any]):
    types = [model_types] if isinstance(model_types, str) else model_types
    for mt in types:
      _MODEL_PATCH_REGISTRY[mt] = func
    return func

  return wrapper


@contextlib.contextmanager
def get_patch_context(model_type: str, export_config: Any = None):
  """Enters registered and `ModelExtension` load-time patches for `model_type`.

  Combines any `@register_patch` context for `model_type` with
  `ext.patch_context()` from the resolved `ModelExtension` (if any), or acts as
  a no-op when neither is registered.

  Args:
    model_type: Target `model_type` string being loaded.
    export_config: Optional `ExportableModuleConfig` used to resolve custom
      extensions.

  Yields:
    Control to the caller while class-level load-time patches are active.
  """
  ext = extension_lib.resolve_model_extension(model_type, export_config)
  if model_type not in _CONTEXT_REGISTRY:
    try:
      __import__(
          f"litert_torch.generative.export_hf.model_ext.{model_type}.patch"
      )
    except ImportError:
      pass

  patch_cm = _CONTEXT_REGISTRY.get(model_type)
  with contextlib.ExitStack() as stack:
    if patch_cm is not None:
      stack.enter_context(patch_cm())
    if ext is not None:
      stack.enter_context(ext.patch_context())
    yield


@contextlib.contextmanager
def patch_model(model: Any, model_type: str, export_config: Any):
  """Enters registered and `ModelExtension` instance patches around tracing.

  Combines any `@register_model_patch` context for `model_type` with
  `ext.patch_model(model, export_config)` from the resolved `ModelExtension`
  (if any), restoring original submodules on exit.

  Args:
    model: Instantiated `torch.nn.Module` or task wrapper to patch before
      tracing.
    model_type: Target `model_type` string.
    export_config: Active `ExportableModuleConfig` controlling fusions and
      composites.

  Yields:
    Control to the caller while `model` instance patches are active.
  """
  ext = extension_lib.resolve_model_extension(model_type, export_config)
  if model_type not in _MODEL_PATCH_REGISTRY:
    try:
      __import__(
          f"litert_torch.generative.export_hf.model_ext.{model_type}.patch"
      )
    except ImportError:
      pass

  patch_cm = _MODEL_PATCH_REGISTRY.get(model_type)
  with contextlib.ExitStack() as stack:
    if patch_cm is not None:
      stack.enter_context(patch_cm(model, export_config))
    if ext is not None:
      stack.enter_context(ext.patch_model(model, export_config))
    yield


def maybe_ones_final_scale(elide, size, device):
  """Returns `None` to elide RMSNorm's final scale, else an all-ones tensor.

  Gemma4's `final_scale` is always all-ones, so the multiply is a no-op;
  tracing it emits an identical constant per call site that some NPU backends
  merge and fail to compile. `None` lets a trace-time patch skip it.
  """
  if elide:
    return None
  return torch.ones((size,), dtype=torch.float32, device=device)


@contextlib.contextmanager
def elide_final_scale_during_trace(model, norm_type, enabled):
  """Flips `elide_final_scale=True` on matching norms, restoring on exit.

  Walks `model.modules()` so it reaches `norm_type` instances that fused
  modules reuse by reference. No-op when `enabled` is falsy.
  """
  patched = []
  if enabled:
    for module in model.modules():
      if isinstance(module, norm_type):
        patched.append((module, module.elide_final_scale))
        module.elide_final_scale = True
  try:
    yield
  finally:
    for module, original in patched:
      module.elide_final_scale = original
