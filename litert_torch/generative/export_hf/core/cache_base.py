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
"""Base class for cache."""

import abc
from typing import Any, Dict
from litert_torch.generative.export_hf.core import exportable_module_config
from transformers import cache_utils

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig


class LiteRTLMCacheLayerMixin(cache_utils.CacheLayerMixin, abc.ABC):
  """Optimized Cache layer class mixin for HuggingFace integration."""

  @abc.abstractmethod
  def get_batch_size(self) -> int:
    """Returns the batch size of the cache."""
    ...

  @abc.abstractmethod
  def get_k_ts_idx(self) -> int:
    """Returns the index of the sequence dimension in K cache."""
    ...

  @abc.abstractmethod
  def get_v_ts_idx(self) -> int:
    """Returns the index of the sequence dimension in V cache."""
    ...

  def set_cache_runtime_args(self, cache_runtime_args: Dict[str, Any]) -> None:
    """Sets the cache runtime arguments."""
    setattr(self, "cache_runtime_args", cache_runtime_args)

  def get_cache_runtime_args(self) -> Dict[str, Any]:
    """Returns the cache runtime arguments."""
    return getattr(self, "cache_runtime_args", {})

  @classmethod
  @abc.abstractmethod
  def create_from_config(
      cls,
      model_config,
      layer_index,
      export_config: ExportableModuleConfig,
      **kwargs
  ) -> "LiteRTLMCacheLayerMixin":
    ...


class LiteRTLMCacheMixin(cache_utils.Cache, abc.ABC):
  """Optimized Cache class mixin for HuggingFace integration."""

  @classmethod
  @abc.abstractmethod
  def create_from_config(
      cls, model_config, export_config: ExportableModuleConfig, **kwargs
  ) -> "LiteRTLMCacheMixin":
    """Creates a KV cache from the model config."""
    ...

  def set_cache_runtime_args(self, cache_runtime_args: Dict[str, Any]) -> None:
    """Sets the cache runtime arguments."""
    for layer in self.layers:
      assert isinstance(layer, LiteRTLMCacheLayerMixin)
      layer.set_cache_runtime_args(cache_runtime_args)

  def to(self, *args, **kwargs) -> "LiteRTLMCacheMixin":
    """Moves all cache layers to the specified device/dtype."""
    for layer in self.layers:
      if hasattr(layer, "to"):
        layer.to(*args, **kwargs)
    return self


CACHE_REGISTRY: dict[str, type[LiteRTLMCacheMixin]] = {}


def register_cache_implementation(cls):
  """Decorator to register a cache implementation."""
  CACHE_REGISTRY[cls.__name__] = cls
  return cls
