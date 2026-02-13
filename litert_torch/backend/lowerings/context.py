# Copyright 2024 The LiteRT Torch Authors.
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
"""Define context object for export and MLIR lowerings."""

import abc
import dataclasses
from typing import TypeVar
from ai_edge_litert.mlir import ir
import torch


T = TypeVar("T", bound="LoweringContextPlugin")


@dataclasses.dataclass
class LoweringContext:
  """The context object used in export interpreter and MLIR lowerings."""

  ir_context: ir.Context
  ir_module: ir.Module
  ir_location: ir.Location | None = None
  node: torch.fx.Node | None = None

  # Use a private registry for plugins
  _plugins: dict[type["LoweringContextPlugin"], "LoweringContextPlugin"] = (
      dataclasses.field(default_factory=dict)
  )

  @property
  def ctx(self) -> ir.Context:
    return self.ir_context

  @property
  def loc(self) -> ir.Location:
    return self.ir_location or ir.Location.unknown(self.ctx)

  def add_plugin(self, plugin: "LoweringContextPlugin"):
    """Explicitly register a plugin instance."""
    self._plugins[type(plugin)] = plugin

  def has_plugin(self, plugin_cls: type[T]) -> bool:
    """Check if a plugin is registered in the context."""
    return plugin_cls in self._plugins

  def get_plugin(self, plugin_cls: type[T]) -> T:
    """Retrieve a plugin with full type-hinting support.

    Args:
      plugin_cls: The class of the plugin to retrieve.

    Returns:
      The registered plugin instance.

    Raises:
      KeyError: If the plugin of the given class is not registered.
    """
    plugin = self._plugins.get(plugin_cls)
    if plugin is None:
      raise KeyError(f"Plugin {plugin_cls.__name__} not registered in context.")
    return plugin


class LoweringContextPlugin(abc.ABC):
  """Base class for context payloads (e.g., Quantization state, Debug info)."""

  pass
