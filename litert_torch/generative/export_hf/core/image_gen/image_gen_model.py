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
"""Base class of ImageGen models to be converted to LiteRT."""

import abc
import os

import huggingface_hub
from litert_torch.generative.export_hf.core import exportable_module_config

from litert_lm_builder.runtime.proto import image_gen_metadata_pb2


class ImageGenModel(abc.ABC):
  """Base class for ImageGen models."""

  def __init__(
      self,
      model_path: str,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ):
    self.model_path = model_path
    self.model_dir = (
        model_path
        if os.path.exists(model_path)
        else huggingface_hub.snapshot_download(model_path)
    )
    self.export_config = export_config

  @abc.abstractmethod
  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports ImageGen model components to LiteRT TFLite files."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_image_gen_metadata(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> image_gen_metadata_pb2.ImageGenMetadata:
    """Returns ImageGenMetadata proto populated from the model."""
    raise NotImplementedError
