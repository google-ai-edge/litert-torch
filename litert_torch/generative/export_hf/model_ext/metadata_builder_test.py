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
"""Unit tests verifying serialization and defaults for LiteRT executor metadata."""

import types
from absl.testing import absltest
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.model_ext import metadata_builder
import transformers


class MetadataBuilderTest(absltest.TestCase):

  def _create_source_artifacts(self, num_layers: int = 2):
    config = transformers.PretrainedConfig()
    config.model_type = "gemma"
    config.num_hidden_layers = num_layers
    return export_lib.SourceModelArtifacts(
        model=types.SimpleNamespace(config=config),
        model_config=config,
        text_model_config=config,
        tokenizer=types.SimpleNamespace(),
    )

  def test_build_executor_metadata_npu_flags(self):
    source_artifacts = self._create_source_artifacts()
    export_config = exportable_module.ExportableModuleConfig(
        model="dummy",
        enable_neon_for_npu_greedy_sampling=True,
        use_hw_masking_for_npu=False,
        use_hw_cache_update_for_npu=True,
        use_hw_ple_for_npu=False,
    )
    metadata = metadata_builder.build_executor_metadata(
        source_artifacts,
        export_config,
        export_lib.ExportedModelArtifacts(),
    )
    self.assertTrue(metadata.HasField("llm_npu_executor_metadata"))
    npu_meta = metadata.llm_npu_executor_metadata
    self.assertTrue(npu_meta.enable_neon_for_npu_greedy_sampling)
    self.assertFalse(npu_meta.use_hw_masking_for_npu)
    self.assertTrue(npu_meta.use_hw_cache_update_for_npu)
    self.assertFalse(npu_meta.use_hw_ple_for_npu)

  def test_build_executor_metadata_ring_buffer_defaults_hw_cache_update(self):
    source_artifacts = self._create_source_artifacts()
    export_config = exportable_module.ExportableModuleConfig(
        model="dummy",
        sliding_window_ring_buffer_size=512,
    )
    metadata = metadata_builder.build_executor_metadata(
        source_artifacts,
        export_config,
        export_lib.ExportedModelArtifacts(),
    )
    self.assertTrue(metadata.HasField("llm_npu_executor_metadata"))
    self.assertTrue(
        metadata.llm_npu_executor_metadata.use_hw_cache_update_for_npu
    )

  def test_build_executor_metadata_split_cache_populates_npu_metadata(self):
    source_artifacts = self._create_source_artifacts()
    export_config = exportable_module.ExportableModuleConfig(
        model="dummy",
        split_cache=True,
    )
    metadata = metadata_builder.build_executor_metadata(
        source_artifacts,
        export_config,
        export_lib.ExportedModelArtifacts(),
    )
    self.assertTrue(metadata.HasField("llm_npu_executor_metadata"))


if __name__ == "__main__":
  absltest.main()
