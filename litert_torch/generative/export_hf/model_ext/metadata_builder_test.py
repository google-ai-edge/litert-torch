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
"""Tests for the executor metadata builder."""

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import metadata_builder
import transformers

from absl.testing import absltest as googletest


class MetadataBuilderTest(googletest.TestCase):

  def test_mamba2_and_stateless_layers(self):
    build_executor_metadata = metadata_builder.build_executor_metadata
    if build_executor_metadata is None:
      self.skipTest("litert_lm_builder is not available.")
    config = transformers.NemotronHConfig(
        hidden_size=64,
        layers_block_type=[
            "linear_attention",
            "mlp",
            "full_attention",
            "moe",
        ],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        mamba_num_heads=8,
        mamba_head_dim=16,
        ssm_state_size=16,
        n_groups=2,
    )
    # The executor metadata is built from the config alone.
    source_model_artifacts = export_lib.SourceModelArtifacts(
        model=None,  # pyrefly: ignore[bad-argument-type]
        model_config=config,
        text_model_config=config,
        tokenizer=None,  # pyrefly: ignore[bad-argument-type]
    )
    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy", cache_length=64
    )

    executor_metadata = build_executor_metadata(
        source_model_artifacts,
        export_config,
        export_lib.ExportedModelArtifacts(),
    )

    llm_metadata = executor_metadata.llm_executor_metadata
    state_buffer_type = (
        metadata_builder.executor_metadata_pb2.StateBuffer
    )
    self.assertEqual(
        [
            (buffer.prefill_input_name, buffer.type)
            for buffer in llm_metadata.state_buffers
        ],
        [
            ("kv_cache_c_0", state_buffer_type.TYPE_LINEAR_ATTENTION),
            ("kv_cache_r_0", state_buffer_type.TYPE_LINEAR_ATTENTION),
            ("kv_cache_k_2", state_buffer_type.TYPE_GLOBAL_KEY_CACHE),
            ("kv_cache_v_2", state_buffer_type.TYPE_GLOBAL_VALUE_CACHE),
        ],
    )
    # Linear attention states cannot be rolled back.
    self.assertEqual(llm_metadata.max_history_size, 0)

  def test_hybrid_layers(self):
    build_executor_metadata = metadata_builder.build_executor_metadata
    if build_executor_metadata is None:
      self.skipTest("litert_lm_builder is not available.")
    config = transformers.Zamba2Config(
        hidden_size=64,
        num_hidden_layers=3,
        layers_block_type=["linear_attention", "hybrid", "linear_attention"],
        num_attention_heads=4,
        n_mamba_heads=8,
        mamba_d_state=16,
    )
    source_model_artifacts = export_lib.SourceModelArtifacts(
        model=None,  # pyrefly: ignore[bad-argument-type]
        model_config=config,
        text_model_config=config,
        tokenizer=None,  # pyrefly: ignore[bad-argument-type]
    )
    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy", cache_length=64
    )

    executor_metadata = build_executor_metadata(
        source_model_artifacts,
        export_config,
        export_lib.ExportedModelArtifacts(),
    )

    llm_metadata = executor_metadata.llm_executor_metadata
    state_buffer_type = (
        metadata_builder.executor_metadata_pb2.StateBuffer
    )
    # A hybrid layer has a KV cache and the Mamba2 states at its index.
    self.assertEqual(
        [
            (buffer.prefill_input_name, buffer.type)
            for buffer in llm_metadata.state_buffers
        ],
        [
            ("kv_cache_c_0", state_buffer_type.TYPE_LINEAR_ATTENTION),
            ("kv_cache_r_0", state_buffer_type.TYPE_LINEAR_ATTENTION),
            ("kv_cache_k_1", state_buffer_type.TYPE_GLOBAL_KEY_CACHE),
            ("kv_cache_v_1", state_buffer_type.TYPE_GLOBAL_VALUE_CACHE),
            ("kv_cache_c_1", state_buffer_type.TYPE_LINEAR_ATTENTION),
            ("kv_cache_r_1", state_buffer_type.TYPE_LINEAR_ATTENTION),
            ("kv_cache_c_2", state_buffer_type.TYPE_LINEAR_ATTENTION),
            ("kv_cache_r_2", state_buffer_type.TYPE_LINEAR_ATTENTION),
        ],
    )
    key_buffer = llm_metadata.state_buffers[2]
    self.assertEqual(key_buffer.sequence_axis, export_config.k_ts_idx)
    self.assertEqual(key_buffer.maximum_sequence_length, 64)
    self.assertEqual(llm_metadata.max_history_size, 0)

    # FalconH1: every layer is a hybrid layer.
    config = transformers.FalconH1Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_d_ssm=128,
        mamba_n_heads=8,
        mamba_d_state=16,
    )
    source_model_artifacts = export_lib.SourceModelArtifacts(
        model=None,  # pyrefly: ignore[bad-argument-type]
        model_config=config,
        text_model_config=config,
        tokenizer=None,  # pyrefly: ignore[bad-argument-type]
    )
    executor_metadata = build_executor_metadata(
        source_model_artifacts,
        export_config,
        export_lib.ExportedModelArtifacts(),
    )
    self.assertEqual(
        [
            buffer.prefill_input_name
            for buffer in executor_metadata.llm_executor_metadata.state_buffers
        ],
        [
            "kv_cache_k_0",
            "kv_cache_v_0",
            "kv_cache_c_0",
            "kv_cache_r_0",
            "kv_cache_k_1",
            "kv_cache_v_1",
            "kv_cache_c_1",
            "kv_cache_r_1",
        ],
    )


if __name__ == "__main__":
  googletest.main()
