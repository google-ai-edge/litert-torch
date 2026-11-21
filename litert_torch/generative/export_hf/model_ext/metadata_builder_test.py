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
from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.model_ext import metadata_builder
import transformers

from litert_lm_builder.runtime.proto import executor_metadata_pb2

# Expected NPU executor metadata for every supported model type.
_ALL_HW_COMPONENTS = {
    "enable_neon_for_npu_greedy_sampling": True,
    "use_hw_masking_for_npu": True,
    "use_hw_cache_update_for_npu": True,
    "use_hw_ple_for_npu": True,
}

_UNSET = object()


def _make_config(model_type: str, num_layers: int = 2):
  config = transformers.PretrainedConfig()
  config.model_type = model_type
  config.num_hidden_layers = num_layers
  return config


def _create_source_artifacts(
    model_type: str = "gemma", text_model_type: str | None = None
) -> export_lib.SourceModelArtifacts:
  """Creates source artifacts, optionally with a multimodal text config."""
  config = _make_config(model_type)
  text_config = config
  if text_model_type is not None:
    text_config = _make_config(text_model_type)
    config.text_config = text_config
  return export_lib.SourceModelArtifacts(
      model=types.SimpleNamespace(config=config),
      model_config=config,
      text_model_config=text_config,
      tokenizer=types.SimpleNamespace(),
  )


def _create_sliding_window_source_artifacts() -> (
    export_lib.SourceModelArtifacts
):
  """Creates source artifacts whose layers alternate sliding/full attention."""
  source_artifacts = _create_source_artifacts()
  config = source_artifacts.model_config
  config.layer_types = ["sliding_attention", "full_attention"]
  config.sliding_window = 512
  return source_artifacts


def _build(
    source_artifacts: export_lib.SourceModelArtifacts | None = None,
    **config_kwargs: Any,
):
  # NPU executor metadata is only written for split-cache (NPU) exports.
  config_kwargs.setdefault("split_cache", True)
  export_config = exportable_module.ExportableModuleConfig(
      model="dummy", **config_kwargs
  )
  return metadata_builder.build_executor_metadata(
      source_artifacts or _create_source_artifacts(),
      export_config,
      export_lib.ExportedModelArtifacts(),
  )


def _npu_fields(executor_metadata) -> dict[str, bool]:
  """Returns the NPU executor fields that are explicitly set."""
  npu_metadata = executor_metadata.llm_npu_executor_metadata
  return {
      field: getattr(npu_metadata, field)
      for field in metadata_builder.NPU_EXECUTOR_SETTING_FIELDS
      if npu_metadata.HasField(field)
  }


def _make_attention_mask_inputs(
    *,
    has_vision: bool,
    use_bidirectional_attention: Any = _UNSET,
    sliding_window: int = 512,
    ring_buffer_size: int = 512,
) -> tuple[
    export_lib.SourceModelArtifacts,
    exportable_module.ExportableModuleConfig,
    export_lib.ExportedModelArtifacts,
]:
  text_config = types.SimpleNamespace(
      sliding_window=sliding_window,
      num_hidden_layers=2,
      layer_types=["sliding_attention", "full_attention"],
  )
  if use_bidirectional_attention is not _UNSET:
    text_config.use_bidirectional_attention = use_bidirectional_attention

  vision_config = types.SimpleNamespace() if has_vision else None
  model_config = types.SimpleNamespace(
      text_config=text_config,
      vision_config=vision_config,
  )
  source_model_artifacts = cast(
      export_lib.SourceModelArtifacts,
      types.SimpleNamespace(
          model_config=model_config,
          text_model_config=text_config,
      ),
  )
  export_config = cast(
      exportable_module.ExportableModuleConfig,
      types.SimpleNamespace(
          cache_length=4096,
          sliding_window_ring_buffer_size=ring_buffer_size,
          k_ts_idx=2,
          v_ts_idx=3,
          split_cache=False,
      ),
  )
  exported_model_artifacts = cast(
      export_lib.ExportedModelArtifacts,
      types.SimpleNamespace(),
  )
  return source_model_artifacts, export_config, exported_model_artifacts


# TODO(b/542876170): Remove the skip once litert-torch requires a
# litert-lm-builder release that includes `LlmNpuExecutorMetadata`.
@absltest.skipUnless(
    metadata_builder.HAS_NPU_EXECUTOR_METADATA,
    "Installed litert_lm executor_metadata proto predates"
    " LlmNpuExecutorMetadata.",
)
class MetadataBuilderTest(parameterized.TestCase):

  def test_explicit_npu_flags(self):
    metadata = _build(
        enable_neon_for_npu_greedy_sampling=True,
        use_hw_masking_for_npu=False,
        use_hw_cache_update_for_npu=True,
        use_hw_ple_for_npu=False,
    )
    self.assertEqual(
        _npu_fields(metadata),
        {
            "enable_neon_for_npu_greedy_sampling": True,
            "use_hw_masking_for_npu": False,
            "use_hw_cache_update_for_npu": True,
            "use_hw_ple_for_npu": False,
        },
    )

  def test_explicit_user_setting_overrides_model_default(self):
    metadata = _build(
        litert_lm_model_type_override="gemma3",
        use_hw_ple_for_npu=False,
    )
    self.assertEqual(
        _npu_fields(metadata),
        _ALL_HW_COMPONENTS | {"use_hw_ple_for_npu": False},
    )

  def test_unknown_model_has_no_npu_metadata(self):
    metadata = _build()
    self.assertFalse(metadata.HasField("llm_npu_executor_metadata"))

  def test_no_npu_metadata_without_split_cache(self):
    metadata = _build(
        split_cache=False,
        litert_lm_model_type_override="gemma3",
        use_hw_masking_for_npu=True,
    )
    self.assertFalse(metadata.HasField("llm_npu_executor_metadata"))

  def test_ring_buffer_sets_hw_cache_update_for_unknown_model(self):
    metadata = _build(
        _create_sliding_window_source_artifacts(),
        sliding_window_ring_buffer_size=512,
    )
    self.assertEqual(
        _npu_fields(metadata), {"use_hw_cache_update_for_npu": True}
    )

  def test_explicit_user_setting_overrides_ring_buffer(self):
    metadata = _build(
        _create_sliding_window_source_artifacts(),
        sliding_window_ring_buffer_size=512,
        use_hw_cache_update_for_npu=False,
    )
    self.assertEqual(
        _npu_fields(metadata), {"use_hw_cache_update_for_npu": False}
    )

  def test_ring_buffer_size_without_sliding_layers_is_not_a_ring_buffer(self):
    # Full-attention models have no ringbuffer even if the size is set.
    metadata = _build(sliding_window_ring_buffer_size=512)
    self.assertFalse(metadata.HasField("llm_npu_executor_metadata"))

  def test_combined_layers_state_buffers(self):
    # Two sliding layers (ring buffer) and one global layer: one K and one V
    # state buffer per group instead of one per layer.
    config = transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=8,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
    )
    # A model type without NPU defaults, so that only the ring buffer can turn
    # on the handwritten cache update.
    config.model_type = "unknown_ring_model"
    source_artifacts = export_lib.SourceModelArtifacts(
        model=types.SimpleNamespace(config=config),
        model_config=config,
        text_model_config=config,
        tokenizer=types.SimpleNamespace(),
    )
    metadata = _build(
        source_artifacts,
        split_cache_combined_layers=True,
        cache_length=128,
        sliding_window_ring_buffer_size=32,
    )
    buffers = {
        b.prefill_input_name: b
        for b in metadata.llm_executor_metadata.state_buffers
    }
    self.assertCountEqual(
        buffers,
        ["kv_cache_k_0", "kv_cache_v_0", "kv_cache_k_1", "kv_cache_v_1"],
    )
    state_buffer = metadata_builder.executor_metadata_pb2.StateBuffer
    self.assertEqual(
        buffers["kv_cache_k_0"].type, state_buffer.TYPE_LOCAL_KEY_CACHE
    )
    self.assertEqual(
        buffers["kv_cache_v_0"].type, state_buffer.TYPE_LOCAL_VALUE_CACHE
    )
    self.assertEqual(buffers["kv_cache_k_0"].maximum_sequence_length, 32)
    self.assertEqual(buffers["kv_cache_k_0"].minimum_sequence_length, 8)
    self.assertEqual(
        buffers["kv_cache_k_1"].type, state_buffer.TYPE_GLOBAL_KEY_CACHE
    )
    self.assertEqual(
        buffers["kv_cache_v_1"].type, state_buffer.TYPE_GLOBAL_VALUE_CACHE
    )
    self.assertEqual(buffers["kv_cache_v_1"].maximum_sequence_length, 128)
    self.assertEqual(buffers["kv_cache_k_1"].sequence_axis, 2)
    self.assertEqual(buffers["kv_cache_v_1"].sequence_axis, 3)
    # The sliding groups are ring buffers, which need the handwritten cache
    # update path, as with per-layer caches.
    self.assertTrue(_npu_fields(metadata)["use_hw_cache_update_for_npu"])

  def test_sliding_layers_without_ring_buffer_size_is_not_a_ring_buffer(self):
    # Without a ringbuffer size, sliding layers are exported as full caches.
    metadata = _build(_create_sliding_window_source_artifacts())
    self.assertFalse(metadata.HasField("llm_npu_executor_metadata"))

  @parameterized.named_parameters(
      ("gemma3", "gemma3"),
      ("gemma3n", "gemma3n"),
      ("gemma4", "gemma4"),
      ("qwen3", "qwen3"),
      ("minicpm5", "minicpm5"),
  )
  def test_defaults_from_model_type_override(self, model_type):
    metadata = _build(litert_lm_model_type_override=model_type)
    self.assertEqual(_npu_fields(metadata), _ALL_HW_COMPONENTS)

  def test_model_type_override_matching_is_exact(self):
    # `litert_lm_builder` writes GenericModel for "Gemma3", so no defaults.
    metadata = _build(litert_lm_model_type_override="Gemma3")
    self.assertFalse(metadata.HasField("llm_npu_executor_metadata"))

  @parameterized.named_parameters(
      ("gemma3_text_only", "gemma3_text", None),
      ("gemma3_multimodal", "gemma3", "gemma3_text"),
      ("gemma3n_multimodal", "gemma3n", "gemma3n_text"),
      ("gemma4_multimodal", "gemma4", "gemma4_text"),
      ("gemma4_unified", "gemma4_unified", "gemma4_unified_text"),
      ("qwen3", "qwen3", None),
      ("qwen3_asr", "qwen3_asr", "qwen3"),
  )
  def test_defaults_from_hf_config(self, model_type, text_model_type):
    metadata = _build(_create_source_artifacts(model_type, text_model_type))
    self.assertEqual(_npu_fields(metadata), _ALL_HW_COMPONENTS)

  def test_unsupported_override_is_authoritative(self):
    metadata = _build(
        _create_source_artifacts("gemma3", "gemma3_text"),
        litert_lm_model_type_override="function_gemma",
    )
    self.assertFalse(metadata.HasField("llm_npu_executor_metadata"))


class AttentionMaskMetadataBuilderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="e2b_like",
          has_vision=True,
          use_bidirectional_attention=None,
          expected_local="ATTENTION_MASK_TYPE_CAUSAL",
      ),
      dict(
          testcase_name="false_like",
          has_vision=True,
          use_bidirectional_attention=False,
          expected_local="ATTENTION_MASK_TYPE_CAUSAL",
      ),
      dict(
          testcase_name="31b_like",
          has_vision=True,
          use_bidirectional_attention="vision",
          expected_local="ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL",
      ),
      dict(
          testcase_name="gemma3_like",
          has_vision=True,
          use_bidirectional_attention=_UNSET,
          expected_local="ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL",
      ),
      dict(
          testcase_name="all_bidirectional",
          has_vision=True,
          use_bidirectional_attention="all",
          expected_local="ATTENTION_MASK_TYPE_BIDIRECTIONAL",
      ),
      dict(
          testcase_name="true_bidirectional",
          has_vision=True,
          use_bidirectional_attention=True,
          expected_local="ATTENTION_MASK_TYPE_BIDIRECTIONAL",
      ),
      dict(
          testcase_name="text_only",
          has_vision=False,
          use_bidirectional_attention=_UNSET,
          expected_local="ATTENTION_MASK_TYPE_CAUSAL",
      ),
  )
  def test_global_attention_mask_type_stays_causal_in_every_case(
      self,
      has_vision: bool,
      use_bidirectional_attention: Any,
      expected_local: str,
  ):
    assert executor_metadata_pb2 is not None
    assert metadata_builder.build_executor_metadata is not None
    source_artifacts, export_config, exported_artifacts = (
        _make_attention_mask_inputs(
            has_vision=has_vision,
            use_bidirectional_attention=use_bidirectional_attention,
        )
    )
    if use_bidirectional_attention is _UNSET:
      self.assertFalse(
          hasattr(
              source_artifacts.model_config.text_config,
              "use_bidirectional_attention",
          )
      )
    if not has_vision:
      self.assertIsNone(source_artifacts.model_config.vision_config)
    executor_metadata = metadata_builder.build_executor_metadata(
        source_artifacts, export_config, exported_artifacts
    )
    mask_settings = (
        executor_metadata.llm_executor_metadata.attention_mask_settings
    )
    self.assertEqual(
        mask_settings.attention_mask_type,
        executor_metadata_pb2.ATTENTION_MASK_TYPE_CAUSAL,
    )
    self.assertEqual(
        mask_settings.local_attention_mask_type,
        getattr(executor_metadata_pb2, expected_local),
    )
    self.assertEqual(mask_settings.sliding_window_size, 512)

  @parameterized.named_parameters(
      dict(testcase_name="unknown_string", value="unknown"),
      dict(testcase_name="invalid_int", value=123),
  )
  def test_unknown_use_bidirectional_attention_raises_value_error(self, value):
    assert metadata_builder.build_executor_metadata is not None
    source_artifacts, export_config, exported_artifacts = (
        _make_attention_mask_inputs(
            has_vision=True,
            use_bidirectional_attention=value,
        )
    )
    with self.assertRaises(ValueError):
      metadata_builder.build_executor_metadata(
          source_artifacts, export_config, exported_artifacts
      )


class NpuExecutorSettingsTest(absltest.TestCase):
  """Tests that don't depend on the executor_metadata proto version."""

  def test_get_npu_executor_settings_for_unknown_model_type(self):
    self.assertEmpty(
        metadata_builder.get_npu_executor_settings_for_model_type("unknown")
    )
    self.assertEmpty(
        metadata_builder.get_npu_executor_settings_for_model_type(None)
    )

  def test_get_npu_executor_settings_returns_copy(self):
    settings = metadata_builder.get_npu_executor_settings_for_model_type(
        "gemma3"
    )
    settings["use_hw_masking_for_npu"] = False
    self.assertTrue(
        metadata_builder.get_npu_executor_settings_for_model_type("gemma3")[
            "use_hw_masking_for_npu"
        ]
    )


class StateBufferTest(absltest.TestCase):
  """Tests the state buffers that `build_executor_metadata` adds per layer."""

  def _build_executor_metadata(self, config: transformers.PretrainedConfig):
    build_executor_metadata = metadata_builder.build_executor_metadata
    if build_executor_metadata is None:
      self.skipTest("litert_lm_builder is not available.")
    # The executor metadata is built from the config alone.
    source_model_artifacts = export_lib.SourceModelArtifacts(
        model=None,  # pyrefly: ignore[bad-argument-type]
        model_config=config,
        text_model_config=config,
        tokenizer=None,  # pyrefly: ignore[bad-argument-type]
    )
    export_config = exportable_module.ExportableModuleConfig(
        model="dummy", cache_length=64
    )
    return build_executor_metadata(
        source_model_artifacts,
        export_config,
        export_lib.ExportedModelArtifacts(),
    )

  def test_mamba2_and_stateless_layers(self):
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

    executor_metadata = self._build_executor_metadata(config)

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

  def test_unsupported_layer_type_raises(self):
    # The stateless layer types are skipped; any other unknown type still
    # raises instead of being exported without its state buffers.
    config = _make_config("dummy", num_layers=3)
    config.layer_types = ["mlp", "moe", "unknown_layer"]
    with self.assertRaisesRegex(
        ValueError, "Unsupported layer type: unknown_layer"
    ):
      self._build_executor_metadata(config)


if __name__ == "__main__":
  absltest.main()
