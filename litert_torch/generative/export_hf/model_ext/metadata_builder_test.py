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
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.model_ext import metadata_builder
import transformers

# Expected NPU executor metadata for every supported model type.
_ALL_HW_COMPONENTS = {
    "enable_neon_for_npu_greedy_sampling": True,
    "use_hw_masking_for_npu": True,
    "use_hw_cache_update_for_npu": True,
    "use_hw_ple_for_npu": True,
}


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


if __name__ == "__main__":
  absltest.main()
