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
"""Reusable test harness for validating ModelExtension implementations."""

import types
from typing import Any

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import extension
import torch
import transformers

from litert_lm_builder.runtime.proto import llm_metadata_pb2


class ModelExtensionTestMixin:
  """Mixin for testing custom or built-in `ModelExtension` implementations.

  Usage:
    class MyExtensionTest(ModelExtensionTestMixin, googletest.TestCase):
      extension_cls = MyExtension

      def get_sample_model_config(self):
        return transformers.PretrainedConfig(model_type="my_arch")
  """

  extension_cls: (
      type[extension.ModelExtension] | extension.ModelExtension | None
  ) = None

  def get_extension(self) -> extension.ModelExtension:
    """Returns the `ModelExtension` instance under test."""
    if self.extension_cls is None:
      raise NotImplementedError(
          'Subclasses of ModelExtensionTestMixin must set `extension_cls` or'
          ' override `get_extension()`.'
      )
    if isinstance(self.extension_cls, type) and issubclass(
        self.extension_cls, extension.ModelExtension
    ):
      return self.extension_cls()
    if isinstance(self.extension_cls, extension.ModelExtension):
      return self.extension_cls
    raise TypeError(f'Invalid extension_cls: {self.extension_cls!r}')

  def get_sample_model_config(self) -> transformers.PretrainedConfig:
    """Returns a sample `PretrainedConfig` for testing hooks."""
    ext = self.get_extension()
    model_type = ext.model_types[0] if ext.model_types else 'test_model'
    return transformers.PretrainedConfig.from_dict({
        'model_type': model_type,
        'num_hidden_layers': 2,
        'hidden_size': 16,
        'num_attention_heads': 2,
        'num_key_value_heads': 2,
        'head_dim': 8,
        'vocab_size': 64,
    })

  def get_sample_export_config(
      self,
  ) -> exportable_module_config.ExportableModuleConfig:
    """Returns a sample `ExportableModuleConfig` for testing hooks."""
    return exportable_module_config.ExportableModuleConfig(
        model='dummy',
        externalize_embedder=True,
    )

  def get_sample_model(self) -> torch.nn.Module:
    """Returns a minimal `torch.nn.Module` for testing `patch_model`."""
    return torch.nn.Linear(16, 16)

  def test_extension_instance_and_model_types_contract(self):
    ext = self.get_extension()
    self.assertIsInstance(ext, extension.ModelExtension)  # pyrefly: ignore[missing-attribute]
    for mt in ext.model_types:
      self.assertIsInstance(mt, str)  # pyrefly: ignore[missing-attribute]
      self.assertTrue(mt)  # pyrefly: ignore[missing-attribute]

  def test_extension_update_export_config_contract(self):
    ext = self.get_extension()
    model_cfg = self.get_sample_model_config()
    export_cfg = self.get_sample_export_config()
    updated = ext.update_export_config(export_cfg, model_cfg)
    self.assertIsInstance(  # pyrefly: ignore[missing-attribute]
        updated, exportable_module_config.ExportableModuleConfig
    )

  def test_extension_patch_contexts_are_reentrant(self):
    ext = self.get_extension()
    model_cfg = self.get_sample_model_config()
    export_cfg = ext.update_export_config(
        self.get_sample_export_config(), model_cfg
    )
    sample_model = self.get_sample_model()
    with ext.patch_context():
      with ext.patch_model(sample_model, export_cfg):
        pass
    # Verify a second entry/exit cycle also succeeds cleanly.
    with ext.patch_context():
      with ext.patch_model(sample_model, export_cfg):
        pass

  def test_extension_exportables_contracts(self):
    ext = self.get_extension()
    model_cfg = self.get_sample_model_config()
    export_cfg = ext.update_export_config(
        self.get_sample_export_config(), model_cfg
    )

    pd = ext.get_prefill_decode_exportables(model_cfg, export_cfg)
    if pd is not None:
      self.assertIsInstance(pd, tuple)  # pyrefly: ignore[missing-attribute]
      self.assertLen(pd, 2)  # pyrefly: ignore[missing-attribute]

    vis = ext.get_vision_exportables(model_cfg, export_cfg)
    if vis is not None:
      self.assertLen(tuple(vis), 3)  # pyrefly: ignore[missing-attribute]
      self.assertIsNotNone(tuple(vis)[0])  # pyrefly: ignore[missing-attribute]

    aud = ext.get_audio_exportables(model_cfg, export_cfg)
    if aud is not None:
      self.assertLen(tuple(aud), 3)  # pyrefly: ignore[missing-attribute]
      self.assertIsNotNone(tuple(aud)[0])  # pyrefly: ignore[missing-attribute]

    add = ext.get_additional_exportables(model_cfg, export_cfg)
    self.assertIsInstance(add, dict)  # pyrefly: ignore[missing-attribute]

  def test_extension_metadata_hooks_contract(self):
    ext = self.get_extension()
    model_cfg = self.get_sample_model_config()
    export_cfg = ext.update_export_config(
        self.get_sample_export_config(), model_cfg
    )

    fake_source: Any = types.SimpleNamespace(
        model=self.get_sample_model(),
        model_config=model_cfg,
        text_model_config=model_cfg,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
    )
    fake_exported: Any = types.SimpleNamespace(
        prefill_decode_model_path=None,
        embedder_model_path=None,
        vision_encoder_model_path=None,
        vision_adapter_model_path=None,
        eoi_model_path=None,
        audio_encoder_model_path=None,
        audio_adapter_model_path=None,
        eoa_model_path=None,
        auxiliary_model_path=None,
        tokenizer_model_path=None,
        additional_model_paths=None,
        litert_lm_model_path=None,
    )
    llm_meta = llm_metadata_pb2.LlmMetadata()
    updated_meta = ext.build_llm_metadata(
        fake_source, export_cfg, fake_exported, llm_meta
    )
    self.assertIsInstance(updated_meta, llm_metadata_pb2.LlmMetadata)  # pyrefly: ignore[missing-attribute]
