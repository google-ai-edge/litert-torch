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
"""Tests for pluggable ModelExtension architecture and registry."""

import contextlib
import dataclasses
import sys
import types
from unittest import mock

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.model_ext import builtin_extensions
from litert_torch.generative.export_hf.model_ext import exportables
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext import metadata_builder
from litert_torch.generative.export_hf.model_ext import patches
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
import transformers

from absl.testing import absltest as googletest
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig
ModelExtension = extension.ModelExtension
VisionExportablesSpec = extension.VisionExportablesSpec
AudioExportablesSpec = extension.AudioExportablesSpec


class _DummyPrefill:
  pass


class _DummyDecode:
  pass


class _DummyVisionEncoder:
  pass


class _DummyVisionAdapter:
  pass


class _DummyEOI:
  pass


class _DummyAudioEncoder:
  pass


class _DummyAudioAdapter:
  pass


class _DummyEOA:
  pass


class _DummyExtraSubmodel:
  pass


class _DummySpeechModel:
  pass


class _DummyTtsModel:
  pass


class _DummyImageGenModel:
  pass


class ModelExtensionTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self._saved_registry = dict(extension._EXTENSION_REGISTRY)
    self._saved_entry_points_loaded = extension._ENTRY_POINTS_LOADED

  def tearDown(self):
    extension._EXTENSION_REGISTRY.clear()
    extension._EXTENSION_REGISTRY.update(self._saved_registry)
    extension._ENTRY_POINTS_LOADED = self._saved_entry_points_loaded
    super().tearDown()

  def test_register_and_lookup_model_extension(self):

    @extension.register_model_extension(['custom_arch', 'custom_arch_text'])
    class CustomArchExtension(ModelExtension):

      def get_prefill_decode_exportables(self, model_config, export_config):
        return (_DummyPrefill, _DummyDecode)

    ext = extension.get_model_extension('custom_arch')
    self.assertIsInstance(ext, CustomArchExtension)
    self.assertIs(ext, extension.get_model_extension('custom_arch_text'))
    self.assertIn('custom_arch', extension.registered_model_types())

    model_cfg = transformers.PretrainedConfig(model_type='custom_arch')
    exp_cfg = ExportableModuleConfig(model='dummy')
    self.assertEqual(
        exportables.get_prefill_decode_exportables(model_cfg, exp_cfg),
        (_DummyPrefill, _DummyDecode),
    )

    removed = extension.unregister_model_extension('custom_arch')
    self.assertIs(removed, ext)
    self.assertNotIn('custom_arch', extension.registered_model_types())

  def test_direct_model_extension_in_export_config(self):
    events: list[str] = []

    class InlineExtension(ModelExtension):

      def update_export_config(self, export_config, model_config):
        events.append(f'update_config:{model_config.model_type}')
        return dataclasses.replace(export_config, externalize_embedder=True)

      @contextlib.contextmanager
      def patch_context(self):
        events.append('enter_patch_context')
        try:
          yield
        finally:
          events.append('exit_patch_context')

      @contextlib.contextmanager
      def patch_model(self, model, export_config):
        events.append(f'enter_patch_model:{model}')
        try:
          yield
        finally:
          events.append('exit_patch_model')

      def get_vision_exportables(self, model_config, export_config=None):
        return VisionExportablesSpec(
            encoder_cls=_DummyVisionEncoder,
            adapter_cls=_DummyVisionAdapter,
            eoi_cls=_DummyEOI,
        )

      def get_audio_exportables(self, model_config, export_config=None):
        return AudioExportablesSpec(
            encoder_cls=_DummyAudioEncoder,
            adapter_cls=_DummyAudioAdapter,
            eoa_cls=_DummyEOA,
        )

      def get_additional_exportables(self, model_config, export_config=None):
        return {'custom_submodel': _DummyExtraSubmodel}

      def get_llm_model_type_proto(self, model_config, export_config):
        return llm_model_type_pb2.LlmModelType(qwen3=llm_model_type_pb2.Qwen3())

      def build_llm_metadata(
          self,
          source_model_artifacts,
          export_config,
          exported_model_artifacts,
          llm_metadata,
      ):
        llm_metadata.max_num_tokens = 777
        return llm_metadata

    model_cfg = transformers.PretrainedConfig(model_type='inline_test_arch')
    exp_cfg = ExportableModuleConfig(
        model='dummy',
        model_extension=InlineExtension(),
    )
    # Verify __repr__ works with arbitrary object in model_extension.
    self.assertIn('InlineExtension', repr(exp_cfg))

    updated_cfg = extension.update_export_config(exp_cfg, model_cfg)
    self.assertTrue(updated_cfg.externalize_embedder)

    with patches.get_patch_context('inline_test_arch', updated_cfg):
      events.append('inside_load')
    with patches.patch_model('fake_model', 'inline_test_arch', updated_cfg):
      events.append('inside_export')

    self.assertEqual(
        events,
        [
            'update_config:inline_test_arch',
            'enter_patch_context',
            'inside_load',
            'exit_patch_context',
            'enter_patch_model:fake_model',
            'inside_export',
            'exit_patch_model',
        ],
    )

    self.assertEqual(
        exportables.get_vision_exportables(model_cfg, updated_cfg),
        (_DummyVisionEncoder, _DummyVisionAdapter, _DummyEOI),
    )
    self.assertEqual(
        exportables.get_audio_exportables(model_cfg, updated_cfg),
        (_DummyAudioEncoder, _DummyAudioAdapter, _DummyEOA),
    )
    self.assertEqual(
        exportables.get_additional_exportables(model_cfg, updated_cfg),
        {'custom_submodel': _DummyExtraSubmodel},
    )

    fake_model = types.SimpleNamespace(config=model_cfg, generation_config=None)
    fake_tokenizer = types.SimpleNamespace(
        bos_token=None,
        pad_token=None,
        pad_token_id=None,
        eos_token='</s>',
        chat_template=None,
    )
    source_artifacts = export_lib.SourceModelArtifacts(
        model=fake_model,  # pyrefly: ignore[bad-argument-type]
        model_config=model_cfg,
        text_model_config=model_cfg,
        tokenizer=fake_tokenizer,  # pyrefly: ignore[bad-argument-type]
    )
    exported_artifacts = export_lib.ExportedModelArtifacts()
    llm_meta = litert_lm_builder.build_llm_metadata(
        source_artifacts,
        updated_cfg,
        ((None, None), (None, None), (None, None)),
        exported_artifacts,
    )
    self.assertTrue(llm_meta.llm_model_type.HasField('qwen3'))
    self.assertEqual(llm_meta.max_num_tokens, 777)

  def test_composite_extension_and_registered_override_priority(self):
    order: list[str] = []

    @extension.register_model_extension('comp_arch')
    class BaseExtension(ModelExtension):

      @contextlib.contextmanager
      def patch_context(self):
        order.append('base_enter')
        try:
          yield
        finally:
          order.append('base_exit')

      def get_prefill_decode_exportables(self, model_config, export_config):
        return (_DummyPrefill, _DummyDecode)

      def get_additional_exportables(self, model_config, export_config=None):
        return {'sub_a': _DummyPrefill, 'sub_b': _DummyDecode}

      def build_llm_metadata(
          self,
          source_model_artifacts,
          export_config,
          exported_model_artifacts,
          llm_metadata,
      ):
        llm_metadata.max_num_tokens = 100
        return llm_metadata

    class UserOverrideExtension(ModelExtension):

      @contextlib.contextmanager
      def patch_context(self):
        order.append('override_enter')
        try:
          yield
        finally:
          order.append('override_exit')

      def get_vision_exportables(self, model_config, export_config=None):
        return (_DummyVisionEncoder, None, None)

      def get_additional_exportables(self, model_config, export_config=None):
        return {'sub_b': _DummyExtraSubmodel, 'sub_c': _DummyVisionAdapter}

      def build_llm_metadata(
          self,
          source_model_artifacts,
          export_config,
          exported_model_artifacts,
          llm_metadata,
      ):
        llm_metadata.max_num_tokens += 50
        return llm_metadata

    model_cfg = transformers.PretrainedConfig(model_type='comp_arch')
    exp_cfg = ExportableModuleConfig(
        model='dummy',
        model_extension=UserOverrideExtension(),
    )

    with patches.get_patch_context('comp_arch', exp_cfg):
      order.append('body')
    self.assertEqual(
        order,
        ['base_enter', 'override_enter', 'body', 'override_exit', 'base_exit'],
    )

    # Inherited from BaseExtension since UserOverrideExtension returns None.
    self.assertEqual(
        exportables.get_prefill_decode_exportables(model_cfg, exp_cfg),
        (_DummyPrefill, _DummyDecode),
    )
    # Provided by UserOverrideExtension.
    self.assertEqual(
        exportables.get_vision_exportables(model_cfg, exp_cfg),
        (_DummyVisionEncoder, None, None),
    )
    # Merged dict with UserOverrideExtension taking precedence on 'sub_b'.
    self.assertEqual(
        exportables.get_additional_exportables(model_cfg, exp_cfg),
        {
            'sub_a': _DummyPrefill,
            'sub_b': _DummyExtraSubmodel,
            'sub_c': _DummyVisionAdapter,
        },
    )

    builder_fn = metadata_builder.get_metadata_builder(model_cfg, exp_cfg)
    meta = llm_metadata_pb2.LlmMetadata()
    meta = builder_fn(None, exp_cfg, None, meta)
    self.assertEqual(meta.max_num_tokens, 150)

  def test_load_extension_modules_via_config(self):
    mod_name = '_fake_user_litert_ext_mod'
    fake_mod = types.ModuleType(mod_name)

    @extension.register_model_extension('dynamic_mod_arch')
    class DynamicModExtension(ModelExtension):

      def get_speech_model_cls(self, model_type):
        return _DummySpeechModel

    # Unregister first, then attach registration inside the fake module import.
    extension.unregister_model_extension('dynamic_mod_arch')

    def _register_on_import():
      extension.register_model_extension('dynamic_mod_arch')(
          DynamicModExtension
      )

    fake_mod._register_on_import = _register_on_import  # pylint: disable=protected-access
    sys.modules[mod_name] = fake_mod
    try:
      with mock.patch.object(
          extension.importlib,
          'import_module',
          side_effect=lambda name: _register_on_import()
          if name == mod_name
          else None,
      ):
        exp_cfg = ExportableModuleConfig(
            model='dummy',
            extension_modules=f' {mod_name} , ',
        )
        self.assertEqual(exp_cfg.extension_modules, [mod_name])
        cls = exportables.get_speech_model_cls('dynamic_mod_arch', exp_cfg)
        self.assertIs(cls, _DummySpeechModel)
    finally:
      sys.modules.pop(mod_name, None)

  def test_entry_point_discovery_for_oss_wheels(self):

    class WheelPluginExtension(ModelExtension):
      model_types = ('oss_wheel_arch',)

      def get_tts_model_cls(self, model_type):
        return _DummyTtsModel

      @classmethod
      def is_matching_tts_checkpoint(cls, config_dict, model_path):
        return config_dict.get('custom_tts_flag') is True

      def get_image_gen_model_cls(self, model_type):
        return _DummyImageGenModel

    fake_ep = mock.MagicMock()
    fake_ep.name = 'oss_wheel_arch'
    fake_ep.load.return_value = WheelPluginExtension

    extension._ENTRY_POINTS_LOADED = False
    with mock.patch.object(
        extension.importlib_metadata,
        'entry_points',
        return_value=[fake_ep],
    ) as mock_eps:
      inferred = exportables.infer_tts_model_type(
          {'custom_tts_flag': True}, '/tmp/fake_ckpt'
      )
      self.assertEqual(inferred, 'oss_wheel_arch')
      self.assertIs(
          exportables.get_tts_model_cls('oss_wheel_arch'), _DummyTtsModel
      )
      self.assertIs(
          exportables.get_image_gen_model_cls('oss_wheel_arch'),
          _DummyImageGenModel,
      )
      mock_eps.assert_called_once_with(group=extension.ENTRY_POINT_GROUP)

  def test_builtin_extensions_registered(self):
    registered = extension.registered_model_types()
    for expected in (
        'gemma3',
        'gemma3n',
        'gemma4',
        'qwen3',
        'qwen3_5',
        'lfm2',
        'lfm2_vl',
        'parakeet_ctc',
        'whisper',
        'qwen3_asr',
        'qwen3_tts',
        'kokoro',
        'bonsai_flux2',
        'flux2_klein',
        'minicpmo',
    ):
      self.assertIn(expected, registered)

  def test_non_text_generation_checkpoint_matching_and_section_types(self):

    class CustomDiffusionExtension(ModelExtension):
      model_types = ('custom_diffusion',)

      def get_image_gen_model_cls(self, model_type):
        return _DummyImageGenModel

      def get_litert_lm_section_types(self):
        return {'custom_denoiser': 'tf_lite_custom_denoiser'}

      @classmethod
      def match_checkpoint(cls, model_path, task=None, config_dict=None):
        return (
            str(task) == 'text_to_image'
            and config_dict is not None
            and config_dict.get('_class_name') == 'CustomDiffusionPipeline'
        )

    exp_cfg = ExportableModuleConfig(
        model='dummy',
        model_extension=CustomDiffusionExtension(),
    )
    inferred = exportables.infer_model_type(
        model_path='/tmp/custom_diff',
        task='text_to_image',
        config_dict={'_class_name': 'CustomDiffusionPipeline'},
        export_config=exp_cfg,
    )
    self.assertEqual(inferred, 'custom_diffusion')
    ext = extension.resolve_model_extension('custom_diffusion', exp_cfg)
    self.assertIsNotNone(ext)
    self.assertEqual(
        ext.get_litert_lm_section_types(),
        {'custom_denoiser': 'tf_lite_custom_denoiser'},
    )


class BuiltinQwen3ExtensionContractTest(
    ext_testing.ModelExtensionTestMixin, googletest.TestCase
):
  extension_cls = builtin_extensions.Qwen3Extension


class BuiltinQwen35ExtensionContractTest(
    ext_testing.ModelExtensionTestMixin, googletest.TestCase
):
  extension_cls = builtin_extensions.Qwen3_5Extension


class BuiltinLfm2ExtensionContractTest(
    ext_testing.ModelExtensionTestMixin, googletest.TestCase
):
  extension_cls = builtin_extensions.Lfm2Extension


class BuiltinQwen3AsrExtensionContractTest(
    ext_testing.ModelExtensionTestMixin, googletest.TestCase
):
  extension_cls = builtin_extensions.Qwen3AsrExtension


class BuiltinQwen3TtsExtensionContractTest(
    ext_testing.ModelExtensionTestMixin, googletest.TestCase
):
  extension_cls = builtin_extensions.Qwen3TtsExtension


class BuiltinGemma3nExtensionContractTest(
    ext_testing.ModelExtensionTestMixin, googletest.TestCase
):
  extension_cls = builtin_extensions.Gemma3nExtension


if __name__ == '__main__':
  googletest.main()
