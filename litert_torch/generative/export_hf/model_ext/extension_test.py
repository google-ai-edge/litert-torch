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
import json
import os
import sys
import textwrap
import types
from unittest import mock

from litert_torch.generative.export_hf import export as export_pkg
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.model_ext import builtin_extensions
from litert_torch.generative.export_hf.model_ext import exportables
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext import inspector
from litert_torch.generative.export_hf.model_ext import metadata_builder
from litert_torch.generative.export_hf.model_ext import patches
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
import safetensors.torch
import torch
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


class _W13Mlp(torch.nn.Module):
  """Toy MLP whose checkpoint names gate / up projections `w1` / `w3`."""

  def __init__(self, dim: int = 8, hidden: int = 16):
    super().__init__()
    self.w1 = torch.nn.Linear(dim, hidden, bias=False)
    self.w2 = torch.nn.Linear(hidden, dim, bias=False)
    self.w3 = torch.nn.Linear(dim, hidden, bias=False)

  def forward(self, x):
    return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class _FusedW13Mlp(torch.nn.Module):
  """Fuses `_W13Mlp.w1` / `w3` into `gate_up_proj`, keeping `w2`."""

  def __init__(self, mlp: _W13Mlp):
    super().__init__()
    self.w2 = mlp.w2
    self.gate_up_proj = torch.nn.Linear(
        mlp.w1.in_features, 2 * mlp.w1.out_features, bias=False
    )
    self.gate_up_proj.weight = torch.nn.Parameter(
        torch.cat([mlp.w1.weight.data, mlp.w3.weight.data], dim=0)
    )

  def forward(self, x):
    gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
    return self.w2(torch.nn.functional.silu(gate) * up)


class _W13Model(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.layers = torch.nn.ModuleList([_W13Mlp(), _W13Mlp()])


class _W13FusingExtension(ModelExtension):
  """Fuses `w1` / `w3` in `patch_model` without declaring a fused spec."""

  model_types = ('v2_weights_toy_arch',)

  @contextlib.contextmanager
  def patch_model(self, model, export_config):
    replaced = []
    if export_config.fuse_gate_up:
      for idx, layer in enumerate(model.layers):
        model.layers[idx] = _FusedW13Mlp(layer)
        replaced.append((idx, layer))
    try:
      yield
    finally:
      for idx, layer in replaced:
        model.layers[idx] = layer


class _W13FusingExtensionWithSpec(_W13FusingExtension):
  fused_weight_specs = (
      extension.FusedWeightSpec('gate_up_proj', ('w1', 'w3'), dim=0),
  )


class ModelExtensionTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self._saved_registry = dict(extension._EXTENSION_REGISTRY)
    self._saved_entry_points_loaded = extension._ENTRY_POINTS_LOADED
    self._saved_extra_kwargs = (
        exportable_module_config.get_registered_extra_kwargs()
    )

  def tearDown(self):
    extension._EXTENSION_REGISTRY.clear()
    extension._EXTENSION_REGISTRY.update(self._saved_registry)
    extension._ENTRY_POINTS_LOADED = self._saved_entry_points_loaded
    exportable_module_config.reset_registered_extra_kwargs(
        self._saved_extra_kwargs
    )
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
    class BaseExtension(ModelExtension):  # pylint: disable=unused-variable

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
        return bool(config_dict.get('custom_tts_flag'))

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

  def test_load_extension_from_py_file_and_auto_register(self):
    temp_dir = self.create_tempdir().full_path
    ext_file = os.path.join(temp_dir, 'bare_model_ext.py')
    with open(ext_file, 'w') as f:
      f.write(
          textwrap.dedent("""
          from litert_torch.generative.export_hf import model_ext

          class BareFileExtension(model_ext.ModelExtension):
            model_types = ('bare_file_arch',)
            extra_kwargs_keys = ('bare_file_flag',)
          """)
      )

    extension.load_extension_modules(ext_file)
    exp_cfg = ExportableModuleConfig(
        model='dummy',
        extension_modules=ext_file,
        extra_kwargs={'bare_file_flag': 'ok'},
    )
    self.assertIsNotNone(extension.get_model_extension('bare_file_arch'))
    self.assertEqual(exp_cfg.extra_kwargs['bare_file_flag'], 'ok')

    # Verify an undecorated subclass in `--extension_modules` overrides an
    # existing model_type registration and registers its `extra_kwargs_keys`.
    override_file = os.path.join(temp_dir, 'override_bare_ext.py')
    with open(override_file, 'w') as f:
      f.write(textwrap.dedent("""
          from litert_torch.generative.export_hf import model_ext


          class OverrideBareFileExtension(model_ext.ModelExtension):
            model_types = ('bare_file_arch',)
            extra_kwargs_keys = ('override_bare_flag',)
          """))
    extension.load_extension_modules(override_file)
    override_cfg = ExportableModuleConfig(
        model='dummy',
        extension_modules=override_file,
        extra_kwargs={'override_bare_flag': 'overridden'},
    )
    self.assertEqual(
        override_cfg.extra_kwargs['override_bare_flag'], 'overridden'
    )
    self.assertEqual(
        type(extension.get_model_extension('bare_file_arch')).__name__,
        'OverrideBareFileExtension',
    )

  def test_composite_extension_merges_zero_touch_hooks(self):

    class BaseZeroTouchExt(ModelExtension):
      extra_kwargs_keys = ('flag_a',)
      checkpoint_key_prefixes = ('backbone.',)
      fused_weight_specs = (extension.FusedWeightSpec('w13', ('w1', 'w3')),)
      is_causal_text_model = True
      sliding_window_layers = 'all'

      def resolve_checkpoint_tensor(self, param_name, try_load):
        if param_name == 'base_tensor':
          return torch.ones(2)
        return None

    class OverrideZeroTouchExt(ModelExtension):
      extra_kwargs_keys = ('flag_b',)
      checkpoint_key_prefixes = ('decoder.',)
      fused_weight_specs = (
          extension.FusedWeightSpec('col_ab', ('col_a', 'col_b'), dim=1),
      )
      is_causal_text_model = False
      sliding_window_layers = 'none'

      def resolve_checkpoint_tensor(self, param_name, try_load):
        if param_name == 'override_tensor':
          return torch.zeros(3)
        return None

      def load_config(self, model_path, export_config, trust_remote_code=False):
        return transformers.PretrainedConfig(model_type='merged_arch')

      def get_hf_model_cls(self, task, model_config, export_config):
        return _DummySpeechModel

    comp = extension.compose_extensions(
        BaseZeroTouchExt(), OverrideZeroTouchExt()
    )
    self.assertEqual(comp.extra_kwargs_keys, ('flag_a', 'flag_b'))
    self.assertEqual(
        comp.checkpoint_key_prefixes, ('backbone.', 'decoder.')
    )
    # Override extension specs come first so they take precedence when tried in
    # order by HFCheckpointWeightsLoader.
    self.assertEqual(
        [s.fused_name for s in comp.fused_weight_specs], ['col_ab', 'w13']
    )
    dummy_cfg = ExportableModuleConfig(model='dummy')
    self.assertFalse(comp.is_causal_text_model)
    self.assertEqual(comp.sliding_window_layers, 'none')
    self.assertEqual(
        comp.resolve_checkpoint_tensor('base_tensor', lambda _: None).shape,
        torch.Size([2]),
    )
    self.assertEqual(
        comp.resolve_checkpoint_tensor('override_tensor', lambda _: None).shape,
        torch.Size([3]),
    )
    loaded_cfg = comp.load_config('/tmp/fake', dummy_cfg)
    self.assertEqual(loaded_cfg.model_type, 'merged_arch')
    self.assertIs(
        comp.get_hf_model_cls('text_generation', loaded_cfg, dummy_cfg),
        _DummySpeechModel,
    )

  def test_zero_touch_out_of_tree_new_model_end_to_end(self):
    temp_dir = self.create_tempdir().full_path
    ckpt_dir = os.path.join(temp_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
      json.dump(
          {
              'model_type': 'zero_touch_arch',
              'vocab_size': 8,
              'hidden_size': 4,
              'intermediate_size': 6,
          },
          f,
      )

    # Minimal fast tokenizer JSON so out-of-tree checkpoints without a built-in
    # AutoTokenizer mapping still load cleanly.
    with open(os.path.join(ckpt_dir, 'tokenizer.json'), 'w') as f:
      json.dump(
          {
              'version': '1.0',
              'truncation': None,
              'padding': None,
              'added_tokens': [],
              'normalizer': None,
              'pre_tokenizer': {'type': 'Whitespace'},
              'post_processor': None,
              'decoder': None,
              'model': {
                  'type': 'WordLevel',
                  'vocab': {'<unk>': 0, '<bos>': 1, '<eos>': 2, 'hi': 3},
                  'unk_token': '<unk>',
              },
          },
          f,
      )

    ckpt_tensors = {
        'backbone.embed_tokens.weight': torch.randn(8, 4),
        'backbone.layers.0.w1.weight': torch.randn(6, 4),
        'backbone.layers.0.w3.weight': torch.randn(6, 4),
        'backbone.layers.0.layer_scale': torch.tensor([1.5]),
    }
    safetensors.torch.save_file(
        ckpt_tensors, os.path.join(ckpt_dir, 'model.safetensors')
    )

    ext_py = os.path.join(temp_dir, 'zero_touch_ext.py')
    with open(ext_py, 'w') as f:
      f.write(
          textwrap.dedent("""
          import dataclasses
          import json
          import os
          from litert_torch.generative.export_hf import model_ext
          import torch
          from torch import nn
          import transformers


          class ZeroTouchConfig(transformers.PretrainedConfig):
            model_type = 'zero_touch_arch'


          class _ZeroTouchLayer(nn.Module):

            def __init__(self, hidden_size: int, intermediate_size: int):
              super().__init__()
              self.w13 = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
              self.register_buffer('layer_scale', torch.ones(1))


          class ZeroTouchCausalLM(nn.Module):
            _can_compile_fullgraph = True
            _supports_attention_backend = True

            def __init__(self, config: ZeroTouchConfig):
              super().__init__()
              self.config = config
              self.embed_tokens = nn.Embedding(
                  config.vocab_size, config.hidden_size
              )
              self.layers = nn.ModuleList([
                  _ZeroTouchLayer(config.hidden_size, config.intermediate_size)
              ])
              self.lm_head = nn.Linear(
                  config.hidden_size, config.vocab_size, bias=False
              )
              self.lm_head.weight = self.embed_tokens.weight

            def get_input_embeddings(self):
              return self.embed_tokens

            def get_output_embeddings(self):
              return self.lm_head


          class ZeroTouchExtension(model_ext.ModelExtension):
            model_types = ('zero_touch_arch',)
            extra_kwargs_keys = ('zero_touch_scale',)
            is_causal_text_model = True
            checkpoint_key_prefixes = ('backbone.',)
            fused_weight_specs = (
                model_ext.FusedWeightSpec('w13', ('w1', 'w3')),
            )

            def load_config(
                self, model_path, export_config, trust_remote_code=False
            ):
              with open(os.path.join(model_path, 'config.json'), 'r') as cfg_f:
                return ZeroTouchConfig.from_dict(json.load(cfg_f))

            def get_hf_model_cls(self, task, model_config, export_config):
              return ZeroTouchCausalLM

            def update_export_config(self, export_config, model_config):
              return dataclasses.replace(
                  export_config, externalize_embedder=True
              )
          """)
      )

    # 1. Verify Converter V2 meta loading + custom fused/prefixed weights + CLI
    # extra_kwargs_keys purely from the standalone .py extension file.
    extension.load_extension_modules(ext_py)
    v2_cfg = ExportableModuleConfig(
        model=ckpt_dir,
        output_dir=os.path.join(temp_dir, 'out_v2'),
        extension_modules=ext_py,
        use_v2=True,
        bundle_litert_lm=False,
        extra_kwargs={'zero_touch_scale': 2.5},
    )
    artifacts_v2 = export_lib.load_model(ckpt_dir, v2_cfg)
    self.assertEqual(artifacts_v2.model_config.model_type, 'zero_touch_arch')
    self.assertTrue(
        all(p.is_meta for p in artifacts_v2.model.parameters())
    )
    self.assertIsNotNone(artifacts_v2.weights_loader)
    torch.testing.assert_close(
        artifacts_v2.model.layers[0].layer_scale,
        ckpt_tensors['backbone.layers.0.layer_scale'],
    )
    torch.testing.assert_close(
        artifacts_v2.weights_loader('model.layers.0.w13.weight'),
        torch.cat(
            [
                ckpt_tensors['backbone.layers.0.w1.weight'],
                ckpt_tensors['backbone.layers.0.w3.weight'],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        artifacts_v2.weights_loader('model.lm_head.weight'),
        ckpt_tensors['backbone.embed_tokens.weight'],
    )
    artifacts_v2.weights_loader.close()

    # 2. Verify eager (use_v2=False) parameter loading onto custom nn.Module
    # AND task list rebuilding when update_export_config flips
    # externalize_embedder=True.
    v1_cfg = ExportableModuleConfig(
        model=ckpt_dir,
        output_dir=os.path.join(temp_dir, 'out_v1'),
        extension_modules=ext_py,
        use_v2=False,
        bundle_litert_lm=False,
        extra_kwargs={'zero_touch_scale': 1.0},
    )
    initial_tasks, _ = export_pkg._build_export_tasks(v1_cfg)  # pylint: disable=protected-access
    self.assertNotIn(export_lib.export_embedder_model, initial_tasks)

    called_steps: list[str] = []

    def _record_step(name):
      def _fn(src, cfg, out):
        del src, cfg
        called_steps.append(name)
        return out

      return _fn

    with (
        mock.patch.object(
            export_lib,
            'export_embedder_model',
            side_effect=_record_step('embedder'),
        ),
        mock.patch.object(
            export_lib,
            'export_additional_models',
            side_effect=_record_step('additional'),
        ),
        mock.patch.object(
            export_lib,
            'export_text_prefill_decode_model',
            side_effect=_record_step('prefill_decode'),
        ),
        mock.patch.object(
            export_lib,
            'export_tokenizer',
            side_effect=_record_step('tokenizer'),
        ),
    ):
      rebuilt_initial_tasks, _ = export_pkg._build_export_tasks(v1_cfg)  # pylint: disable=protected-access
      export_pkg.run_export_tasks(rebuilt_initial_tasks, v1_cfg)

    self.assertEqual(
        called_steps,
        ['embedder', 'additional', 'prefill_decode', 'tokenizer'],
    )

  def test_assert_v2_weights_resolvable_reports_missing_fused_spec(self):
    model = _W13Model()
    with self.assertRaisesRegex(
        AssertionError,
        r'2 parameter\(s\)[\s\S]*model\.layers\.0\.gate_up_proj\.weight: not'
        r' found[\s\S]*model\.layers\.1\.gate_up_proj\.weight: not found',
    ):
      ext_testing.assert_v2_weights_resolvable(_W13FusingExtension(), model)
    # `patch_model` is restored even when the check fails.
    self.assertIsInstance(model.layers[0], _W13Mlp)

  def test_assert_v2_weights_resolvable_with_fused_spec(self):
    resolved = ext_testing.assert_v2_weights_resolvable(
        _W13FusingExtensionWithSpec(), _W13Model()
    )
    self.assertEqual(
        resolved,
        [
            'model.layers.0.gate_up_proj.weight',
            'model.layers.0.w2.weight',
            'model.layers.1.gate_up_proj.weight',
            'model.layers.1.w2.weight',
        ],
    )

  def test_assert_v2_weights_resolvable_builtin_qwen3_defaults(self):
    ext = builtin_extensions.Qwen3Extension()
    qwen3_cfg = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    with ext.patch_context():
      tiny_model = transformers.Qwen3ForCausalLM(qwen3_cfg).eval()
    resolved = ext_testing.assert_v2_weights_resolvable(ext, tiny_model)
    self.assertIn('model.model.layers.0.mlp.gate_up_proj.weight', resolved)
    self.assertIn('model.model.layers.0.self_attn.qkv_proj.weight', resolved)

  def test_assert_patch_numerical_parity_and_tiny_prefill_decode_export(self):
    ext = builtin_extensions.Qwen3Extension()
    qwen3_cfg = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    with ext.patch_context():
      tiny_model = transformers.Qwen3ForCausalLM(qwen3_cfg).eval()

    export_cfg = ExportableModuleConfig(
        model='dummy',
        prefill_lengths=[8],
        cache_length=16,
        cache_lengths=[16],
        externalize_embedder=True,
        fuse_gate_up=True,
        fuse_qkv=True,
    )
    sample_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ext_testing.assert_patch_numerical_parity(
        ext=ext,
        model=tiny_model,
        sample_kwargs={'input_ids': sample_ids, 'use_cache': False},
        export_config=export_cfg,
    )

    exported_programs = ext_testing.assert_tiny_prefill_decode_torch_export(
        ext=ext,
        model=tiny_model,
        model_config=qwen3_cfg,
        export_config=export_cfg,
    )
    self.assertIn('prefill_8', exported_programs)
    self.assertIn('decode', exported_programs)

  def test_inspector_analyzes_checkpoint_and_runs_dry_run_export(self):
    temp_dir = self.create_tempdir().full_path
    qwen3_cfg = {
        'model_type': 'qwen3',
        'architectures': ['Qwen3ForCausalLM'],
        'vocab_size': 64,
        'hidden_size': 16,
        'intermediate_size': 32,
        'num_hidden_layers': 4,
        'num_attention_heads': 2,
        'num_key_value_heads': 2,
        'head_dim': 8,
        'max_position_embeddings': 64,
    }
    with open(os.path.join(temp_dir, 'config.json'), 'w') as f:
      json.dump(qwen3_cfg, f)
    safetensors.torch.save_file(
        {'backbone.layers.0.weight': torch.ones(4, 4)},
        os.path.join(temp_dir, 'model.safetensors'),
    )

    report = inspector.inspect_model_architecture(
        model=temp_dir,
        run_dry_run_export=True,
    )
    self.assertEqual(report.model_type, 'qwen3')
    self.assertEqual(report.registered_extension_cls, 'Qwen3Extension')
    self.assertIn('backbone.', report.unrecognized_safetensors_prefixes)
    self.assertTrue(report.dry_run_results)
    self.assertTrue(report.dry_run_results[0].passed)
    md = report.to_markdown()
    self.assertIn('Qwen3Extension', md)
    self.assertIn("checkpoint_key_prefixes = ('backbone.',)", md)

  def test_sliding_window_layers_extension_hook_enables_sliding_window(self):
    @extension.register_model_extension(override=True)
    class _UniformSwaExt(ModelExtension):
      model_types = ('custom_swa_arch',)
      sliding_window_layers = 'all'

    del _UniformSwaExt
    cfg = transformers.PretrainedConfig(
        model_type='custom_swa_arch',
        sliding_window=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=32,
    )
    export_cfg = ExportableModuleConfig(
        model='dummy',
        cache_length=256,
        sliding_window_ring_buffer_size=128,
    )
    mod = exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill(
        model=torch.nn.Module(),
        export_config=export_cfg,
    )
    self.assertTrue(mod._has_sliding_window(cfg))

    # Verify `_apply_extension_config_defaults` normalizes `layer_types` on
    # loaded configs and `build_executor_metadata` emits local KV cache states.
    ext_instance = extension.resolve_model_extension(cfg, export_cfg)
    export_lib._apply_extension_config_defaults(cfg, ext_instance)  # pylint: disable=protected-access
    self.assertEqual(
        cfg.layer_types, ['sliding_attention', 'sliding_attention']
    )

    # Without the extension hook (or when set to 'none'), a non-Mistral model
    # without `layer_types` defaults to False.
    cfg_no_ext = transformers.PretrainedConfig(
        model_type='unregistered_swa_arch',
        sliding_window=128,
    )
    self.assertFalse(mod._has_sliding_window(cfg_no_ext))

  def test_shrink_config_preserves_distinct_layer_types_and_clamps_token_ids(
      self,
  ):
    tri_hybrid_cfg = transformers.PretrainedConfig.from_dict({
        'model_type': 'lfm3_like',
        'vocab_size': 256,
        'pad_token_id': 124893,
        'bos_token_id': 124800,
        'eos_token_id': [124801, 124809],
        'num_hidden_layers': 24,
        'layer_types': (
            [
                'conv',
                'conv',
                'sliding_attention',
                'conv',
                'full_attention',
            ]
            + ['conv'] * 19
        ),
        'dtype': 'bfloat16',
    })
    shrunk = inspector._shrink_config_for_tiny_model(tri_hybrid_cfg)  # pylint: disable=protected-access
    self.assertEqual(shrunk.num_hidden_layers, 3)
    self.assertEqual(
        shrunk.layer_types,
        ['conv', 'sliding_attention', 'full_attention'],
    )
    self.assertEqual(shrunk.pad_token_id, 0)
    self.assertEqual(shrunk.bos_token_id, 1)
    self.assertEqual(shrunk.eos_token_id, [2, 2])
    self.assertEqual(shrunk.dtype, torch.float32)

  def test_assert_patch_numerical_parity_casts_bfloat16_models_to_fp32(self):
    ext = builtin_extensions.Qwen3Extension()
    qwen3_cfg = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    with ext.patch_context():
      bf16_model = (
          transformers.Qwen3ForCausalLM(qwen3_cfg).to(torch.bfloat16).eval()
      )
    export_cfg = ExportableModuleConfig(
        model='dummy',
        externalize_embedder=True,
        fuse_gate_up=True,
        fuse_qkv=True,
    )
    sample_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    ext_testing.assert_patch_numerical_parity(
        ext=ext,
        model=bf16_model,
        sample_kwargs={'input_ids': sample_ids, 'use_cache': False},
        export_config=export_cfg,
        atol=1e-4,
        rtol=1e-4,
    )
    # Original model remains in bfloat16 after parity check and restoration.
    self.assertEqual(next(bf16_model.parameters()).dtype, torch.bfloat16)

  def test_generate_extension_skeleton_sanitizes_hyphenated_model_type(self):
    skeleton = inspector._generate_extension_skeleton(  # pylint: disable=protected-access
        model_type='qwen2-vl',
        unrecognized_prefixes=(),
        unpatched_rmsnorm_classes=('Qwen2VLTextRMSNorm',),
        has_dual_rope_attrs=False,
        rotary_accepts_layer_type=True,
        has_dual_rope_parameters=False,
        has_auto_map=True,
        architectures=('Qwen2VLForConditionalGeneration',),
        mlp_proj_names=('w1', 'w2', 'w3'),
    )
    # Must be syntactically valid Python even with a hyphenated model_type.
    compile(skeleton, '<skeleton>', 'exec')
    self.assertIn('class Qwen2VlExtension(extension.ModelExtension):', skeleton)
    self.assertIn('def _find_qwen2_vl_modeling_module():', skeleton)
    self.assertIn('mod = _find_qwen2_vl_modeling_module()', skeleton)

  def test_tiny_prefill_decode_export_auto_enables_gpu_composites_after_update_export_config(
      self,
  ):
    class _RingBufferInUpdateConfigExt(builtin_extensions.Gemma3Extension):
      model_types = ('gemma3_text_ring_test',)

      def update_export_config(self, export_config, model_config):
        del model_config
        return dataclasses.replace(
            export_config, sliding_window_ring_buffer_size=32
        )

    ext = _RingBufferInUpdateConfigExt()
    gemma3_cfg = transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=8,
        layer_types=['sliding_attention', 'full_attention'],
        max_position_embeddings=64,
    )
    with ext.patch_context():
      tiny_model = transformers.Gemma3ForCausalLM(gemma3_cfg).eval()
    # Default export_config=None; `sliding_window_ring_buffer_size` is only set
    # inside `ext.update_export_config`.
    exported = ext_testing.assert_tiny_prefill_decode_torch_export(
        ext=ext,
        model=tiny_model,
    )
    self.assertIn('decode', exported)


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
