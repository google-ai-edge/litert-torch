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
"""Tests for the Gemma 4 Unified extension and exportable modules."""

import re
import types

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.gemma4 import audio_exportable as gemma4_audio_exportable
from litert_torch.generative.export_hf.model_ext.gemma4 import vision_budgets
from litert_torch.generative.export_hf.model_ext.gemma4_unified import audio_exportable
from litert_torch.generative.export_hf.model_ext.gemma4_unified import extension
from litert_torch.generative.export_hf.model_ext.gemma4_unified import vision_exportable
import torch
from transformers import conversion_mapping
from transformers.models.gemma4_unified import configuration_gemma4_unified
from transformers.models.gemma4_unified import modeling_gemma4_unified

from absl.testing import absltest as googletest
from litert_lm_builder.runtime.proto import llm_metadata_pb2

_ExportTask = extension_lib.ExportTask


def _make_tiny_config() -> configuration_gemma4_unified.Gemma4UnifiedConfig:
  return configuration_gemma4_unified.Gemma4UnifiedConfig(
      text_config=configuration_gemma4_unified.Gemma4UnifiedTextConfig(
          vocab_size=64,
          hidden_size=32,
          intermediate_size=64,
          num_hidden_layers=2,
          num_attention_heads=2,
          num_key_value_heads=1,
          head_dim=16,
          global_head_dim=16,
          layer_types=['sliding_attention', 'full_attention'],
          sliding_window=16,
      ),
      vision_config=configuration_gemma4_unified.Gemma4UnifiedVisionConfig(
          patch_size=2,
          pooling_kernel_size=2,
          mm_embed_dim=24,
          output_proj_dims=24,
          mm_posemb_size=16,
      ),
      audio_config=configuration_gemma4_unified.Gemma4UnifiedAudioConfig(
          audio_embed_dim=20,
      ),
  )


def _make_tiny_model() -> torch.nn.Module:
  torch.manual_seed(0)
  model = modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration(
      _make_tiny_config()
  ).eval()
  # Non-trivial norm/position parameters so a wrong wiring is detectable.
  with torch.no_grad():
    for name, param in model.model.embed_vision.named_parameters():
      if 'patch_dense' not in name and 'embedding_projection' not in name:
        param.normal_()
  return model


def _export_config(**kwargs) -> exportable_module_config.ExportableModuleConfig:
  return exportable_module_config.ExportableModuleConfig(
      model='unused', output_dir='unused', **kwargs
  )


class Gemma4UnifiedVisionEncoderTest(googletest.TestCase):

  def test_matches_hf_embed_vision(self):
    model = _make_tiny_model()
    patch_dim = model.config.vision_config.model_patch_size**2 * 3
    pixel_values = torch.randn(1, 6, patch_dim)
    positions_xy = torch.tensor(
        [[[0, 0], [1, 0], [0, 1], [3, 2], [-1, -1], [-1, -1]]],
        dtype=torch.int32,
    )
    module = (
        vision_exportable.LiteRTExportableModuleForGemma4UnifiedVisionEncoder(
            model, _export_config()
        )
    )

    with torch.no_grad():
      out = module(pixel_values, positions_xy)
      expected = model.model.embed_vision(
          pixel_values, positions_xy.long()
      ).pooler_output

    torch.testing.assert_close(out['features'], expected)
    torch.testing.assert_close(
        out['mask'], torch.tensor([[True, True, True, True, False, False]])
    )


class Gemma4UnifiedAudioEncoderTest(googletest.TestCase):

  def test_matches_hf_get_audio_features(self):
    model = _make_tiny_model()
    module = (
        audio_exportable.LiteRTExportableModuleForGemma4UnifiedAudioEncoder(
            model, _export_config()
        )
    )
    sample_inputs = module.get_sample_inputs(model.config)
    self.assertEqual(
        list(sample_inputs),
        [f'audio_{audio_exportable.DEFAULT_AUDIO_TOKENS}'],
    )
    (inputs, _), = sample_inputs.values()
    self.assertEqual(list(inputs), ['src_inputs'])
    self.assertEqual(
        inputs['src_inputs'].shape,
        (1, audio_exportable.DEFAULT_AUDIO_TOKENS, 20),
    )
    src_inputs = torch.randn(1, 5, 20)

    with torch.no_grad():
      out = module(src_inputs)
      expected = model.model.get_audio_features(src_inputs).pooler_output

    self.assertEqual(list(out), ['features'])
    torch.testing.assert_close(out['features'], expected)


class Gemma4UnifiedExtensionTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    self.ext = extension.Gemma4UnifiedExtension()

  def test_default_task_with_audio_is_multimodal(self):
    defaults = self.ext.get_export_defaults(
        {'model_type': 'gemma4_unified', 'audio_config': {'a': 1}}
    )
    self.assertEqual(defaults.task, _ExportTask.MULTIMODAL_LM)
    self.assertTrue(defaults.export_kwargs['export_vision_encoder'])
    self.assertTrue(defaults.export_kwargs['export_audio_encoder'])

  def test_default_task_without_audio_is_vlm(self):
    defaults = self.ext.get_export_defaults({'model_type': 'gemma4_unified'})
    self.assertEqual(defaults.task, _ExportTask.IMAGE_TEXT_TO_TEXT)
    self.assertTrue(defaults.export_kwargs['export_vision_encoder'])
    self.assertFalse(defaults.export_kwargs['export_audio_encoder'])

  def test_ring_buffer_default_for_vision_bidirectional_attention(self):
    config = {
        'model_type': 'gemma4_unified',
        'text_config': {'sliding_window': 1024},
    }
    cpu_gpu = self.ext.get_export_defaults(config, target='cpu_gpu')
    self.assertEqual(
        cpu_gpu.export_kwargs['sliding_window_ring_buffer_size'], 1152
    )
    npu = self.ext.get_export_defaults(config, target='npu:qualcomm')
    self.assertNotIn('sliding_window_ring_buffer_size', npu.export_kwargs)
    no_window = self.ext.get_export_defaults({'model_type': 'gemma4_unified'})
    self.assertNotIn('sliding_window_ring_buffer_size', no_window.export_kwargs)

  def test_explicit_tasks(self):
    config = {'model_type': 'gemma4_unified', 'audio_config': {'a': 1}}
    vlm = self.ext.get_export_defaults(config, _ExportTask.IMAGE_TEXT_TO_TEXT)
    self.assertTrue(vlm.export_kwargs['export_vision_encoder'])
    self.assertFalse(vlm.export_kwargs['export_audio_encoder'])
    text = self.ext.get_export_defaults(config, _ExportTask.TEXT_GENERATION)
    self.assertFalse(text.export_kwargs['export_vision_encoder'])
    self.assertFalse(text.export_kwargs['export_audio_encoder'])

  def test_update_export_config_defaults_to_all_vision_budgets(self):
    export_config = _export_config(task=_ExportTask.IMAGE_TEXT_TO_TEXT)

    export_config = self.ext.update_export_config(
        export_config, _make_tiny_config()
    )

    self.assertEqual(
        vision_budgets.get_vision_token_budgets(export_config.extra_kwargs),
        vision_budgets.SUPPORTED_VISION_TOKEN_BUDGETS,
    )

  def test_update_export_config_keeps_requested_vision_budgets(self):
    for flag in (
        vision_budgets.VISION_TOKEN_BUDGETS_FLAG,
        vision_budgets.LEGACY_VISION_MAX_SOFT_TOKENS_FLAG,
    ):
      with self.subTest(flag=flag):
        export_config = _export_config(
            task=_ExportTask.IMAGE_TEXT_TO_TEXT, extra_kwargs={flag: 280}
        )

        export_config = self.ext.update_export_config(
            export_config, _make_tiny_config()
        )

        self.assertEqual(
            vision_budgets.get_vision_token_budgets(export_config.extra_kwargs),
            (280,),
        )

  def test_audio_exportables(self):
    spec = self.ext.get_audio_exportables(_make_tiny_config())
    self.assertIs(
        spec.encoder_cls,
        audio_exportable.LiteRTExportableModuleForGemma4UnifiedAudioEncoder,
    )
    self.assertIsNone(spec.adapter_cls)
    self.assertIs(
        spec.eoa_cls,
        gemma4_audio_exportable.LiteRTExportableModuleForGemma4EndOfAudio,
    )

  def test_checkpoint_renames_invert_hf_conversion_mapping(self):
    renamings = conversion_mapping.get_checkpoint_conversion_mapping(
        'gemma4_unified'
    )
    model = _make_tiny_model()
    names = [
        n for n, _ in model.named_parameters() if 'embed_vision' in n
    ]
    self.assertNotEmpty(names)
    for name in names:
      checkpoint_name = extension.to_checkpoint_name(name)
      self.assertIsNotNone(checkpoint_name, name)
      self.assertNotIn('embed_vision.patch', checkpoint_name)
      # Applying HF's load-time renames to the checkpoint name must give back
      # the module name.
      loaded_name = checkpoint_name
      for renaming in renamings:
        for src, tgt in zip(renaming.source_patterns, renaming.target_patterns):
          loaded_name = re.sub(src, tgt, loaded_name)
      self.assertEqual(loaded_name, name)

  def test_resolve_checkpoint_tensor(self):
    weight = torch.ones(2)
    loaded = {}

    def try_load(key):
      loaded['key'] = key
      return weight

    self.assertIs(
        self.ext.resolve_checkpoint_tensor(
            'model.model.embed_vision.pos_norm.weight', try_load
        ),
        weight,
    )
    self.assertEqual(
        loaded['key'], 'model.model.vision_embedder.pos_norm.weight'
    )
    self.assertIsNone(
        self.ext.resolve_checkpoint_tensor(
            'model.model.embed_audio.embedding_projection.weight', try_load
        )
    )

  def test_build_llm_metadata_sets_unified_preprocessing(self):
    llm_metadata = llm_metadata_pb2.LlmMetadata()
    source = types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(
            special_tokens_map={
                'boi_token': '<|image>',
                'eoi_token': '<image|>',
                'boa_token': '<|audio>',
                'eoa_token': '<audio|>',
            }
        ),
        image_processor=types.SimpleNamespace(
            patch_size=16, pooling_kernel_size=3
        ),
        feature_extractor=object(),
    )
    exported = types.SimpleNamespace(
        vision_encoder_model_path='vision.tflite',
        audio_encoder_model_path='audio.tflite',
    )

    llm_metadata = self.ext.build_llm_metadata(
        source, _export_config(), exported, llm_metadata
    )

    gemma4 = llm_metadata.llm_model_type.gemma4
    self.assertTrue(gemma4.merge_patches)
    self.assertTrue(gemma4.skip_mel_spectrogram_extraction)
    self.assertEqual(gemma4.start_of_audio_token.token_str, '<|audio>')


if __name__ == '__main__':
  googletest.main()
