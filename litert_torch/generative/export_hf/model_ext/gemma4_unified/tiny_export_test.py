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
"""Hermetic end-to-end export test for a tiny Gemma 4 Unified checkpoint.

Exports a 2-layer random-weight `Gemma4UnifiedForConditionalGeneration` (text,
vision and audio) with the default `cpu_gpu` Converter V2 settings, then checks
the parts of the artifact LiteRT-LM depends on:

- the checkpoint is written by `save_pretrained`, so it has the released
  `vision_embedder.*` key layout that Converter V2 must map back;
- the vision encoder has a signature for every supported soft-token budget and
  the audio encoder has the `src_inputs` -> `features` contract;
- `ExecutorMetadata` declares local (ring buffer) KV caches and the
  `VISION_BIDIRECTIONAL` local attention mask;
- the exported vision and audio encoders match the HF reference numerically.
"""

import json
import os
import tempfile

from litert_torch._convert import converter_v2
from litert_torch.generative.export_hf import export as export_lib
from litert_torch.generative.export_hf.model_ext.gemma4 import vision_budgets
from litert_torch.generative.export_hf.model_ext.gemma4_unified import audio_exportable
from litert_torch.generative.export_hf.model_ext.gemma4_unified import extension
import numpy as np
import safetensors
import torch
import transformers
from transformers.models.gemma4_unified import configuration_gemma4_unified
from transformers.models.gemma4_unified import modeling_gemma4_unified

from absl.testing import absltest as googletest
from litert_lm_builder.runtime.proto import executor_metadata_pb2
from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import

_VOCAB_SIZE = 64
_SLIDING_WINDOW = 16
_PATCH_SIZE = 2
_POOLING_KERNEL_SIZE = 2
_AUDIO_SAMPLES_PER_TOKEN = 20

# Named special tokens read by the Gemma 4 metadata builder and EOI/EOA modules.
_SPECIAL_TOKENS = {
    'pad_token': '<pad>',
    'eos_token': '<eos>',
    'bos_token': '<bos>',
    'unk_token': '<unk>',
    'boi_token': '<|image>',
    'eoi_token': '<image|>',
    'image_token': '<|image|>',
    'boa_token': '<|audio>',
    'eoa_token': '<audio|>',
    'audio_token': '<|audio|>',
}


def _make_config() -> configuration_gemma4_unified.Gemma4UnifiedConfig:
  return configuration_gemma4_unified.Gemma4UnifiedConfig(
      text_config=configuration_gemma4_unified.Gemma4UnifiedTextConfig(
          vocab_size=_VOCAB_SIZE,
          hidden_size=32,
          intermediate_size=64,
          num_hidden_layers=2,
          num_attention_heads=2,
          num_key_value_heads=1,
          num_global_key_value_heads=1,
          attention_k_eq_v=True,
          head_dim=16,
          global_head_dim=16,
          layer_types=['sliding_attention', 'full_attention'],
          sliding_window=_SLIDING_WINDOW,
          final_logit_softcapping=30.0,
          pad_token_id=0,
          eos_token_id=1,
          bos_token_id=2,
      ),
      vision_config=configuration_gemma4_unified.Gemma4UnifiedVisionConfig(
          patch_size=_PATCH_SIZE,
          pooling_kernel_size=_POOLING_KERNEL_SIZE,
          mm_embed_dim=24,
          output_proj_dims=24,
          mm_posemb_size=64,
      ),
      audio_config=configuration_gemma4_unified.Gemma4UnifiedAudioConfig(
          audio_embed_dim=_AUDIO_SAMPLES_PER_TOKEN,
      ),
      boi_token_id=4,
      eoi_token_id=5,
      image_token_id=6,
      boa_token_id=7,
      eoa_token_index=8,
      audio_token_id=9,
      video_token_id=10,
  )


def _write_tokenizer(checkpoint_dir: str) -> None:
  """Writes a word-level tokenizer whose ids match `_make_config`."""
  ordered = [
      '<pad>', '<eos>', '<bos>', '<unk>', '<|image>', '<image|>', '<|image|>',
      '<|audio>', '<audio|>', '<|audio|>', '<|video|>',
  ]
  vocab = {tok: i for i, tok in enumerate(ordered)}
  for i in range(len(ordered), _VOCAB_SIZE):
    vocab[f'tok_{i}'] = i
  tokenizer_json = {
      'version': '1.0',
      'truncation': None,
      'padding': None,
      'added_tokens': [
          {
              'id': vocab[tok],
              'content': tok,
              'single_word': False,
              'lstrip': False,
              'rstrip': False,
              'normalized': False,
              'special': True,
          }
          for tok in ordered
      ],
      'normalizer': None,
      'pre_tokenizer': {'type': 'Whitespace'},
      'post_processor': None,
      'decoder': None,
      'model': {'type': 'WordLevel', 'vocab': vocab, 'unk_token': '<unk>'},
  }
  with open(os.path.join(checkpoint_dir, 'tokenizer.json'), 'w') as f:
    json.dump(tokenizer_json, f)
  with open(os.path.join(checkpoint_dir, 'tokenizer_config.json'), 'w') as f:
    json.dump(
        {
            'tokenizer_class': 'PreTrainedTokenizerFast',
            'extra_special_tokens': ['<|video|>'],
            **_SPECIAL_TOKENS,
        },
        f,
    )


def _write_checkpoint(checkpoint_dir: str) -> torch.nn.Module:
  torch.manual_seed(0)
  model = modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration(
      _make_config()
  ).eval()
  with torch.no_grad():
    # Non-trivial norm / position parameters so wiring mistakes are visible.
    for name, param in model.model.embed_vision.named_parameters():
      if 'patch_dense' not in name and 'embedding_projection' not in name:
        param.normal_()
  model.save_pretrained(checkpoint_dir, safe_serialization=True)
  _write_tokenizer(checkpoint_dir)
  # Same nested layout as the released `processor_config.json`.
  with open(os.path.join(checkpoint_dir, 'processor_config.json'), 'w') as f:
    json.dump(
        {
            'processor_class': 'Gemma4UnifiedProcessor',
            'audio_seq_length': audio_exportable.DEFAULT_AUDIO_TOKENS,
            'image_seq_length': 280,
            'feature_extractor': {
                'feature_extractor_type': 'Gemma4UnifiedAudioFeatureExtractor',
                'audio_samples_per_token': _AUDIO_SAMPLES_PER_TOKEN,
                'feature_size': _AUDIO_SAMPLES_PER_TOKEN,
                'padding_side': 'right',
                'padding_value': 0.0,
                'return_attention_mask': True,
                'sampling_rate': 16000,
            },
            'image_processor': {
                'image_processor_type': 'Gemma4UnifiedImageProcessor',
                'do_convert_rgb': True,
                'do_normalize': False,
                'do_rescale': True,
                'do_resize': True,
                'image_mean': [0.0, 0.0, 0.0],
                'image_std': [1.0, 1.0, 1.0],
                'max_soft_tokens': 280,
                'patch_size': _PATCH_SIZE,
                'pooling_kernel_size': _POOLING_KERNEL_SIZE,
                'resample': 3,
                'rescale_factor': 1 / 255,
            },
        },
        f,
    )
  return model


def _checkpoint_keys(checkpoint_dir: str) -> set[str]:
  keys = set()
  for name in os.listdir(checkpoint_dir):
    if name.endswith('.safetensors'):
      with safetensors.safe_open(
          os.path.join(checkpoint_dir, name), framework='pt'
      ) as f:
        keys.update(f.keys())
  return keys


def _run_signature(tflite_path: str, signature: str, **inputs):
  interp = tfl_interpreter.Interpreter(
      model_path=tflite_path,
      experimental_op_resolver_type=(
          tfl_interpreter.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
      ),
  )
  interp.allocate_tensors()
  return interp.get_signature_runner(signature)(**inputs)


def _signature_io(tflite_path: str) -> dict[str, dict[str, list[str]]]:
  interp = tfl_interpreter.Interpreter(model_path=tflite_path)
  return {
      key: {'inputs': sorted(io['inputs']), 'outputs': sorted(io['outputs'])}
      for key, io in interp.get_signature_list().items()
  }


class Gemma4UnifiedTinyExportTest(googletest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if not converter_v2.is_supported():
      raise googletest.SkipTest('Converter V2 is not supported.')
    cls._tmp = tempfile.TemporaryDirectory()
    cls.checkpoint_dir = os.path.join(cls._tmp.name, 'ckpt')
    cls.output_dir = os.path.join(cls._tmp.name, 'out')
    os.makedirs(cls.checkpoint_dir)
    cls.model = _write_checkpoint(cls.checkpoint_dir)
    cls.plan = export_lib.export(
        model=cls.checkpoint_dir,
        output_dir=cls.output_dir,
        target='cpu_gpu',
        use_v2=True,
        delete_in_memory_params=True,
        prefill_lengths=[8],
        cache_length=64,
        externalize_weights=False,
        quantization_recipe='none',
        vision_encoder_quantization_recipe='none',
        audio_encoder_quantization_recipe='none',
        litert_lm_llm_metadata_override='none',
        jinja_chat_template_override=None,
        keep_temporary_files=True,
    )

  @classmethod
  def tearDownClass(cls):
    cls._tmp.cleanup()
    super().tearDownClass()

  def _path(self, name: str) -> str:
    path = os.path.join(self.output_dir, name)
    self.assertTrue(os.path.exists(path), f'{name} was not exported.')
    return path

  def test_checkpoint_uses_released_key_layout(self):
    keys = _checkpoint_keys(self.checkpoint_dir)
    self.assertTrue(any(k.startswith('model.vision_embedder.') for k in keys))
    self.assertIn('model.embed_vision.embedding_projection.weight', keys)

  def test_default_plan(self):
    self.assertEqual(self.plan.task, 'multimodal_lm')
    kwargs = self.plan.export_kwargs
    self.assertTrue(kwargs['export_vision_encoder'])
    self.assertTrue(kwargs['export_audio_encoder'])
    self.assertEqual(
        kwargs['sliding_window_ring_buffer_size'],
        _SLIDING_WINDOW + extension.RING_BUFFER_PREFILL_MARGIN,
    )
    self.assertTrue(os.path.exists(self._path('model.litertlm')))

  def test_vision_encoder_has_every_budget(self):
    io = _signature_io(self._path('vision_encoder.tflite'))
    self.assertCountEqual(
        io,
        [f'vision_{b}' for b in vision_budgets.SUPPORTED_VISION_TOKEN_BUDGETS],
    )
    for sig in io.values():
      self.assertEqual(sig['inputs'], ['images', 'positions_xy'])
      self.assertEqual(sig['outputs'], ['features', 'mask'])

  def test_audio_encoder_contract(self):
    io = _signature_io(self._path('audio_encoder.tflite'))
    self.assertEqual(
        io,
        {
            f'audio_{audio_exportable.DEFAULT_AUDIO_TOKENS}': {
                'inputs': ['src_inputs'],
                'outputs': ['features'],
            }
        },
    )
    self.assertFalse(
        os.path.exists(os.path.join(self.output_dir, 'audio_adapter.tflite'))
    )

  def test_executor_metadata_vision_bidirectional_local_mask(self):
    metadata = executor_metadata_pb2.ExecutorMetadata()
    with open(self._path('executor_metadata.pb'), 'rb') as f:
      metadata.ParseFromString(f.read())
    llm = metadata.llm_executor_metadata
    settings = llm.attention_mask_settings
    self.assertEqual(
        settings.attention_mask_type,
        executor_metadata_pb2.ATTENTION_MASK_TYPE_CAUSAL,
    )
    self.assertEqual(
        settings.local_attention_mask_type,
        executor_metadata_pb2.ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL,
    )
    self.assertEqual(settings.sliding_window_size, _SLIDING_WINDOW)
    buffer_types = {b.type for b in llm.state_buffers}
    self.assertIn(
        executor_metadata_pb2.StateBuffer.TYPE_LOCAL_KEY_CACHE, buffer_types
    )
    self.assertIn(
        executor_metadata_pb2.StateBuffer.TYPE_GLOBAL_KEY_CACHE, buffer_types
    )

  def test_vision_encoder_matches_hf(self):
    reference = (
        modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration.from_pretrained(
            self.checkpoint_dir
        ).eval()
    )
    processor = transformers.AutoImageProcessor.from_pretrained(
        self.checkpoint_dir
    )
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, (64, 48, 3)).astype(np.uint8)
    inputs = processor(images=[image], max_soft_tokens=70, return_tensors='pt')
    pixel_values = inputs.pixel_values.float()
    positions = inputs.image_position_ids.int()

    out = _run_signature(
        self._path('vision_encoder.tflite'),
        'vision_70',
        images=pixel_values.numpy(),
        positions_xy=positions.numpy(),
    )
    with torch.no_grad():
      expected = reference.model.embed_vision(
          pixel_values, positions.long()
      ).pooler_output

    valid = (positions != -1).all(dim=-1).numpy()
    np.testing.assert_array_equal(out['mask'].astype(bool), valid)
    np.testing.assert_allclose(
        out['features'][valid], expected.numpy()[valid], rtol=1e-4, atol=1e-4
    )

  def test_audio_encoder_matches_hf(self):
    reference = (
        modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration.from_pretrained(
            self.checkpoint_dir
        ).eval()
    )
    rng = np.random.default_rng(0)
    frames = rng.uniform(
        -1,
        1,
        (1, audio_exportable.DEFAULT_AUDIO_TOKENS, _AUDIO_SAMPLES_PER_TOKEN),
    ).astype(np.float32)

    out = _run_signature(
        self._path('audio_encoder.tflite'),
        f'audio_{audio_exportable.DEFAULT_AUDIO_TOKENS}',
        src_inputs=frames,
    )
    with torch.no_grad():
      expected = reference.model.get_audio_features(
          torch.from_numpy(frames)
      ).pooler_output

    np.testing.assert_allclose(
        out['features'], expected.numpy(), rtol=1e-4, atol=1e-4
    )


if __name__ == '__main__':
  googletest.main()
