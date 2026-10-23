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
"""Tests for export_lib."""

import json
import os
import shutil
import tempfile

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from litert_torch import backend
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.layers import moe
import torch
import transformers
from transformers.models.gemma4 import modeling_gemma4
from transformers.models.lfm2_moe import modeling_lfm2_moe
from transformers.models.qwen3_moe import modeling_qwen3_moe


class ExportLibTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_maybe_patch_tokenizer_patches_incorrect_config(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # BPE + Metaspace + BPE chars in vocab
    incorrect_config = {
        "model": {"type": "BPE", "vocab": {"ĠI": 1, "Ċ": 2, "hello": 3}},
        "pre_tokenizer": {"type": "Metaspace", "replacement": " "},
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": {"String": " "},
                    "content": " ",
                }
            ],
        },
    }

    with open(tokenizer_path, "w") as f:
      json.dump(incorrect_config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      patched_config = json.load(f)

    self.assertEqual(patched_config["pre_tokenizer"]["type"], "ByteLevel")
    self.assertEqual(patched_config["decoder"]["type"], "ByteLevel")
    self.assertTrue(patched_config["pre_tokenizer"]["use_regex"])
    self.assertTrue(patched_config["decoder"]["use_regex"])

  def test_maybe_patch_tokenizer_skips_correct_config(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # BPE + ByteLevel
    correct_config = {
        "model": {"type": "BPE", "vocab": {"ĠI": 1, "Ċ": 2}},
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
    }

    with open(tokenizer_path, "w") as f:
      json.dump(correct_config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      config = json.load(f)

    self.assertEqual(config["pre_tokenizer"]["type"], "ByteLevel")
    self.assertEqual(config["decoder"]["type"], "ByteLevel")
    self.assertNotIn("use_regex", config["pre_tokenizer"])

  def test_maybe_patch_tokenizer_skips_non_bpe(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # WordPiece + Metaspace (dummy)
    config = {
        "model": {"type": "WordPiece", "vocab": {"hello": 1}},
        "pre_tokenizer": {"type": "Metaspace"},
    }

    with open(tokenizer_path, "w") as f:
      json.dump(config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      result_config = json.load(f)

    self.assertEqual(result_config["pre_tokenizer"]["type"], "Metaspace")

  def test_maybe_patch_tokenizer_skips_no_bpe_chars(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # BPE + Metaspace but NO BPE chars in vocab
    config = {
        "model": {"type": "BPE", "vocab": {"hello": 1, "world": 2}},
        "pre_tokenizer": {"type": "Metaspace"},
    }

    with open(tokenizer_path, "w") as f:
      json.dump(config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      result_config = json.load(f)

    self.assertEqual(result_config["pre_tokenizer"]["type"], "Metaspace")

  def test_asr_export_config_rules(self):
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_asr_model",
        task=export_lib.exportable_module_config.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        split_cache=True,
        export_vision_encoder=True,
    )
    self.assertEqual(
        config.task,
        export_lib.exportable_module_config.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
    )
    self.assertTrue(config.export_audio_encoder)
    self.assertTrue(config.bundle_litert_lm)
    self.assertFalse(config.export_vision_encoder)
    self.assertFalse(config.split_cache)
    self.assertTrue(config.externalize_embedder)
    self.assertTrue(config.single_token_embedder)
    self.assertFalse(config.externalize_rope)
    self.assertEqual(config.input_sec, 1.0)
    self.assertEqual(config.stateful_after, -1)

    config_legacy = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_asr_model",
        task=export_lib.exportable_module_config.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        bundle_litert_lm=False,
    )
    self.assertFalse(config_legacy.bundle_litert_lm)
    self.assertFalse(config_legacy.export_audio_encoder)

  def test_exported_model_artifacts_audio_fields(self):
    artifacts = export_lib.ExportedModelArtifacts(
        audio_encoder_model_path="path/to/audio_encoder.tflite",
    )
    self.assertEqual(
        artifacts.audio_encoder_model_path, "path/to/audio_encoder.tflite"
    )

  def test_cache_lengths_config_rules(self):
    # Test default initialization
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        cache_length=1024,
    )
    self.assertEqual(config.cache_lengths, [1024])

    # Test explicit initialization
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        cache_lengths=[1024, 4096],
    )
    self.assertEqual(config.cache_lengths, [1024, 4096])

    # Test empty cache_lengths defaults to cache_length
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        cache_length=2048,
        cache_lengths=[],
    )
    self.assertEqual(config.cache_lengths, [2048])

    # Test error when dynamic shape is enabled with multiple cache lengths
    with self.assertRaisesRegex(
        ValueError,
        "Dynamic shape is not supported with multiple cache lengths.",
    ):
      export_lib.exportable_module_config.ExportableModuleConfig(
          model="dummy_model",
          enable_dynamic_shape=True,
          cache_lengths=[1024, 4096],
      )

  def test_gpu_dynamic_shapes_config_rules(self):
    # Only prefill
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        prefill_lengths=[32, 128],
        cache_length=1024,
        enable_gpu_dynamic_prefill=True,
    )
    self.assertEqual(config.prefill_lengths, [37, 131])
    self.assertEqual(config.cache_length, 1024)

    # Only cache
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        prefill_lengths=[32, 128],
        cache_length=1024,
        enable_gpu_dynamic_cache=True,
    )
    self.assertEqual(config.prefill_lengths, [32, 128])
    self.assertEqual(config.cache_length, 1031)

    # Both
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        prefill_lengths=[32, 128],
        cache_length=1024,
        enable_gpu_dynamic_prefill=True,
        enable_gpu_dynamic_cache=True,
    )
    self.assertEqual(config.prefill_lengths, [37, 131])
    self.assertEqual(config.cache_length, 1031)

  def test_gpu_dynamic_shapes_conflict(self):
    with self.assertRaises(ValueError):
      export_lib.exportable_module_config.ExportableModuleConfig(
          model="dummy_model",
          enable_dynamic_shape=True,
          enable_gpu_dynamic_prefill=True,
      )
    with self.assertRaises(ValueError):
      export_lib.exportable_module_config.ExportableModuleConfig(
          model="dummy_model",
          enable_dynamic_shape=True,
          enable_gpu_dynamic_cache=True,
      )

  @mock.patch(
      "litert_torch.generative.export_hf.core.export_lib.mu_pass_lib"
      ".update_model"
  )
  @mock.patch(
      "litert_torch.generative.export_hf.core.export_lib"
      ".get_prefill_decode_exportable_cls"
  )
  @mock.patch(
      "litert_torch.generative.export_hf.core.export_lib.converter_utils"
      ".Converter"
  )
  @mock.patch(
      "litert_torch.generative.export_hf.core.export_lib.model_ext_patches"
      ".get_patch_context"
  )
  def test_export_signature_naming(
      self,
      mock_patch_context,
      mock_converter_cls,
      mock_get_cls,
      mock_update_model,
  ):
    del mock_patch_context  # Unused.
    # Setup mocks
    mock_update_model.side_effect = lambda x: x
    mock_prefill_cls = mock.MagicMock()
    mock_decode_cls = mock.MagicMock()
    mock_get_cls.return_value = (mock_prefill_cls, mock_decode_cls)

    def mock_prefill_init(model, export_config, source_model_artifacts):
      del model, source_model_artifacts  # Unused.
      instance = mock.MagicMock()
      def get_sample_inputs(model_config):
        del model_config  # Unused.
        return {
            f"prefill_{length}": ({"input": 1}, None)
            for length in export_config.prefill_lengths
        }
      instance.get_sample_inputs.side_effect = get_sample_inputs
      return instance

    mock_prefill_cls.side_effect = mock_prefill_init

    mock_decode_instance = mock_decode_cls.return_value
    mock_decode_instance.get_sample_inputs.return_value = {
        "decode": ({"input": 2}, None)
    }

    mock_converter = mock_converter_cls.return_value
    added_signatures = []

    def mock_add_signature(name, module, sample_kwargs, dynamic_shapes=None):
      del module, sample_kwargs, dynamic_shapes  # Unused.
      added_signatures.append(name)
    mock_converter.add_signature.side_effect = mock_add_signature

    mock_model = mock.MagicMock()
    mock_artifacts = mock.MagicMock()
    mock_artifacts.model = mock_model
    mock_artifacts.model_config.model_type = "dummy"
    mock_artifacts.text_model_config = mock.MagicMock()

    # Scenario 1: Single cache length (backward compatible)
    config_single = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        cache_length=1024,
        quantization_recipe=None,
        work_dir=self.test_dir,
    )

    export_lib.export_text_prefill_decode_model(
        mock_artifacts, config_single, export_lib.ExportedModelArtifacts()
    )

    expected_sigs_single = [
        f"prefill_{length}" for length in config_single.prefill_lengths
    ] + ["decode"]
    self.assertEqual(added_signatures, expected_sigs_single)

    # Scenario 2: Multiple cache lengths
    added_signatures.clear()
    config_multi = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_model",
        cache_lengths=[1024, 2048],
        quantization_recipe=None,
        work_dir=self.test_dir,
    )

    export_lib.export_text_prefill_decode_model(
        mock_artifacts, config_multi, export_lib.ExportedModelArtifacts()
    )

    expected_sigs_multi = []
    for cache_len in config_multi.cache_lengths:
      for length in config_multi.prefill_lengths:
        expected_sigs_multi.append(f"prefill_{length}_cache_{cache_len}")
      expected_sigs_multi.append(f"decode_cache_{cache_len}")
    self.assertEqual(added_signatures, expected_sigs_multi)


def _model_with_experts(experts_cls, config):
  torch.manual_seed(0)
  model = torch.nn.Module()
  model.experts = experts_cls(config)
  with torch.no_grad():
    model.experts.gate_up_proj.normal_()
    model.experts.down_proj.normal_()
  return model


def _qwen3_moe_config(hidden_act):
  return transformers.Qwen3MoeConfig(
      hidden_size=16,
      moe_intermediate_size=8,
      num_experts=4,
      num_experts_per_tok=2,
      hidden_act=hidden_act,
  )


class LitertMoeActivationCheckTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("silu_module", "silu", "SiLUActivation"),
      ("exact_gelu", "gelu", "GELUActivation"),
  )
  def test_rejects_non_gelu_tanh_experts(self, hidden_act, act_name):
    model = _model_with_experts(
        modeling_qwen3_moe.Qwen3MoeExperts, _qwen3_moe_config(hidden_act)
    )

    with self.assertRaisesRegex(
        ValueError, f"`{act_name}` activation.*litert_moe_sequential"
    ):
      export_lib._check_litert_moe_activations(model)

  def test_rejects_silu_function_experts(self):
    # LFM2-MoE experts hold `F.silu` itself rather than an `ACT2FN` module.
    config = transformers.Lfm2MoeConfig(
        hidden_size=16, moe_intermediate_size=8, num_experts=4
    )
    model = _model_with_experts(modeling_lfm2_moe.Lfm2MoeExperts, config)

    with self.assertRaisesRegex(
        ValueError, "`silu` activation.*litert_moe_sequential"
    ):
      export_lib._check_litert_moe_activations(model)

  def test_accepts_gemma4_experts(self):
    config = modeling_gemma4.Gemma4TextConfig(
        hidden_size=16, moe_intermediate_size=8, num_experts=4, top_k_experts=2
    )
    model = _model_with_experts(modeling_gemma4.Gemma4TextExperts, config)

    export_lib._check_litert_moe_activations(model)

  @parameterized.named_parameters(
      ("gelu_pytorch_tanh", "gelu_pytorch_tanh"),
      ("gelu_new", "gelu_new"),
  )
  def test_gelu_tanh_experts_lower_to_moe_op(self, hidden_act):
    model = _model_with_experts(
        modeling_qwen3_moe.Qwen3MoeExperts, _qwen3_moe_config(hidden_act)
    )
    export_lib._check_litert_moe_activations(model)
    export_lib.pre_flatten_model_experts(model)

    class _LitertMoeExperts(torch.nn.Module):

      def __init__(self, experts):
        super().__init__()
        self.experts = experts

      def forward(self, hidden_states, top_k_index, top_k_weights):
        return moe.litert_moe_experts_forward(
            self.experts, hidden_states, top_k_index, top_k_weights
        )

    hidden_states = torch.randn(3, 16)
    top_k_weights, top_k_index = torch.topk(
        torch.randn(3, 4).softmax(dim=-1), 2, dim=-1
    )
    ep = torch.export.export(
        _LitertMoeExperts(model.experts).eval(),
        (hidden_states, top_k_index, top_k_weights),
    )
    ir_text = backend.export.exported_program_to_mlir(ep).get_text()
    self.assertIn('custom_code = "moe"', ir_text)


if __name__ == "__main__":
  absltest.main()
