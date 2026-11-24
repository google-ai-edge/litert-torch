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
"""Unit tests for the `transformers.exporters` LiteRT backend."""

import os
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch import model as lrt_model_lib
from litert_torch._convert import converter_v2
from litert_torch.generative.export_hf import export
from litert_torch.generative.export_hf import transformers_exporter
from litert_torch.generative.export_hf.core import export_lib
import torch
import transformers
from transformers.exporters import auto as exporters_auto
from transformers.exporters import utils as exporters_utils


def _make_tiny_qwen3_model() -> transformers.Qwen3ForCausalLM:
  """Builds a tiny randomly-initialized Qwen3ForCausalLM for fast tests."""
  config = transformers.Qwen3Config(
      vocab_size=64,
      hidden_size=32,
      intermediate_size=64,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=8,
      max_position_embeddings=64,
      tie_word_embeddings=True,
  )
  torch.manual_seed(0)
  return transformers.Qwen3ForCausalLM(config).eval()


class LiteRTConfigTest(parameterized.TestCase):

  def test_auto_registration_and_flat_from_dict_routing(self):
    cfg_dict = {
        "export_format": "litert",
        "backend": "npu:qualcomm",
        "prefill_lengths": [64],
        "quantization_recipe": "dynamic_wi8_afp32",
        "aot_backend": "qualcomm",
        "aot_soc_model": "sm8850",
    }
    supports_fn = getattr(
        exporters_auto.AutoHfExporter, "supports_export_format", None
    )
    if supports_fn is not None:
      self.assertTrue(supports_fn(cfg_dict))
    else:
      self.assertIn("litert", getattr(exporters_auto, "EXPORT_BACKENDS"))
    cfg = exporters_auto.AutoExportConfig.from_dict(cfg_dict)
    self.assertIsInstance(cfg, transformers_exporter.LiteRTConfig)
    self.assertEqual(cfg.target, "npu:qualcomm")
    self.assertEqual(
        cfg.export_kwargs,
        {
            "prefill_lengths": [64],
            "quantization_recipe": "dynamic_wi8_afp32",
            "aot_backend": "qualcomm",
            "aot_soc_model": "sm8850",
        },
    )

    exporter = exporters_auto.AutoHfExporter.from_config(cfg)
    self.assertIsInstance(exporter, transformers_exporter.LiteRTExporter)

  @parameterized.named_parameters(
      dict(testcase_name="default", target="cpu_gpu", expected="cpu_gpu"),
      dict(testcase_name="cpu_alias", target="cpu", expected="cpu_gpu"),
      dict(testcase_name="gpu_alias", target="gpu", expected="cpu_gpu"),
      dict(testcase_name="npu", target="npu", expected="npu"),
      dict(
          testcase_name="npu_vendor_alias",
          target="npu:qc",
          expected="npu:qualcomm",
      ),
  )
  def test_target_normalization(self, target, expected):
    cfg = transformers_exporter.LiteRTConfig(target=target)
    self.assertEqual(cfg.target, expected)

  def test_invalid_target_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown target"):
      transformers_exporter.LiteRTConfig(target="tpu")


class LiteRTExporterTest(parameterized.TestCase):

  def test_bundle_litert_lm_requires_tokenizer(self):
    model = _make_tiny_qwen3_model()
    exporter = transformers_exporter.LiteRTExporter()
    cfg = transformers_exporter.LiteRTConfig(
        export_kwargs={"bundle_litert_lm": True}
    )
    with self.assertRaisesRegex(ValueError, "tokenizer"):
      exporter.export_for_generation(model, config=cfg, tokenizer=None)

  def test_export_for_generation_delegates_to_export(self):
    output_dir = self.create_tempdir("out").full_path
    model = _make_tiny_qwen3_model()
    exporter = transformers_exporter.LiteRTExporter()
    cfg = transformers_exporter.LiteRTConfig.from_dict({
        "export_format": "litert",
        "target": "npu:qualcomm",
        "output_dir": output_dir,
        "aot_backend": "qualcomm",
        "aot_soc_model": "sm8850",
        "calibration_dataset_dir": "/fake/calib",
    })
    expected_path = os.path.join(output_dir, "model.litertlm")
    mock_export = self.enter_context(
        mock.patch.object(
            export,
            "export",
            autospec=True,
            return_value=export.ExportPlan(
                model="",
                task="text_generation",
                task_source="model",
                model_type="qwen3",
                export_kwargs={},
                sources={},
                target="npu:qualcomm",
                artifacts=export_lib.ExportedModelArtifacts(
                    prefill_decode_model_path="/tmp/model.tflite",
                    embedder_model_path="/tmp/embedder.tflite",
                    auxiliary_model_path="/tmp/auxiliary.tflite",
                    litert_lm_model_path=expected_path,
                ),
            ),
        )
    )

    artifacts = exporter.export_for_generation(
        model, config=cfg, tokenizer=mock.sentinel.tokenizer
    )
    self.assertEqual(artifacts, {"litert_lm": expected_path})
    mock_export.assert_called_once()
    call_kwargs = mock_export.call_args.kwargs
    self.assertIs(call_kwargs["model"], model)
    self.assertIs(call_kwargs["tokenizer"], mock.sentinel.tokenizer)
    self.assertEqual(call_kwargs["output_dir"], output_dir)
    self.assertEqual(call_kwargs["target"], "npu:qualcomm")
    self.assertEqual(call_kwargs["aot_backend"], "qualcomm")
    self.assertEqual(call_kwargs["aot_soc_model"], "sm8850")
    self.assertEqual(call_kwargs["calibration_dataset_dir"], "/fake/calib")
    self.assertTrue(call_kwargs["bundle_litert_lm"])
    self.assertTrue(call_kwargs["use_v2"])

  @absltest.skipUnless(
      converter_v2.is_supported(),
      "Converter V2 requires a newer TensorFlow / tf-nightly build.",
  )
  def test_export_single_graph_tiny_qwen3(self):
    output_dir = self.create_tempdir().full_path
    model = _make_tiny_qwen3_model()
    cfg = exporters_auto.AutoExportConfig.from_dict({
        "export_format": "litert",
        "output_dir": output_dir,
    })
    exporter = exporters_auto.AutoHfExporter.from_config(cfg)
    self.assertIsInstance(exporter, transformers_exporter.LiteRTExporter)
    assert isinstance(exporter, transformers_exporter.LiteRTExporter)
    sample_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "use_cache": False,
    }
    exported = exporter.export(model, sample_inputs, config=cfg)
    self.assertIsInstance(exported, lrt_model_lib.LiteRTModel)
    tflite_path = os.path.join(output_dir, "forward.tflite")
    self.assertTrue(os.path.exists(tflite_path))
    self.assertGreater(os.path.getsize(tflite_path), 0)

  @absltest.skipUnless(
      converter_v2.is_supported(),
      "Converter V2 requires a newer TensorFlow / tf-nightly build.",
  )
  def test_export_decomposed_hf_static_components_tiny_qwen3(self):
    """Mirrors `ExportGenerateTesterMixin`: `decompose_for_generation` + `export`."""
    output_dir = self.create_tempdir().full_path
    model = _make_tiny_qwen3_model()
    cfg = exporters_auto.AutoExportConfig.from_dict({
        "export_format": "litert",
        "output_dir": output_dir,
    })
    exporter = exporters_auto.AutoHfExporter.from_config(cfg)
    self.assertIsInstance(exporter, transformers_exporter.LiteRTExporter)
    assert isinstance(exporter, transformers_exporter.LiteRTExporter)
    sample_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    components = exporters_utils.decompose_for_generation(
        cast(Any, model),
        sample_inputs,
        generation_config=transformers.GenerationConfig(
            cache_implementation="static", max_cache_len=32
        ),
    )
    self.assertNotEmpty(components)
    for name, (submodel, sub_inputs) in components.items():
      artifact = exporter.export(
          submodel, sub_inputs, config=cfg, signature_name=name
      )
      self.assertIsInstance(artifact, lrt_model_lib.LiteRTModel)
      tflite_path = os.path.join(output_dir, f"{name}.tflite")
      self.assertTrue(os.path.exists(tflite_path))
      self.assertGreater(os.path.getsize(tflite_path), 0)

  @absltest.skipUnless(
      converter_v2.is_supported(),
      "Converter V2 requires a newer TensorFlow / tf-nightly build.",
  )
  def test_export_for_generation_litert_lm_tiny_qwen3(self):
    output_dir = self.create_tempdir().full_path
    model = _make_tiny_qwen3_model()
    cfg = exporters_auto.AutoExportConfig.from_dict({
        "export_format": "litert",
        "output_dir": output_dir,
        "prefill_lengths": [8],
        "cache_length": 32,
        "quantization_recipe": None,
        "apply_gpu_composites": False,
    })
    exporter = exporters_auto.AutoHfExporter.from_config(cfg)
    self.assertIsInstance(exporter, transformers_exporter.LiteRTExporter)
    assert isinstance(exporter, transformers_exporter.LiteRTExporter)
    artifacts = exporter.export_for_generation(model, config=cfg)
    self.assertIn("prefill_decode", artifacts)
    tflite_path = artifacts["prefill_decode"]
    self.assertTrue(os.path.exists(tflite_path))
    self.assertGreater(os.path.getsize(tflite_path), 0)

  @absltest.skipUnless(
      converter_v2.is_supported(),
      "Converter V2 requires a newer TensorFlow / tf-nightly build.",
  )
  def test_export_for_generation_npu_split_cache_tiny_qwen3(self):
    output_dir = self.create_tempdir().full_path
    model = _make_tiny_qwen3_model()
    cfg = exporters_auto.AutoExportConfig.from_dict({
        "export_format": "litert",
        "target": "npu",
        "output_dir": output_dir,
        "prefill_lengths": [8],
        "cache_length": 32,
        "quantization_recipe": None,
    })
    exporter = exporters_auto.AutoHfExporter.from_config(cfg)
    self.assertIsInstance(exporter, transformers_exporter.LiteRTExporter)
    assert isinstance(exporter, transformers_exporter.LiteRTExporter)
    artifacts = exporter.export_for_generation(model, config=cfg)
    for key in ("prefill_decode", "embedder", "auxiliary"):
      self.assertIn(key, artifacts)
      self.assertTrue(os.path.exists(artifacts[key]))
      self.assertGreater(os.path.getsize(artifacts[key]), 0)


if __name__ == "__main__":
  absltest.main()
