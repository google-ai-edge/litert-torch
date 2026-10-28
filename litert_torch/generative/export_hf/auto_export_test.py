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
"""Tests for auto_export (local checkpoint configs only; no model loading)."""

import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf import auto_export


class AutoExportTest(parameterized.TestCase):

  def _checkpoint(self, config=None, model_index=None, name=None):
    ckpt_dir = self.create_tempdir(name)
    if config is not None:
      ckpt_dir.create_file("config.json", content=json.dumps(config))
    if model_index is not None:
      ckpt_dir.create_file("model_index.json", content=json.dumps(model_index))
    return ckpt_dir.full_path

  @parameterized.named_parameters(
      ("qwen3_0_6b_int4", 1024, "dynamic_wi4b32_afp32"),
      ("qwen3_1_7b_int8", 2048, "dynamic_wi8_afp32"),
      ("qwen3_4b_int4", 2560, "dynamic_wi4b32_afp32"),
  )
  def test_qwen3_llm_defaults(self, hidden_size, expected_quantization):
    model = self._checkpoint(
        {"model_type": "qwen3", "hidden_size": hidden_size}
    )

    plan = auto_export.resolve(model)

    self.assertEqual(plan.task, "text_generation")
    self.assertEqual(plan.task_source, auto_export.SOURCE_MODEL)
    self.assertEqual(plan.litert_lm_model_family, "qwen3")
    self.assertEqual(
        plan.export_kwargs,
        {
            "apply_gpu_composites": True,
            "externalize_weights": True,
            "prefill_lengths": [128, 1024],
            "cache_length": 32768,
            "quantization_recipe": expected_quantization,
        },
    )
    self.assertEqual(plan.sources["apply_gpu_composites"], "preset")
    self.assertEqual(plan.sources["quantization_recipe"], "model")

  def test_user_overrides_win_and_none_is_ignored(self):
    model = self._checkpoint({"model_type": "qwen3", "hidden_size": 1024})

    plan = auto_export.resolve(
        model,
        quantization_recipe="dynamic_wi8_afp32",
        apply_gpu_composites=False,
        cache_length=None,
    )

    self.assertEqual(
        plan.export_kwargs["quantization_recipe"], "dynamic_wi8_afp32"
    )
    self.assertFalse(plan.export_kwargs["apply_gpu_composites"])
    self.assertEqual(plan.export_kwargs["cache_length"], 32768)
    self.assertEqual(plan.sources["quantization_recipe"], "user")
    self.assertEqual(plan.sources["cache_length"], "preset")

  def test_preset_none_keeps_export_defaults(self):
    model = self._checkpoint({"model_type": "qwen3", "hidden_size": 1024})

    plan = auto_export.resolve(model, preset="none", cache_length=4096)

    self.assertEqual(plan.task, "text_generation")
    self.assertEqual(plan.export_kwargs, {"cache_length": 4096})

  def test_whisper_is_asr_without_llm_settings(self):
    model = self._checkpoint({"model_type": "whisper"})

    plan = auto_export.resolve(model)

    self.assertEqual(plan.task, "automatic_speech_recognition")
    self.assertEqual(plan.runtime_model_name, "whisper")
    self.assertEqual(plan.export_kwargs, {"input_sec": 30.0})

  def test_gemma4_e2b_is_vlm_with_e2b_chat_template(self):
    model = self._checkpoint({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size_per_layer_input": 256,
        },
    })

    plan = auto_export.resolve(model)

    self.assertEqual(plan.task, "image_text_to_text")
    self.assertTrue(plan.export_kwargs["export_vision_encoder"])
    self.assertTrue(plan.export_kwargs["externalize_embedder"])
    self.assertEqual(
        plan.litert_lm_chat_template, "gemma4/chat_template_e2b_e4b.jinja"
    )

  def test_vlm_exported_as_text_drops_vision_encoder(self):
    model = self._checkpoint(
        {"model_type": "lfm2_vl", "text_config": {"model_type": "lfm2"}}
    )

    plan = auto_export.resolve(model, task="text_generation")

    self.assertEqual(plan.task, "text_generation")
    self.assertEqual(plan.task_source, auto_export.SOURCE_USER)
    self.assertFalse(plan.export_kwargs["export_vision_encoder"])
    self.assertEqual(plan.litert_lm_model_family, "lfm2")

  def test_text_only_gemma3_is_text_generation(self):
    model = self._checkpoint(
        {"model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"]}
    )

    plan = auto_export.resolve(model)

    self.assertEqual(plan.task, "text_generation")
    self.assertFalse(plan.export_kwargs["export_vision_encoder"])

  def test_kokoro_without_model_type_is_tts(self):
    model = self._checkpoint({"istftnet": {}, "n_token": 178})

    plan = auto_export.resolve(model)

    self.assertEqual(plan.task, "text_to_speech")
    self.assertEqual(plan.runtime_model_name, "kokoro")

  def test_bonsai_diffusers_pipeline_is_text_to_image(self):
    model = self._checkpoint(
        model_index={"_class_name": "Flux2KleinPipeline"}, name="bonsai_ckpt"
    )

    plan = auto_export.resolve(model)

    self.assertEqual(plan.task, "text_to_image")
    self.assertEqual(plan.runtime_model_name, "bonsai-flux2")
    self.assertEqual(plan.export_kwargs["t2i_output_image_size"], 512)

  def test_unknown_causal_lm_gets_generic_llm_preset(self):
    model = self._checkpoint(
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    )

    plan = auto_export.resolve(model)

    self.assertEqual(plan.task, "text_generation")
    self.assertEqual(plan.task_source, auto_export.SOURCE_DEFAULT)
    self.assertIsNone(plan.litert_lm_model_family)
    self.assertTrue(plan.export_kwargs["apply_gpu_composites"])

  def test_unknown_task_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown task"):
      auto_export.resolve(self._checkpoint({}), task="video")

  def test_unknown_preset_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown preset"):
      auto_export.resolve(self._checkpoint({}), preset="fast")

  def test_export_forwards_plan_to_export_hf(self):
    model = self._checkpoint({"model_type": "whisper"})

    with mock.patch.object(
        auto_export.export_lib, "export", autospec=True
    ) as mock_export:
      plan = auto_export.export(model, "/tmp/out", input_sec=10.0)

    mock_export.assert_called_once_with(
        model=model,
        output_dir="/tmp/out",
        task="automatic_speech_recognition",
        input_sec=10.0,
    )
    self.assertEqual(plan.sources["input_sec"], "user")


if __name__ == "__main__":
  absltest.main()
