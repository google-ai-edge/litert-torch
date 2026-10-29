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
"""Tests for HF model export orchestration and PTQ pipeline."""

from unittest import mock

from absl.testing import absltest
from litert_torch.generative.export_hf import export
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import litert_lm_builder


class ExportNpuCompilationPipelineTest(absltest.TestCase):

  @mock.patch("litert_torch.generative.export_hf.export.run_export_tasks")
  @mock.patch("litert_torch.generative.export_hf.export_utils.calib_func")
  @mock.patch("litert_torch.generative.export_hf.export_utils.quant_func")
  @mock.patch("litert_torch.generative.export_hf.export_utils.compile_func")
  @mock.patch("transformers.AutoConfig.from_pretrained")
  def test_npu_compilation_pipeline_orchestration(
      self,
      mock_auto_config,
      mock_compile,
      mock_quant,
      mock_calib,
      mock_run_export,
  ):
    # Setup mocks
    mock_run_export.return_value = mock.MagicMock(
        litert_lm_model_path="/tmp/fake_model.litertlm"
    )
    mock_config = mock.MagicMock()
    mock_config.model_type = "gemma3"
    mock_auto_config.return_value = mock_config

    # Execute
    export.export(
        model="google/gemma-3-270m-it",
        output_dir="/tmp/fake_output",
        calibration_dataset_dir="/tmp/fake_dataset",
        quantization_recipe="dynamic_wi8_afp32",
        static_quantization_recipe="custom_recipe.json",
        aot_backend="qualcomm",
        aot_soc_model="sm8850",
    )

    # Verify Stage 1 (Export)
    mock_run_export.assert_called_once()
    # Check that quantization_recipe on config was set to Stage 1 weight recipe
    passed_config = mock_run_export.call_args[0][1]
    self.assertEqual(passed_config.quantization_recipe, "dynamic_wi8_afp32")

    # Verify Stage 2 (Calibrate)
    mock_calib.assert_called_once_with(
        input_litertlm="/tmp/fake_model.litertlm",
        calibration_dataset_dir="/tmp/fake_dataset",
        calibration_result_save_dir=mock.ANY,
        kv_cache_max_len=mock.ANY,
        calibration_dataset_format="jsonl",
        calibration_eval_task_names="ALL",
        max_calibration_decode_steps=32,
        use_profiler_based_calibration=True,
        enable_min_max_calibration_update=True,
        ema_smoothing_factor=0.1,
    )

    # Verify Stage 3 (Quantize)
    mock_quant.assert_called_once_with(
        input_litertlm="/tmp/fake_model.litertlm",
        output_litertlm=mock.ANY,
        calibration_dir=mock.ANY,
        use_float_input_output_normalizer=False,
        skip_mlir_passes=True,
        calibration_range_scale=1.0,
        quantization_recipe="custom_recipe.json",
    )

    # Verify Stage 4 (Compile)
    mock_compile.assert_called_once_with(
        input_litertlm=mock.ANY,
        output_litertlm="/tmp/fake_output/model.litertlm",
        backend="qualcomm",
        soc_model="sm8850",
        compile_configs=None,
        model_name="gemma3",
        overwrite=True,
    )

  def test_npu_compilation_pipeline_early_validation(self):
    with self.assertRaisesRegex(ValueError, "static quantization was disabled"):
      export.export(
          model="google/gemma-3-270m-it",
          output_dir="/tmp/fake_output",
          calibration_dataset_dir="/tmp/fake_dataset",
          static_quantization_recipe="none",
      )

  @mock.patch("litert_torch.generative.export_hf.export.run_export_tasks")
  @mock.patch("litert_torch.generative.export_hf.export_utils.compile_func")
  @mock.patch("transformers.AutoConfig.from_pretrained")
  def test_export_and_compile_only_orchestration(
      self, mock_auto_config, mock_compile, mock_run_export
  ):
    mock_run_export.return_value = mock.MagicMock(
        litert_lm_model_path="/tmp/fake_model.litertlm"
    )
    mock_config = mock.MagicMock()
    mock_config.model_type = "gemma3"
    mock_auto_config.return_value = mock_config

    export.export(
        model="google/gemma-3-270m-it",
        output_dir="/tmp/fake_output",
        aot_backend="qualcomm",
        aot_soc_model="sm8850",
        use_litert_lm_compiler=True,
    )

    mock_run_export.assert_called_once()
    passed_tasks = mock_run_export.call_args[0][0]

    # Verify that bundle packaging is included, but no legacy compilation
    # task is.
    self.assertIn(litert_lm_builder.package_model, passed_tasks)
    self.assertNotIn(export_lib.compile_litertlm_bundle, passed_tasks)
    self.assertNotIn(export_lib.aot_compile_model, passed_tasks)

    # Verify Stage 4 Compilation was called on the unquantized bundle directly
    mock_compile.assert_called_once_with(
        input_litertlm="/tmp/fake_model.litertlm",
        output_litertlm="/tmp/fake_output/model.litertlm",
        backend="qualcomm",
        soc_model="sm8850",
        compile_configs=None,
        model_name="gemma3",
        overwrite=True,
    )

  @mock.patch("litert_torch.generative.export_hf.export.run_export_tasks")
  def test_text_to_image_export_orchestration(self, mock_run_export):
    mock_run_export.return_value = mock.MagicMock(
        litert_lm_model_path="/tmp/fake_image_gen.litertlm"
    )
    export.export(
        model="/tmp/fake_bonsai_flux2",
        output_dir="/tmp/fake_output",
        task="text_to_image",
        t2i_output_image_size=256,
        bundle_litert_lm=True,
    )
    mock_run_export.assert_called_once()
    passed_tasks = mock_run_export.call_args[0][0]
    passed_config = mock_run_export.call_args[0][1]
    self.assertEqual(
        passed_config.task,
        export.ExportTask.TEXT_TO_IMAGE,
    )
    self.assertEqual(passed_config.t2i_output_image_size, 256)
    self.assertEqual(
        passed_tasks,
        [
            export_lib.export_image_gen_models,
            export_lib.export_tokenizer,
            litert_lm_builder.package_model,
        ],
    )

  @mock.patch("transformers.AutoTokenizer.from_pretrained")
  @mock.patch(
      "litert_torch.generative.export_hf.model_ext.exportables.get_image_gen_model_cls"
  )
  @mock.patch("huggingface_hub.snapshot_download")
  def test_load_model_text_to_image_remote_hf_repo(
      self, mock_snapshot_download, mock_get_cls, mock_tokenizer
  ):
    import json  # pylint: disable=g-import-not-at-top
    import os  # pylint: disable=g-import-not-at-top
    import tempfile  # pylint: disable=g-import-not-at-top

    with tempfile.TemporaryDirectory() as tmp_dir:
      with open(os.path.join(tmp_dir, "model_index.json"), "w") as f:
        json.dump({"_class_name": "Flux2KleinPipeline"}, f)
      mock_snapshot_download.return_value = tmp_dir
      fake_model_instance = mock.MagicMock()
      fake_model_instance.model_dir = tmp_dir
      mock_model_cls = mock.MagicMock(return_value=fake_model_instance)
      mock_get_cls.return_value = mock_model_cls
      mock_tokenizer.return_value = mock.MagicMock()

      config = export.exportable_module_config.ExportableModuleConfig(
          model="prism-ml/bonsai-image-ternary-4B-unpacked",
          output_dir=tmp_dir,
          task=export.ExportTask.TEXT_TO_IMAGE,
      )
      export_lib.load_model(
          "prism-ml/bonsai-image-ternary-4B-unpacked",
          export_config=config,
          task=export.ExportTask.TEXT_TO_IMAGE,
      )

      mock_snapshot_download.assert_called_once_with(
          "prism-ml/bonsai-image-ternary-4B-unpacked"
      )
      mock_get_cls.assert_called_once_with("bonsai_flux2")
      mock_model_cls.assert_called_once_with(tmp_dir, export_config=config)

  @mock.patch("transformers.AutoTokenizer.from_pretrained")
  @mock.patch(
      "litert_torch.generative.export_hf.model_ext.exportables.get_image_gen_model_cls"
  )
  @mock.patch("huggingface_hub.snapshot_download")
  def test_load_model_text_to_image_flux2_klein_remote_hf_repo(
      self, mock_snapshot_download, mock_get_cls, mock_tokenizer
  ):
    import json  # pylint: disable=g-import-not-at-top
    import os  # pylint: disable=g-import-not-at-top
    import tempfile  # pylint: disable=g-import-not-at-top

    with tempfile.TemporaryDirectory() as tmp_dir:
      with open(os.path.join(tmp_dir, "model_index.json"), "w") as f:
        json.dump({"_class_name": "Flux2KleinPipeline"}, f)
      mock_snapshot_download.return_value = tmp_dir
      fake_model_instance = mock.MagicMock()
      fake_model_instance.model_dir = tmp_dir
      mock_model_cls = mock.MagicMock(return_value=fake_model_instance)
      mock_get_cls.return_value = mock_model_cls
      mock_tokenizer.return_value = mock.MagicMock()

      config = export.exportable_module_config.ExportableModuleConfig(
          model="black-forest-labs/FLUX.2-klein-4B",
          output_dir=tmp_dir,
          task=export.ExportTask.TEXT_TO_IMAGE,
      )
      export_lib.load_model(
          "black-forest-labs/FLUX.2-klein-4B",
          export_config=config,
          task=export.ExportTask.TEXT_TO_IMAGE,
      )

      mock_snapshot_download.assert_called_once_with(
          "black-forest-labs/FLUX.2-klein-4B"
      )
      mock_get_cls.assert_called_once_with("Flux2KleinPipeline")
      mock_model_cls.assert_called_once_with(tmp_dir, export_config=config)

  @mock.patch("litert_torch.generative.export_hf.export.run_export_tasks")
  def test_bundled_speech_to_text_dispatches_after_model_load(
      self, mock_run_export
  ):
    mock_run_export.return_value = mock.MagicMock(
        litert_lm_model_path="/tmp/fake_whisper_output/model.litertlm"
    )
    export.export(
        model="openai/whisper-tiny",
        output_dir="/tmp/fake_whisper_output",
        task="automatic_speech_recognition",
    )
    passed_tasks = mock_run_export.call_args[0][0]
    self.assertEqual(passed_tasks, [export.export_and_package_speech_to_text])

  @mock.patch("litert_torch.generative.export_hf.export.run_export_tasks")
  def test_text_to_speech_kokoro_export_orchestration(self, mock_run_export):
    mock_run_export.return_value = mock.MagicMock(
        litert_lm_model_path=None,
        additional_model_paths={
            "kokoro_acoustic": "/tmp/kokoro_acoustic.tflite"
        },
    )
    export.export(
        model="hexgrad/Kokoro-82M",
        output_dir="/tmp/fake_kokoro_output",
        task="text_to_speech",
    )
    mock_run_export.assert_called_once()
    passed_tasks = mock_run_export.call_args[0][0]
    passed_config = mock_run_export.call_args[0][1]
    self.assertEqual(passed_config.task, export.ExportTask.TEXT_TO_SPEECH)
    self.assertFalse(passed_config.bundle_litert_lm)
    self.assertEqual(passed_tasks, [export_lib.export_tts_models])


if __name__ == "__main__":
  absltest.main()
