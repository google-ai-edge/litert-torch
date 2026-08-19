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


if __name__ == "__main__":
  absltest.main()
