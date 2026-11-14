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
"""Tests for HF model export orchestration, plan resolution, and PTQ pipeline."""

import dataclasses
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf import export
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import litert_lm_builder
from litert_torch.generative.export_hf.model_ext import extension


class ExportNpuCompilationPipelineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(export, "_fetch_hub_json", return_value=None)
    )

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

  def _asr_tasks_after_model_load(self, model_type, **kwargs):
    config = export.exportable_module_config.ExportableModuleConfig(
        model="fake/asr",
        output_dir="/tmp/fake_asr_output",
        task=export.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        **kwargs,
    )
    source = export_lib.SourceModelArtifacts(
        model=mock.MagicMock(),
        model_config=mock.MagicMock(model_type=model_type),
        text_model_config=None,
        tokenizer=None,
    )
    updated = export_lib.update_export_config(config, source)
    return updated, export._build_export_tasks(updated)[0]

  def test_bundled_standalone_asr_packages_speech2text_model(self):
    config, tasks = self._asr_tasks_after_model_load("whisper")

    self.assertFalse(config.externalize_embedder)
    self.assertFalse(config.export_audio_encoder)
    self.assertFalse(config.single_token_embedder)
    self.assertEqual(
        tasks,
        [
            export_lib.export_asr_models,
            export_lib.export_tokenizer,
            litert_lm_builder.package_model,
        ],
    )

  def test_bundled_llm_asr_uses_llm_export_path(self):
    config, tasks = self._asr_tasks_after_model_load("qwen3_asr")

    self.assertTrue(config.export_audio_encoder)
    self.assertTrue(config.externalize_embedder)
    self.assertTrue(config.single_token_embedder)
    replaced = dataclasses.replace(config, cache_length=512)
    self.assertTrue(replaced.export_audio_encoder)
    self.assertTrue(replaced.externalize_embedder)
    self.assertTrue(replaced.single_token_embedder)
    self.assertEqual(
        tasks,
        [
            export_lib.export_text_prefill_decode_model,
            export_lib.export_embedder_model,
            export_lib.export_audio_encoder_models,
            export_lib.export_tokenizer,
            litert_lm_builder.package_model,
        ],
    )

  def test_unbundled_llm_asr_exports_submodels_without_packaging(self):
    config, tasks = self._asr_tasks_after_model_load(
        "qwen3_asr", bundle_litert_lm=False
    )

    self.assertTrue(config.export_audio_encoder)
    self.assertTrue(config.externalize_embedder)
    self.assertTrue(config.single_token_embedder)
    self.assertEqual(
        tasks,
        [
            export_lib.export_text_prefill_decode_model,
            export_lib.export_embedder_model,
            export_lib.export_audio_encoder_models,
        ],
    )

  def test_explicit_export_audio_encoder_false_preserves_standalone_asr_path(
      self,
  ):
    config, tasks = self._asr_tasks_after_model_load(
        "qwen3_asr", bundle_litert_lm=True, export_audio_encoder=False
    )

    self.assertFalse(config.export_audio_encoder)
    self.assertFalse(config.externalize_embedder)
    self.assertFalse(config.single_token_embedder)
    self.assertFalse(config.bundle_litert_lm)
    self.assertEqual(tasks, [export_lib.export_asr_models])

    # Also verify dataclasses.replace on a default ASR config before
    # update_export_config does not flip _explicit_export_audio_encoder=True.
    default_cfg = export.exportable_module_config.ExportableModuleConfig(
        model="qwen3_asr",
        task=export.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
    )
    replaced_default = dataclasses.replace(default_cfg, cache_length=512)
    self.assertFalse(replaced_default._explicit_export_audio_encoder)

  def test_unbundled_asr_exports_only_the_tflite_model(self):
    _, tasks = self._asr_tasks_after_model_load(
        "whisper", bundle_litert_lm=False
    )

    self.assertEqual(tasks, [export_lib.export_asr_models])

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


class ExportPlanAndPresetResolutionTest(parameterized.TestCase):

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

    plan = export.export(model, dry_run=True)

    self.assertEqual(plan.task, "text_generation")
    self.assertEqual(plan.task_source, export.SOURCE_MODEL)
    self.assertEqual(plan.target, "cpu_gpu")
    self.assertEqual(
        plan.export_kwargs,
        {
            "apply_gpu_composites": True,
            "externalize_weights": True,
            "prefill_lengths": [128, 1024],
            "cache_length": 32771,
            "llm_metadata_max_num_tokens_override": 4096,
            "quantization_recipe": expected_quantization,
            "use_jinja_template": True,
            "litert_lm_llm_metadata_override": "qwen3",
        },
    )
    self.assertEqual(plan.sources["apply_gpu_composites"], "target")
    self.assertEqual(plan.sources["cache_length"], "target")
    self.assertEqual(
        plan.sources["llm_metadata_max_num_tokens_override"], "target"
    )
    self.assertEqual(plan.sources["quantization_recipe"], "model")
    self.assertEqual(plan.sources["use_jinja_template"], "model")
    self.assertEqual(
        plan.sources["litert_lm_llm_metadata_override"], "model"
    )

  def test_user_overrides_win_and_none_is_ignored(self):
    model = self._checkpoint({"model_type": "qwen3", "hidden_size": 1024})

    plan = export.export(
        model,
        dry_run=True,
        quantization_recipe="dynamic_wi8_afp32",
        apply_gpu_composites=False,
        cache_length=None,
    )

    self.assertEqual(
        plan.export_kwargs["quantization_recipe"], "dynamic_wi8_afp32"
    )
    self.assertFalse(plan.export_kwargs["apply_gpu_composites"])
    self.assertEqual(plan.export_kwargs["cache_length"], 32771)
    self.assertEqual(
        plan.export_kwargs["llm_metadata_max_num_tokens_override"], 4096
    )
    self.assertEqual(plan.sources["quantization_recipe"], "user")
    self.assertEqual(plan.sources["cache_length"], "target")

  def test_explicit_quantization_recipe_none_disables_quantization(self):
    model = self._checkpoint({"model_type": "qwen3", "hidden_size": 1024})

    plan = export.export(
        model,
        dry_run=True,
        quantization_recipe=None,
    )

    self.assertIsNone(plan.export_kwargs["quantization_recipe"])
    self.assertEqual(plan.sources["quantization_recipe"], "user")

  @parameterized.parameters(None, "", "cpu_gpu", "cpu", "gpu", "CPU", "GPU")
  def test_cpu_gpu_target_aliases_resolve_identically(self, target):
    model = self._checkpoint({"model_type": "qwen3", "hidden_size": 1024})

    plan = export.export(model, target=target, dry_run=True)

    self.assertEqual(plan.target, "cpu_gpu")
    self.assertEqual(plan.export_kwargs["cache_length"], 32771)
    self.assertEqual(
        plan.export_kwargs["llm_metadata_max_num_tokens_override"], 4096
    )
    self.assertTrue(plan.export_kwargs["apply_gpu_composites"])
    self.assertTrue(plan.export_kwargs["externalize_weights"])

  def test_npu_generic_target_resolves_split_cache_defaults(self):
    model = self._checkpoint({"model_type": "qwen3", "hidden_size": 1024})

    plan = export.export(model, target="npu", dry_run=True)

    self.assertEqual(plan.target, "npu")
    self.assertEqual(
        plan.export_kwargs,
        {
            "split_cache": True,
            "apply_gpu_composites": False,
            "externalize_embedder": True,
            "prefill_lengths": [128],
            "cache_length": 1024,
            "quantization_recipe": "dynamic_wi8_afp32",
            "use_jinja_template": True,
            "litert_lm_llm_metadata_override": "qwen3",
        },
    )
    self.assertEqual(plan.sources["split_cache"], "target")
    self.assertEqual(plan.sources["cache_length"], "target")

  @parameterized.named_parameters(
      ("qualcomm", "npu:qualcomm", "npu:qualcomm", 3, 2),
      ("qc_alias", "npu:qc", "npu:qualcomm", 3, 2),
      ("qnn_alias", "npu:qnn", "npu:qualcomm", 3, 2),
      ("mediatek", "npu:mediatek", "npu:mediatek", 2, 3),
      ("mtk_alias", "npu:mtk", "npu:mediatek", 2, 3),
      ("intel", "npu:intel", "npu:intel", 2, 3),
      ("google_tensor", "npu:google_tensor", "npu:google_tensor", 2, 3),
      ("tensor_alias", "npu:tensor", "npu:google_tensor", 2, 3),
      ("samsung", "npu:samsung", "npu:samsung", 2, 3),
      ("slsi_alias", "npu:slsi", "npu:samsung", 2, 3),
  )
  def test_npu_vendor_targets_resolve_transpose_indices(
      self, target, expected_canonical, expected_k_ts_idx, expected_v_ts_idx
  ):
    model = self._checkpoint({"model_type": "lfm2"})

    plan = export.export(model, target=target, dry_run=True)

    self.assertEqual(plan.target, expected_canonical)
    self.assertTrue(plan.export_kwargs["split_cache"])
    self.assertFalse(plan.export_kwargs["apply_gpu_composites"])
    self.assertTrue(plan.export_kwargs["externalize_embedder"])
    self.assertEqual(plan.export_kwargs["prefill_lengths"], [128])
    self.assertEqual(plan.export_kwargs["cache_length"], 1024)
    self.assertEqual(
        plan.export_kwargs["quantization_recipe"], "dynamic_wi8_afp32"
    )
    self.assertEqual(plan.export_kwargs["k_ts_idx"], expected_k_ts_idx)
    self.assertEqual(plan.export_kwargs["v_ts_idx"], expected_v_ts_idx)
    self.assertEqual(plan.sources["k_ts_idx"], "target")
    self.assertEqual(plan.sources["v_ts_idx"], "target")

  @parameterized.parameters(
      "npu:qualcomm:sm8750",
      "npu:qc:SM8850",
      "npu:mediatek:mt6991",
      "npu:intel:lunar_lake",
  )
  def test_npu_soc_target_raises_not_implemented_error(self, target):
    model = self._checkpoint({"model_type": "qwen3", "hidden_size": 1024})

    with self.assertRaisesRegex(
        NotImplementedError, "SoC-targeted NPU AOT compilation"
    ):
      export.export(model, target=target, dry_run=True)

  def test_whisper_is_asr_without_llm_settings(self):
    model = self._checkpoint({"model_type": "whisper"})

    plan = export.export(model, dry_run=True)

    self.assertEqual(plan.task, "automatic_speech_recognition")
    self.assertEqual(plan.export_kwargs, {"input_sec": 30.0})

  def test_gemma4_vlm_without_audio_is_image_text_to_text(self):
    model = self._checkpoint({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size_per_layer_input": 256,
        },
    })

    plan = export.export(model, dry_run=True)

    self.assertEqual(plan.task, "image_text_to_text")
    self.assertTrue(plan.export_kwargs["export_vision_encoder"])
    self.assertFalse(plan.export_kwargs["export_audio_encoder"])
    self.assertTrue(plan.export_kwargs["externalize_embedder"])

  def test_gemma4_with_audio_config_is_multimodal_lm(self):
    model = self._checkpoint({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size_per_layer_input": 256,
        },
        "vision_config": {"model_type": "gemma4_vision"},
        "audio_config": {"model_type": "gemma4_audio"},
    })

    plan = export.export(model, dry_run=True)

    self.assertEqual(plan.task, "multimodal_lm")
    self.assertTrue(plan.export_kwargs["export_vision_encoder"])
    self.assertTrue(plan.export_kwargs["export_audio_encoder"])
    self.assertTrue(plan.export_kwargs["externalize_embedder"])

  def test_vlm_exported_as_text_drops_vision_encoder(self):
    model = self._checkpoint(
        {"model_type": "lfm2_vl", "text_config": {"model_type": "lfm2"}}
    )

    plan = export.export(model, task="text_generation", dry_run=True)

    self.assertEqual(plan.task, "text_generation")
    self.assertEqual(plan.task_source, export.SOURCE_USER)
    self.assertFalse(plan.export_kwargs["export_vision_encoder"])

  def test_text_only_gemma3_is_text_generation(self):
    model = self._checkpoint(
        {"model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"]}
    )

    plan = export.export(model, dry_run=True)

    self.assertEqual(plan.task, "text_generation")
    self.assertFalse(plan.export_kwargs["export_vision_encoder"])

  def test_kokoro_without_model_type_is_tts(self):
    model = self._checkpoint({"istftnet": {}, "n_token": 178})

    plan = export.export(model, dry_run=True)

    self.assertEqual(plan.task, "text_to_speech")
    self.assertEqual(plan.model_type, "kokoro")

  def test_bonsai_diffusers_pipeline_is_text_to_image(self):
    model = self._checkpoint(
        model_index={"_class_name": "Flux2KleinPipeline"}, name="bonsai_ckpt"
    )

    plan = export.export(model, dry_run=True)

    self.assertEqual(plan.task, "text_to_image")
    self.assertEqual(plan.model_type, "bonsai_flux2")
    self.assertEqual(plan.export_kwargs["t2i_output_image_size"], 512)

  def test_unknown_causal_lm_gets_generic_llm_preset(self):
    model = self._checkpoint(
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    )

    plan = export.export(model, dry_run=True)

    self.assertEqual(plan.task, "text_generation")
    self.assertEqual(plan.task_source, export.SOURCE_DEFAULT)
    self.assertTrue(plan.export_kwargs["apply_gpu_composites"])
    self.assertFalse(plan.export_kwargs["use_jinja_template"])
    self.assertEqual(plan.sources["use_jinja_template"], "target")

  def test_minicpm5_llama_checkpoint_matches_minicpm5_extension(self):
    model = self._checkpoint(
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "_name_or_path": "openbmb/MiniCPM5-1B",
        }
    )

    plan = export.export(model, dry_run=True)

    self.assertEqual(plan.task, "text_generation")
    self.assertEqual(plan.model_type, "llama")
    self.assertTrue(plan.export_kwargs["use_jinja_template"])
    self.assertEqual(
        plan.export_kwargs["litert_lm_llm_metadata_override"], "minicpm5"
    )
    self.assertEqual(
        plan.sources["litert_lm_llm_metadata_override"], export.SOURCE_MODEL
    )

  @parameterized.parameters(None, "none", "NONE")
  def test_disabling_llm_metadata_override_resets_use_jinja_template(
      self, none_value
  ):
    model = self._checkpoint({"model_type": "qwen3", "hidden_size": 1024})

    plan = export.export(
        model,
        dry_run=True,
        litert_lm_llm_metadata_override=none_value,
    )

    self.assertIsNone(plan.export_kwargs["litert_lm_llm_metadata_override"])
    self.assertEqual(
        plan.sources["litert_lm_llm_metadata_override"], export.SOURCE_USER
    )
    self.assertFalse(plan.export_kwargs["use_jinja_template"])

  @parameterized.named_parameters(
      (
          "qwen2",
          {"model_type": "qwen2"},
          "text_generation",
          {
              "use_jinja_template": True,
              "litert_lm_llm_metadata_override": "qwen2_5",
          },
      ),
      (
          "qwen3_5",
          {"model_type": "qwen3_5"},
          "text_generation",
          {
              "quantization_recipe": "dynamic_wi8_afp32",
              "use_jinja_template": True,
              "litert_lm_llm_metadata_override": "qwen3_5",
          },
      ),
      (
          "lfm2",
          {"model_type": "lfm2"},
          "text_generation",
          {
              "quantization_recipe": "dynamic_wi4b32_afp32",
              "use_jinja_template": True,
              "litert_lm_llm_metadata_override": "lfm2",
          },
      ),
      (
          "lfm2_vl",
          {"model_type": "lfm2_vl", "text_config": {"model_type": "lfm2"}},
          "image_text_to_text",
          {
              "quantization_recipe": "dynamic_wi4b32_afp32",
              "export_vision_encoder": True,
              "use_jinja_template": True,
              "litert_lm_llm_metadata_override": "lfm2",
          },
      ),
      (
          "gemma3_vlm",
          {
              "model_type": "gemma3",
              "architectures": ["Gemma3ForConditionalGeneration"],
          },
          "image_text_to_text",
          {
              "export_vision_encoder": True,
              "externalize_embedder": True,
              "vision_encoder_quantization_recipe": "weight_only_wi8_afp32",
              "use_jinja_template": False,
          },
      ),
      (
          "gemma3n",
          {"model_type": "gemma3n"},
          "image_text_to_text",
          {
              "export_vision_encoder": True,
              "externalize_embedder": True,
              "single_token_embedder": True,
              "vision_encoder_quantization_recipe": "weight_only_wi8_afp32",
              "use_jinja_template": False,
          },
      ),
      (
          "gemma4",
          {
              "model_type": "gemma4",
              "text_config": {
                  "model_type": "gemma4_text",
                  "hidden_size_per_layer_input": 256,
              },
          },
          "image_text_to_text",
          {
              "quantization_recipe": "dynamic_wi4b32_afp32",
              "externalize_embedder": True,
              "single_token_embedder": True,
              "export_vision_encoder": True,
              "export_audio_encoder": False,
              "use_jinja_template": True,
              "jinja_chat_template_override": (
                  "litert-community/gemma-4-E2B-it-litert-lm"
              ),
          },
      ),
      (
          "gemma4_26b_moe",
          {
              "model_type": "gemma4",
              "text_config": {
                  "model_type": "gemma4_text",
                  "enable_moe_block": True,
                  "num_experts": 128,
              },
          },
          "image_text_to_text",
          {
              "quantization_recipe": "dynamic_wi4b32_afp32",
              "externalize_embedder": True,
              "single_token_embedder": True,
              "export_vision_encoder": True,
              "export_audio_encoder": False,
              "use_jinja_template": True,
              "litert_lm_llm_metadata_override": "gemma4_26b_31b",
          },
      ),
      (
          "gemma4_31b",
          {
              "model_type": "gemma4",
              "text_config": {
                  "model_type": "gemma4_text",
                  "hidden_size": 5376,
                  "hidden_size_per_layer_input": 0,
              },
          },
          "image_text_to_text",
          {
              "quantization_recipe": "dynamic_wi4b32_afp32",
              "externalize_embedder": True,
              "single_token_embedder": True,
              "export_vision_encoder": True,
              "export_audio_encoder": False,
              "use_jinja_template": True,
              "litert_lm_llm_metadata_override": "gemma4_26b_31b",
          },
      ),
      (
          "gemma4_unified",
          {"model_type": "gemma4_unified"},
          "image_text_to_text",
          {
              "export_vision_encoder": True,
              "export_audio_encoder": False,
              "use_jinja_template": True,
              "litert_lm_llm_metadata_override": "gemma4_12b",
          },
      ),
      (
          "gemma4_unified_audio",
          {
              "model_type": "gemma4_unified",
              "audio_config": {"model_type": "gemma4_unified_audio"},
          },
          "multimodal_lm",
          {
              "export_vision_encoder": True,
              "export_audio_encoder": True,
              "use_jinja_template": True,
              "litert_lm_llm_metadata_override": "gemma4_12b",
          },
      ),
      (
          "moonshine",
          {"model_type": "moonshine"},
          "automatic_speech_recognition",
          {"input_sec": 5.0},
      ),
      (
          "parakeet_ctc",
          {"model_type": "parakeet_ctc"},
          "automatic_speech_recognition",
          {"input_sec": 5.0},
      ),
      (
          "parakeet_tdt",
          {"model_type": "parakeet_tdt"},
          "automatic_speech_recognition",
          {"input_sec": 5.0},
      ),
      (
          "qwen3_asr",
          {"model_type": "qwen3_asr"},
          "automatic_speech_recognition",
          {"input_sec": 5.0},
      ),
      (
          "qwen3_tts",
          {"model_type": "qwen3_tts"},
          "text_to_speech",
          {},
      ),
      ("minicpm_o", {"model_type": "minicpmo"}, "any_to_any", {}),
  )
  def test_builtin_extension_defaults(self, config, task, model_kwargs):
    plan = export.export(self._checkpoint(config), dry_run=True)

    self.assertEqual(plan.task, task)
    model_values = {
        key: value
        for key, value in plan.export_kwargs.items()
        if plan.sources[key] == export.SOURCE_MODEL
    }
    self.assertEqual(model_values, model_kwargs)

  def test_flux2_klein_pipeline_is_text_to_image(self):
    model = self._checkpoint(
        model_index={"_class_name": "Flux2KleinPipeline"}, name="klein_ckpt"
    )

    plan = export.export(model, dry_run=True)

    self.assertEqual(plan.task, "text_to_image")
    self.assertEqual(plan.model_type, "Flux2KleinPipeline")

  def test_gemma3n_exported_as_text_keeps_external_embedder(self):
    plan = export.export(
        self._checkpoint({"model_type": "gemma3n"}),
        task="text_generation",
        dry_run=True,
    )

    self.assertTrue(plan.export_kwargs["externalize_embedder"])
    self.assertFalse(plan.export_kwargs["export_vision_encoder"])

  def test_gemma4_text_only_is_text_generation(self):
    plan = export.export(
        self._checkpoint({"model_type": "gemma4_text"}), dry_run=True
    )

    self.assertEqual(plan.task, "text_generation")
    self.assertFalse(plan.export_kwargs["export_vision_encoder"])
    self.assertFalse(plan.export_kwargs["export_audio_encoder"])

  @parameterized.named_parameters(
      ("whisper", "whisper", "whisper", 0, 0, False),
      ("moonshine", "moonshine", "moonshine", 10000, 0, False),
      ("parakeet_ctc", "parakeet_ctc", "parakeet", 10000, 1000, False),
      ("parakeet_tdt", "parakeet_tdt", "parakeet", 10000, 1000, False),
      ("qwen3_asr_is_packaged_as_llm", "qwen3_asr", None, 0, 0, True),
  )
  def test_speech2text_packaging_info(
      self,
      model_type,
      asr_model_type,
      expected_input_ms,
      expected_mel_frames,
      is_llm_asr,
  ):
    ext = extension.get_model_extension(model_type)
    model_cls = ext.get_speech_model_cls(model_type)
    cfg = export.exportable_module_config.ExportableModuleConfig(
        model=model_type,
        task=export.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        input_sec=10.0,
    )
    cfg = ext.update_export_config(cfg, mock.MagicMock(model_type=model_type))
    self.assertEqual(cfg.export_audio_encoder, is_llm_asr)
    if asr_model_type is not None:
      metadata = model_cls.get_speech2text_metadata(model_cls, cfg)
      self.assertTrue(metadata.asr_model_type.HasField(asr_model_type))
      self.assertEqual(metadata.input_milliseconds, expected_input_ms)
      self.assertEqual(
          metadata.log_mel_spectrogram_config.n_frames, expected_mel_frames
      )

  def test_composite_extension_uses_last_non_empty_hooks(self):
    whisper = extension.get_model_extension("whisper")
    base = extension.ModelExtension()
    composite = extension.compose_extensions(whisper, base)

    self.assertEqual(
        composite.get_export_defaults({}, None).export_kwargs,
        {"input_sec": 30.0},
    )

  def test_remote_checkpoint_with_model_type_skips_model_index(self):
    with mock.patch.object(
        export,
        "_fetch_hub_json",
        autospec=True,
        return_value={"model_type": "whisper"},
    ) as fetch:
      config, model_index = export.load_checkpoint_config("openai/whisper-tiny")

    fetch.assert_called_once_with("openai/whisper-tiny", "config.json")
    self.assertEqual(config, {"model_type": "whisper"})
    self.assertIsNone(model_index)

  def test_remote_checkpoint_without_model_type_reads_model_index(self):
    responses = {
        "config.json": None,
        "model_index.json": {"_class_name": "Flux2KleinPipeline"},
    }
    with mock.patch.object(
        export,
        "_fetch_hub_json",
        autospec=True,
        side_effect=lambda repo_id, filename: responses[filename],
    ):
      config, model_index = export.load_checkpoint_config("org/pipeline")

    self.assertEqual(config, {})
    self.assertEqual(model_index, {"_class_name": "Flux2KleinPipeline"})

  def test_in_process_model_extension_is_consulted(self):
    class _MyExtension(extension.ModelExtension):
      model_types = ("my_encoder",)

      def get_export_defaults(self, config_dict, task=None, target="cpu_gpu"):
        del config_dict, task, target
        return extension.ExportDefaults(
            task=export.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
            export_kwargs={"input_sec": 7.0},
        )

    model = self._checkpoint(
        {"model_type": "my_encoder", "architectures": ["MyEncoderModel"]}
    )

    plan = export.export(
        model, model_extension=_MyExtension, dry_run=True
    )

    self.assertEqual(plan.task, "automatic_speech_recognition")
    self.assertEqual(plan.task_source, export.SOURCE_MODEL)
    self.assertEqual(plan.model_type, "my_encoder")
    self.assertEqual(plan.export_kwargs["input_sec"], 7.0)
    self.assertEqual(plan.sources["input_sec"], export.SOURCE_MODEL)
    self.assertIs(plan.export_kwargs["model_extension"], _MyExtension)

  def test_in_process_model_extension_overrides_registered_defaults(self):
    class _MyWhisperExtension(extension.ModelExtension):

      def get_export_defaults(self, config_dict, task=None, target="cpu_gpu"):
        del config_dict, task, target
        return extension.ExportDefaults(
            task=export.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
            export_kwargs={"input_sec": 10.0},
        )

    plan = export.export(
        self._checkpoint({"model_type": "whisper"}),
        model_extension=[_MyWhisperExtension()],
        dry_run=True,
    )

    self.assertEqual(plan.model_type, "whisper")
    self.assertEqual(plan.export_kwargs["input_sec"], 10.0)

  def test_out_of_tree_extension_can_override_target_settings(self):
    class _MyOutOfTreeLlmExtension(extension.ModelExtension):
      model_types = ("my_custom_llm",)

      def get_export_defaults(self, config_dict, task=None, target="cpu_gpu"):
        del config_dict, task
        if target.startswith("npu"):
          kwargs = {"cache_length": 2048}
          if target == "npu:qualcomm":
            kwargs["prefill_lengths"] = (64, 256)
          return extension.ExportDefaults(
              task=export.ExportTask.TEXT_GENERATION,
              export_kwargs=kwargs,
          )
        return extension.ExportDefaults(
            task=export.ExportTask.TEXT_GENERATION,
            export_kwargs={
                "cache_length": 8192,
                "llm_metadata_max_num_tokens_override": 2048,
            },
        )

    model = self._checkpoint({"model_type": "my_custom_llm"})

    cpu_plan = export.export(
        model,
        target="cpu_gpu",
        model_extension=_MyOutOfTreeLlmExtension,
        dry_run=True,
    )
    self.assertEqual(cpu_plan.export_kwargs["cache_length"], 8192)
    self.assertEqual(
        cpu_plan.export_kwargs["llm_metadata_max_num_tokens_override"], 2048
    )
    self.assertEqual(cpu_plan.sources["cache_length"], export.SOURCE_MODEL)

    qc_plan = export.export(
        model,
        target="npu:qc",
        model_extension=_MyOutOfTreeLlmExtension,
        dry_run=True,
    )
    self.assertEqual(qc_plan.target, "npu:qualcomm")
    self.assertEqual(qc_plan.export_kwargs["cache_length"], 2048)
    self.assertEqual(qc_plan.export_kwargs["prefill_lengths"], [64, 256])
    self.assertEqual(qc_plan.export_kwargs["k_ts_idx"], 3)
    self.assertEqual(qc_plan.export_kwargs["v_ts_idx"], 2)
    self.assertEqual(
        qc_plan.sources["prefill_lengths"], export.SOURCE_MODEL
    )

  def test_summary_rows_list_sources(self):
    plan = export.export(
        self._checkpoint({"model_type": "whisper"}), dry_run=True
    )

    self.assertEqual(
        plan.summary_rows(),
        [
            ("model_type", "whisper", "detected"),
            ("task", "automatic_speech_recognition", "model"),
            ("target", "cpu_gpu", "target"),
            ("input_sec", "30.0", "model"),
        ],
    )
    rendered = str(plan)
    self.assertIn("ExportPlan(", rendered)
    self.assertIn("automatic_speech_recognition", rendered)
    self.assertIn("input_sec", rendered)
    self.assertIsNot(type(plan).__str__, object.__str__)

  def test_non_causal_lm_without_extension_raises(self):
    model = self._checkpoint(
        {"model_type": "bert", "architectures": ["BertForMaskedLM"]}
    )

    with self.assertRaisesRegex(ValueError, "AutoModelForCausalLM"):
      export.export(model, dry_run=True)

  def test_non_causal_lm_with_explicit_task_is_allowed(self):
    model = self._checkpoint(
        {"model_type": "bert", "architectures": ["BertForMaskedLM"]}
    )

    plan = export.export(model, task="text_generation", dry_run=True)

    self.assertEqual(plan.task, "text_generation")

  def test_unknown_task_raises(self):
    with self.assertRaisesRegex(ValueError, "Unknown task"):
      export.export(self._checkpoint({}), task="video", dry_run=True)

  @parameterized.parameters(
      ("fast", "Unknown target 'fast'"),
      ("qualcomm", "Prefix NPU vendors with 'npu:'"),
      ("npu:sm8750", "npu:<vendor>:sm8750"),
      ("npu:", "Invalid NPU target"),
      ("npu:qualcomm:sm8750:extra", "Invalid NPU target"),
  )
  def test_invalid_target_raises(self, invalid_target, expected_regex):
    with self.assertRaisesRegex(ValueError, expected_regex):
      export.export(
          self._checkpoint({}), target=invalid_target, dry_run=True
      )

  def test_export_forwards_resolved_plan_to_run_export_tasks(self):
    model = self._checkpoint({"model_type": "whisper"})

    with mock.patch.object(
        export,
        "run_export_tasks",
        autospec=True,
        return_value=mock.MagicMock(
            litert_lm_model_path="/tmp/out/model.litertlm"
        ),
    ) as mock_run_export:
      plan = export.export(model, "/tmp/out", input_sec=10.0)

    mock_run_export.assert_called_once()
    passed_config = mock_run_export.call_args[0][1]
    self.assertEqual(passed_config.model, model)
    self.assertEqual(passed_config.output_dir, "/tmp/out")
    self.assertEqual(
        passed_config.task, export.ExportTask.AUTOMATIC_SPEECH_RECOGNITION
    )
    self.assertEqual(passed_config.input_sec, 10.0)
    self.assertEqual(plan.sources["input_sec"], "user")

  def test_failed_export_cleans_up_temporary_work_dir(self):
    import os  # pylint: disable=g-import-not-at-top

    model = self._checkpoint({"model_type": "whisper"})
    out_dir = self.create_tempdir().full_path

    with mock.patch.object(
        export,
        "run_export_tasks",
        autospec=True,
        side_effect=ModuleNotFoundError("No module named 'nemo'"),
    ):
      with self.assertRaises(ModuleNotFoundError):
        export.export(model, out_dir)

    self.assertEqual(os.listdir(out_dir), [])

  def test_failed_export_with_keep_temporary_files_preserves_output_dir(self):
    import os  # pylint: disable=g-import-not-at-top

    model = self._checkpoint({"model_type": "whisper"})
    out_dir = self.create_tempdir().full_path
    sentinel = os.path.join(out_dir, "keep.txt")
    with open(sentinel, "w") as f:
      f.write("keep")

    with mock.patch.object(
        export,
        "run_export_tasks",
        autospec=True,
        side_effect=RuntimeError("export failed"),
    ):
      with self.assertRaises(RuntimeError):
        export.export(model, out_dir, keep_temporary_files=True)

    self.assertTrue(os.path.exists(sentinel))

  def test_auto_docstring_undocumented_parameter_errors_are_silenced(self):
    import contextlib  # pylint: disable=g-import-not-at-top
    import importlib  # pylint: disable=g-import-not-at-top
    import io  # pylint: disable=g-import-not-at-top
    from litert_torch.generative.export_hf.model_ext.qwen3_tts.tokenizer_v2 import (  # pylint: disable=g-import-not-at-top
        modeling_qwen3_tts_tokenizer_v2,
    )
    from transformers.models.qwen3_5 import configuration_qwen3_5  # pylint: disable=g-import-not-at-top

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      importlib.reload(configuration_qwen3_5)
      importlib.reload(modeling_qwen3_tts_tokenizer_v2)

    self.assertNotIn("not documented", buf.getvalue())


if __name__ == "__main__":
  absltest.main()
