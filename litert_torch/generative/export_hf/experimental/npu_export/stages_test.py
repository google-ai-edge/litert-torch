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
"""Unit tests for NPU stage wrappers and dynamic config binding."""

import json
import os
from typing import Any
from unittest import mock
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.experimental.npu_export import config_manager
from litert_torch.generative.export_hf.experimental.npu_export import stages
from litert_torch.generative.export_hf.experimental.npu_export.configs import vendor_configs


def dummy_func(
    model: str | None = None,
    cache_length: int = 0,
    max_calibration_decode_steps: int = 0,
    custom_flag: bool = False,
) -> dict[str, Any]:
  return {
      "model": model,
      "cache_length": cache_length,
      "max_calibration_decode_steps": max_calibration_decode_steps,
      "custom_flag": custom_flag,
  }


class StagesTest(absltest.TestCase):

  def test_bind_cfg_automatic_reflection(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="SM8850",
        cache_length=2048,
        max_calibration_decode_steps=64,
    )
    bound = stages._bind_cfg(
        cfg, dummy_func, {"custom_flag": True}, name_map={"model": "model_id"}
    )
    self.assertEqual(bound["model"], "google/gemma-3-270m-it")
    self.assertEqual(bound["cache_length"], 2048)
    self.assertEqual(bound["max_calibration_decode_steps"], 64)
    self.assertTrue(bound["custom_flag"])

  def test_bind_cfg_user_override_precedence(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="SM8850",
        cache_length=1280,
    )
    bound = stages._bind_cfg(cfg, dummy_func, {"cache_length": 512})
    self.assertEqual(bound["cache_length"], 512)

  def test_dynamic_summary_and_user_overrides(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="SM8850",
        custom_test_property="hello_world",
    )
    self.assertEqual(getattr(cfg, "custom_test_property"), "hello_world")

  def test_bind_cfg_against_all_target_functions(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="SM8850",
        cache_length=2048,
        max_calibration_decode_steps=64,
    )
    bound_export = stages._bind_cfg(
        cfg,
        stages.export_func,
        {"output_dir": "/tmp/exp"},
        name_map={
            "model": "model_id",
            "quantization_recipe": "weight_quantization_recipe",
        },
    )
    self.assertEqual(bound_export["model"], "google/gemma-3-270m-it")
    self.assertEqual(bound_export["quantization_recipe"], "dynamic_wi8_afp32")
    self.assertEqual(bound_export["cache_length"], 2048)

    bound_calib = stages._bind_cfg(
        cfg,
        stages.calib_func,
        {
            "input_litertlm": "/tmp/model.litertlm",
            "calibration_dataset_dir": "/tmp/data",
            "calibration_result_save_dir": "/tmp/out",
        },
        name_map={
            "kv_cache_max_len": "cache_length",
        },
    )
    self.assertEqual(bound_calib["max_calibration_decode_steps"], 64)
    self.assertEqual(bound_calib["kv_cache_max_len"], 2048)
    self.assertEqual(bound_calib["calibration_dataset_format"], "jsonl")
    self.assertEqual(bound_calib["calibration_eval_task_names"], "ALL")
    self.assertEqual(bound_calib["calibration_result_save_dir"], "/tmp/out")

    bound_quant = stages._bind_cfg(
        cfg,
        stages.quant_func,
        {"input_litertlm": "/tmp/model.litertlm", "output_litertlm": "/tmp/q"},
    )
    self.assertEqual(bound_quant["use_float_input_output_normalizer"], False)

    bound_compile = stages._bind_cfg(
        cfg,
        stages.compile_func,
        {"input_litertlm": "/tmp/q", "output_litertlm": "/tmp/npu"},
    )
    self.assertEqual(bound_compile["backend"], "qualcomm")
    self.assertEqual(bound_compile["soc_model"], "SM8850")

  def test_json_serialization_and_ad_hoc_attributes(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="SM8850",
        cache_length=2048,
        custom_ad_hoc_property="hello_json",
    )
    json_str = cfg.to_json()
    loaded_cfg = config_manager.NpuPipelineConfig.from_json(json_str)

    self.assertEqual(loaded_cfg.model_id, "google/gemma-3-270m-it")
    self.assertEqual(loaded_cfg.backend, "qualcomm")
    self.assertEqual(loaded_cfg.cache_length, 2048)
    self.assertEqual(
        getattr(loaded_cfg, "custom_ad_hoc_property"), "hello_json"
    )

  def test_load_preconfigured_qwen_json(self):
    json_path = os.path.join(
        os.path.dirname(__file__), "configs/qwen3_0.6b_qualcomm_sm8750.json"
    )
    if not os.path.exists(json_path):
      json_path = "third_party/py/litert_torch/generative/export_hf/experimental/npu_export/configs/qwen3_0.6b_qualcomm_sm8750.json"
    cfg = config_manager.NpuPipelineConfig.load_json(json_path)
    self.assertEqual(cfg.model_id, "Qwen/Qwen3-0.6B")
    self.assertEqual(cfg.backend, "qualcomm")
    self.assertEqual(cfg.soc_model, "SM8750")
    self.assertEqual(cfg.weight_quantization_recipe, "dynamic_wi8_afp32")
    self.assertEqual(cfg.quantization_recipe, "dynamic_wi8_afp32")


class QuantizeOptionsTest(parameterized.TestCase):
  """Checks that the stage 3 options of a config reach quantize()."""

  @parameterized.parameters(*sorted(vendor_configs.VENDOR_CONFIGS))
  def test_vendor_defaults_reach_quantize(self, vendor):
    vendor_def = vendor_configs.VENDOR_CONFIGS[vendor]
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor=vendor,
        target_soc=vendor_def["default_soc"],
    )
    self.assertEqual(cfg.a16w8, vendor_def["a16w8"])
    self.assertEqual(
        cfg.use_float_input_output_normalizer,
        vendor_def["use_float_input_output_normalizer"],
    )
    bound_quant = stages._bind_cfg(cfg, stages.quant_func, {})
    self.assertEqual(bound_quant["a16w8"], vendor_def["a16w8"])
    self.assertEqual(
        bound_quant["use_float_input_output_normalizer"],
        vendor_def["use_float_input_output_normalizer"],
    )

  def test_mediatek_keeps_normalizer_in_float(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="mediatek",
        target_soc="mt6993",
    )
    self.assertTrue(cfg.a16w8)
    self.assertTrue(cfg.use_float_input_output_normalizer)

  @parameterized.named_parameters(
      ("field_name", {"use_float_input_output_normalizer": False}, False),
      ("allow_float_operations", {"allow_float_operations": False}, False),
      (
          "field_name_wins",
          {
              "use_float_input_output_normalizer": True,
              "allow_float_operations": False,
          },
          True,
      ),
  )
  def test_user_override_wins_over_vendor_default(self, overrides, expected):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="mediatek",
        target_soc="mt6993",
        **overrides,
    )
    self.assertEqual(cfg.use_float_input_output_normalizer, expected)
    self.assertNotIn("allow_float_operations", cfg.__dict__)

  def test_a16w8_user_override(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="SM8850",
        a16w8=False,
    )
    self.assertFalse(cfg.a16w8)
    self.assertFalse(stages._bind_cfg(cfg, stages.quant_func, {})["a16w8"])

  def test_allow_float_operations_attribute_alias(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="SM8850",
    )
    self.assertFalse(cfg.allow_float_operations)
    cfg.allow_float_operations = True
    self.assertTrue(cfg.use_float_input_output_normalizer)
    bound_quant = stages._bind_cfg(cfg, stages.quant_func, {})
    self.assertTrue(bound_quant["use_float_input_output_normalizer"])

  def test_json_round_trip_keeps_quantize_options(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="mediatek",
        target_soc="mt6993",
        a16w8=False,
    )
    data = json.loads(cfg.to_json())
    self.assertFalse(data["a16w8"])
    self.assertTrue(data["use_float_input_output_normalizer"])
    self.assertNotIn("allow_float_operations", data)
    loaded_cfg = config_manager.NpuPipelineConfig.from_json(cfg.to_json())
    self.assertFalse(loaded_cfg.a16w8)
    self.assertTrue(loaded_cfg.use_float_input_output_normalizer)

  @parameterized.named_parameters(
      ("allow_float_operations_only", {"allow_float_operations": True}, True),
      (
          "field_name_wins",
          {
              "use_float_input_output_normalizer": False,
              "allow_float_operations": True,
          },
          False,
      ),
  )
  def test_from_dict_maps_allow_float_operations(self, extra, expected):
    cfg = config_manager.NpuPipelineConfig.from_dict({
        "model_id": "google/gemma-3-270m-it",
        "backend": "mediatek",
        "soc_model": "mt6993",
        **extra,
    })
    self.assertEqual(cfg.use_float_input_output_normalizer, expected)
    self.assertNotIn("allow_float_operations", cfg.__dict__)

  @parameterized.named_parameters(
      ("qualcomm", "qualcomm", "SM8850", {}, True, False),
      ("mediatek", "mediatek", "mt6993", {}, True, True),
      ("mediatek_a8", "mediatek", "mt6993", {"a16w8": False}, False, True),
  )
  def test_npu_quantize_passes_options_to_quantize(
      self, vendor, soc, overrides, expected_a16w8, expected_normalizer
  ):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor=vendor,
        target_soc=soc,
        **overrides,
    )
    with mock.patch.object(
        stages, "quant_func", autospec=True
    ) as mock_quant_func:
      stages.npu_quantize(
          cfg=cfg,
          input_litertlm="/tmp/model.litertlm",
          output_litertlm="/tmp/model_srq.litertlm",
          calibration_dir="/tmp/calib",
      )
    _, kwargs = mock_quant_func.call_args
    self.assertEqual(kwargs["a16w8"], expected_a16w8)
    self.assertEqual(
        kwargs["use_float_input_output_normalizer"], expected_normalizer
    )
    self.assertEqual(kwargs["calibration_dir"], "/tmp/calib")


if __name__ == "__main__":
  flags.FLAGS.set_default("calibration_result_save_dir", "/tmp")
  absltest.main()
