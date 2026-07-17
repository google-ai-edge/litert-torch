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

import os
from typing import Any
from absl import flags
from absl.testing import absltest
from litert_torch.generative.export_hf.experimental.npu_export import config_manager
from litert_torch.generative.export_hf.experimental.npu_export import stages


def dummy_func(
    model: str | None = None,
    cache_length: int = 0,
    max_decode_steps: int = 0,
    custom_flag: bool = False,
) -> dict[str, Any]:
  return {
      "model": model,
      "cache_length": cache_length,
      "max_decode_steps": max_decode_steps,
      "custom_flag": custom_flag,
  }


class StagesTest(absltest.TestCase):

  def test_bind_cfg_automatic_reflection(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="sm8850",
        cache_length=2048,
        max_decode_steps=64,
    )
    bound = stages._bind_cfg(
        cfg, dummy_func, {"custom_flag": True}, name_map={"model": "model_id"}
    )
    self.assertEqual(bound["model"], "google/gemma-3-270m-it")
    self.assertEqual(bound["cache_length"], 2048)
    self.assertEqual(bound["max_decode_steps"], 64)
    self.assertTrue(bound["custom_flag"])

  def test_bind_cfg_user_override_precedence(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="sm8850",
        cache_length=1280,
    )
    bound = stages._bind_cfg(cfg, dummy_func, {"cache_length": 512})
    self.assertEqual(bound["cache_length"], 512)

  def test_dynamic_summary_and_user_overrides(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="sm8850",
        custom_test_property="hello_world",
    )
    self.assertEqual(getattr(cfg, "custom_test_property"), "hello_world")

  def test_bind_cfg_against_all_target_functions(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="sm8850",
        cache_length=2048,
        max_decode_steps=64,
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
            "dataset_dir": "/tmp/data",
            "calibration_result_save_dir": "/tmp/out",
        },
        name_map={
            "kv_cache_max_len": "cache_length",
            "dataset_format": "calib_dataset_format",
            "eval_task_names": "calib_eval_task_names",
        },
    )
    self.assertEqual(bound_calib["max_decode_steps"], 64)
    self.assertEqual(bound_calib["kv_cache_max_len"], 2048)
    self.assertEqual(bound_calib["dataset_format"], "jsonl")
    self.assertEqual(bound_calib["eval_task_names"], "ALL")
    self.assertEqual(bound_calib["calibration_result_save_dir"], "/tmp/out")

    bound_quant = stages._bind_cfg(
        cfg,
        stages.quant_func,
        {"input_litertlm": "/tmp/model.litertlm", "output_litertlm": "/tmp/q"},
        name_map={"a16w8": "use_16bits_activations"},
    )
    self.assertEqual(bound_quant["a16w8"], True)
    self.assertEqual(bound_quant["allow_float_operations"], False)

    bound_compile = stages._bind_cfg(
        cfg,
        stages.compile_func,
        {"input_litertlm": "/tmp/q", "output_litertlm": "/tmp/npu"},
    )
    self.assertEqual(bound_compile["backend"], "qualcomm")
    self.assertEqual(bound_compile["soc_model"], "sm8850")

  def test_json_serialization_and_ad_hoc_attributes(self):
    cfg = config_manager.build_pipeline_config(
        model_id="google/gemma-3-270m-it",
        target_vendor="qualcomm",
        target_soc="sm8850",
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
    self.assertEqual(cfg.soc_model, "sm8750")
    self.assertEqual(cfg.weight_quantization_recipe, "dynamic_wi8_afp32")
    self.assertEqual(cfg.quantization_recipe, "dynamic_wi8_afp32")
    self.assertTrue(cfg.use_16bits_activations)
    self.assertTrue(cfg.a16w8)


if __name__ == "__main__":
  flags.FLAGS.set_default("calibration_result_save_dir", "/tmp")
  absltest.main()
