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
"""Tests for SDPA composite normalization, GPU presets, and extra_kwargs validation."""

import types
import warnings

from absl.testing import parameterized
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import extension
import torch

from absl.testing import absltest as googletest

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig


def _config(**kwargs):
  return ExportableModuleConfig(model="dummy", **kwargs)


class SdpaCompositeTest(parameterized.TestCase):

  def test_defaults_to_false(self):
    config = _config()
    self.assertFalse(config.use_sdpa_composite)
    self.assertFalse(config.apply_gpu_composites)
    self.assertFalse(config.use_bool_mask)

  def test_composite_implies_apply_gpu_composites_and_bool_mask(self):
    config = _config(use_sdpa_composite=True)
    self.assertTrue(config.use_sdpa_composite)
    self.assertTrue(config.apply_gpu_composites)
    self.assertTrue(config.use_bool_mask)

  def test_composite_respects_explicit_bool_mask_false(self):
    config = _config(use_sdpa_composite=True, use_bool_mask=False)
    self.assertTrue(config.use_sdpa_composite)
    self.assertTrue(config.apply_gpu_composites)
    self.assertFalse(config.use_bool_mask)

  def test_non_bool_raises(self):
    with self.assertRaisesRegex(TypeError, "use_sdpa_composite must be bool"):
      _config(use_sdpa_composite="invalid")  # pyrefly: ignore[bad-argument-type]


class CompositePrerequisitesTest(parameterized.TestCase):

  def test_qkv_norm_rope_composite_implies_fuse_qkv_and_prefill_logits(self):
    config = _config(use_qkv_norm_rope_composite=True)
    self.assertTrue(config.use_qkv_norm_rope_composite)
    self.assertTrue(config.fuse_qkv)
    self.assertTrue(config.prefill_logits)

  def test_swiglu_composite_implies_fuse_gate_up(self):
    config = _config(use_swiglu_composite=True)
    self.assertTrue(config.use_swiglu_composite)
    self.assertTrue(config.fuse_gate_up)

  def test_explicit_false_overrides_prerequisite(self):
    config = _config(
        use_qkv_norm_rope_composite=True,
        fuse_qkv=False,
        prefill_logits=False,
        use_swiglu_composite=True,
        fuse_gate_up=False,
    )
    self.assertFalse(config.fuse_qkv)
    self.assertFalse(config.prefill_logits)
    self.assertFalse(config.fuse_gate_up)


class ModelFamilyGpuPresetsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="qwen3",
          model_type="qwen3",
          expected_flags={
              "fuse_qkv": True,
              "fuse_gate_up": True,
              "use_qkv_norm_rope_composite": True,
              "use_swiglu_composite": True,
              "use_sdpa_composite": True,
              "use_bool_mask": True,
              "enable_gpu_dynamic_cache": True,
              "use_short_conv_composite": False,
              "use_rope_composite": False,
              "prefill_logits": True,
          },
      ),
      dict(
          testcase_name="qwen3_text",
          model_type="qwen3_text",
          expected_flags={
              "fuse_qkv": True,
              "fuse_gate_up": True,
              "use_qkv_norm_rope_composite": True,
              "use_swiglu_composite": True,
              "use_sdpa_composite": True,
              "use_bool_mask": True,
              "enable_gpu_dynamic_cache": True,
              "use_short_conv_composite": False,
              "use_rope_composite": False,
              "prefill_logits": True,
          },
      ),
      dict(
          testcase_name="qwen3_5",
          model_type="qwen3_5",
          expected_flags={
              "fuse_qkv": True,
              "fuse_gate_up": True,
              "use_qkv_norm_rope_composite": False,
              "use_swiglu_composite": True,
              "use_sdpa_composite": True,
              "use_bool_mask": True,
              "enable_gpu_dynamic_cache": True,
              "use_short_conv_composite": False,
              "use_rope_composite": True,
              "prefill_logits": False,
          },
      ),
      dict(
          testcase_name="lfm2",
          model_type="lfm2",
          expected_flags={
              "fuse_qkv": True,
              "fuse_gate_up": True,
              "use_qkv_norm_rope_composite": True,
              "use_swiglu_composite": True,
              "use_short_conv_composite": True,
              "use_sdpa_composite": True,
              "use_bool_mask": True,
              "enable_gpu_dynamic_cache": True,
              "use_rope_composite": False,
              "prefill_logits": True,
          },
      ),
      dict(
          testcase_name="gemma4",
          model_type="gemma4",
          expected_flags={
              "fuse_qkv": True,
              "fuse_gate_up": True,
              "use_qkv_norm_rope_composite": True,
              "use_swiglu_composite": True,
              "use_short_conv_composite": False,
              "use_sdpa_composite": True,
              "use_bool_mask": True,
              "enable_gpu_dynamic_cache": True,
              "use_rope_composite": False,
              "prefill_logits": True,
          },
      ),
      dict(
          testcase_name="gemma4_text",
          model_type="gemma4_text",
          expected_flags={
              "fuse_qkv": True,
              "fuse_gate_up": True,
              "use_qkv_norm_rope_composite": True,
              "use_swiglu_composite": True,
              "use_short_conv_composite": False,
              "use_sdpa_composite": True,
              "use_bool_mask": True,
              "enable_gpu_dynamic_cache": True,
              "use_rope_composite": False,
              "prefill_logits": True,
          },
      ),
      dict(
          testcase_name="gemma3",
          model_type="gemma3",
          expected_flags={
              "fuse_qkv": True,
              "fuse_gate_up": True,
              "use_qkv_norm_rope_composite": False,
              "use_swiglu_composite": False,
              "use_short_conv_composite": False,
              "use_sdpa_composite": True,
              "use_bool_mask": True,
              "enable_gpu_dynamic_cache": True,
              "use_rope_composite": True,
              "prefill_logits": False,
          },
      ),
      dict(
          testcase_name="gemma3_text",
          model_type="gemma3_text",
          expected_flags={
              "fuse_qkv": True,
              "fuse_gate_up": True,
              "use_qkv_norm_rope_composite": False,
              "use_swiglu_composite": False,
              "use_short_conv_composite": False,
              "use_sdpa_composite": True,
              "use_bool_mask": True,
              "enable_gpu_dynamic_cache": True,
              "use_rope_composite": True,
              "prefill_logits": False,
          },
      ),
  )
  def test_apply_gpu_composites_enables_model_preset(
      self, model_type, expected_flags
  ):
    config = _config(apply_gpu_composites=True, cache_length=4096)
    model_config = types.SimpleNamespace(model_type=model_type)
    config = extension.update_export_config(config, model_config)
    for flag_name, expected_val in expected_flags.items():
      self.assertEqual(
          getattr(config, flag_name),
          expected_val,
          msg=f"Flag {flag_name} mismatch for {model_type}",
      )
    # Verify magic number cache length was applied for dynamic cache.
    self.assertGreater(config.cache_length, 4096)
    self.assertEqual(config.cache_lengths, [4096])

  def test_multimodal_wrapper_uses_text_config_model_type(self):
    config = _config(apply_gpu_composites=True)
    model_config = types.SimpleNamespace(
        model_type="lfm2_vl",
        text_config=types.SimpleNamespace(model_type="lfm2"),
    )
    config = extension.update_export_config(config, model_config)
    self.assertTrue(config.use_short_conv_composite)
    self.assertTrue(config.use_qkv_norm_rope_composite)
    self.assertTrue(config.use_swiglu_composite)
    self.assertTrue(config.use_sdpa_composite)

  @parameterized.parameters(
      "qwen3", "qwen3_5", "lfm2", "gemma4", "gemma3", "llama", "dummy"
  )
  def test_cpu_default_no_regression_across_all_models(self, model_type):
    config = _config(cache_length=4096)
    model_config = types.SimpleNamespace(model_type=model_type)
    config = extension.update_export_config(config, model_config)
    for flag_name in (
        "apply_gpu_composites",
        "fuse_qkv",
        "fuse_gate_up",
        "use_rope_composite",
        "use_swiglu_composite",
        "use_qkv_norm_rope_composite",
        "use_short_conv_composite",
        "use_sdpa_composite",
        "use_bool_mask",
        "enable_gpu_dynamic_cache",
        "prefill_logits",
    ):
      self.assertFalse(
          getattr(config, flag_name),
          msg=f"CPU default should have {flag_name}=False for {model_type}",
      )
    self.assertEqual(config.cache_length, 4096)

  @parameterized.parameters("llama", "mistral", "dummy")
  def test_other_models_no_regression_with_apply_gpu_composites(
      self, model_type
  ):
    config = _config(apply_gpu_composites=True, cache_length=4096)
    model_config = types.SimpleNamespace(model_type=model_type)
    config = extension.update_export_config(config, model_config)
    self.assertTrue(config.apply_gpu_composites)
    for flag_name in (
        "fuse_qkv",
        "fuse_gate_up",
        "use_rope_composite",
        "use_swiglu_composite",
        "use_qkv_norm_rope_composite",
        "use_short_conv_composite",
        "use_sdpa_composite",
        "use_bool_mask",
        "enable_gpu_dynamic_cache",
        "prefill_logits",
    ):
      self.assertFalse(
          getattr(config, flag_name),
          msg=f"Other model {model_type} should keep {flag_name}=False",
      )
    self.assertEqual(config.cache_length, 4096)

  def test_bisection_overrides_respected(self):
    config = _config(
        apply_gpu_composites=True,
        use_short_conv_composite=False,
        use_sdpa_composite=False,
        use_bool_mask=False,
    )
    model_config = types.SimpleNamespace(model_type="lfm2")
    config = extension.update_export_config(config, model_config)
    self.assertFalse(config.use_short_conv_composite)
    self.assertFalse(config.use_sdpa_composite)
    self.assertFalse(config.use_bool_mask)
    self.assertTrue(config.use_qkv_norm_rope_composite)
    self.assertTrue(config.use_swiglu_composite)

  def test_conflicting_explicit_flags_disable_dependent_presets(self):
    # Explicitly passing use_rope_composite=True or fuse_qkv=False should not
    # enable use_qkv_norm_rope_composite; fuse_gate_up=False should not enable
    # use_swiglu_composite; enable_dynamic_shape=True should not enable
    # enable_gpu_dynamic_cache.
    config = _config(
        apply_gpu_composites=True,
        use_rope_composite=True,
        fuse_gate_up=False,
        enable_dynamic_shape=True,
        cache_length=4096,
    )
    model_config = types.SimpleNamespace(model_type="qwen3")
    config = extension.update_export_config(config, model_config)
    self.assertTrue(config.use_rope_composite)
    self.assertFalse(config.use_qkv_norm_rope_composite)
    self.assertFalse(config.fuse_gate_up)
    self.assertFalse(config.use_swiglu_composite)
    self.assertFalse(config.enable_gpu_dynamic_cache)
    self.assertEqual(config.cache_length, 4096)


class LegacyFlagTest(parameterized.TestCase):

  def test_legacy_prefill_flag_maps_to_true(self):
    with self.assertWarns(DeprecationWarning):
      config = _config(
          use_sdpa_composite=True,
          extra_kwargs={"use_sdpa_composite_for_prefill": True},
      )
    self.assertTrue(config.use_sdpa_composite)
    self.assertTrue(config.apply_gpu_composites)

  def test_legacy_prefill_flag_alone_maps_to_true(self):
    """The previously-broken combination is normalized rather than preserved.

    Setting only the prefill flag used to trace prefill with the composite while
    the sample inputs were built without `param_tensor`.
    """
    with self.assertWarns(DeprecationWarning):
      config = _config(
          extra_kwargs={"use_sdpa_composite_for_prefill": True},
      )
    self.assertTrue(config.use_sdpa_composite)
    self.assertTrue(config.apply_gpu_composites)

  def test_legacy_flags_are_consumed(self):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      config = _config(
          extra_kwargs={"use_sdpa_composite_for_prefill": True},
      )
    self.assertNotIn("use_sdpa_composite_for_prefill", config.extra_kwargs)


class ExtraKwargsValidationTest(parameterized.TestCase):

  def test_unknown_key_raises(self):
    with self.assertRaisesRegex(ValueError, "Unrecognized export flag"):
      _config(extra_kwargs={"definitely_not_a_flag": True})

  def test_typo_suggests_close_match(self):
    with self.assertRaisesRegex(ValueError, "did you mean.*use_sdpa_composite"):
      _config(extra_kwargs={"use_sdpa_composit": True})

  def test_known_passthrough_key_is_allowed(self):
    config = _config(extra_kwargs={"targets": ["a"]})
    self.assertEqual(config.extra_kwargs["targets"], ["a"])

  def test_promoted_flags_are_mirrored_for_legacy_readers(self):
    config = _config(use_bool_mask=True, use_sdpa_composite=True)
    self.assertTrue(config.extra_kwargs["use_bool_mask"])
    self.assertTrue(config.extra_kwargs["apply_gpu_composites"])

  def test_text_to_image_config_defaults_and_max_seq_len(self):
    config = _config(
        task="text_to_image",
        t2i_output_image_size=256,
        extra_kwargs={"max_seq_len": 128},
    )
    self.assertEqual(
        config.task, exportable_module_config.ExportTask.TEXT_TO_IMAGE
    )
    self.assertEqual(config.t2i_output_image_size, 256)
    self.assertEqual(config.extra_kwargs["max_seq_len"], 128)


class ModelDtypeTest(parameterized.TestCase):

  def test_default_is_float32(self):
    config = _config()
    self.assertEqual(config.get_torch_dtype(), torch.float32)
    self.assertEqual(config.get_cache_dtype(), torch.float32)

  @parameterized.parameters("bfloat16", "bf16", "BF16")
  def test_bfloat16(self, model_dtype):
    config = _config(model_dtype=model_dtype)
    self.assertEqual(config.get_torch_dtype(), torch.bfloat16)
    self.assertEqual(config.get_cache_dtype(), torch.bfloat16)

  @parameterized.parameters("float16", "fp16", "int8")
  def test_unsupported_raises(self, model_dtype):
    with self.assertRaisesRegex(ValueError, "Unsupported dtype"):
      _config(model_dtype=model_dtype)

  def test_fp16_flag_keeps_model_float32(self):
    config = _config(experimental_use_fp16=True)
    self.assertEqual(config.get_torch_dtype(), torch.float32)
    self.assertEqual(config.get_cache_dtype(), torch.float16)

  def test_fp16_flag_allows_explicit_float32(self):
    config = _config(model_dtype="float32", experimental_use_fp16=True)
    self.assertEqual(config.get_cache_dtype(), torch.float16)

  def test_fp16_flag_rejects_bfloat16(self):
    with self.assertRaisesRegex(ValueError, "requires `model_dtype` float32"):
      _config(model_dtype="bfloat16", experimental_use_fp16=True)


if __name__ == "__main__":
  googletest.main()
