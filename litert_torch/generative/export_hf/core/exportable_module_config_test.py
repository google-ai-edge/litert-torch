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
"""Tests for SDPA composite normalization and extra_kwargs validation."""

import warnings

from absl.testing import parameterized
from litert_torch.generative.export_hf.core import exportable_module_config
from absl.testing import absltest as googletest

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig


def _config(**kwargs):
  return ExportableModuleConfig(model="dummy", **kwargs)


class SdpaCompositeTest(parameterized.TestCase):

  def test_defaults_to_false(self):
    config = _config()
    self.assertFalse(config.use_sdpa_composite)
    self.assertFalse(config.apply_gpu_composites)

  def test_composite_implies_apply_gpu_composites(self):
    config = _config(use_sdpa_composite=True)
    self.assertTrue(config.use_sdpa_composite)
    self.assertTrue(config.apply_gpu_composites)

  def test_non_bool_raises(self):
    with self.assertRaisesRegex(TypeError, "use_sdpa_composite must be bool"):
      _config(use_sdpa_composite="invalid")  # pyrefly: ignore[bad-argument-type]


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


if __name__ == "__main__":
  googletest.main()
