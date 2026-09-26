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
"""Tests for the SDPA composite scope and extra_kwargs validation."""

import warnings

from absl.testing import parameterized
from litert_torch.generative.export_hf.core import exportable_module_config
from absl.testing import absltest as googletest

SdpaComposite = exportable_module_config.SdpaComposite
ExportableModuleConfig = exportable_module_config.ExportableModuleConfig


def _config(**kwargs):
  return ExportableModuleConfig(model="dummy", **kwargs)


class SdpaCompositeTest(parameterized.TestCase):

  def test_defaults_to_none(self):
    config = _config()
    self.assertIs(config.use_sdpa_composite, SdpaComposite.NONE)
    self.assertFalse(config.sdpa_composite_for_decode)
    self.assertFalse(config.sdpa_composite_for_prefill)
    self.assertFalse(config.apply_gpu_composites)

  @parameterized.named_parameters(
      ("none", "none", False, False),
      ("decode", "decode", True, False),
      ("all", "all", True, True),
  )
  def test_scope(self, value, want_decode, want_prefill):
    config = _config(use_sdpa_composite=value)
    self.assertEqual(config.sdpa_composite_for_decode, want_decode)
    self.assertEqual(config.sdpa_composite_for_prefill, want_prefill)

  def test_scope_is_case_insensitive(self):
    self.assertIs(
        _config(use_sdpa_composite="ALL").use_sdpa_composite, SdpaComposite.ALL
    )

  def test_invalid_scope_raises(self):
    with self.assertRaisesRegex(ValueError, "Invalid use_sdpa_composite"):
      _config(use_sdpa_composite="prefill")

  def test_composite_implies_apply_gpu_composites(self):
    self.assertTrue(_config(use_sdpa_composite="decode").apply_gpu_composites)
    self.assertTrue(_config(use_sdpa_composite="all").apply_gpu_composites)

  def test_none_is_falsy(self):
    """NONE must be falsy despite being a non-empty string enum value."""
    self.assertFalse(SdpaComposite.NONE)
    self.assertTrue(SdpaComposite.DECODE)
    self.assertTrue(SdpaComposite.ALL)


class LegacyFlagTest(parameterized.TestCase):

  def test_legacy_bool_true_maps_to_all(self):
    with self.assertWarns(DeprecationWarning):
      config = _config(use_sdpa_composite=True)
    self.assertIs(config.use_sdpa_composite, SdpaComposite.ALL)
    self.assertTrue(config.sdpa_composite_for_decode)
    self.assertTrue(config.sdpa_composite_for_prefill)

  def test_legacy_bool_false_maps_to_none(self):
    config = _config(use_sdpa_composite=False)
    self.assertIs(config.use_sdpa_composite, SdpaComposite.NONE)

  def test_legacy_prefill_flag_maps_to_all(self):
    with self.assertWarns(DeprecationWarning):
      config = _config(
          use_sdpa_composite=True,
          extra_kwargs={"use_sdpa_composite_for_prefill": True},
      )
    self.assertIs(config.use_sdpa_composite, SdpaComposite.ALL)
    self.assertTrue(config.sdpa_composite_for_decode)
    self.assertTrue(config.sdpa_composite_for_prefill)

  def test_legacy_prefill_flag_alone_maps_to_all(self):
    """The previously-broken combination is normalized rather than preserved.

    Setting only the prefill flag used to trace prefill with the composite while
    the sample inputs were built without `param_tensor`.
    """
    with self.assertWarns(DeprecationWarning):
      config = _config(
          extra_kwargs={"use_sdpa_composite_for_prefill": True},
      )
    self.assertIs(config.use_sdpa_composite, SdpaComposite.ALL)
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
    config = _config(use_bool_mask=True, use_sdpa_composite="all")
    self.assertTrue(config.extra_kwargs["use_bool_mask"])
    self.assertTrue(config.extra_kwargs["apply_gpu_composites"])


if __name__ == "__main__":
  googletest.main()
