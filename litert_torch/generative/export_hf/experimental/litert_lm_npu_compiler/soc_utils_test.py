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
"""Unit tests for soc_utils."""

from absl.testing import absltest
from litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler import soc_utils


class SocUtilsTest(absltest.TestCase):

  def test_get_supported_socs_qualcomm(self):
    q_socs = soc_utils.get_supported_socs("qualcomm")
    self.assertNotEmpty(q_socs)
    self.assertIn("sm8850", q_socs)
    self.assertIn("sw6100", q_socs)
    self.assertIn("sm8750", q_socs)

  def test_get_supported_socs_mediatek(self):
    m_socs = soc_utils.get_supported_socs("mediatek")
    self.assertNotEmpty(m_socs)
    self.assertIn("mt6993", m_socs)
    self.assertIn("mt6989", m_socs)

  def test_validate_soc_success(self):
    resolved = soc_utils.validate_soc("qualcomm", "SM8850 (Qualcomm)")
    self.assertEqual(resolved, "sm8850")

    resolved_mtk = soc_utils.validate_soc("mediatek", "mt6993")
    self.assertEqual(resolved_mtk, "mt6993")

  def test_validate_soc_unsupported_vendor(self):
    with self.assertRaises(FileNotFoundError):
      soc_utils.get_supported_socs("unknown_vendor_9999")

  def test_validate_soc_unsupported_chip(self):
    with self.assertRaises(ValueError):
      soc_utils.validate_soc("qualcomm", "invalid_chip_9999")


if __name__ == "__main__":
  absltest.main()
