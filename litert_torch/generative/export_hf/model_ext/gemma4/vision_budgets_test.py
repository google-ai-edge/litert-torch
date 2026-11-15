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
"""Tests for Gemma 4 vision soft-token budget flag parsing."""

from absl.testing import parameterized
from litert_torch.generative.export_hf.model_ext.gemma4 import vision_budgets

from absl.testing import absltest as googletest


class VisionBudgetsTest(parameterized.TestCase):

  def test_default(self):
    self.assertEqual(vision_budgets.get_vision_token_budgets({}), (140,))

  @parameterized.named_parameters(
      ("int", 280, (280,)),
      ("str_single", "140", (140,)),
      ("str_list", "1120,70, 280", (70, 280, 1120)),
      ("str_bracketed", "[70,140]", (70, 140)),
      ("tuple", (140, 70), (70, 140)),
      ("list_dedup", [70, 70, 560], (70, 560)),
      ("list_of_str", ["70", "140"], (70, 140)),
      ("all", (70, 140, 280, 560, 1120), (70, 140, 280, 560, 1120)),
  )
  def test_budgets_flag(self, value, expected):
    self.assertEqual(
        vision_budgets.get_vision_token_budgets(
            {"gemma4_vision_token_budgets": value}
        ),
        expected,
    )

  def test_legacy_flag(self):
    self.assertEqual(
        vision_budgets.get_vision_token_budgets(
            {"gemma4_vision_max_soft_tokens": 140}
        ),
        (140,),
    )

  def test_both_flags_raises(self):
    with self.assertRaisesRegex(ValueError, "only one"):
      vision_budgets.get_vision_token_budgets({
          "gemma4_vision_token_budgets": 70,
          "gemma4_vision_max_soft_tokens": 140,
      })

  @parameterized.named_parameters(
      ("unsupported", [70, 100]),
      ("empty_list", []),
      ("empty_str", ""),
      ("non_numeric", "70,abc"),
      ("bool", True),
      ("float_type", 70.0),
  )
  def test_invalid_raises(self, value):
    with self.assertRaises(ValueError):
      vision_budgets.get_vision_token_budgets(
          {"gemma4_vision_token_budgets": value}
      )


if __name__ == "__main__":
  googletest.main()
