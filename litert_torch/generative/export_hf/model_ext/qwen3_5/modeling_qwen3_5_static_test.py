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

"""Tests for modeling_qwen3_5_static."""

import importlib.metadata

orig_version = importlib.metadata.version


def patched_version(distribution_name: str) -> str:
  if distribution_name == "torchao":
    return "0.4.0"
  return orig_version(distribution_name)


importlib.metadata.version = patched_version

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from litert_torch.generative.export_hf.model_ext.qwen3_5 import modeling_qwen3_5_static  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top
import torch.nn.functional as F  # pylint: disable=g-import-not-at-top


class StableSoftplusTest(absltest.TestCase):

  def test_matches_softplus_in_fp32(self):
    x = torch.linspace(-60.0, 60.0, steps=241)
    torch.testing.assert_close(
        modeling_qwen3_5_static.stable_softplus(x), F.softplus(x)
    )

  def test_is_finite_in_fp16_where_log1p_exp_overflows(self):
    x = torch.tensor([11.0, 12.0, 30.0, 1000.0], dtype=torch.float16)
    # The naive form overflows to inf for the larger inputs.
    self.assertFalse(torch.isfinite(torch.log1p(torch.exp(x))).all())
    y = modeling_qwen3_5_static.stable_softplus(x)
    self.assertTrue(torch.isfinite(y).all())
    torch.testing.assert_close(
        y.float(), F.softplus(x.float()), rtol=1e-3, atol=1e-3
    )


if __name__ == "__main__":
  absltest.main()
