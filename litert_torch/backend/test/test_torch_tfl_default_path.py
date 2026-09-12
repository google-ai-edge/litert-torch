# Copyright 2024 The LiteRT Torch Authors.
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
"""Tests that the default path applies the full torch_tfl decomp table."""

from unittest import mock

import numpy as np
import torch

from litert_torch import fx_infra
from litert_torch.backend import export
from litert_torch.backend import export_utils

from absl.testing import absltest as googletest


def _lower_to_mlir_text(model, args):
  """Export + lower via the default path; return the MLIR text."""
  exported = torch.export.export(model.eval(), args)
  exported = fx_infra.safe_run_decompositions(
      exported, fx_infra.decomp.pre_convert_decomp()
  )
  lowered = export.exported_program_to_mlir(exported)
  return lowered.get_text()


class TestTorchTflDefaultPath(googletest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_default_without_flag_restores_multinomial_only(self):
    """By default (without flag), gelu must NOT go through torch_tfl."""

    class GeluModel(torch.nn.Module):

      def forward(self, x):
        return torch.nn.functional.gelu(x)

    mlir_text = _lower_to_mlir_text(GeluModel(), (torch.randn(2, 8),))
    self.assertNotIn("@tfl.gelu", mlir_text)
    self.assertNotIn('call_target_name = "tfl.gelu"', mlir_text)

  def test_flag_on_enables_gelu_lowered_via_torch_tfl(self):
    """With flag on, aten.gelu must reach tfl.gelu."""

    class GeluModel(torch.nn.Module):

      def forward(self, x):
        return torch.nn.functional.gelu(x)

    self.enter_context(
        mock.patch.dict("os.environ", {"LITERT_TORCH_FULL_TFL_DECOMPS": "1"})
    )
    mlir_text = _lower_to_mlir_text(GeluModel(), (torch.randn(2, 8),))
    self.assertTrue(
        "@tfl.gelu" in mlir_text or 'call_target_name = "tfl.gelu"' in mlir_text
    )

  def test_flag_on_enables_mean_dim_lowered_via_torch_tfl(self):
    """With flag on, aten.mean.dim must reach tfl.mean."""

    class MeanModel(torch.nn.Module):

      def forward(self, x):
        return x.mean(dim=(2, 3))

    self.enter_context(
        mock.patch.dict("os.environ", {"LITERT_TORCH_FULL_TFL_DECOMPS": "1"})
    )
    mlir_text = _lower_to_mlir_text(MeanModel(), (torch.randn(1, 4, 8, 8),))
    self.assertTrue(
        "@tfl.mean" in mlir_text or 'call_target_name = "tfl.mean"' in mlir_text
    )


if __name__ == "__main__":
  googletest.main()
