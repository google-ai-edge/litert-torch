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
"""Tests for NHWC layout rewrite of aten.amax (issue #1126)."""

import litert_torch
import numpy as np
import torch

from absl.testing import absltest as googletest


class TestLayoutAmax(googletest.TestCase):
  """Tests for NHWC layout optimization involving aten.amax."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def _convert_and_compare(self, model, args):
    model = model.eval()
    with torch.no_grad():
      expected = model(*args)
    edge_model = litert_torch.convert(model, args)
    actual = edge_model(*args)
    np.testing.assert_allclose(
        expected.detach().numpy(),
        np.asarray(actual),
        rtol=1e-5,
        atol=1e-3,
    )

  def test_conv_then_amax_channel_dim(self):
    """amax over the channel dim of a 4-D conv output (NHWC partition)."""

    class ConvAmax(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

      def forward(self, x):
        x = self.conv(x)
        return torch.amax(x, dim=1, keepdim=True)

    self._convert_and_compare(ConvAmax(), (torch.randn(1, 3, 16, 16),))

  def test_conv_then_amax_last_dim(self):
    """amax over the last dim with keepdim, the #1126 repro shape."""

    class ConvAmaxLast(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

      def forward(self, x):
        x = self.conv(x)
        return torch.amax(x, dim=-1, keepdim=True)

    self._convert_and_compare(ConvAmaxLast(), (torch.randn(1, 3, 16, 16),))

  def test_conv_then_amax_two_dims(self):
    """amax over both spatial dims exercises the int[1] list form."""

    class ConvAmaxHW(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

      def forward(self, x):
        x = self.conv(x)
        return torch.amax(x, dim=[2, 3], keepdim=False)

    self._convert_and_compare(ConvAmaxHW(), (torch.randn(1, 3, 16, 16),))

  def test_standalone_amax(self):
    """Standalone amax should not force unnecessary NHWC transposes."""

    class StandaloneAmax(torch.nn.Module):

      def forward(self, x):
        return torch.amax(x, dim=1, keepdim=True)

    self._convert_and_compare(StandaloneAmax(), (torch.randn(1, 3, 16, 16),))


if __name__ == "__main__":
  googletest.main()
