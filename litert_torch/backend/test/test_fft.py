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
import litert_torch
from litert_torch import backend
import numpy as np
import torch

from absl.testing import absltest as googletest
from absl.testing import parameterized


class RfftModule(torch.nn.Module):
  def forward(self, x):
    return torch.fft.rfft(x)

class TestFft(parameterized.TestCase):

  def test_rfft(self):
    x = torch.randn(1, 2, 2048)
    ep = torch.export.export(RfftModule(), (x,))
    edge_model = litert_torch.convert(ep.module(), (x,), {})
    expected = torch.fft.rfft(x).numpy()
    actual = edge_model(x.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  googletest.main()
