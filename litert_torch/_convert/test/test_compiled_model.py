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
"""Tests for running converted PyTorch models via LiteRT CompiledModel API."""

import litert_torch
import numpy as np
import torch
from torch import nn

from absl.testing import absltest as googletest
try:
  from ai_edge_litert.litert_wrapper.compiled_model_wrapper.hardware_accelerator import HardwareAccelerator  # pylint: disable=g-import-not-at-top
except ImportError:
  from ai_edge_litert.compiled_model import HardwareAccelerator  # pylint: disable=g-import-not-at-top


class SimpleMLP(nn.Module):
  """Simple MLP module for testing CompiledModel execution."""

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(16, 32)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(32, 8)

  def forward(self, x):
    return self.fc2(self.relu(self.fc1(x)))


class TestCompiledModelRunner(googletest.TestCase):
  """Test suite for LiteRTModel.run_compiled()."""

  def test_run_compiled_cpu_parity(self):
    """Verifies LiteRTModel.run_compiled() executes via CompiledModel with exact PyTorch parity."""
    torch.manual_seed(42)
    model = SimpleMLP().eval()
    sample_input = (torch.randn(2, 16),)

    edge_model = litert_torch.convert(model, sample_input)
    expected = model(*sample_input).detach().numpy()
    actual = edge_model.run_compiled(
        *sample_input,
        hardware_accel=HardwareAccelerator.CPU,
        require_fully_accelerated=True,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  googletest.main()
