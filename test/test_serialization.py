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
"""Tests for the serialization of models."""

import os
import tempfile

import litert_torch
from litert_torch.testing import model_coverage
import torch
import torchvision

from absl.testing import absltest as googletest


class TestSerialization(googletest.TestCase):
  """Tests for the serialization of models."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_read_write(self):
    """Tests if an exported model can be loaded and run.

    (1) Creates an litert_torch model from a torch model
    (2) Saves and then loads the model
    (3) Checks to make sure the model is still runnable and produces the right
    results.
    """
    resnet18 = torchvision.models.resnet18().eval()
    sample_input = (torch.randn(4, 3, 224, 224),)

    edge_model = litert_torch.convert(resnet18, sample_input)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
      edge_model.export(os.path.join(tmp_dir_name, "test.model"))
      loaded_model = litert_torch.load(os.path.join(tmp_dir_name, "test.model"))

    result = model_coverage.compare_tflite_torch(
        loaded_model, resnet18, sample_input
    )
    self.assertTrue(result)


if __name__ == "__main__":
  googletest.main()
