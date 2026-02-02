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
"""Tests for the quantizer."""

import os
import tempfile

import litert_torch
from litert_torch.quantize import pt2e_quantizer
from litert_torch.quantize import quant_config
import torch
from torchao.quantization.pt2e import quantize_pt2e
import torchvision

from absl.testing import absltest as googletest


class TestQuantizerSanityBasic(googletest.TestCase):
  """Test the basic sanity of the quantizer.

  This test is to ensure that the quantizer is working as expected.
  """

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_quantizer_arg(self):
    """Compare the sizes of models.

    Compare the sizes of models with and without PT2EQuantizer passed in. Expect
    a smaller binary size for the model with PT2EQuantizer.
    """
    model = torchvision.models.vgg16().eval()
    sample_input = (torch.randn(4, 3, 224, 224),)

    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config()
    )
    model = torch.export.export(model, sample_input).module()
    model = quantize_pt2e.prepare_pt2e(model, quantizer)
    model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)

    without_quantizer = litert_torch.convert(model, sample_input)
    with_quantizer = litert_torch.convert(
        model,
        sample_input,
        quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
    )

    without_quantizer_size = len(without_quantizer.tflite_model())
    with_quantizer_size = len(with_quantizer.tflite_model())

    self.assertNotEqual(
        with_quantizer_size,
        without_quantizer_size,
        "Quantized model size is expected to differ from unquantized's.",
    )


if __name__ == "__main__":
  googletest.main()
