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

import copy
import operator

import litert_torch
from litert_torch.quantize import pt2e_quantizer
from litert_torch.quantize import quant_config
import torch
from torchao.quantization.pt2e import quantize_pt2e
import torchvision

from absl.testing import absltest as googletest
from absl.testing import parameterized


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

    without_quantizer_size = len(without_quantizer.model_content())
    with_quantizer_size = len(with_quantizer.model_content())

    self.assertNotEqual(
        with_quantizer_size,
        without_quantizer_size,
        "Quantized model size is expected to differ from unquantized's.",
    )


class GruModel(torch.nn.Module):
  """A GRU with or without initial state, returning its final state or not."""

  def __init__(self, return_state: bool):
    super().__init__()
    self.gru = torch.nn.GRU(4, 8, batch_first=True)
    self.return_state = return_state

  def forward(self, x, h0=None):
    output, state = self.gru(x, h0)
    return (output, state) if self.return_state else output


class TestQuantizerGru(parameterized.TestCase):
  """Test the GRU annotation of the quantizer.

  The GRU input, initial state and outputs are annotated whether the model
  passes the initial state or not and uses the final state or not.
  """

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  @parameterized.named_parameters(
      ("initial_state_final_state", True, True),
      ("final_state", False, True),
      ("initial_state", True, False),
      ("no_state", False, False),
  )
  def test_gru_io_annotation(self, initial_state, return_state):
    sample_input = (torch.randn(1, 5, 4),)
    if initial_state:
      sample_input += (torch.randn(1, 1, 8),)
    config = pt2e_quantizer.get_symmetric_quantization_config()
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(config)
    model = GruModel(return_state).eval()
    model = torch.export.export(model, sample_input).module()

    # Annotate a copy: prepare_pt2e skips the nodes already annotated.
    annotated = quantizer.annotate(copy.deepcopy(model))
    (gru,) = [
        n for n in annotated.graph.nodes if n.target == torch.ops.aten.gru.input
    ]
    input_qspec_map = gru.meta["quantization_annotation"].input_qspec_map
    input_qspecs = {n.name: qspec for n, qspec in input_qspec_map.items()}
    expected_input_qspecs = {"x": config.input_activation}
    if initial_state:
      expected_input_qspecs["h0"] = config.input_activation
    self.assertEqual(input_qspecs, expected_input_qspecs)
    output_qspecs = {
        n.args[1]: n.meta["quantization_annotation"].output_qspec
        for n in gru.users
        if n.target == operator.getitem and n.users
    }
    expected_output_qspecs = {0: config.output_activation}
    if return_state:
      expected_output_qspecs[1] = config.output_activation
    self.assertEqual(output_qspecs, expected_output_qspecs)

    model = quantize_pt2e.prepare_pt2e(model, quantizer)
    model(*sample_input)
    model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)
    model(*sample_input)
    litert_torch.convert(
        model,
        sample_input,
        quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
    )


if __name__ == "__main__":
  googletest.main()
