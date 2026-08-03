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
"""Tests conversion modules that are meant to be wrapped as composites."""

from collections.abc import Callable

import litert_torch
from litert_torch.quantize import pt2e_quantizer
from litert_torch.quantize.quant_config import QuantConfig
from litert_torch.testing import model_coverage
import numpy as np
import parameterized
import torch
from torch import nn
from torchao.quantization.pt2e import quantize_pt2e

from absl.testing import absltest as googletest
from ai_edge_litert import interpreter as tfl_interpreter


def _func_to_torch_module(func: Callable[..., torch.Tensor]):
  """Wraps a function into a torch module."""

  class TestModule(torch.nn.Module):

    def __init__(self, func):
      super().__init__()
      self._func = func

    def forward(self, *args, **kwargs):
      return self._func(*args, **kwargs)

  return TestModule(func).eval()


class TestConvertComposites(googletest.TestCase):
  """Tests conversion modules that are meant to be wrapped as composites."""

  def test_convert_hardswish(self):
    """Tests conversion of a HardSwish module."""

    args = (torch.randn((5, 10)),)
    torch_module = torch.nn.Hardswish().eval()
    edge_model = litert_torch.convert(torch_module, args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )

  @parameterized.parameterized.expand([
      # (input_size, kernel_size, stride, padding, ceil_mode,
      # count_include_pad, divisor_override)
      # no padding, stride = 1
      ([1, 3, 6, 6], [3, 3], [1, 1], [0, 0], False, True, None),
      # add stride
      ([1, 3, 6, 6], [3, 3], [2, 2], [0, 0], False, True, None),
      # default values
      ([1, 3, 6, 6], [3, 3]),
      # add padding
      ([1, 3, 6, 6], [3, 3], [1, 1], [1, 1], False, True, None),
      # add different padding for different dims
      ([1, 3, 6, 6], [3, 3], [1, 1], [0, 1], False, True, None),
      # add both stride and padding
      ([1, 3, 6, 6], [3, 3], [2, 2], [1, 1], False, True, None),
      # padding set to one number
      ([1, 3, 6, 6], [3, 3], [1, 1], 1, False, True, None),
      # count_include_pad = False
      ([1, 3, 6, 6], [3, 3], [1, 1], [1, 1], False, False, None),
      # ceil_mode = True
      ([1, 3, 6, 6], [3, 3], [1, 1], [1, 1], True, True, None),
      # ceil_mode = True, stride=[3, 3]
      ([1, 3, 6, 6], [3, 3], [3, 3], [1, 1], True, True, None),
      # set divisor_override
      ([1, 3, 6, 6], [3, 3], [1, 1], 0, False, True, 6),
  ])
  def test_convert_avg_pool2d(self, input_size, *args):
    """Tests conversion of a module containing an avg_pool2d aten."""
    torch_module = _func_to_torch_module(
        lambda input_tensor: torch.ops.aten.avg_pool2d(input_tensor, *args)
    )
    tracing_args = (torch.randn(*input_size),)
    edge_model = litert_torch.convert(torch_module, tracing_args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, tracing_args
        )
    )

  def test_convert_adaptive_avg_pool2d_pt2e_static(self):
    class ConvAdaptivePoolConv(nn.Module):

      def __init__(self):
        super().__init__()
        self.pre = nn.Conv2d(3, 4, 1)
        self.post = nn.Conv2d(4, 5, 1)

      def forward(self, x):
        return self.post(torch.nn.functional.adaptive_avg_pool2d(
            self.pre(x), (1, 1)
        ))

    args = (torch.randn(1, 3, 4, 4),)
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True)
    )
    model = torch.export.export(ConvAdaptivePoolConv().eval(), args).module()
    model = quantize_pt2e.prepare_pt2e(model, quantizer)
    for _ in range(4):
      model(torch.randn_like(args[0]))
    model = quantize_pt2e.convert_pt2e(model, fold_quantize=False)

    edge_model = litert_torch.convert(
        model, args, quant_config=QuantConfig(pt2e_quantizer=quantizer)
    )
    interpreter = tfl_interpreter.Interpreter(
        model_content=edge_model.model_content()
    )
    interpreter.allocate_tensors()
    op_names = [op["op_name"] for op in interpreter._get_ops_details()]

    self.assertIn("AVERAGE_POOL_2D", op_names)
    self.assertNotIn("DEQUANTIZE", op_names)
    self.assertNotIn("SUM", op_names)
    self.assertEqual(interpreter.get_input_details()[0]["dtype"], np.int8)
    self.assertEqual(interpreter.get_output_details()[0]["dtype"], np.int8)

  @parameterized.parameterized.expand([
      # use scale_factor with align_corners=False
      (
          [1, 3, 10, 10],
          dict(scale_factor=3.0, mode='bilinear', align_corners=False),
      ),
      # use scale_factor with align_corners=true
      (
          [1, 3, 10, 10],
          dict(scale_factor=3.0, mode='bilinear', align_corners=True),
      ),
      # use size
      ([1, 3, 10, 10], dict(size=[15, 20], mode='bilinear')),
      # use size with align_corners=true
      (
          [1, 3, 10, 10],
          dict(size=[15, 20], mode='bilinear', align_corners=True),
      ),
  ])
  def test_convert_upsample_bilinear_functional(self, input_size, kwargs):
    """Tests conversion of a torch.nn.functional.upsample module."""
    torch_module = _func_to_torch_module(
        lambda input_tensor: torch.nn.functional.upsample(  # pylint: disable=unnecessary-lambda
            input_tensor, **kwargs
        )
    )
    tracing_args = (torch.randn(*input_size),)
    edge_model = litert_torch.convert(torch_module, tracing_args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, tracing_args
        )
    )

  @parameterized.parameterized.expand([
      # use scale_factor with align_corners=False
      (
          [1, 3, 10, 10],
          dict(scale_factor=3.0, mode='bilinear', align_corners=False),
      ),
      # use scale_factor with align_corners=true
      (
          [1, 3, 10, 10],
          dict(scale_factor=3.0, mode='bilinear', align_corners=True),
      ),
      # use size
      ([1, 3, 10, 10], dict(size=[15, 20], mode='bilinear')),
      # use size with align_corners=true
      (
          [1, 3, 10, 10],
          dict(size=[15, 20], mode='bilinear', align_corners=True),
      ),
  ])
  def test_convert_upsample_bilinear(self, input_size, kwargs):
    """Tests conversion of a torch.nn.Upsample module."""
    torch_module = _func_to_torch_module(
        lambda input_tensor: torch.nn.Upsample(**kwargs)(input_tensor)  # pylint: disable=unnecessary-lambda
    )
    tracing_args = (torch.randn(*input_size),)
    edge_model = litert_torch.convert(torch_module, tracing_args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, tracing_args
        )
    )

  @parameterized.parameterized.expand([
      # use scale_factor with align_corners=False
      (
          [1, 3, 10, 10],
          dict(scale_factor=3.0, mode='bilinear', align_corners=False),
      ),
      # use scale_factor with align_corners=true
      (
          [1, 3, 10, 10],
          dict(scale_factor=3.0, mode='bilinear', align_corners=True),
      ),
      # use size
      ([1, 3, 10, 10], dict(size=[15, 20], mode='bilinear')),
      # use size with align_corners=true
      (
          [1, 3, 10, 10],
          dict(size=[15, 20], mode='bilinear', align_corners=True),
      ),
  ])
  def test_convert_interpolate_bilinear_functional(self, input_size, kwargs):
    """Tests conversion of a torch.nn.functional.interpolate module."""
    torch_module = _func_to_torch_module(
        lambda input_tensor: torch.nn.functional.interpolate(  # pylint: disable=unnecessary-lambda
            input_tensor, **kwargs
        )
    )
    tracing_args = (torch.randn(*input_size),)
    edge_model = litert_torch.convert(torch_module, tracing_args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, tracing_args
        )
    )

  def test_convert_gelu(self):
    """Tests conversion of a GELU module."""

    args = (torch.randn((5, 10)),)
    torch_module = torch.nn.GELU().eval()
    edge_model = litert_torch.convert(torch_module, args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )

  def test_convert_gelu_approximate(self):
    """Tests conversion of an Approximate GELU module."""

    args = (torch.randn((5, 10)),)
    torch_module = torch.nn.GELU('tanh').eval()
    edge_model = litert_torch.convert(torch_module, args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )

  def test_convert_embedding_lookup(self):
    """Tests conversion of an Embedding module."""

    args = (torch.full((1, 10), 0, dtype=torch.long),)
    torch_module = torch.nn.Embedding(10, 10)
    edge_model = litert_torch.convert(torch_module, args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )


if __name__ == '__main__':
  googletest.main()
