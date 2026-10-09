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
from litert_torch import testing
from litert_torch.testing import model_coverage
import parameterized
import torch

from absl.testing import absltest as googletest
from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import


def _func_to_torch_module(func: Callable[..., torch.Tensor]):
  """Wraps a function into a torch module."""

  class TestModule(torch.nn.Module):

    def __init__(self, func):
      super().__init__()
      self._func = func

    def forward(self, *args, **kwargs):
      return self._func(*args, **kwargs)

  return TestModule(func).eval()


def _op_names(edge_model) -> list[str]:
  """Returns the builtin op names of the converted model, in graph order."""
  interpreter = tfl_interpreter.Interpreter(
      model_content=edge_model.model_content()
  )
  return [op['op_name'] for op in interpreter._get_ops_details()]  # pylint: disable=protected-access


@testing.parameterized_class(testing.V1_V2_PARAMETERS)
class TestConvertComposites(testing.V1V2TestCase):
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
    # The comparison feeds the int64 ids straight to the TFLite model.
    edge_model = litert_torch.convert(torch_module, args, enable_x64=True)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )

  def _assert_single_mirror_pad(self, edge_model):
    if self.use_v2:
      # The V2 path skips the pre-convert decompositions
      # (`export_torch_signature(..., run_decompositions=False)`), so
      # `aten.pad` never becomes `aten.reflection_pad2d` there and the
      # mirror_pad composite is not built. Numerics are still checked above.
      return
    op_names = _op_names(edge_model)
    self.assertEqual(op_names.count('MIRROR_PAD'), 1, op_names)
    self.assertNotIn('GATHER_ND', op_names)

  def test_convert_pad_reflect(self):
    """Tests conversion of reflection pad2d."""
    torch_module = _func_to_torch_module(
        lambda x: torch.nn.functional.pad(x, [1, 1, 2, 2], mode='reflect')
    )
    tracing_args = (torch.randn(1, 3, 10, 10),)
    edge_model = litert_torch.convert(torch_module, tracing_args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, tracing_args
        )
    )
    self._assert_single_mirror_pad(edge_model)

  def test_convert_reflection_pad2d_module(self):
    """Tests nn.ReflectionPad2d feeding a conv, as in AOT-GAN's stem."""
    torch_module = torch.nn.Sequential(
        torch.nn.ReflectionPad2d(3),
        torch.nn.Conv2d(3, 8, kernel_size=7),
    ).eval()
    tracing_args = (torch.randn(1, 3, 16, 16),)
    edge_model = litert_torch.convert(torch_module, tracing_args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, tracing_args
        )
    )
    self._assert_single_mirror_pad(edge_model)

  def test_convert_pad_replicate(self):
    """Tests conversion of replication pad2d."""
    torch_module = _func_to_torch_module(
        lambda x: torch.nn.functional.pad(x, [1, 1, 2, 2], mode='replicate')
    )
    tracing_args = (torch.randn(1, 3, 10, 10),)
    edge_model = litert_torch.convert(torch_module, tracing_args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, tracing_args
        )
    )
    # Replicate padding > 1 is not a MIRROR_PAD; it stays decomposed.
    self.assertNotIn('MIRROR_PAD', _op_names(edge_model))

  def test_convert_pad_replicate_one(self):
    """Tests replication pad2d by 1, which maps to MIRROR_PAD SYMMETRIC."""
    torch_module = _func_to_torch_module(
        lambda x: torch.nn.functional.pad(x, [1, 1, 1, 1], mode='replicate')
    )
    tracing_args = (torch.randn(1, 3, 10, 10),)
    edge_model = litert_torch.convert(torch_module, tracing_args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, tracing_args
        )
    )
    self._assert_single_mirror_pad(edge_model)

  def test_convert_conv2d_reflect_padding(self):
    """Tests conversion of Conv2d with reflect padding_mode."""
    torch_module = torch.nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        padding_mode='reflect',
    ).eval()
    tracing_args = (torch.randn(1, 3, 10, 10),)
    edge_model = litert_torch.convert(torch_module, tracing_args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, tracing_args
        )
    )
    self._assert_single_mirror_pad(edge_model)

  def test_convert_conv2d_replicate_padding(self):
    """Tests conversion of Conv2d with replicate padding_mode."""
    torch_module = torch.nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        padding_mode='replicate',
    ).eval()
    tracing_args = (torch.randn(1, 3, 10, 10),)
    edge_model = litert_torch.convert(torch_module, tracing_args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, torch_module, tracing_args
        )
    )
    self._assert_single_mirror_pad(edge_model)


if __name__ == '__main__':
  googletest.main()
