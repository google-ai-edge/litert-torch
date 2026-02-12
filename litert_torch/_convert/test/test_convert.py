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
"""Tests for litert_torch.convert."""

import dataclasses
from typing import Tuple

import litert_torch
from litert_torch.quantize import pt2e_quantizer
from litert_torch.testing import model_coverage
import numpy as np
import torch
from torch import nn
from torchao.quantization.pt2e import quantize_pt2e
import torchvision

from absl.testing import absltest as googletest
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors import fallback_backend
from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import


@dataclasses.dataclass
class TestContainer1:
  data_1: torch.Tensor
  data_2: Tuple[torch.Tensor, torch.Tensor]


torch.export.register_dataclass(
    TestContainer1, serialized_type_name="TestContainer1"
)


class TestConvert(googletest.TestCase):
  """Tests conversion of various modules."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_convert_add(self):
    """Tests conversion of a simple Add module."""

    class Add(nn.Module):

      def forward(self, a, b):
        return a + b

    args = (
        torch.randn((5, 10)),
        torch.randn((5, 10)),
    )
    torch_module = Add().eval()
    edge_model = litert_torch.convert(torch_module, args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )

  def test_convert_dot_add(self):
    """Tests conversion of a matrix multiplication followed by an add."""

    class DotAdd(nn.Module):

      def forward(self, a, b, c):
        return a @ b + c

    args = (
        torch.randn((5, 10)),
        torch.randn((10, 5)),
        torch.randn((5, 5)),
    )
    torch_module = DotAdd().eval()
    edge_model = litert_torch.convert(torch_module, args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )

  def test_convert_resnet18(self):
    args = (torch.randn(4, 3, 224, 224),)
    torch_module = torchvision.models.resnet18().eval()
    edge_model = litert_torch.convert(torch_module, args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )

  def test_signature_args_ordering(self):
    """Tests conversion of a model with more than 10 arguments."""

    class AddChainWith11Args(nn.Module):
      """A model with 11 arguments."""

      def forward(
          self,
          arg0: torch.Tensor,
          arg1: torch.Tensor,
          arg2: torch.Tensor,
          arg3: torch.Tensor,
          arg4: torch.Tensor,
          arg5: torch.Tensor,
          arg6: torch.Tensor,
          arg7: torch.Tensor,
          arg8: torch.Tensor,
          arg9: torch.Tensor,
          arg10: torch.Tensor,
      ):
        add0 = torch.add(arg0, arg1)
        add1 = torch.add(add0, arg2)
        add2 = torch.add(add1, arg3)
        add3 = torch.add(add2, arg4)
        add4 = torch.add(add3, arg5)
        add5 = torch.add(add4, arg6)
        add6 = torch.add(add5, arg7)
        add7 = torch.add(add6, arg8)
        add8 = torch.add(add7, arg9)
        add9 = torch.add(add8, arg10)
        return add9

    sample_input = lambda: (
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
    )
    torch_model = AddChainWith11Args().eval()
    edge_model = litert_torch.convert(torch_model, sample_input())

    result = model_coverage.compare_tflite_torch(
        edge_model, torch_model, sample_input, num_valid_inputs=10
    )
    self.assertTrue(result)

  def test_multi_output_model(self):
    """Tests conversion of a model that returns multiple outputs."""

    class BasicAddModelWithMultipleOutputs(nn.Module):
      """A model that returns multiple outputs."""

      def forward(self, arg0, arg1):
        add0 = arg0 + arg1
        mul0 = arg0 * arg1
        return add0, mul0

    sample_input = (
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
    )

    torch_model = BasicAddModelWithMultipleOutputs().eval()
    edge_model = litert_torch.convert(torch_model, sample_input)

    result = model_coverage.compare_tflite_torch(
        edge_model, torch_model, sample_input
    )
    self.assertTrue(result)

  def test_12_outputs_model(self):
    """Tests conversion of a model that returns more than 10 outputs."""

    class BasicAddModelWithMultipleOutputs(nn.Module):
      """A model that returns multiple outputs."""

      def forward(self, arg0, arg1):
        add0 = arg0 + arg1
        mul0 = arg0 * arg1
        add1 = add0 + mul0
        mul1 = add0 * mul0
        add2 = add1 + mul1
        mul2 = add1 * mul1
        add3 = add2 + mul2
        mul3 = add2 * mul2
        add4 = add3 + mul3
        mul4 = add3 * mul3
        add5 = add4 + mul4
        mul5 = add4 * mul4

        return (
            add0,
            mul0,
            add1,
            mul1,
            add2,
            mul2,
            add3,
            mul3,
            add4,
            mul4,
            add5,
            mul5,
        )

    sample_input = (
        torch.rand((64,), dtype=torch.float32),
        torch.rand((64,), dtype=torch.float32),
    )

    torch_model = BasicAddModelWithMultipleOutputs().eval()
    edge_model = litert_torch.convert(torch_model, sample_input)

    result = model_coverage.compare_tflite_torch(
        edge_model, torch_model, sample_input
    )
    self.assertTrue(result)

  def test_convert_conv_transpose_batch_norm(self):
    """Tests conversion of a model with ConvTranspose2d and BatchNorm2d."""

    channels = 2
    size = 2
    torch_model = nn.Sequential(
        nn.ConvTranspose2d(
            channels, channels, 1, stride=2, dilation=1, bias=False
        ),
        nn.BatchNorm2d(channels),
    )

    torch_model.eval()
    sample_input = (torch.rand(1, channels, size, size),)
    edge_model = litert_torch.convert(torch_model, sample_input)

    result = model_coverage.compare_tflite_torch(
        edge_model, torch_model, sample_input
    )
    self.assertTrue(result)

  def test_convert_model_with_kwargs(self):
    """Test converting a simple model with sample_kwargs."""

    class SampleModel(nn.Module):

      def forward(self, x, y):
        return x + y

    kwargs_gen = lambda: dict(x=torch.randn(10, 10), y=torch.randn(10, 10))

    model = SampleModel().eval()
    edge_model = litert_torch.convert(model, sample_kwargs=kwargs_gen())

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, model, kwargs=kwargs_gen
        )
    )

  def test_convert_model_with_args_kwargs(self):
    """Test converting a simple model with both sample_args and sample_kwargs."""

    class SampleModel(nn.Module):

      def forward(self, x, y):
        return x + y

    args_gen = lambda: (torch.randn(10, 10),)
    kwargs_gen = lambda: dict(y=torch.randn(10, 10))

    model = SampleModel().eval()
    edge_model = litert_torch.convert(model, args_gen(), kwargs_gen())

    self.assertTrue(
        model_coverage.compare_tflite_torch(
            edge_model, model, args_gen, kwargs_gen
        )
    )

  def test_convert_model_with_args_nested_kwargs_1(self):
    """Test converting a simple model with both sample_args and nested sample_kwargs."""

    class SampleModel(nn.Module):

      def forward(self, x: torch.Tensor, y: torch.Tensor, z: TestContainer1):
        return x + y + z.data_1 + z.data_2[0] + z.data_2[1]

    args = (torch.randn(10, 10),)
    kwargs = dict(
        y=torch.randn(10, 10),
        z=TestContainer1(
            data_1=torch.randn(10, 10),
            data_2=(torch.randn(10, 10), torch.randn(10, 10)),
        ),
    )
    flat_inputs = {
        "args_0": args[0].numpy(),
        "y": kwargs["y"].numpy(),
        "z_data_1": kwargs["z"].data_1.numpy(),
        "z_data_2_0": kwargs["z"].data_2[0].numpy(),
        "z_data_2_1": kwargs["z"].data_2[1].numpy(),
    }
    self._compare_tflite_torch_args_kwargs(
        SampleModel(), args, kwargs, flat_inputs
    )

  def test_convert_model_with_args_nested_kwargs_2(self):
    """Test converting a simple model with both sample_args and nested sample_kwargs."""

    class SampleModel(nn.Module):

      def forward(self, x, y, z):
        return x + y + z.data_1 + z.data_2[0][0] + z.data_2[1]

    args = (torch.randn(10, 10),)
    kwargs = dict(
        y=torch.randn(10, 10),
        z=TestContainer1(
            data_1=torch.randn(10, 10),
            data_2=[(torch.randn(10, 10),), torch.randn(10, 10)],
        ),
    )
    flat_inputs = {
        "args_0": args[0].numpy(),
        "y": kwargs["y"].numpy(),
        "z_data_1": kwargs["z"].data_1.numpy(),
        "z_data_2_0_0": kwargs["z"].data_2[0][0].numpy(),
        "z_data_2_1": kwargs["z"].data_2[1].numpy(),
    }
    self._compare_tflite_torch_args_kwargs(
        SampleModel(), args, kwargs, flat_inputs
    )

  def test_convert_model_with_args_nested_kwargs_3(self):
    """Test converting a simple model with both sample_args and nested sample_kwargs."""

    class SampleModel(nn.Module):

      def forward(self, x, y, z):
        return x + y + z.data_1 + z.data_2[0]["foo"] + z.data_2[1]

    args = (torch.randn(10, 10),)
    kwargs = dict(
        y=torch.randn(10, 10),
        z=TestContainer1(
            data_1=torch.randn(10, 10),
            data_2=(dict(foo=torch.randn(10, 10)), torch.randn(10, 10)),
        ),
    )
    flat_inputs = {
        "args_0": args[0].numpy(),
        "y": kwargs["y"].numpy(),
        "z_data_1": kwargs["z"].data_1.numpy(),
        "z_data_2_0_foo": kwargs["z"].data_2[0]["foo"].numpy(),
        "z_data_2_1": kwargs["z"].data_2[1].numpy(),
    }
    self._compare_tflite_torch_args_kwargs(
        SampleModel(), args, kwargs, flat_inputs
    )

  def test_convert_model_non_flat_output_dict(self):
    """Test converting a model with non-flat output structure."""

    class SampleModel(nn.Module):

      def forward(self, x, y, z):
        return {"x": x, "y": TestContainer1(data_1=y, data_2=[y, z])}

    args = (torch.randn(10, 10), torch.randn(10, 10), torch.randn(10, 10))
    kwargs = dict()
    flat_inputs = {
        "args_0": args[0].numpy(),
        "args_1": args[1].numpy(),
        "args_2": args[2].numpy(),
    }

    edge_model = litert_torch.convert(SampleModel().eval(), args, kwargs)
    edge_output = edge_model(**flat_inputs)
    np.testing.assert_almost_equal(edge_output["x"], args[0])
    np.testing.assert_almost_equal(edge_output["y_data_1"], args[1])
    np.testing.assert_almost_equal(edge_output["y_data_2_0"], args[1])
    np.testing.assert_almost_equal(edge_output["y_data_2_1"], args[2])

    interpreter = tfl_interpreter.Interpreter(
        model_content=edge_model.model_content()
    )
    runner = interpreter.get_signature_runner("serving_default")
    output_details = runner.get_output_details()
    self.assertIn("x", output_details.keys())
    self.assertIn("y_data_1", output_details.keys())
    self.assertIn("y_data_2_0", output_details.keys())
    self.assertIn("y_data_2_1", output_details.keys())

  def _compare_tflite_torch_args_kwargs(self, model, args, kwargs, flat_inputs):
    model.eval()
    edge_model = litert_torch.convert(model, args, kwargs)
    interpreter = tfl_interpreter.Interpreter(
        model_content=edge_model.model_content()
    )
    runner = interpreter.get_signature_runner("serving_default")
    input_details = runner.get_input_details()
    self.assertEqual(input_details.keys(), flat_inputs.keys())

    reference_output = model(*args, **kwargs)
    tflite_output = edge_model(**flat_inputs)
    np.testing.assert_almost_equal(reference_output, tflite_output)

  def test_convert_model_with_input_mutation(self):
    class SampleModel(nn.Module):

      def forward(self, x):
        x /= 1
        x = x + 10
        return x

    args = (torch.randn(10, 10),)
    torch_module = SampleModel().eval()
    edge_model = litert_torch.convert(torch_module, args)

    self.assertTrue(
        model_coverage.compare_tflite_torch(edge_model, torch_module, args)
    )

  def test_convert_resnet18_pt2e_per_layer(self):
    # Step 1: export resnet18
    args = (torch.randn(1, 3, 224, 224),)
    m = torchvision.models.resnet18().eval()
    m = torch.export.export(m, args).module()

    # Step 2: Insert observers or fake quantize modules
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=False)
    )
    m = quantize_pt2e.prepare_pt2e(m, quantizer)

    # Step 3: Quantize the model
    m = quantize_pt2e.convert_pt2e(m, fold_quantize=False)

    # pylint: disable=broad-except
    try:
      litert_torch.convert(m, args)
    except Exception as err:
      self.fail(f"PT2E conversion failed: {err}")
    # pylint: enable=broad-except

  def test_convert_resnet18_pt2e_per_channel(self):
    # Step 1: export resnet18
    args = (torch.randn(1, 3, 224, 224),)
    m = torchvision.models.resnet18().eval()
    m = torch.export.export(m, args).module()

    # Step 2: Insert observers or fake quantize modules
    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True)
    )
    m = quantize_pt2e.prepare_pt2e(m, quantizer)
    # Step 3: Run through example inputs, otherwise per-channel
    # quant may have scalar scale/zero_point
    m(*args)
    # Step 4: Quantize the model
    m = quantize_pt2e.convert_pt2e(m, fold_quantize=False)

    # pylint: disable=broad-except
    try:
      litert_torch.convert(m, args)
    except Exception as err:
      self.fail(f"PT2E conversion failed: {err}")
    # pylint: enable=broad-except

  def test_convert_model_with_bfloat16_inputs(self):
    """Test converting a simple model with torch.bfloat16 input.

    bf16 inputs would remain in converted model signature but be casted to f32
    right after the model inputs.
    """

    class SampleModel(nn.Module):

      def forward(self, x: torch.Tensor):
        return (x + 1) * 1.2

    model = SampleModel().eval()
    args = (torch.randn(10, 10).to(torch.bfloat16),)
    # pylint: disable=broad-except
    try:
      litert_torch.convert(model, args)
    except Exception as err:
      self.fail(f"Conversion failed with bloat16 inputs: {err}")
    # pylint: enable=broad-except

  def test_convert_model_with_torch_linspace_operation(self):
    """Test converting a simple model with torch.linspace operation."""

    class SampleModel(nn.Module):

      def forward(self, x: torch.Tensor):
        return torch.linspace(0.5, 10.5, steps=x.shape[0], dtype=torch.float64)

    model = SampleModel().eval()
    args = (torch.randint(0, 100, (10, 10), dtype=torch.int64),)

    try:
      # Expect this to fix the error during conversion
      litert_torch.convert(model, args)
    except Exception as err:
      self.fail(f"Conversion failed with int64 inputs: {err}")
    # pylint: enable=broad-except

  def test_convert_model_with_torch_div_operation(self):
    """Test converting a simple model with torch.div operation."""

    class SampleModel(nn.Module):

      def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x / y

    model = SampleModel().eval()
    args = (
        torch.randint(0, 100, (10, 10), dtype=torch.int64),
        torch.randint(0, 100, (10, 10), dtype=torch.int64),
    )

    try:
      # Expect this to fix the error during conversion
      litert_torch.convert(model, args)
    except Exception as err:
      self.fail(f"Conversion failed with int64 inputs: {err}")
    # pylint: enable=broad-except

  def test_convert_model_with_torch_div_operation_6d_inputs(self):
    """Test converting a simple model with torch.div operation and 6d inputs."""

    class SampleModel(nn.Module):

      def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x / y

    model = SampleModel().eval()
    args = (
        torch.randn((1, 2, 3, 4, 5, 6)),
        torch.randn((1, 1, 1, 1, 1, 1)),
    )

    try:
      # Expect this to fix the error during conversion
      litert_torch.convert(model, args)
    except Exception as err:
      self.fail(f"Conversion failed with 6d inputs: {err}")
    # pylint: enable=broad-except

  def test_convert_model_with_slice_6d_inputs(self):
    """Test converting a simple model with slice and 6d inputs."""

    class SampleModel(nn.Module):

      def forward(self, x: torch.Tensor):
        return x[0:1, 0:2, 0:3, 0:4, 0:5, 0:1]

    model = SampleModel().eval()
    args = (torch.randn((1, 2, 3, 4, 5, 6)),)

    try:
      # Expect this to fix the error during conversion
      litert_torch.convert(model, args)
    except Exception as err:
      self.fail(f"Conversion failed with 6d inputs for slice: {err}")
    # pylint: enable=broad-except

  def test_convert_model_with_strided_slice_6d_inputs(self):
    """Test converting a simple model with strided_slice and 6d inputs."""

    class SampleModel(nn.Module):

      def forward(self, x: torch.Tensor):
        return x[:, :, :, :, :, ::2]

    model = SampleModel().eval()
    args = (torch.randn((1, 2, 3, 4, 5, 6)),)

    try:
      # Expect this to fix the error during conversion
      litert_torch.convert(model, args)
    except Exception as err:
      self.fail(f"Conversion failed with 6d inputs for strided_slice: {err}")
    # pylint: enable=broad-except

  @googletest.skipIf(
      litert_torch.config.in_oss,
      "wait until converter next is completely OSS ready.",
  )
  def test_convert_large_model_with_lightweight_conversion(self):
    """Test converting a simple model with large lazy constants."""

    class SampleModel(nn.Module):

      def __init__(self):
        super().__init__()
        self.weights = [torch.randn(32 * 1024 * 1024) for _ in range(3)]

      def forward(self, x):
        for w in self.weights:
          x = torch.sin(x * w)
        return x

    model = SampleModel().eval()
    args = (torch.randn(1),)
    try:
      em = litert_torch.convert(model, args, lightweight_conversion=True)
    except Exception as err:
      self.fail(f"Conversion failed with large lazy constants: {err}")

    np.testing.assert_allclose(
        em(args[0]),
        model(args[0]).detach().numpy(),
        atol=1e-4,
    )

  def test_compile_model(self):
    """Tests AOT compilation of a simple Add module."""

    class Add(nn.Module):

      def forward(self, a, b):
        return a + b

    args = (
        torch.randn((5, 10)),
        torch.randn((5, 10)),
    )
    torch_module = Add().eval()
    compilation_result = litert_torch.experimental_add_compilation_backend(
        fallback_backend.FallbackTarget()
    ).convert(torch_module, args)
    assert isinstance(compilation_result, litert_types.CompilationResult)
    self.assertLen(compilation_result.models_with_backend, 1)
    self.assertEqual(
        compilation_result.models_with_backend[0][0].id(),
        fallback_backend.FallbackBackend.id(),
    )


if __name__ == "__main__":
  googletest.main()
