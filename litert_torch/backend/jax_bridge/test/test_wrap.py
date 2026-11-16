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
"""jax_bridge.wrap related tests."""

import collections
import types

import jax
from litert_torch import backend
from litert_torch.backend import jax_bridge
from litert_converter.mlir import ir
from litert_converter.mlir.dialects import func
import torch

from absl.testing import absltest as googletest


class TestWrap(googletest.TestCase):

  def test_wrap(self):
    with (
        backend.export_utils.create_ir_context() as context,
        ir.Location.unknown(),
    ):
      module = ir.Module.create()
      lctx = backend.export.LoweringContext(context, module)

      @jax_bridge.wrap
      def wrapped_add(a: jax.Array, b: jax.Array):
        return a + b

      with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
            ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
            name="main",
        )
        def _main(a, b):
          return wrapped_add(lctx, a, b)

    ir_text = str(module)
    self.assertIn("%0 = call @wrapped_add_", ir_text)
    self.assertIn(
        "%0 = stablehlo.add %arg0, %arg1 : tensor<10x10xf32>", ir_text
    )

  def test_wrap_with_ir_input_names(self):
    with (
        backend.export_utils.create_ir_context() as context,
        ir.Location.unknown(),
    ):
      module = ir.Module.create()
      lctx = backend.export.LoweringContext(context, module)

      def retb(a: jax.Array, b: jax.Array):
        return b

      wrapped_retb = jax_bridge.wrap(retb, ir_input_names=["b"])

      with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
            ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
            name="main",
        )
        def _main(a, b):
          return wrapped_retb(lctx, a, b)

    ir_text = str(module)
    self.assertIn("%0 = call @retb_", ir_text)
    self.assertIn("return %arg0 : tensor<10x10xf32>", ir_text)

  def test_wrap_single_element_multi_output_and_namedtuple_meta(self):
    """Checks 1-element multi-output lists vs single NamedTuple tensor meta."""
    tensor_meta_cls = collections.namedtuple("TensorMeta", ["dtype"])
    with (
        backend.export_utils.create_ir_context() as context,
        ir.Location.unknown(),
    ):
      module = ir.Module.create()

      @jax_bridge.wrap
      def wrapped_list(a: jax.Array):
        return [a]

      @jax_bridge.wrap
      def wrapped_single(a: jax.Array):
        return a

      multi_out_res = []
      single_out_res = []
      with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
            name="main",
        )
        def _main(a):
          multi_lctx = backend.export.LoweringContext(
              context,
              module,
              node=types.SimpleNamespace(
                  meta={"val": [tensor_meta_cls(torch.float16)]}
              ),
          )
          res_list = wrapped_list(multi_lctx, a)
          multi_out_res.append(res_list)

          single_lctx = backend.export.LoweringContext(
              context,
              module,
              node=types.SimpleNamespace(
                  meta={"tensor_meta": tensor_meta_cls(torch.float32)}
              ),
          )
          res_single = wrapped_single(single_lctx, res_list[0])
          single_out_res.append(res_single)
          return res_single

      self.assertIsInstance(multi_out_res[0], list)
      self.assertLen(multi_out_res[0], 1)
      self.assertIsInstance(multi_out_res[0][0], ir.Value)
      self.assertEqual(
          multi_out_res[0][0].type,
          ir.RankedTensorType.get((10, 10), ir.F16Type.get()),
      )
      self.assertIsInstance(single_out_res[0], ir.Value)
      self.assertEqual(
          single_out_res[0].type,
          ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
      )

    ir_text = str(module)
    self.assertIn("stablehlo.convert", ir_text)
    self.assertEqual(ir_text.count("stablehlo.convert"), 2)

  def test_wrap_mismatched_out_avals_raises(self):
    """Checks mismatched results/out_avals raises ValueError."""
    tensor_meta_cls = collections.namedtuple("TensorMeta", ["dtype"])
    with (
        backend.export_utils.create_ir_context() as context,
        ir.Location.unknown(),
    ):
      module = ir.Module.create()

      @jax_bridge.wrap
      def wrapped_list(a: jax.Array):
        return [a]

      with ir.InsertionPoint(module.body):

        @func.FuncOp.from_py_func(
            ir.RankedTensorType.get((10, 10), ir.F32Type.get()),
            name="main",
        )
        def _main(a):
          mismatched_lctx = backend.export.LoweringContext(
              context,
              module,
              node=types.SimpleNamespace(
                  meta={
                      "val": [
                          tensor_meta_cls(torch.float32),
                          tensor_meta_cls(torch.float32),
                      ]
                  }
              ),
          )
          with self.assertRaises(ValueError):
            wrapped_list(mismatched_lctx, a)
          return a


if __name__ == "__main__":
  googletest.main()
