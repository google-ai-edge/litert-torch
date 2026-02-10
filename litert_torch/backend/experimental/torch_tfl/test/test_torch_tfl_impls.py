# Copyright 2025 The LiteRT Torch Authors.
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
"""Numerical validation tests for torch ops and Torch-TFL ops."""

from typing import Any, Dict, Sequence

import litert_torch
from litert_torch import testing
from litert_torch.backend.experimental import torch_tfl
import numpy as np
import torch
from torch.utils import _pytree as pytree

from absl.testing import absltest as googletest
from absl.testing import parameterized


export_with_tensor_inputs_only = testing.export_with_tensor_inputs_only


def tree_map_list_to_tuple(x):
  if isinstance(x, (list, tuple)):
    return tuple(tree_map_list_to_tuple(y) for y in x)
  if isinstance(x, dict):
    return {k: tree_map_list_to_tuple(v) for k, v in x.items()}
  return x


def rnd(dtype, shape, min_v=None, max_v=None):
  """Shortcut for creating a random torch tensor."""
  if dtype in (torch.int32, torch.int64, torch.bool):
    min_v = min_v if min_v else 1
    max_v = max_v if max_v else 10
    return torch.randint(min_v, max_v, shape).to(dtype)
  else:
    min_v = min_v if min_v else 0.0
    max_v = max_v if max_v else 1.0
    return (torch.rand(shape) * (max_v - min_v) + min_v).to(dtype)


class TestTorchTFLImpls(parameterized.TestCase):
  """Numerical validation tests for torch ops and Torch-TFL ops.

  The op test suite is forked from
  litert_torch/backend/test/test_core_aten_ops.py. Eventually, we should
  merge the two test suites.
  """

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def _assert_export_and_close(
      self,
      func,
      args,
      kwargs,
      dynamic_shapes: Dict[str, Any] | Sequence[Any] | None = None,
      atol=1e-3,
      rtol=1e-5,
      equal_nan=True,
      check_values=True,
  ):
    """Assert func, args, and kwargs can be lowered and pass numerical validation."""
    with self.subTest("torch_eval"):
      expected = func(*args, **kwargs)

      with self.subTest("export_and_decompse"):
        exported_program = export_with_tensor_inputs_only(
            func, args, kwargs, dynamic_shapes
        )
        exported_program = exported_program.run_decompositions(
            torch_tfl.decomps
        )

        with self.subTest("decomp_eval"):
          args, kwargs = exported_program.example_inputs
          actual = exported_program.module()(*args, **kwargs)

          with self.subTest("torch_lower_eval_diff:" + str(atol)):
            expected_flat, expected_spec = pytree.tree_flatten(expected)
            actual_flat, actual_spec = pytree.tree_flatten(actual)

            self.assertEqual(expected_spec, actual_spec)
            for v1, v2 in zip(expected_flat, actual_flat):
              if check_values:
                torch.testing.assert_close(
                    v1, v2, atol=atol, rtol=rtol, equal_nan=equal_nan
                )
              else:
                self.assertEqual(v1.shape, v2.shape)
                self.assertEqual(v1.dtype, v2.dtype)

        with self.subTest("convert_eval"):
          args, kwargs = exported_program.example_inputs
          edge_model = litert_torch.convert(
              exported_program.module(), args, dynamic_shapes=dynamic_shapes
          )
          actual = edge_model(*args, **kwargs)

          with self.subTest("torch_convert_eval_diff:" + str(atol)):
            expected = tree_map_list_to_tuple(expected)
            actual = tree_map_list_to_tuple(actual)

            expected_flat, expected_spec = pytree.tree_flatten(expected)
            actual_flat, actual_spec = pytree.tree_flatten(actual)

            self.assertEqual(expected_spec, actual_spec)
            for v1, v2 in zip(expected_flat, actual_flat):
              # Convert NumPy arrays to PyTorch tensors if necessary
              if isinstance(v1, torch.Tensor) and isinstance(v2, np.ndarray):
                v2 = torch.from_numpy(v2)
              if check_values:
                torch.testing.assert_close(
                    v1, v2, atol=atol, rtol=rtol, equal_nan=equal_nan
                )
              else:
                self.assertEqual(v1.shape, v2.shape)
                self.assertEqual(v1.dtype, v2.dtype)

  @parameterized.named_parameters(
      # fmt: off
      # pyformat: disabledef
      ("aten_mm_0", torch.ops.aten.mm.default, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_mm_1", torch.ops.aten.mm.default, (rnd(torch.float32, (2, 10)), rnd(torch.float32, (10, 5)),), dict()),
      ("aten_bmm_0", torch.ops.aten.bmm.default, (rnd(torch.float32, (10, 10, 10)), rnd(torch.float32, (10, 10, 10)),), dict()),
      ("aten_bmm_1", torch.ops.aten.bmm.default, (rnd(torch.float32, (10, 2, 3)), rnd(torch.float32, (10, 3, 4)),), dict()),
      ("aten_add_Tensor_0", torch.ops.aten.add.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_add_Tensor_1", torch.ops.aten.add.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict(alpha=10)),
      ("aten_add_Tensor_2", torch.ops.aten.add.Tensor, (rnd(torch.float32, (10, 10)), np.random.rand(),), dict()),
      ("aten_sub_Tensor_0", torch.ops.aten.sub.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_sub_Tensor_1", torch.ops.aten.sub.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict(alpha=10)),
      ("aten_sub_Tensor_2", torch.ops.aten.sub.Tensor, (rnd(torch.float32, (10, 10)), np.random.rand(),), dict()),
      ("aten_mul_Tensor_0", torch.ops.aten.mul.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_mul_Tensor_1", torch.ops.aten.mul.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict()),
      ("aten_mul_Tensor_2", torch.ops.aten.mul.Tensor, (rnd(torch.float32, (10, 10)), np.random.rand(),), dict()),
      ("aten_mul_Scalar_0", torch.ops.aten.mul.Scalar, (rnd(torch.float32, (10, 10)), np.random.rand(),), dict()),
      ("aten_div_Tensor_0", torch.ops.aten.div.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_div_Tensor_1", torch.ops.aten.div.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict()),
      ("aten_div_Tensor_2", torch.ops.aten.div.Tensor, (rnd(torch.float32, (10, 10)), np.random.rand(),), dict()),
      ("aten_pow_Scalar_0", torch.ops.aten.pow.Scalar, (np.random.rand(), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_pow_Tensor_Scalar_0", torch.ops.aten.pow.Tensor_Scalar, (rnd(torch.float32, (10, 10)), np.random.rand(),), dict()),
      ("aten_pow_Tensor_Scalar_1", torch.ops.aten.pow.Tensor_Scalar, (rnd(torch.float32, (10, 10)), np.random.randint(2, 5),), dict()),
      ("aten_pow_Tensor_Tensor_0", torch.ops.aten.pow.Tensor_Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_bitwise_and_Tensor_0", torch.ops.aten.bitwise_and.Tensor, (rnd(torch.bool, (10, 10)), rnd(torch.bool, (10, 10)),), dict()),
      ("aten_mean_dim_0", torch.ops.aten.mean.dim, (rnd(torch.float32, (10, 10)), 0), dict()),
      ("aten_mean_dim_1", torch.ops.aten.mean.dim, (rnd(torch.float32, (10, 10)), 0, True), dict()),
      ("aten_mean_dim_2", torch.ops.aten.mean.dim, (rnd(torch.float32, (10, 10)), 1), dict()),
      ("aten_gt_Tensor_0", torch.ops.aten.gt.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_gt_Tensor_1", torch.ops.aten.gt.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict()),
      ("aten_lt_Tensor_0", torch.ops.aten.lt.Tensor, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_lt_Tensor_1", torch.ops.aten.lt.Tensor, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict()),
      ("aten_maximum_0", torch.ops.aten.maximum.default, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_maximum_1", torch.ops.aten.maximum.default, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict()),
      ("aten_minimum_0", torch.ops.aten.minimum.default, (rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_minimum_1", torch.ops.aten.minimum.default, (rnd(torch.float32, (1, 10)), rnd(torch.float32, (10, 1)),), dict()),
      ("aten_sin_0", torch.ops.aten.sin.default, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_sin_1", torch.ops.aten.sin.default, (rnd(torch.float32, (1, 10)),), dict()),
      ("aten_cos_0", torch.ops.aten.cos.default, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_cos_1", torch.ops.aten.cos.default, (rnd(torch.float32, (1, 10)),), dict()),
      ("aten_rsqrt_0", torch.ops.aten.rsqrt.default, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_rsqrt_1", torch.ops.aten.rsqrt.default, (rnd(torch.float32, (1, 10)),), dict()),
      ("aten_neg_0", torch.ops.aten.neg.default, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_gelu_0", torch.ops.aten.gelu.default, (rnd(torch.float32, (10, 10)),), dict()),
      ("aten_gelu_1", torch.ops.aten.gelu.default, (rnd(torch.float32, (10, 10)),), dict(approximate="tanh")),
      ("aten_gelu_2", torch.ops.aten.gelu.default, (rnd(torch.float32, (1, 10)),), dict()),
      ("aten_gelu_3", torch.ops.aten.gelu.default, (rnd(torch.float32, (1, 10)),), dict(approximate="tanh")),
      ("aten_permute_0", torch.ops.aten.permute.default, (rnd(torch.float32, (10, 10)), [0, 1],), dict()),
      ("aten_permute_1", torch.ops.aten.permute.default, (rnd(torch.float32, (1, 10)), [0, 1],), dict()),
      ("aten_arange_start_step_0", torch.ops.aten.arange.start_step, (0, 100, 5), dict()),
      ("aten_arange_start_step_1", torch.ops.aten.arange.start_step, (0, 100), dict()),
      ("aten_arange_start_step_2", torch.ops.aten.arange.start_step, (100, 0, -1), dict()),
      ("aten_arange_start_step_3", torch.ops.aten.arange.start_step, (0, 10.5, 0.5), dict()),
      ("aten_cat_0", torch.ops.aten.cat.default, ([rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10))], 0,), dict()),
      ("aten_cat_1", torch.ops.aten.cat.default, ([rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10))], 1,), dict()),
      ("aten_cat_2", torch.ops.aten.cat.default, ([rnd(torch.float32, (10, 10)), rnd(torch.float32, (0, 10))], 0,), dict()),
      ("aten_cat_3", torch.ops.aten.cat.default, ([rnd(torch.float32, (10, 10)), rnd(torch.float32, (0,))], 0,), dict()),
      ("aten_cat_4", torch.ops.aten.cat.default, ([rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10))],), dict()),
      ("aten_full_0", torch.ops.aten.full.default, ([10, 10], 0.123,), dict()),
      ("aten_full_1", torch.ops.aten.full.default, ([10, 10], 123,), dict()),
      ("aten_full_like_0", torch.ops.aten.full_like.default, (rnd(torch.float32, (10, 10)), 0.123,), dict()),
      ("aten_full_like_1", torch.ops.aten.full_like.default, (rnd(torch.int64, (10, 10)), 123,), dict()),
      ("aten_view_0", torch.ops.aten.view.default, (rnd(torch.float32, (10, 10)), [1, 100],), dict()),
      ("aten_view_1", torch.ops.aten.view.default, (rnd(torch.float32, (1, 10)), [10, 1],), dict()),
      ("aten_view_2", torch.ops.aten.view.default, (rnd(torch.float32, (10, 10)), [2, 5, 10],), dict()),
      ("aten_view_3", torch.ops.aten.view.default, (rnd(torch.float32, (10, 10)), [1, -1],), dict()),
      ("aten_view_4", torch.ops.aten.view.default, (rnd(torch.float32, (10, 10)), [2, -1, 10],), dict()),
      ("aten_view_5", torch.ops.aten.view.default, (rnd(torch.float32, (10, 10)), [-1, 2, 10],), dict()),
      ("aten_split_with_sizes_0", torch.ops.aten.split_with_sizes.default, (rnd(torch.float32, (10, 10)), [1, 2, 3, 4],), dict()),
      ("aten_split_with_sizes_1", torch.ops.aten.split_with_sizes.default, (rnd(torch.float32, (10, 10)), [1, 2, 3, 4], 1), dict()),
      ("aten_unsqueeze_0", torch.ops.aten.unsqueeze.default, (rnd(torch.float32, (10, 10)), 0,), dict()),
      ("aten_unsqueeze_1", torch.ops.aten.unsqueeze.default, (rnd(torch.float32, (10, 10)), 1,), dict()),
      ("aten_unsqueeze_2", torch.ops.aten.unsqueeze.default, (rnd(torch.float32, (10, 10)), 2,), dict()),
      ("aten_unsqueeze_3", torch.ops.aten.unsqueeze.default, (rnd(torch.float32, (10, 10)), -1,), dict()),
      ("aten_expand_0", torch.ops.aten.expand.default, (rnd(torch.float32, (10, 1)), [10, 10],), dict()),
      ("aten_expand_1", torch.ops.aten.expand.default, (rnd(torch.float32, (1, 10)), [10, 10],), dict()),
      ("aten_squeeze_dims_0", torch.ops.aten.squeeze.dims, (rnd(torch.float32, (2, 1, 2, 1, 2)), [1, 2, 3],), dict()),
      ("aten_select_int_0", torch.ops.aten.select.int, (rnd(torch.float32, (2, 3, 4)), 0, 1,), dict()),
      ("aten_select_int_1", torch.ops.aten.select.int, (rnd(torch.float32, (2, 3, 4)), 1, 1,), dict()),
      ("aten_slice_tensor_0", torch.ops.aten.slice.Tensor, (rnd(torch.float32, (10, 10)),), dict(dim=0, start=1, end=3)),
      ("aten_slice_tensor_1", torch.ops.aten.slice.Tensor, (rnd(torch.float32, (10, 10)),), dict(dim=1, start=2, end=5)),
      ("aten_slice_tensor_2", torch.ops.aten.slice.Tensor, (rnd(torch.float32, (10, 10)),), dict(dim=0, start=None, end=5)),
      ("aten_slice_tensor_3", torch.ops.aten.slice.Tensor, (rnd(torch.float32, (10, 10)),), dict(dim=0, start=2, end=None)),
      ("aten_slice_tensor_4", torch.ops.aten.slice.Tensor, (rnd(torch.float32, (10, 10)),), dict(dim=0, start=-5, end=-2)),
      ("aten_slice_tensor_5", torch.ops.aten.slice.Tensor, (rnd(torch.float32, (10, 10)),), dict(dim=0, start=1, end=8, step=2)),
      ("aten_slice_tensor_6", torch.ops.aten.slice.Tensor, (rnd(torch.float32, (10, 10)),), dict(dim=1, start=2, end=100)),
      ("aten_slice_tensor_7", torch.ops.aten.slice.Tensor, (rnd(torch.float32, (10, 10)),), dict(dim=0, start=None, end=None)),
      ("aten_where_self_0", torch.ops.aten.where.self, (rnd(torch.bool, (10, 10)), rnd(torch.float32, (10, 10)), rnd(torch.float32, (10, 10)),), dict()),
      ("aten_embedding_0", torch.ops.aten.embedding.default, (rnd(torch.float32, (10, 10)), torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]),), dict()),
      ("aten__softmax_0", torch.ops.aten._softmax.default, (rnd(torch.float32, (10, 10)), -1, False), dict()),
      ("aten__softmax_1", torch.ops.aten._softmax.default, (rnd(torch.float32, (1, 10)), -1, False), dict()),
      ("aten__softmax_2", torch.ops.aten._softmax.default, (rnd(torch.float32, (10, 10)), 0, False), dict()),
      ("aten__softmax_3", torch.ops.aten._softmax.default, (rnd(torch.float32, (1, 10)), 0, False), dict()),
      ("aten__softmax_4", torch.ops.aten._softmax.default, (rnd(torch.float32, (10, 10)), 1, False), dict()),
      ("aten__softmax_5", torch.ops.aten._softmax.default, (rnd(torch.float32, (1, 10)), 1, False), dict()),
      ("aten_topk_0", torch.ops.aten.topk.default, (rnd(torch.float32, (4, 10)), 3), dict()),
      ("aten_topk_1", torch.ops.aten.topk.default, (rnd(torch.float32, (4, 10)), 3), dict(dim=0)),
      # fmt: on
      # pyformat: enable
  )
  def test_op(
      self,
      op,
      args,
      kwargs,
  ):
    self._assert_export_and_close(op, args, kwargs)

  @parameterized.named_parameters(
      # fmt: off
      # pyformat: disable
      ("aten_multinomial_0", torch.ops.aten.multinomial.default, (rnd(torch.float32, (10,)), 5, False), dict()),
      ("aten_multinomial_1", torch.ops.aten.multinomial.default, (rnd(torch.float32, (3, 10)), 5, False), dict()),
      # fmt: on
      # pyformat: enable
  )
  def test_stochastic_op(
      self,
      op,
      args,
      kwargs,
  ):
    self._assert_export_and_close(op, args, kwargs, check_values=False)

  @parameterized.named_parameters(
      # fmt: off
      # pyformat: disabledef
      ("reshape_without_dynamic_shape_0", (rnd(torch.float32, (10, 2, 3)),), dict(), None),
      ("reshape_with_dynamic_shape_1", (rnd(torch.float32, (10, 2, 3)),), dict(), ((torch.export.Dim("batch"), None, None),)),
      ("reshape_with_dynamic_shape_2", (rnd(torch.float32, (10, 2, 3)),), dict(), ({0: torch.export.Dim("batch")},)),
      ("reshape_with_dynamic_shape_3", (rnd(torch.float32, (10, 2, 3)),), dict(), ((torch.export.Dim("batch"), torch.export.Dim("height"), torch.export.Dim("width")),)),
      ("reshape_with_dynamic_shape_4", (rnd(torch.float32, (10, 2, 3)),), dict(), ({0: torch.export.Dim("batch"), 1: torch.export.Dim("height"), 2: torch.export.Dim("width")},)),
      # fmt: on
      # pyformat: enable
  )
  def test_reshape_op(
      self,
      args,
      kwargs,
      dynamic_shapes: Dict[str, Any] | Sequence[Any] | None = None,
  ):

    class ReshapeModel(torch.nn.Module):

      def forward(self, x):
        x = x + x
        x = x.reshape([x.shape[0], x.shape[1] * x.shape[2]])
        return x

    self._assert_export_and_close(ReshapeModel(), args, kwargs, dynamic_shapes)

  @parameterized.named_parameters(
      # fmt: off
      # pyformat: disabledef
      ("split_with_sizes_without_dynamic_shape_0", (rnd(torch.float32, (10, 10)),), dict(), None),
      ("split_with_sizes_with_dynamic_shape_1", (rnd(torch.float32, (10, 10)),), dict(), ({1: torch.export.Dim("batch", min=10)},)),
      # fmt: on
      # pyformat: enable
  )
  def test_split_with_sizes_op(
      self,
      args,
      kwargs,
      dynamic_shapes: Dict[str, Any] | Sequence[Any] | None = None,
  ):

    class SplitWithSizesModel(torch.nn.Module):

      def forward(self, x):
        variable_split_size = x.shape[-1] - 8
        y = torch.ops.aten.split_with_sizes(
            x, [1, variable_split_size, 3, 4], dim=-1
        )
        return y

    self._assert_export_and_close(
        SplitWithSizesModel(), args, kwargs, dynamic_shapes
    )


if __name__ == "__main__":
  googletest.main()
