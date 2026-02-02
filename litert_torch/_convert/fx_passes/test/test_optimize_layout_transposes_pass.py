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
"""Tests for OptimizeLayoutTransposesPass."""

from typing import Callable, Union

import litert_torch
from litert_torch import backend
from litert_torch import fx_infra
from litert_torch._convert import fx_passes
import torch
import torch.utils._pytree as pytree
import torchvision

from absl.testing import absltest as googletest


def export_with_pass(
    func: Union[torch.nn.Module, Callable[..., torch.Tensor]],
    export_args: list[torch.Tensor],
) -> torch.export.ExportedProgram:
  """Exports a function with OptimizeLayoutTransposesPass."""
  if not isinstance(func, torch.nn.Module):

    class TestModule(torch.nn.Module):

      def forward(self, *args, **kwargs):
        return func(*args, **kwargs)

    module = TestModule().eval()
  else:
    module = func

  exported_program = torch.export.export(module, export_args)
  exported_program = fx_infra.safe_run_decompositions(
      exported_program,
      fx_infra.decomp.pre_convert_decomp(),
  )
  exported_program = fx_infra.run_passes(
      exported_program,
      [fx_passes.OptimizeLayoutTransposesPass()],
  )
  return exported_program


class TestOptimizeLayoutTransposesPass(googletest.TestCase):
  """Tests for OptimizeLayoutTransposesPass."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def assert_outputs_allclose(self, m1, m2, args):
    out1 = m1(*args)
    out2 = m2(*args)
    out1, _ = pytree.tree_flatten(out1)
    out2, _ = pytree.tree_flatten(out2)
    self.assertEqual(len(out1), len(out2))
    for o1, o2 in zip(out1, out2):
      self.assertTrue(torch.allclose(o1, o2, atol=1e-5))

  def test_torchvision_mobilenet_v3_small(self):
    model = torchvision.models.mobilenet_v3_small().eval()
    forward_args = lambda: (torch.rand(1, 3, 224, 224),)

    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(
        model, exported_program.module(), forward_args()
    )

  def test_torchvision_resnet18(self):
    model = torchvision.models.resnet18().eval()
    forward_args = lambda: (torch.rand(1, 3, 224, 224),)

    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(
        model, exported_program.module(), forward_args()
    )

  def test_group_norm_affine_false(self):

    class SampleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=640, affine=False, eps=1e-6
        )

      def forward(self, x):
        x = torch.nn.AvgPool2d(2)(x)
        x = self.group_norm(x)
        x = torch.nn.AvgPool2d(2)(x)
        return x

    model = SampleModel().eval()
    forward_args = lambda: (torch.rand(16, 640, 32, 32) * 1000,)
    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(
        model, exported_program.module(), forward_args()
    )

  def test_group_norm_large_affine_true(self):

    class SampleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=640, affine=True, eps=1e-6
        )

      def forward(self, x):
        x = torch.nn.AvgPool2d(2)(x)
        x = self.group_norm(x)
        x = torch.nn.AvgPool2d(2)(x)
        return x

    model = SampleModel().eval()
    forward_args = lambda: (torch.rand(16, 640, 32, 32) * 1000,)
    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(
        model, exported_program.module(), forward_args()
    )

  def test_group_norm_with_composite_enabled(self):
    litert_torch.config.enable_group_norm_composite = True

    class SampleModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(
            num_groups=2, num_channels=10, affine=True
        )

      def forward(self, x):
        x = torch.nn.AvgPool2d(2)(x)
        x = self.group_norm(x)
        x = torch.nn.AvgPool2d(2)(x)
        return x

    model = SampleModel().eval()
    forward_args = lambda: (torch.rand(1, 10, 32, 32),)
    exported_program = export_with_pass(model, forward_args())
    self.assert_outputs_allclose(
        model, exported_program.module(), forward_args()
    )

    ir_text = backend.export.exported_program_to_mlir(
        exported_program
    ).get_text()
    self.assertEqual(ir_text.count("stablehlo.custom_call @mark_tensor"), 4)


if __name__ == "__main__":
  googletest.main()
