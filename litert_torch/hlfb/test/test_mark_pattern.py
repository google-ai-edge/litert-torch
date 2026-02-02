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
"""Tests for mark_pattern."""

import re

from litert_torch import backend
from litert_torch import fx_infra
from litert_torch.hlfb import mark_pattern
from litert_torch.hlfb.mark_pattern import pattern as pattern_module
import torch

from absl.testing import absltest as googletest


def _export_and_decomp(mod, args):
  ep = torch.export.export(mod, args)
  ep = fx_infra.safe_run_decompositions(ep, fx_infra.decomp.pre_lower_decomp())
  return ep


def _to_mlir(ep: torch.export.ExportedProgram):
  return backend.export.exported_program_to_mlir(ep).get_text()


def _extract_backend_configs(mlir):
  mlir = mlir.replace("\\22", '"')
  configs = []
  for match in re.finditer(r"backend_config\s*=\s*\"(\{.*\})\"", mlir):
    configs.append(match.group(1))
  return "\n".join(configs)


class TestMarkPattern(googletest.TestCase):

  def test_mark_pattern(self):

    class TestModel(torch.nn.Module):

      def forward(self, x):
        return x * x + x + x

    pattern = pattern_module.Pattern(
        "test.add",
        lambda a, b: a + b,
        export_args=(torch.rand(2, 2), torch.rand(2, 2)),
    )

    model = TestModel().eval()
    args = (torch.rand(20, 20),)
    exported_program = _export_and_decomp(model, args)
    mark_pattern.mark_pattern(exported_program.graph_module, pattern)
    mlir = _to_mlir(exported_program)

    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 6)

  def test_mark_pattern_with_clone_inputs(self):

    class TestModel(torch.nn.Module):

      def forward(self, x):
        return torch.ops.aten.clone.default(x * x) + x

    pattern = pattern_module.Pattern(
        "test.add",
        lambda a, b: a + b,
        export_args=(torch.rand(2, 2), torch.rand(2, 2)),
    )

    model = TestModel().eval()
    args = (torch.rand(20, 20),)
    exported_program = _export_and_decomp(model, args)
    mark_pattern.mark_pattern(exported_program.graph_module, pattern)
    mlir = _to_mlir(exported_program)

    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 3)

  def test_mark_pattern_with_attr_builder(self):
    class TestModel(torch.nn.Module):

      def forward(self, x):
        return x * x * x + x - x * x + x

    pattern = pattern_module.Pattern(
        "test.add",
        lambda a, b: a + b,
        export_args=(torch.rand(2, 2), torch.rand(2, 2)),
        attr_builder=lambda *args: {"alias": "test.test_add"},
    )

    model = TestModel().eval()
    args = (torch.rand(20, 20),)
    exported_program = _export_and_decomp(model, args)
    mark_pattern.mark_pattern(exported_program.graph_module, pattern)
    mlir = _to_mlir(exported_program)

    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 6)
    backend_configs = _extract_backend_configs(mlir)
    self.assertEqual(backend_configs.count('{"alias": "test.test_add"}'), 2)

  def test_mark_pattern_with_scalar_attr_tracker(self):
    class TestModel(torch.nn.Module):

      def forward(self, x):
        r = x
        for idx in range(5):
          r = torch.nn.LogSoftmax(dim=idx % 2)(r) * x
        return r

    pattern = pattern_module.Pattern(
        "test.log_softmax",
        lambda x, dim: torch.nn.functional.log_softmax(x, dim=dim),
        export_args=(torch.rand(10, 10, 10), 1),
        scalar_attr_trackers=[
            pattern_module.ScalarAttrTracker("dim", pattern_arg_pos=1)
            .track(0)
            .track(1)
            .track(2),
        ],
    )

    model = TestModel().eval()
    args = (torch.rand(10, 10),)
    exported_program = _export_and_decomp(model, args)
    mark_pattern.mark_pattern(exported_program.graph_module, pattern)
    mlir = _to_mlir(exported_program)

    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 10)
    backend_configs = _extract_backend_configs(mlir)
    self.assertEqual(backend_configs.count('{"dim": 0}'), 3)
    self.assertEqual(backend_configs.count('{"dim": 1}'), 2)

  def test_mark_tangent_model_and_pattern_input(self):
    class TestModel(torch.nn.Module):

      def forward(self, x, y):
        z = torch.ops.aten.relu(x)
        z = z + y
        return z

    pattern = pattern_module.Pattern(
        "test.relu",
        lambda x: torch.ops.aten.relu(x),
        export_args=(torch.rand(2, 2),),
    )

    model = TestModel().eval()
    args = (torch.rand(20, 20), torch.rand(20, 20))
    exported_program = _export_and_decomp(model, args)
    mark_pattern.mark_pattern(exported_program.graph_module, pattern)
    mlir = _to_mlir(exported_program)

    self.assertEqual(mlir.count("stablehlo.custom_call @mark_tensor"), 2)


if __name__ == "__main__":
  googletest.main()
