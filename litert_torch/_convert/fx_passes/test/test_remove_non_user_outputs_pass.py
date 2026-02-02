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
"""Tests for RemoveNonUserOutputsPass."""

import re
from typing import Any, Callable, Union

from litert_torch import backend
from litert_torch import fx_infra
from litert_torch._convert import fx_passes
import torch

from absl.testing import absltest as googletest


def _export_to_stablehlo(
    func: Union[torch.nn.Module, Callable[..., Any]], export_args
):
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
      can_skip=False,
  )
  exported_program = fx_infra.run_passes(
      exported_program,
      [
          fx_passes.RemoveNonUserOutputsPass(),
          fx_infra.CanonicalizePass(),
      ],
  )

  return backend.export.exported_program_to_mlir(exported_program).get_text()


class TestRemoveNonUserOutputsPass(googletest.TestCase):
  """Tests for RemoveNonUserOutputsPass."""

  def test_remove_input_mutations(self):
    def f(x):
      x += 1
      x = x + 2
      x = x + 3
      return x

    stablehlo = _export_to_stablehlo(
        f,
        (torch.rand(10, 10),),
    )
    first_return = re.search(
        r"^\s+return[^\n]+\n",
        stablehlo,
        re.MULTILINE,
    ).group(0)

    self.assertRegex(first_return.strip(), r"return %\d+ : tensor<10x10xf32>")


if __name__ == "__main__":
  googletest.main()
