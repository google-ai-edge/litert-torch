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
from litert_torch import backend
from litert_torch import fx_infra
import torch

fx_infra.decomp.remove_pre_convert_decomp(torch.ops.aten.zeros.default)
fx_infra.decomp.remove_pre_convert_decomp(torch.ops.aten.zeros_like.default)


class RemoveSDPACompositeZeroMaskPass(fx_infra.ExportedProgramPassBase):

  def is_zero_tensor_node(self, node: torch.fx.Node):
    return node.target in (
        torch.ops.aten.zeros.default,
        torch.ops.aten.zeros_like.default,
    )

  def call(self, exported_program: torch.export.ExportedProgram):
    graph = exported_program.graph_module.graph
    for node in graph.nodes:
      if not (
          node.op == "call_function"
          and node.target == backend.composite.mark_tensor_op
      ):
        continue

      source, name, io_position, id, is_input = node.args[:5]
      # Composite info:
      # - name: odml.scaled_dot_product_attention
      # - inputs: q, k, v, mask
      if (
          name == "odml.scaled_dot_product_attention"
          and is_input
          and io_position == 3
      ):
        if self.is_zero_tensor_node(source):
          # Remove the mark_tensor call on the mask input by
          # replacing the target with an identity function.
          node.target = lambda *args, **kwargs: torch.zeros_like(args[0])

    exported_program.graph_module.graph.lint()
    exported_program.graph_module.recompile()
    return fx_infra.ExportedProgramPassResult(exported_program, True)
