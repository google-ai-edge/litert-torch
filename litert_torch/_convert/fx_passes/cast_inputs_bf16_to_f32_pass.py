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
"""Pass to cast all inputs with torch.bfloat16 type to torch.float32."""

from litert_torch import fx_infra
import torch


def cast_f32(x):
  return x.to(torch.float32)


class CastInputsBf16ToF32Pass(fx_infra.ExportedProgramPassBase):
  """This pass casts all inputs with torch.bfloat16 type to torch.float32."""

  def call(self, exported_program: torch.export.ExportedProgram):
    # If the model holds bfloat16 weights, it is a bfloat16 model: casting the
    # inputs to float32 would only create a dtype mismatch against those
    # weights, so leave the graph alone. Intermediate activations are
    # deliberately not inspected here - a bfloat16 input always produces
    # bfloat16 intermediates, so checking them would make this pass a no-op for
    # exactly the models it exists to handle.
    for p in exported_program.parameters():
      if getattr(p, "dtype", None) == torch.bfloat16:
        return fx_infra.ExportedProgramPassResult(exported_program, False)
    for b in exported_program.buffers():
      if getattr(b, "dtype", None) == torch.bfloat16:
        return fx_infra.ExportedProgramPassResult(exported_program, False)
    for c in exported_program.constants.values():
      if getattr(c, "dtype", None) == torch.bfloat16:
        return fx_infra.ExportedProgramPassResult(exported_program, False)

    modified = False
    for node in exported_program.graph.nodes:
      val = node.meta.get("val", None)
      if (
          node.op == "placeholder"
          and getattr(val, "dtype", None) == torch.bfloat16
      ):
        if not node.users:
          continue

        modified = True
        user = next(iter(node.users))
        with exported_program.graph.inserting_before(user):
          cast_node = exported_program.graph.call_function(
              cast_f32,
              (node,),
          )
          node.replace_all_uses_with(cast_node)
          cast_node.replace_input_with(cast_node, node)

    exported_program.graph_module.recompile()
    return fx_infra.ExportedProgramPassResult(exported_program, modified)
