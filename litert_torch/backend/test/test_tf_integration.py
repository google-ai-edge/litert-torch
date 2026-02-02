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
import numpy as np
import torch
import torchvision

from absl.testing import absltest as googletest


def _reset_from_node_meta_and_lower(ep: torch.export.ExportedProgram):
  """Lower the exported program with canonical history stack."""
  ep = fx_infra.graph_utils.reset_from_node_meta(ep)
  return backend.export.exported_program_to_mlir(ep)


def _is_aten_op(node: torch.fx.Node) -> bool:
  return node.op == "call_function" and not node.name.startswith("getitem")


class AddModel(torch.nn.Module):
  """A simple model that does addition."""

  def forward(self, x, y):
    return x + y + x + y


class TensorflowIntegrationTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_mlir_lowered_call(self):
    """Test a simple model with MLIR lowered call."""

    model = AddModel().eval()
    forward_args = lambda: (torch.rand((10, 10)), torch.rand((10, 10)))
    ep = torch.export.export(model, forward_args())

    lowered = backend.export.exported_program_to_mlir(ep)

    args = forward_args()
    torch_output = model(*args).detach().numpy()
    lowering_output = np.array(lowered(*args))

    self.assertTrue(np.allclose(lowering_output, torch_output))

  def test_resnet18(self):
    """Test Resnet18 model with MLIR lowered call."""

    model = torchvision.models.resnet18().eval()
    forward_args = lambda: (torch.rand((1, 3, 224, 224)),)

    ep = torch.export.export(model, forward_args())
    lowered = _reset_from_node_meta_and_lower(ep)

    args = forward_args()
    torch_output = model(*args).detach().numpy()
    lowering_output = np.array(lowered(*args))

    # Check value and debug info.
    self.assertTrue(np.allclose(lowering_output, torch_output, atol=1e-5))

  def test_debuginfo_from_export_lower(self):
    """Test the debuginfo with export lower."""

    model = torchvision.models.resnet18().eval()
    forward_args = lambda: (torch.rand((1, 3, 224, 224)),)

    ep = torch.export.export(model, forward_args())
    lowered = _reset_from_node_meta_and_lower(ep)

    lowered_text = lowered.get_text(enable_debug_info=True)
    # Check the file info.
    self.assertIn("torchvision/models/resnet.py", lowered_text)
    # Check the fx node names.
    for n in ep.graph.nodes:
      # Record all aten op nodes from the original graph and check if they
      # are lowered to the same name in the lowered graph.
      if _is_aten_op(n):
        # Ensure strings like `loc("relu__1"` are present in the lowered text.
        self.assertIn(f'loc("{n.name}"', lowered_text)

  def test_debuginfo_from_loaded_reexport_lower(self):
    """Test the debuginfo with loaded reexport lower."""

    model = AddModel().eval()
    forward_args = lambda: (torch.rand((10, 10)), torch.rand((10, 10)))

    # Ensure the debuginfo is preserved after saving, loading and reexporting.
    ep = torch.export.export(model, forward_args())
    torch.export.save(ep, "/tmp/add_model.pt2")
    loaded_ep = torch.export.load("/tmp/add_model.pt2")
    reexported_ep = torch.export.export(loaded_ep.module(), forward_args())
    lowered = _reset_from_node_meta_and_lower(reexported_ep)

    lowered_text = lowered.get_text(enable_debug_info=True)
    # Check the file info.
    self.assertIn(
        "litert_torch/backend/test/test_tf_integration.py", lowered_text
    )
    # Check the fx node names.
    for n in reexported_ep.graph.nodes:
      # Record all aten op nodes from the original graph and check if they
      # are lowered to the same name in the lowered graph.
      if _is_aten_op(n):
        self.assertIn(f'loc("{n.name}"', lowered_text)


if __name__ == "__main__":
  googletest.main()
