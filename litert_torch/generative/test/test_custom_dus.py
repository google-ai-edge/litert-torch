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

"""A suite of tests to validate the Dynamic Update Slice Custom Op."""

from litert_torch.generative.custom_ops.dynamic_update_slice import dynamic_update_slice
import torch
from torch import nn

from absl.testing import absltest as googletest, parameterized


def updated_slice_matches(buffer, update, index):
  indexer = [slice(i, i + d) for i, d in zip(index, update.shape)]
  buf = buffer[indexer]
  return torch.allclose(buf, update)


def intT(x):
  return torch.tensor(x).int()


class DUSMod(nn.Module):

  def forward(self, buffer, update, index):
    out = dynamic_update_slice(buffer, update, index)
    out = out * 2
    return out


@googletest.skip('Enable this when backend is default b/373387583')
class TestCustomDUS(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'DUS_whole_buffer',
          torch.randn(1, 1280, 4, 64),
          torch.randn([1, 1024, 4, 64]),
          [intT(0), intT(0), intT(0), intT(0)],
      ),
      (
          'DUS_kv_example',
          torch.randn(2, 1280, 4, 64),
          torch.randn([2, 1024, 4, 64]),
          [intT(0), intT(0), intT(0), intT(0)],
      ),
      (
          'DUS_3d',
          torch.randn(2, 256, 4, 64),
          torch.randn([2, 256, 2, 64]),
          [intT(0), intT(0), intT(2), intT(0)],
      ),
      (
          'DUS_3d_v2',
          torch.randn(2, 256, 4, 64),
          torch.randn([2, 256, 3, 64]),
          [intT(0), intT(0), intT(1), intT(0)],
      ),
      (
          'DUS_3d_v3',
          torch.randn(6, 8, 32),
          torch.randn([6, 3, 32]),
          [intT(0), intT(5), intT(0)],
      ),
      (
          'DUS_2d',
          torch.randn(8, 32),
          torch.randn([8, 12]),
          [intT(0), intT(20)],
      ),
  )
  def test_opcheck_dynamic_update_slice(self, buffer, update, indices):
    torch.library.opcheck(dynamic_update_slice, (buffer, update, indices))
    out = dynamic_update_slice(buffer, update, indices)
    self.assertTrue(updated_slice_matches(out, update, indices))

  def test_exported_program(self):
    buffer = torch.randn(1, 1280, 4, 64)
    update = torch.randn([1, 1024, 4, 64])
    index = [intT(0), intT(0), intT(0), intT(0)]
    dm = DUSMod()
    ep = torch.export.export(dm, (buffer, update, index))
    dus_in_exported_program = False
    for node in ep.graph.nodes:
      if node.op == 'call_function':
        if node.target.__name__.startswith('dynamic_update_slice'):
          dus_in_exported_program = True
          break

    self.assertTrue(dus_in_exported_program)


if __name__ == '__main__':
  googletest.main()
