# Copyright 2026 The LiteRT Torch Authors.
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
"""Tests that loader only deserializes tensor weights from checkpoints."""

import os
import pickle
from unittest import mock

from absl.testing import absltest
from litert_torch.generative.utilities import loader
import torch

_CALLED = []


def _on_unpickle():
  _CALLED.append(True)
  return {}


class _CustomObject:
  """Custom object used to check that only tensor weights are unpickled."""

  def __reduce__(self):
    return (_on_unpickle, ())


class LoadPytorchStatedictTest(absltest.TestCase):

  def test_loads_tensor_state_dict(self):
    tmp_dir = self.create_tempdir().full_path
    torch.save({"w": torch.ones(2)}, os.path.join(tmp_dir, "model.bin"))
    state_dict = loader.load_pytorch_statedict(tmp_dir)
    self.assertEqual(list(state_dict.keys()), ["w"])
    self.assertTrue(torch.equal(state_dict["w"], torch.ones(2)))

  def test_loads_with_weights_only(self):
    # PyTorch 2.6+ defaults to weights_only=True, while PyTorch 2.4 and 2.5 do
    # not, so check that the loader passes weights_only=True explicitly.
    tmp_dir = self.create_tempdir().full_path
    torch.save({"w": torch.ones(2)}, os.path.join(tmp_dir, "model.bin"))
    with mock.patch.object(torch, "load", wraps=torch.load) as mock_load:
      loader.load_pytorch_statedict(tmp_dir)
    mock_load.assert_called_once()
    self.assertIs(mock_load.call_args.kwargs.get("weights_only"), True)

  def test_rejects_custom_pickled_objects(self):
    tmp_dir = self.create_tempdir().full_path
    with open(os.path.join(tmp_dir, "custom.bin"), "wb") as f:
      torch.save(_CustomObject(), f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    _CALLED.clear()
    with self.assertRaises(pickle.UnpicklingError):
      loader.load_pytorch_statedict(tmp_dir)
    self.assertEmpty(_CALLED)


if __name__ == "__main__":
  absltest.main()
