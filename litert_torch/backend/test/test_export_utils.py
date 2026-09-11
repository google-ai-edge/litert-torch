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
"""export_utils related tests."""

from litert_torch import backend
from litert_converter.mlir import ir
import torch

from absl.testing import absltest as googletest
from absl.testing import parameterized


class TestDtypeMapping(parameterized.TestCase):

  @parameterized.named_parameters(
      ("uint8", torch.uint8),
      ("int16", torch.int16),
      ("int32", torch.int32),
      ("long", torch.long),
      ("bool", torch.bool),
      ("half", torch.half),
      ("float32", torch.float32),
      ("float64", torch.float64),
  )
  def test_torch_dtype_ir_element_type_round_trip(self, torch_dtype):
    with backend.export_utils.create_ir_context(), ir.Location.unknown():
      ir_type = backend.export_utils.torch_dtype_to_ir_element_type(torch_dtype)
      self.assertEqual(
          backend.export_utils.ir_element_type_to_torch_dtype(ir_type),
          torch_dtype,
      )

  def test_uint8_maps_to_unsigned_integer(self):
    # Regression test: a torch.uint8 result type must map back to torch.uint8.
    # ir_element_type_to_torch_dtype used to nest the unsigned check under
    # `is_signless`, which is never true for an unsigned integer type, so a
    # `ui8` type fell through and raised "Unsupported ir element type".
    with backend.export_utils.create_ir_context(), ir.Location.unknown():
      ir_type = backend.export_utils.torch_dtype_to_ir_element_type(torch.uint8)
      self.assertTrue(ir_type.is_unsigned)
      self.assertEqual(ir_type.width, 8)
      self.assertEqual(
          backend.export_utils.ir_element_type_to_torch_dtype(ir_type),
          torch.uint8,
      )


if __name__ == "__main__":
  googletest.main()
