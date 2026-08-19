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
"""Provides lowering for coreaten to stablehlo for FFT operations."""

from litert_torch.backend.lowerings import registry
from litert_converter.mlir import ir
from litert_converter.mlir.dialects import stablehlo
import torch

lower = registry.lower


@lower(torch.ops.aten._fft_r2c.default)
def _aten_fft_r2c(
    lctx,
    input: ir.Value,
    dim: list[int],
    normalization: int,
    onesided: bool,
):
  if normalization != 0:
    raise NotImplementedError("FFT normalization != 0 is not yet implemented.")

  input_type: ir.RankedTensorType = input.type
  rank = len(input_type.shape)
  pos_dim = [d if d >= 0 else rank + d for d in dim]

  if len(pos_dim) != 1 or pos_dim[0] != rank - 1:
    raise NotImplementedError(
        "Only 1D FFT on the last dimension is currently supported."
    )

  fft_length = input_type.shape[pos_dim[0]]

  # TFLite only supports 2D FFT (RFFT2D). We implement 1D FFT of length L
  # by reshaping the input to [..., 1, L], performing 2D FFT with length [1, L],
  # and reshaping the output back to [..., L // 2 + 1].
  
  # 1. Reshape input [..., L] -> [..., 1, L]
  reshaped_input_shape = list(input_type.shape)
  reshaped_input_shape.insert(-1, 1)
  reshaped_input_type = ir.RankedTensorType.get(
      reshaped_input_shape, input_type.element_type
  )
  reshaped_input = stablehlo.reshape(reshaped_input_type, input)

  # 2. Setup 2D FFT output type [..., 1, L // 2 + 1]
  out_aval = lctx.node.meta.get("tensor_meta") or lctx.node.meta.get("val")
  complex_elem_type = ir.ComplexType.get(input_type.element_type)
  
  reshaped_output_shape = list(out_aval.shape)
  reshaped_output_shape.insert(-1, 1)
  reshaped_output_type = ir.RankedTensorType.get(
      reshaped_output_shape, complex_elem_type
  )

  # 3. Run 2D FFT
  fft_type = stablehlo.FftTypeAttr.get("RFFT")
  fft_length_attr = ir.DenseI64ArrayAttr.get([1, fft_length])
  
  fft_res = stablehlo.fft(
      results=[reshaped_output_type],
      operand=reshaped_input,
      fft_type=fft_type,
      fft_length=fft_length_attr,
  )

  # 4. Reshape output back [..., 1, L // 2 + 1] -> [..., L // 2 + 1]
  res_type = ir.RankedTensorType.get(out_aval.shape, complex_elem_type)
  return stablehlo.reshape(res_type, fft_res)
