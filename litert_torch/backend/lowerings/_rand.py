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
import uuid

from litert_torch.backend import export_utils
from litert_torch.backend.lowerings import context
from litert_torch.backend.lowerings import registry
from ai_edge_litert.mlir import ir
from ai_edge_litert.mlir.dialects import func
from ai_edge_litert.mlir.dialects import stablehlo
import numpy as np
import torch
import torch.utils._pytree as pytree

LoweringContext = context.LoweringContext
lower = registry.lower


def _random_lowering(
    lctx: LoweringContext,
    size: list[int],
    generator,
    dtype: torch.dtype,
    rand_tensor,
    composite_name: str,
):
  if dtype is None:
    dtype = torch.float32

  rand_tensor = rand_tensor.type(dtype)
  data = rand_tensor.detach().numpy()

  shape, _ = pytree.tree_flatten(size)
  elty = export_utils.torch_dtype_to_ir_element_type(dtype)

  decomp_name = f"{composite_name}.impl_{uuid.uuid4().hex[:8]}"

  with ir.InsertionPoint(lctx.ir_module.body):

    @func.FuncOp.from_py_func(
        ir.RankedTensorType.get(
            [len(shape)],
            ir.IntegerType.get_signless(32),
        ),
        name=decomp_name,
    )
    def _rand_impl(_):
      return [stablehlo.constant(ir.DenseElementsAttr.get(data))]

  seed, seed2 = (
      torch.randint(
          torch.iinfo(torch.int64).min,
          torch.iinfo(torch.int64).max,
          (2,),
          dtype=torch.int64,
          generator=generator,
      )
      .detach()
      .numpy()
  )

  shape_ = stablehlo.constant(
      ir.DenseElementsAttr.get(np.array(shape, dtype=np.int32))
  )
  return stablehlo.CompositeOp(
      result=[ir.RankedTensorType.get(shape, elty)],
      inputs=[shape_],
      name=composite_name,
      composite_attributes=ir.DictAttr.get({
          "seed": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), seed),
          "seed2": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), seed2),
      }),
      decomposition=decomp_name,
  ).results[0]


# Schema:
# - aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None,
#     Device? device=None, bool? pin_memory=None) -> Tensor
# - aten::rand.generator(SymInt[] size, *, Generator? generator,
#     ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#     bool? pin_memory=None) -> Tensor
@registry.lower(torch.ops.aten.rand)
def _aten_rand(
    lctx: LoweringContext,
    size,
    generator=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=False,
):
  return _random_lowering(
      lctx,
      size,
      generator,
      dtype,
      rand_tensor=torch.ops.aten.rand.generator(
          size, generator=generator, dtype=dtype
      ),
      composite_name="odml.random_uniform",
  )


# Schema:
# - aten::randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None,
#     Device? device=None, bool? pin_memory=None) -> Tensor
# - aten::randn.generator(SymInt[] size, *, Generator? generator,
#     ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#     bool? pin_memory=None) -> Tensor
@registry.lower(torch.ops.aten.randn)
def _aten_randn(
    lctx: LoweringContext,
    size,
    generator=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    pin_memory=False,
):
  return _random_lowering(
      lctx,
      size,
      generator,
      dtype,
      rand_tensor=torch.ops.aten.randn.generator(
          size, generator=generator, dtype=dtype
      ),
      composite_name="odml.random_standard_normal",
  )
