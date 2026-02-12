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
"""Model optimization passes."""

from typing import Any, cast
from litert_torch import model as model_lib
from litert_torch._convert import litert_converter
import numpy as np

LazyModelExporter = litert_converter.LazyModelExporter

try:
  # pylint: disable=g-import-not-at-top
  from ai_edge_litert.tools import model_utils as mu
  from ai_edge_litert.tools.model_utils import core
  from ai_edge_litert.tools.model_utils import match as mm
  from ai_edge_litert.tools.model_utils.dialect import mlir
  from ai_edge_litert.tools.model_utils.dialect import tfl
  # pylint: enable=g-import-not-at-top

  _is_mu_available = True
  MuModuleOp = mu.dialect.mlir.ModuleOp

  class HFTransformersOptimize(core.RewritePatternPassBase):
    """Rewrite pass."""

    name = "hf-transformers-optimize"

  @HFTransformersOptimize.register_rewrite_pattern(tfl.SumOp)
  def fuse_mean(op: tfl.SumOp, rewriter) -> None:
    """A pattern that fuse sum-mul with mean."""

    with mm.MatchingContext():
      mm.match(op.name == "tfl.sum")
      reduction_axis = mm.op("arith.constant", None, [op.operands[1]])
      mul_op = mm.op("tfl.mul", [op.results[0], mm.ANY], None)
      reduction_x = mm.op("arith.constant", None, [mul_op.operands[1]])

      mm.match(np.prod(reduction_x.results[0].type.shape) == 1)
      reduction_elements = int(1.0 / reduction_x.numpy().item())

      input_shape = op.operands[0].type.shape
      infered_elements = int(
          np.prod(np.take(input_shape, reduction_axis.numpy()))
      )
      if reduction_elements != infered_elements:
        return
      out = mul_op.results[0]

      print("[HFTransformersOptimize] Applying fuse_mean")
      with core.OpBuildingContext(mul_op):
        mean_op = mlir.MlirOp(
            name="tfl.mean",
            operands=op.operands,
            attributes=op.attributes,
            result_types=op.result_types,
        )
        out.replace_by(mean_op.results[0])
        rewriter.erase_op(mul_op)

except ImportError:
  _is_mu_available = False
  MuModuleOp = Any


def is_mu_available() -> bool:
  return _is_mu_available


def call_pass(mu_module: MuModuleOp) -> MuModuleOp:
  """Calls the pass to optimize the model."""
  if not is_mu_available():
    return mu_module

  # original_module = mu_module
  # mu_module = copy.deepcopy(original_module)

  pass_to_call = HFTransformersOptimize

  pass_to_call()(mu_module)

  # Add verify when bug is fixed with xdsl.
  mu_module.cleanup()
  return mu_module


def update_model(model: model_lib.LiteRTModel) -> model_lib.LiteRTModel:
  """Updates the model with the optimization passes."""
  if not is_mu_available():
    return model

  exporter = cast(model_lib.ModelExporter, model._exporter)
  if isinstance(exporter, LazyModelExporter) and exporter.module is not None:
    # Quick path: avoid serialization and deserialization.
    ir_module = exporter.module
    ctx = ir_module.context
    with ctx:
      mu_module = mu.transform.mlir_to_model_utils(ir_module)
  else:
    # Slow path: reconstruct the model from bytes.
    mu_module, ctx = mu.read_flatbuffer(content=model.model_content())

  with ctx:
    mu_module = call_pass(mu_module)
    ir_module = mu.transform.model_utils_to_mlir(mu_module, ir_context=ctx)

  new_exporter = LazyModelExporter(module=ir_module)
  return model_lib.LiteRTModel(new_exporter)
