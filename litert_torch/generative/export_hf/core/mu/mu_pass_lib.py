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

import copy

import numpy as np

try:
  # pylint: disable=g-import-not-at-top
  from ai_edge_litert.tools import model_utils as mu
  from ai_edge_litert.tools.model_utils import core
  from ai_edge_litert.tools.model_utils import match as mm
  from ai_edge_litert.tools.model_utils.dialect import mlir
  from ai_edge_litert.tools.model_utils.dialect import tfl
  # pylint: enable=g-import-not-at-top

  _is_mu_available = True

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
      if len(reduction_x.numpy().shape) > 0:
        if reduction_x.numpy().size != 1:
          return
        red_x = reduction_x.numpy().flatten()[0]
      else:
        red_x = reduction_x.numpy()
      reduction_elements = int(1.0 / red_x)

      input_shape = op.operands[0].type.shape
      infered_elements = np.prod(
          np.take(input_shape, reduction_axis.numpy())
      ).astype(int)
      if reduction_elements != infered_elements:
        return
      out = mul_op.results[0]

      # print("[HFTransformersOptimize] Applying fuse_mean")
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


def is_mu_available() -> bool:
  return _is_mu_available


def call_pass(input_model: bytes) -> bytes:
  """Calls the pass to optimize the model."""
  if not is_mu_available():
    return input_model

  original_module, ctx = mu.read_flatbuffer(content=input_model)
  module = copy.deepcopy(original_module)

  pass_to_call = HFTransformersOptimize

  with ctx:
    pass_to_call()(module)
    # Add verify when bug is fixed with xdsl.
    module.cleanup()
    return mu.write_flatbuffer(module)


def update_model(input_model_path: str, output_model_path: str):
  """Updates the model."""
  with open(input_model_path, "rb") as f:
    input_model = f.read()
  output_model = call_pass(input_model)
  with open(output_model_path, "wb") as f:
    f.write(output_model)


def apply_mixed_precision(model_path: str, output_model_path: str):
  # TODO(weiyiw): Merge into call_pass.
  from litert_torch.generative.export_hf.core.mu import mixed_precision  # pylint: disable=g-import-not-at-top

  print("Applying mixed precision to model...")
  output_model = mixed_precision.convert_model_to_fp16(
      model_path, mixed_precision.fp32_predicate
  )
  with open(output_model_path, "wb") as f:
    f.write(output_model)
