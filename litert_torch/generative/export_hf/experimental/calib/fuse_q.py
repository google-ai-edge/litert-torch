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
"""Pass to fuse quantized bmm."""

from litert_converter.tools import model_utils as mu
from litert_converter.tools.model_utils import match as mm
from litert_converter.tools.model_utils.dialect import tfl


class FuseQuantizedBmmPass(mu.RewritePatternPassBase):
  name = "fuse-quantized-bmm"
  description = "Fuse quantized bmm"


@FuseQuantizedBmmPass.register_rewrite_pattern(tfl.BatchMatMulOp)
def fuse_quantized_bmm(bmm: tfl.BatchMatMulOp, rewriter) -> None:
  """Fuses a quantize op into a batch_matmul op."""
  with mm.MatchingContext() as mmctx:
    bmm = mm.op(bmm, preds=[lambda bmm: bmm.name == tfl.BatchMatMulOp.name])
    if not hasattr(bmm.operands[1], "op"):
      return
    quant_op = bmm.operands[1].op
    if quant_op is None or quant_op.name != "tfl.quantize":
      return
    if hasattr(quant_op.operands[0], "op"):
      return
    if mmctx.failed:
      return
    print(f"fuse_quantized_bmm on {bmm.location}")
    with mu.OpBuildingContext(bmm, no_insert=True) as opctx:
      _ = tfl.batch_matmul(
          x=bmm.x,
          y=quant_op.operands[0],
          adj_x=bmm.adj_x,
          adj_y=bmm.adj_y,
          asymmetric_quantize_inputs=bmm.asymmetric_quantize_inputs,
          result_type=bmm.result_types[0],
      )
      rewriter.replace_matched_op(opctx.new_ops)
