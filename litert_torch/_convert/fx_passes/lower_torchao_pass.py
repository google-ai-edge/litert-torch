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
"""FX pass to lower TorchAO quantization operations to LiteRT composites.

TorchAO (PyTorch Architecture Optimization) represents quantized neural network
models using high-level affine quantization ops such as:
  - `torch.ops.torchao.quantize_affine`: Quantizes a floating point tensor.
  - `torch.ops.torchao.dequantize_affine`: Dequantizes an integer tensor back to
    float using scale and zero-point.
  - `torch.ops.torchao.choose_qparams_affine`: Dynamically computes per-token
    scale and zero-point from activation tensors.
  - `torch.ops.aten._int_mm`: Computes an integer matrix multiplication between
    quantized activation and weight matrices.

During LiteRT compilation, models are lowered from PyTorch FX graphs to
StableHLO / MLIR. LiteRT's quantization infrastructure and TFLite flatbuffer
exporter rely on quantized composite operations (`quant.fake_quant` and
`quant.dequantize`) tagged with composite attributes e.g. `scale`, `zero_point`,
`dtype`, `quantization_dimension`, `narrow_range`. When strict QDQ mode
(`strict_qdq_mode`) is enabled, LiteRT's MLIR quantization passes convert these
composites into fully fused integer or dynamic range quantized TFLite kernels
(such as int8/int4 `FULLY_CONNECTED` or `BATCH_MATMUL`).

This module implements:
  1. `LowerTorchAOPass`: An FX graph transformation that matches TorchAO
     quantization patterns and rewires them into LiteRT's `quant_composite` ops.
  2. Support for multiple quantization modalities:
     - Weight-Only Quantization (per-tensor, per-axis / per-channel).
     - Dynamic Range Quantization (DRQ) with per-token dynamic activations.
     - Static Quantization (PTQ / QAT) with integer matrix multiplication.
     - Optional promotion of weight-only models to dynamic range quantization
       (`force_weight_only_as_drq`).
     - NOT SUPPORTED:
       Blockwise / Groupwise Weight-Only Quantization e.g. group_size=32/64/128.
  3. Custom PyTorch operator `torch.ops.litert_torch.quant_composite` with both
     functional eager / autograd and meta implementations for TorchDynamo
     tracing.
  4. MLIR lowering rules mapping `quant_composite` to `stablehlo.composite`
     operations with dictionary attributes and private decomposition functions.
"""

import json
from typing import Any, Optional, Union
import uuid
from litert_torch import fx_infra
from litert_torch.backend import _torch_library
from litert_torch.backend import lowerings
from litert_converter.mlir import ir
from litert_converter.mlir.dialects import func
from litert_converter.mlir.dialects import stablehlo
import numpy as np
import torch

try:
  import torchao  # pylint: disable=unused-import

  # TorchAO ops must not be decomposed into primitive aten math ops
  # (div, round, clamp, sub, mul) during the initial FX pre-convert phase.
  # By default, LiteRT-Torch's export pipeline registers decompositions that
  # expand high-level operations. We explicitly deregister pre-convert
  # decompositions for all TorchAO operations so the FX graph retains
  # `quantize_affine`, `dequantize_affine`, and `choose_qparams_affine`
  # for `LowerTorchAOPass` to inspect and lower to composites.
  _TORCHAO_OPS = [
      getattr(torch.ops.torchao, name).default
      for name in dir(torch.ops.torchao)
      if hasattr(getattr(torch.ops.torchao, name, None), "default")
  ]
  for _op in _TORCHAO_OPS:
    fx_infra.decomp.remove_pre_convert_decomp(_op)
except ImportError:
  _TORCHAO_OPS = []


def _infer_dtype_str(
    quant_min: int, quant_max: int, fallback_dtype: torch.dtype = torch.int8
) -> str:
  """Infers the MLIR quantization dtype string from quant min/max bounds.

  LiteRT composites recognize string identifiers such as 'i4', 'ui4', 'i8',
  'ui8', 'i16', and 'i32' to denote the target integer precision.

  Args:
    quant_min: Lower bound of the quantized integer representation.
    quant_max: Upper bound of the quantized integer representation.
    fallback_dtype: Fallback PyTorch dtype if bounds do not match known ranges.

  Returns:
    A string identifier matching LiteRT quantization dialect types (e.g. 'i8').
  """
  if quant_min >= -8 and quant_max <= 7:
    return "i4"
  elif quant_min >= 0 and quant_max <= 15:
    return "ui4"
  elif quant_min >= -128 and quant_max <= 127:
    return "i8"
  elif quant_min >= -32768 and quant_max <= 32767:
    return "i16"
  elif fallback_dtype == torch.int16:
    return "i16"
  elif fallback_dtype == torch.int8:
    return "i8"
  return "i8"


# Maps PyTorch dtypes or string names to LiteRT quantization dtype strings.
_DTYPE_MAP = {
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.uint8: "ui8",
    "i8": "i8",
    "i4": "i4",
    "ui4": "ui4",
    "i16": "i16",
    "i32": "i32",
}


def _infer_quant_dimension(
    tensor_shape: Union[torch.Size, tuple[int, ...], list[int]],
    block_size: Union[tuple[int, ...], list[int]],
    scale_numel: int,
) -> Optional[int]:
  """Infers the quantization axis dimension from tensor shape and scale count.

  In per-axis (per-channel) quantization, the quantization scale has multiple
  elements corresponding to the size of a specific tensor dimension. In TorchAO,
  the `block_size` argument specifies the granularity of quantization across
  each dimension: a dimension with block_size == 1 is quantized at unit
  granularity (per-channel along that axis), whereas block_size == dim_size
  indicates per-tensor granularity along that axis.

  Args:
    tensor_shape: Shape of the tensor being quantized or dequantized.
    block_size: Block size tuple/list specified in TorchAO affine quant op.
    scale_numel: Total number of scale elements in the scale tensor.

  Returns:
    The 0-based index of the quantized dimension, or None for per-tensor
    quantization (where scale_numel <= 1).
  """
  if scale_numel <= 1:
    return None
  if len(tensor_shape) == len(block_size):
    for d, (s, b) in enumerate(zip(tensor_shape, block_size)):
      if b == 1 and s == scale_numel:
        return d
  for d, s in enumerate(tensor_shape):
    if s == scale_numel:
      return d
  return 0


def _get_quant_composite_attributes(
    scale: Optional[Union[float, list[float], torch.Tensor]] = None,
    zero_point: Optional[Union[int, list[int], torch.Tensor]] = None,
    dtype: Union[torch.dtype, str] = "i8",
    quant_dimension: Optional[int] = None,
    narrow_range: bool = True,
) -> dict[str, Any]:
  """Constructs the attributes dictionary for quant composites.

  The resulting dictionary adheres to the schema expected by LiteRT MLIR passes:
    - 'dtype': Target quantized integer bitwidth ('i4', 'ui4', 'i8', 'i16').
    - 'scale': 1D list of floating point scale values.
    - 'zero_point': 1D list of integer zero point values.
    - 'quantization_dimension': Optional, Quantization axis for per-channel ops.
    - 'narrow_range': Whether symmetric signed integer range is used
      (e.g. [-127, 127] for int8 instead of [-128, 127]).

  Args:
    scale: Float scalar, list of floats, or tensor of quantization scales.
    zero_point: Int scalar, list of ints, or tensor of quantization zero points.
    dtype: Target quantization storage dtype (PyTorch dtype or string like
      'i8').
    quant_dimension: Axis along which scales/zero_points are applied. None for
      per-tensor quantization.
    narrow_range: Boolean indicating whether narrow range is used.

  Returns:
    Dictionary containing serialized attributes suitable for MLIR lowering.
  """
  dtype_str = _DTYPE_MAP.get(dtype, str(dtype))

  attrs: dict[str, Any] = {
      "dtype": dtype_str,
      "narrow_range": bool(narrow_range),
  }

  if scale is not None:
    if isinstance(scale, torch.Tensor):
      scale_list = [float(s) for s in scale.flatten().tolist()]
    elif isinstance(scale, (list, tuple)):
      scale_list = [float(s) for s in scale]
    elif isinstance(scale, (int, float)):
      scale_list = [float(scale)]
    else:
      scale_list = []
    attrs["scale"] = scale_list
  else:
    attrs["scale"] = []

  if zero_point is not None:
    if isinstance(zero_point, torch.Tensor):
      zp_list = [int(zp) for zp in zero_point.flatten().tolist()]
    elif isinstance(zero_point, (list, tuple)):
      zp_list = [int(zp) for zp in zero_point]
    elif isinstance(zero_point, (int, float)):
      zp_list = [int(zero_point)]
    else:
      zp_list = []
    attrs["zero_point"] = zp_list
  else:
    attrs["zero_point"] = []

  if quant_dimension is not None:
    attrs["quantization_dimension"] = int(quant_dimension)

  return attrs


class LowerTorchAOPass(fx_infra.ExportedProgramPassBase):
  """Lowers TorchAO quantization ops into LiteRT composite operations.

  This pass scans the FX graph of an `ExportedProgram` for TorchAO quantization
  patterns and transforms them into LiteRT's `quant.fake_quant` and
  `quant.dequantize` composite operations. It handles four primary scenarios:

  1. Full integer matrix multiplication (`torch.ops.aten._int_mm.default`):
     Transforms integer matmul + downstream dequantization scaling into an
     `aten.linear` preceded by `quant.fake_quant` on activations and
     `quant.dequantize` on weights.
  2. Weight dequantization (`torch.ops.torchao.dequantize_affine.default` on
     weights): Transforms weight dequantization into `quant.dequantize`.
     Supports both per-channel quantization and blockwise (grouped) quantization
     by reshaping the weight tensor into 2D blocks `[total_blocks, group_size]`.
  3. Activation dequantization (`torch.ops.torchao.dequantize_affine.default` on
     activations): Transforms paired `quantize_affine` -> `dequantize_affine`
     into a single `quant.fake_quant` composite on the activation tensor.
     Supports both static quantization (fixed scale/zero-point) and dynamic
     range quantization (DRQ, where parameters originate from
     `choose_qparams_affine`).
  4. Optional Dynamic Range Quantization promotion (`force_weight_only_as_drq`):
     When enabled, automatically injects dynamic `quant.fake_quant` on float
     activations entering linear/matmul operations whose weights are
     dequantized, enabling dynamic quantization kernels on runtimes that lack
     weight-only GEMM support.

  Finally, the pass erases unused dead nodes resulting from TorchAO parameter
  calculations and marks the graph metadata with `strict_qdq_mode = True`.

  Attributes:
    force_weight_only_as_drq: If True, promotes weight-only quantized linear and
      matmul operations to dynamic range quantization (DRQ) by inserting dynamic
      fake quant composites on input activations.
  """

  def __init__(self, force_weight_only_as_drq: bool = False):
    super().__init__()
    self.force_weight_only_as_drq = force_weight_only_as_drq

  def call(
      self, exported_program: torch.export.ExportedProgram
  ) -> fx_infra.ExportedProgramPassResult:
    """Executes the LowerTorchAOPass transformation on the ExportedProgram.

    Args:
      exported_program: ExportedProgram produced by torch.export.export.

    Returns:
      ExportedProgramPassResult containing the transformed program and a boolean
      indicating whether any modifications were made.
    """
    gm = exported_program.graph_module
    graph = gm.graph
    state_dict = exported_program.state_dict
    inputs_to_params = exported_program.graph_signature.inputs_to_parameters
    inputs_to_buffers = exported_program.graph_signature.inputs_to_buffers

    modified = False

    def get_tensor_data(
        node: Union[torch.fx.Node, torch.Tensor, None],
    ) -> Optional[torch.Tensor]:
      """Resolves the underlying constant Tensor value for an FX node.

      Extracts tensor data by:
        1. Checking if the argument is already a raw `torch.Tensor`.
        2. Resolving `placeholder` nodes mapped to model parameters or
           persistent buffers in the `ExportedProgram` state_dict.
        3. Resolving `get_attr` nodes representing module attributes.
        4. Tracing through pure tensor view/transformation operations
           (e.g. `view`, `reshape`, `flatten`, `t`, `transpose`) applied to
           constants, evaluating them on the underlying tensor.

      Args:
        node: FX node, Tensor, or None to evaluate.

      Returns:
        The resolved `torch.Tensor` value, or None if the node represents a
        dynamic runtime value.
      """
      if node is None:
        return None
      if isinstance(node, torch.Tensor):
        return node
      if not isinstance(node, torch.fx.Node):
        return None
      if node.op == "placeholder":
        param_name = inputs_to_params.get(node.target) or inputs_to_buffers.get(
            node.target
        )
        if param_name and param_name in state_dict:
          return state_dict[param_name]
      elif node.op == "get_attr":
        try:
          attr_val = getattr(gm, node.target)
          if isinstance(attr_val, torch.Tensor):
            return attr_val
        except AttributeError:
          pass
      elif node.op == "call_function":
        tgt_str = str(node.target)
        if (
            "view" in tgt_str
            or "reshape" in tgt_str
            or "flatten" in tgt_str
            or "t" in tgt_str
        ) and node.args:
          inner = get_tensor_data(node.args[0])
          if inner is not None:
            if "flatten" in tgt_str:
              return inner.flatten()
            elif (
                "t" == getattr(node.target, "__name__", "")
                or "aten.t" in tgt_str
            ):
              return inner.t()
            elif len(node.args) > 1 and isinstance(node.args[1], (list, tuple)):
              try:
                return inner.reshape(node.args[1])
              except Exception:  # pylint: disable=broad-except
                return inner
            return inner
      return None

    # Check whether dynamic activation quantization (DRQ) is used anywhere in
    # the model. In TorchAO, dynamic activation quantization computes qparams
    # on the fly via `torch.ops.torchao.choose_qparams_affine`.
    has_dynamic_act = any(
        isinstance(n, torch.fx.Node)
        and n.op == "call_function"
        and "choose_qparams_affine" in str(n.target)
        for n in graph.nodes
    )
    nodes_to_remove = []

    for node in list(graph.nodes):
      if node.op != "call_function":
        continue

      target = node.target
      target_name = (
          getattr(target, "_name", "")
          if hasattr(target, "_name")
          else str(target)
      )

      # ========================================================================
      # Pattern 1: Match torch.ops.aten._int_mm.default (Static Full-Integer MM)
      # ========================================================================
      # In static int8 quantization (e.g. PTQ or QAT with integer GEMM), TorchAO
      # compiles the linear operation as an integer matrix multiplication:
      #   act -> quantize_affine -> act_q (int8)
      #   weight_placeholder (int8)
      #   out_i32 = aten._int_mm(act_q, weight_placeholder.t())
      #   out_f32 = aten._to_copy(out_i32, dtype=float32)
      #   out_scaled = mul(out_f32, act_scale)
      #   out_scaled = mul(out_scaled, weight_scale)
      #   out_final = add(out_scaled, bias)
      #
      # We transform this pattern into:
      #   fq_act = quant.fake_quant(act, scale=act_scale, zp=act_zp)
      #   dq_w   = quant.dequantize(weight, scale=weight_scale, zp=0)
      #   out    = aten.linear(fq_act, dq_w, bias)
      #
      # LiteRT's MLIR passes match aten.linear with fake_quant activation and
      # dequantize weight to produce a fully fused integer FULLY_CONNECTED op.
      if (
          target == torch.ops.aten._int_mm.default
          or "aten::_int_mm" in target_name
          or "_int_mm" in str(target)
      ):
        act_op = node.args[0]
        weight_t_op = node.args[1]

        # Trace upstream to find the activation quantize_affine node.
        quant_node = act_op
        while (
            isinstance(quant_node, torch.fx.Node)
            and quant_node.op == "call_function"
            and "quantize_affine" not in str(quant_node.target)
        ):
          if quant_node.args:
            quant_node = quant_node.args[0]
          else:
            break

        # Trace upstream from weight transpose to find the weight parameter
        # node.
        weight_placeholder = weight_t_op
        while (
            isinstance(weight_placeholder, torch.fx.Node)
            and weight_placeholder.op == "call_function"
        ):
          if weight_placeholder.args:
            weight_placeholder = weight_placeholder.args[0]
          else:
            break

        if isinstance(quant_node, torch.fx.Node) and "quantize_affine" in str(
            quant_node.target
        ):
          act_x = quant_node.args[0]
          act_scale_node = quant_node.args[2]
          act_zp_node = quant_node.args[3]
          act_dtype = (
              quant_node.args[4] if len(quant_node.args) > 4 else torch.int8
          )
          act_qmin = quant_node.args[5] if len(quant_node.args) > 5 else -128
          act_qmax = quant_node.args[6] if len(quant_node.args) > 6 else 127
          act_dtype_str = _infer_dtype_str(
              act_qmin, act_qmax, fallback_dtype=act_dtype
          )

          act_scale_val = (
              get_tensor_data(act_scale_node)
              if isinstance(act_scale_node, torch.fx.Node)
              else act_scale_node
          )
          act_zp_val = (
              get_tensor_data(act_zp_node)
              if isinstance(act_zp_node, torch.fx.Node)
              else act_zp_node
          )

          # Trace downstream through cast and multiplication ops to extract the
          # weight scale tensor and identify any optional bias addition.
          curr = list(node.users)[0] if node.users else None
          if (
              curr
              and curr.op == "call_function"
              and curr.target == torch.ops.aten._to_copy.default
          ):
            curr = list(curr.users)[0] if curr.users else None

          if curr and curr.op == "call_function" and "mul" in str(curr.target):
            curr = list(curr.users)[0] if curr.users else None

          weight_scale_val = None
          if curr and curr.op == "call_function" and "mul" in str(curr.target):
            for arg in curr.args:
              if arg != curr:
                v = (
                    get_tensor_data(arg)
                    if isinstance(arg, torch.fx.Node)
                    else arg
                )
                if v is not None:
                  weight_scale_val = v
            curr = list(curr.users)[0] if curr.users else None

          while (
              curr and curr.op == "call_function" and "view" in str(curr.target)
          ):
            users = list(curr.users)
            if users and "add" in str(users[0].target):
              curr = users[0]
            else:
              break

          bias_node = None
          if curr and curr.op == "call_function" and "add" in str(curr.target):
            for arg in curr.args:
              if isinstance(arg, torch.fx.Node) and (
                  "bias" in arg.name or "bias" in str(arg.target)
              ):
                bias_node = arg
            final_node = curr
            while final_node.users:
              next_u = list(final_node.users)[0]
              if next_u.op == "call_function" and "view" in str(next_u.target):
                final_node = next_u
              else:
                break
          else:
            final_node = curr or node

          weight_tensor = get_tensor_data(weight_placeholder)
          if weight_tensor is not None and weight_scale_val is not None:
            if (
                isinstance(weight_scale_val, torch.Tensor)
                and weight_tensor.ndim == 2
                and weight_scale_val.numel() > weight_tensor.shape[0]
            ):
              raise NotImplementedError(
                  "Blockwise quantization is not supported for now."
              )
            # Ensure the quantized weight tensor is marked as a Parameter in
            # state_dict so that LiteRT export serializes it as a persistent
            # model weight rather than an inlined constant.
            param_name = inputs_to_params.get(
                weight_placeholder.target
            ) or inputs_to_buffers.get(weight_placeholder.target)
            if param_name and param_name in state_dict:
              if not isinstance(state_dict[param_name], torch.nn.Parameter):
                state_dict[param_name] = torch.nn.Parameter(
                    state_dict[param_name], requires_grad=False
                )

            # Insert fake_quant on activation, dequantize on weight, and
            # aten.linear before final_node.
            with graph.inserting_before(final_node):
              act_attrs = _get_quant_composite_attributes(
                  scale=act_scale_val,
                  zero_point=act_zp_val,
                  dtype=act_dtype_str,
                  quant_dimension=None,
                  narrow_range=True,
              )
              fq_act_op = _create_fake_quant_function(act_attrs)
              fq_act = graph.call_function(fq_act_op, (act_x,))

              w_attrs = _get_quant_composite_attributes(
                  scale=weight_scale_val,
                  zero_point=0,
                  dtype="i8",
                  quant_dimension=0,
                  narrow_range=True,
              )
              dq_w_op = _create_dequantize_function(w_attrs)
              dq_w = graph.call_function(dq_w_op, (weight_placeholder,))

              if bias_node is not None:
                linear_out = graph.call_function(
                    torch.ops.aten.linear.default, (fq_act, dq_w, bias_node)
                )
              else:
                linear_out = graph.call_function(
                    torch.ops.aten.linear.default, (fq_act, dq_w)
                )

              final_node.replace_all_uses_with(linear_out)
              modified = True
          continue

      # ========================================================================
      # Pattern 2 & 3: Match torch.ops.torchao.dequantize_affine.default
      # ========================================================================
      # TorchAO uses `dequantize_affine` to convert quantized integer tensors
      # (both weights and activations) back to floating point format prior to
      # computation:
      #   dequantize_affine(input, block_size, scale, zero_point,
      #                     output_dtype, quant_min, quant_max)
      #
      # We distinguish two major cases:
      #   - Case 1: Dequantizing a constant weight parameter (Weight-Only,
      #     Per-Channel, or Blockwise Quantization).
      #   - Case 2: Dequantizing an activation tensor (Dynamic Range
      #     Quantization or Static Activation Quantization).
      if (
          "torchao::dequantize_affine" in target_name
          or (
              hasattr(torch.ops, "torchao")
              and target
              == getattr(torch.ops.torchao, "dequantize_affine", None)
          )
          or (
              hasattr(torch.ops, "torchao")
              and hasattr(torch.ops.torchao.dequantize_affine, "default")
              and target == torch.ops.torchao.dequantize_affine.default
          )
      ):
        args = node.args
        # Op signature: (input, block_size, scale, zero_point,
        #                output_dtype, quant_min, quant_max)
        input_node = args[0]
        block_size = list(args[1]) if len(args) > 1 else [1, 1]
        scale_node = args[2] if len(args) > 2 else None
        zp_node = args[3] if len(args) > 3 else None
        output_dtype = args[4] if len(args) > 4 else torch.int8
        quant_min = args[5] if len(args) > 5 else -128
        quant_max = args[6] if len(args) > 6 else 127

        dtype_str = _infer_dtype_str(
            quant_min, quant_max, fallback_dtype=output_dtype
        )

        # ----------------------------------------------------------------------
        # Case 1: Dequantizing a Weight Parameter / Constant
        # ----------------------------------------------------------------------
        # If the input is a constant or model parameter stored in state_dict,
        # we lower it to `quant.dequantize`.
        weight_tensor = (
            get_tensor_data(input_node)
            if isinstance(input_node, torch.fx.Node)
            else None
        )
        scale_tensor = (
            get_tensor_data(scale_node)
            if isinstance(scale_node, torch.fx.Node)
            else (scale_node if isinstance(scale_node, torch.Tensor) else None)
        )
        zp_tensor = (
            get_tensor_data(zp_node)
            if isinstance(zp_node, torch.fx.Node)
            else (zp_node if isinstance(zp_node, torch.Tensor) else None)
        )

        if weight_tensor is not None and scale_tensor is not None:
          if zp_tensor is None:
            zp_tensor = torch.zeros_like(scale_tensor, dtype=torch.int32)

          is_2d = weight_tensor.ndim == 2
          oc, ic = weight_tensor.shape if is_2d else (1, weight_tensor.numel())
          is_per_channel = is_2d and (
              len(block_size) == 2
              and block_size[0] == 1
              and block_size[1] == ic
          )
          is_blockwise = is_2d and (
              len(block_size) == 2 and block_size[0] == 1 and block_size[1] < ic
          )

          # Preserve the quantized weight parameter in state_dict so it is
          # exported as an int8/int4 constant tensor instead of being inlined
          # or converted back to float.
          param_name = inputs_to_params.get(
              input_node.target
          ) or inputs_to_buffers.get(input_node.target)
          if param_name and param_name in state_dict:
            is_param = param_name in inputs_to_params.values()
            if is_param and not isinstance(
                state_dict[param_name], torch.nn.Parameter
            ):
              state_dict[param_name] = torch.nn.Parameter(
                  state_dict[param_name], requires_grad=False
              )

          with graph.inserting_before(node):
            if is_blockwise:
              # In blockwise (grouped) weight quantization, weights are divided
              # into sub-channel groups (e.g. group_size=32, 64, or 128).
              # LiteRT's `quant.dequantize` composite operates along a single
              # axis. To map 2D blockwise quantization to LiteRT:
              #   1. Reshape [oc, ic] -> [total_blocks, group_size], where
              #      total_blocks = oc * (ic // group_size).
              #   2. Apply `quant.dequantize` along axis 0 (each block has 1
              #      scale).
              #   3. Reshape back from [total_blocks, group_size] -> [oc, ic].
              if has_dynamic_act:
                raise NotImplementedError(
                    "Blockwise quantization is not supported."
                )

              group_size = block_size[1]
              num_blocks_per_row = ic // group_size
              total_blocks = oc * num_blocks_per_row

              # Reshape to [total_blocks, group_size]
              reshaped_weight = graph.call_function(
                  torch.ops.aten.reshape.default,
                  (input_node, [total_blocks, group_size]),
              )

              # Dequantize composite along dimension 0 with per-block scales.
              attrs = _get_quant_composite_attributes(
                  scale=scale_tensor.flatten().tolist(),
                  zero_point=zp_tensor.flatten().tolist(),
                  dtype=dtype_str,
                  quant_dimension=0,
                  narrow_range=True,
              )
              dq_op = _create_dequantize_function(attrs)
              dq_node = graph.call_function(dq_op, (reshaped_weight,))

              # Reshape back to [oc, ic]
              out_weight = graph.call_function(
                  torch.ops.aten.reshape.default,
                  (dq_node, [oc, ic]),
              )
            else:
              # Per-channel or per-axis quantization along the inferred axis.
              quant_dim = _infer_quant_dimension(
                  weight_tensor.shape, block_size, scale_tensor.numel()
              )
              attrs = _get_quant_composite_attributes(
                  scale=scale_tensor.flatten().tolist(),
                  zero_point=zp_tensor.flatten().tolist(),
                  dtype=dtype_str,
                  quant_dimension=quant_dim,
                  narrow_range=True,
              )
              dq_op = _create_dequantize_function(attrs)
              out_weight = graph.call_function(dq_op, (input_node,))

            node.replace_all_uses_with(out_weight)
            nodes_to_remove.append(node)
            modified = True
          continue

        # ----------------------------------------------------------------------
        # Case 2: Dequantizing an Activation Tensor
        # ----------------------------------------------------------------------
        # In TorchAO, quantized activations appear as a `quantize_affine` op
        # immediately consumed by a `dequantize_affine` op:
        #   act -> quantize_affine -> dequantize_affine -> linear/matmul
        #
        # We collapse this pair into a single `quant.fake_quant` composite:
        #   - For Dynamic Range Quantization (DRQ), scale/zp originate from
        #     `choose_qparams_affine`. In DRQ, scales are computed dynamically
        #     per token at runtime, so static scales in composite attributes
        #     are left empty (`scale=[]`, `zero_point=[]`).
        #   - For Static Activation Quantization, static scale/zp values are
        #     extracted from constants and embedded into composite attributes.
        if (
            isinstance(input_node, torch.fx.Node)
            and input_node.op == "call_function"
        ):
          input_target_name = (
              getattr(input_node.target, "_name", "")
              if hasattr(input_node.target, "_name")
              else str(input_node.target)
          )
          if (
              "torchao::quantize_affine" in input_target_name
              or (
                  hasattr(torch.ops, "torchao")
                  and input_node.target
                  == getattr(torch.ops.torchao, "quantize_affine", None)
              )
              or (
                  hasattr(torch.ops, "torchao")
                  and hasattr(torch.ops.torchao.quantize_affine, "default")
                  and input_node.target
                  == torch.ops.torchao.quantize_affine.default
              )
          ):
            act_input = input_node.args[0]
            is_dynamic = False
            if isinstance(scale_node, torch.fx.Node):
              if (
                  scale_node.op == "call_function"
                  and getattr(scale_node.target, "__name__", "") == "getitem"
              ):
                parent = scale_node.args[0]
                if isinstance(
                    parent, torch.fx.Node
                ) and "choose_qparams_affine" in str(parent.target):
                  is_dynamic = True

            with graph.inserting_before(node):
              if is_dynamic:
                # Dynamic range quantization: dynamic per-token scales.
                attrs = _get_quant_composite_attributes(
                    scale=None,
                    zero_point=None,
                    dtype=dtype_str,
                    quant_dimension=None,
                    narrow_range=True,
                )
              else:
                # Static activation quantization: constant calibration scales.
                static_s = (
                    get_tensor_data(scale_node)
                    if isinstance(scale_node, torch.fx.Node)
                    else scale_node
                )
                static_zp = (
                    get_tensor_data(zp_node)
                    if isinstance(zp_node, torch.fx.Node)
                    else zp_node
                )
                attrs = _get_quant_composite_attributes(
                    scale=static_s,
                    zero_point=static_zp,
                    dtype=dtype_str,
                    quant_dimension=None,
                    narrow_range=True,
                )

              fq_op = _create_fake_quant_function(attrs)
              fq_node = graph.call_function(fq_op, (act_input,))
              node.replace_all_uses_with(fq_node)
              nodes_to_remove.append(node)
              modified = True
            continue

    # ==========================================================================
    # Pattern 4: Optional Dynamic Range Quantization Promotion
    # ==========================================================================
    # When `force_weight_only_as_drq` is enabled, any linear, mm, bmm, matmul,
    # or addmm operation that operates on a dequantized weight but has an
    # unquantized floating-point activation is automatically promoted to dynamic
    # range quantization (DRQ) by injecting a dynamic `quant.fake_quant` op.
    # This enables hardware backends optimized for integer activations to
    # execute weight-only models using efficient dynamic int8 kernels.
    if self.force_weight_only_as_drq:

      def is_dequant_node(n: Any) -> bool:
        """Checks if a node originates from a weight dequant composite."""
        if not isinstance(n, torch.fx.Node):
          return False
        if n.op == "call_function":
          tgt_str = str(n.target)
          if "quant_composite" in tgt_str or "quant.dequantize" in tgt_str:
            return True
          if "dequantize" in tgt_str or "dequant_fn" in tgt_str:
            return True
          # Recursively trace through tensor reshaping and layout operations.
          if (
              "view" in tgt_str
              or "reshape" in tgt_str
              or "permute" in tgt_str
              or "t" in tgt_str
              or "transpose" in tgt_str
              or "to_copy" in tgt_str
              or "slice" in tgt_str
          ) and n.args:
            return is_dequant_node(n.args[0])
        return False

      def is_fake_quant_node(n: Any) -> bool:
        """Checks if a node originates from a fake_quant composite."""
        if not isinstance(n, torch.fx.Node):
          return False
        if n.op == "call_function":
          tgt_str = str(n.target)
          if "fake_quant" in tgt_str:
            return True
          if (
              "quant_composite" in tgt_str
              and len(n.args) > 1
              and n.args[1] == "quant.fake_quant"
          ):
            return True
          # Recursively trace through tensor reshaping and layout operations.
          if (
              "view" in tgt_str
              or "reshape" in tgt_str
              or "permute" in tgt_str
              or "t" in tgt_str
              or "transpose" in tgt_str
              or "to_copy" in tgt_str
          ) and n.args:
            return is_fake_quant_node(n.args[0])
        return False

      drq_fn = _create_fake_quant_function({
          "dtype": "i8",
          "scale": [],
          "zero_point": [],
          "narrow_range": False,
      })

      for node in list(graph.nodes):
        if node.op != "call_function":
          continue
        target = node.target
        target_name = (
            getattr(target, "_name", "")
            if hasattr(target, "_name")
            else str(target)
        )
        if (
            target in (torch.ops.aten.linear.default, torch.ops.aten.linear)
            or "linear" in target_name
        ):
          if len(node.args) >= 2:
            act_node = node.args[0]
            weight_node = node.args[1]
            if is_dequant_node(weight_node) and not is_fake_quant_node(
                act_node
            ):
              with graph.inserting_before(node):
                act_drq = graph.call_function(drq_fn, (act_node,))
              node.args = (act_drq, weight_node, *node.args[2:])
              modified = True
        elif (
            target in (
                torch.ops.aten.mm.default,
                torch.ops.aten.bmm.default,
                torch.ops.aten.matmul.default,
            )
            or "aten.mm" in target_name
            or "aten.bmm" in target_name
            or "aten.matmul" in target_name
        ):
          if len(node.args) >= 2:
            act_node = node.args[0]
            weight_node = node.args[1]
            if is_dequant_node(weight_node) and not is_fake_quant_node(
                act_node
            ):
              with graph.inserting_before(node):
                act_drq = graph.call_function(drq_fn, (act_node,))
              node.args = (act_drq, weight_node, *node.args[2:])
              modified = True
        elif (
            target in (torch.ops.aten.addmm.default,)
            or "aten.addmm" in target_name
        ):
          if len(node.args) >= 3:
            bias_node = node.args[0]
            act_node = node.args[1]
            weight_node = node.args[2]
            if is_dequant_node(weight_node) and not is_fake_quant_node(
                act_node
            ):
              with graph.inserting_before(node):
                act_drq = graph.call_function(drq_fn, (act_node,))
              node.args = (bias_node, act_drq, weight_node, *node.args[3:])
              modified = True

    # ==========================================================================
    # Pattern 5: Iterative Dead Code Elimination
    # ==========================================================================
    # Lowering TorchAO ops leaves behind orphaned parameter calculation nodes
    # (e.g. `choose_qparams_affine`, `quantize_affine`, `getitem`, slicing,
    # casting). We iteratively prune nodes with zero users from output backwards
    # until reaching a fixed point.
    changed = True
    while changed:
      changed = False
      for n in list(reversed(graph.nodes)):
        if n.op not in ("placeholder", "output") and len(n.users) == 0:
          n.args = ()
          n.kwargs = {}
          graph.erase_node(n)
          changed = True

    # When quantization transformations have been applied, mark the GraphModule
    # metadata with `strict_qdq_mode = True`. This signals downstream LiteRT
    # MLIR compilation passes to enforce strict QDQ lowering and fail if any
    # unquantizable pattern is encountered instead of silently emitting float.
    if modified:
      exported_program.graph_module.meta["strict_qdq_mode"] = True

    return fx_infra.ExportedProgramPassResult(exported_program, modified)


# ==============================================================================
# Custom PyTorch Op: quant_composite
# ==============================================================================
# To represent LiteRT quantization composites in the PyTorch FX graph without
# prematurely lowering to MLIR, we register a custom PyTorch operation in
# LiteRT-Torch's internal library (`litert_torch::quant_composite`).
#
# Arguments:
#   - x: Input tensor to quantize or dequantize.
#   - name: Composite name string ('quant.fake_quant' or 'quant.dequantize').
#   - attr_json: Key to look up composite attributes in `_QUANT_ATTR_CACHE`
#     (or fallback JSON string).
_torch_library.LITERT_TORCH_LIB.define(
    "quant_composite(Tensor x, str name, str attr_json) -> Tensor"
)
_quant_composite_op = torch.ops.litert_torch.quant_composite.default


@torch.library.impl(
    _torch_library.LITERT_TORCH_LIB,
    "quant_composite",
    "CompositeExplicitAutograd",
)
def _quant_composite(
    x: torch.Tensor, name: str, attr_json: str
) -> torch.Tensor:
  """Functional eager-mode implementation of quant_composite.

  Simulates numerical behavior during eager execution or testing:
    - 'quant.dequantize': Converts integer tensor to float32 and scales it
      along the specified quantization dimension.
    - 'quant.fake_quant': Simulates affine quantization (divide by scale, round,
      clamp to integer range, multiply by scale) while preserving float32.

  Args:
    x: Input tensor.
    name: Composite op name ('quant.dequantize' or 'quant.fake_quant').
    attr_json: Attribute cache key or JSON string.

  Returns:
    Transformed tensor matching the simulated quantization operation.
  """
  attr = _QUANT_ATTR_CACHE.get(attr_json, {})
  if name == "quant.dequantize":
    scale = attr.get("scale", [])
    if scale:
      scale_t = torch.tensor(scale, dtype=torch.float32, device=x.device)
      q_dim = attr.get("quantization_dimension", 0)
      if q_dim is not None and x.ndim > 1 and scale_t.numel() > 1:
        shape = [1] * x.ndim
        shape[q_dim] = scale_t.numel()
        scale_t = scale_t.view(shape)
      return x.to(torch.float32) * scale_t
    return x.to(torch.float32)
  elif name == "quant.fake_quant":
    scale = attr.get("scale", [])
    if scale:
      scale_t = torch.tensor(scale, dtype=torch.float32, device=x.device)
      dtype_str = attr.get("dtype", "i8")
      if dtype_str == "i4":
        min_v, max_v = -8.0, 7.0
      else:
        min_v, max_v = -128.0, 127.0
      return torch.clamp(torch.round(x / scale_t), min_v, max_v) * scale_t
  return x


@torch.library.impl(_torch_library.LITERT_TORCH_LIB, "quant_composite", "Meta")
def _quant_composite_meta(
    x: torch.Tensor, name: str, attr_json: str
) -> torch.Tensor:
  """Meta-kernel for TorchDynamo and FakeTensor shape/dtype propagation.

  Allows torch.export and AOTAutograd to trace `quant_composite` operations
  symbolically without allocating concrete device buffers.

  Args:
    x: FakeTensor input.
    name: Composite op name.
    attr_json: Attribute cache key or JSON string.

  Returns:
    An unallocated FakeTensor with the appropriate output shape and dtype.
  """
  del attr_json
  if name == "quant.dequantize":
    return torch.empty(x.shape, dtype=torch.float32, device=x.device)
  return torch.empty_like(x)


def _build_composite_attributes(
    attr: dict[str, Any], context: ir.Context
) -> dict[str, ir.Attribute]:
  """Constructs an MLIR NamedAttribute dictionary from Python attribute values.

  Encodes quantization metadata into MLIR Attribute representations:
    - Scales (float list): DenseElementsAttr of F32Type.
    - Zero points (int / int list): DenseElementsAttr of 64-bit signless
      IntegerType.
    - Strings: StringAttr.
    - Booleans: BoolAttr.
    - Integers: 32-bit IntegerAttr.

  Args:
    attr: Python dictionary containing composite attributes.
    context: Active MLIR Context.

  Returns:
    Dictionary mapping attribute names to MLIR ir.Attribute instances.
  """
  composite_attrs = {}
  for k, v in attr.items():
    if k == "scale":
      arr = np.array(v, dtype=np.float32)
      composite_attrs[k] = ir.DenseElementsAttr.get(
          arr,
          type=ir.F32Type.get(context),
          shape=[len(v)],
      )
    elif k == "zero_point":
      if isinstance(v, (int, np.integer)):
        arr = np.array([v], dtype=np.int64)
        composite_attrs[k] = ir.DenseElementsAttr.get(
            arr,
            type=ir.IntegerType.get_signless(64, context),
            shape=[1],
        )
      elif isinstance(v, (list, tuple)):
        if not v:
          continue
        if len(v) == 1 or all(x == v[0] for x in v):
          arr = np.array([v[0]], dtype=np.int64)
          shape = [attr["scale_len"]] if "scale_len" in attr else [len(v)]
          composite_attrs[k] = ir.DenseElementsAttr.get(
              arr,
              type=ir.IntegerType.get_signless(64, context),
              shape=shape,
          )
        else:
          arr = np.array(v, dtype=np.int64)
          composite_attrs[k] = ir.DenseElementsAttr.get(
              arr,
              type=ir.IntegerType.get_signless(64, context),
              shape=[len(v)],
          )
    elif isinstance(v, str):
      composite_attrs[k] = ir.StringAttr.get(v, context)
    elif isinstance(v, bool):
      composite_attrs[k] = ir.BoolAttr.get(v, context)
    elif isinstance(v, int):
      composite_attrs[k] = ir.IntegerAttr.get(
          ir.IntegerType.get_signless(32, context), v
      )
    elif isinstance(v, (list, tuple)):
      if v and isinstance(v[0], float):
        arr = np.array(v, dtype=np.float32)
        composite_attrs[k] = ir.DenseElementsAttr.get(
            arr, type=ir.F32Type.get(context), shape=[len(v)]
        )
      elif v and isinstance(v[0], int):
        arr = np.array(v, dtype=np.int64)
        composite_attrs[k] = ir.DenseElementsAttr.get(
            arr, type=ir.IntegerType.get_signless(64, context), shape=[len(v)]
        )
  return composite_attrs


# In-memory attribute cache indexed by unique string identifiers.
_QUANT_ATTR_CACHE: dict[str, dict[str, Any]] = {}


def _register_quant_attrs(attrs: dict[str, Any]) -> str:
  """Registers composite attributes in in-memory cache and returns UUID key.

  Passing complex attribute dictionaries directly across PyTorch FX custom ops
  can cause serialization issues or loss of numerical precision. Instead, we
  store the dictionary in `_QUANT_ATTR_CACHE` with a unique string ID and
  retrieve it during MLIR lowering.

  Args:
    attrs: Composite attributes dictionary.

  Returns:
    A unique string identifier key.
  """
  attr_id = f"quant_attr_{uuid.uuid4().hex}"
  _QUANT_ATTR_CACHE[attr_id] = attrs
  return attr_id


@lowerings.lower(torch.ops.litert_torch.quant_composite)
def _quant_composite_lowering(
    lctx, x: ir.Value, name: str, attr_json: str
):
  """Lowers torch.ops.litert_torch.quant_composite to stablehlo.composite.

  Constructs a `stablehlo.composite` op with the given composite name (such as
  'quant.dequantize' or 'quant.fake_quant') and attaches composite attributes.
  In accordance with the StableHLO composite specification:
    1. A private decomposition function is synthesized at the module level to
       serve as a fallback representation.
    2. The output tensor type is determined: 'quant.dequantize' casts integer
       inputs to float32, while 'quant.fake_quant' maintains the input type.
    3. The composite op is emitted and returned.

  Downstream LiteRT compilation passes match these composite operations and
  lower them into optimized TFLite quantized kernels.

  Args:
    lctx: LiteRT LoweringContext.
    x: Input MLIR Value.
    name: Composite name string.
    attr_json: Attribute cache key or JSON string.

  Returns:
    The resulting MLIR Value from the stablehlo.composite op.
  """
  if attr_json in _QUANT_ATTR_CACHE:
    attr = _QUANT_ATTR_CACHE[attr_json]
  elif attr_json:
    attr = json.loads(attr_json)
  else:
    attr = {}
  composite_attrs = _build_composite_attributes(attr, lctx.ir_context)

  if name == "quant.dequantize":
    x_shaped = ir.ShapedType(x.type)
    out_type = ir.RankedTensorType.get(
        x_shaped.shape, ir.F32Type.get(lctx.ir_context)
    )
  else:
    out_type = x.type

  # Generate a unique private decomposition function for the StableHLO
  # composite.
  decomp_name = f"litert_torch_{name.replace('.', '_')}_{uuid.uuid4().hex[:8]}"
  with ir.InsertionPoint.at_block_begin(lctx.ir_module.body):
    func_type = ir.FunctionType.get(
        inputs=[x.type],
        results=[out_type],
        context=lctx.ir_context,
    )
    decomp_func = func.FuncOp(
        name=decomp_name,
        type=func_type,
        visibility="private",
    )
    entry_block = decomp_func.add_entry_block()
    with ir.InsertionPoint(entry_block):
      ret_val = entry_block.arguments[0]
      if ret_val.type != out_type:
        ret_val = stablehlo.ConvertOp(out_type, ret_val).result
      func.ReturnOp([ret_val])

  return stablehlo.CompositeOp(
      result=[out_type],
      inputs=[x],
      name=ir.StringAttr.get(name, lctx.ir_context),
      composite_attributes=ir.DictAttr.get(composite_attrs, lctx.ir_context),
      decomposition=ir.FlatSymbolRefAttr.get(decomp_name, lctx.ir_context),
  ).results[0]


def _create_fake_quant_function(attrs: dict[str, Any]):
  """Returns a callable for constructing a quant.fake_quant composite in FX.

  Registers the attributes dictionary into `_QUANT_ATTR_CACHE` and binds the
  unique key to a helper closure that invokes `_quant_composite_op`.

  Args:
    attrs: Composite attributes dictionary.

  Returns:
    A callable `fake_quant_fn(x)` suitable for use in `graph.call_function`.
  """
  attr_key = _register_quant_attrs(attrs)

  def fake_quant_fn(x):
    return _quant_composite_op(x, "quant.fake_quant", attr_key)

  fake_quant_fn.__name__ = "quant_fake_quant"
  return fake_quant_fn


def _create_dequantize_function(attrs: dict[str, Any]):
  """Returns a callable for constructing a quant.dequantize composite in FX.

  Registers the attributes dictionary into `_QUANT_ATTR_CACHE` and binds the
  unique key to a helper closure that invokes `_quant_composite_op`.

  Args:
    attrs: Composite attributes dictionary.

  Returns:
    A callable `dequantize_fn(x)` suitable for use in `graph.call_function`.
  """
  attr_key = _register_quant_attrs(attrs)

  def dequantize_fn(x):
    return _quant_composite_op(x, "quant.dequantize", attr_key)

  dequantize_fn.__name__ = "quant_dequantize"
  return dequantize_fn
