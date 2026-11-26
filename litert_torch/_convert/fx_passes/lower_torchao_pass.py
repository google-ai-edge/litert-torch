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
"""FX pass and MLIR lowerings for TorchAO Q/DQ operations.

TorchAO represents quantized models using affine quantization ops:
  - `torch.ops.torchao.quantize_affine`: Quantizes a floating-point tensor.
  - `torch.ops.torchao.dequantize_affine`: Dequantizes an integer tensor to
    float using scale and zero-point.
  - `torch.ops.torchao.choose_qparams_affine`: Dynamically computes per-token
    scale and zero-point from activation tensors at runtime.

During LiteRT Converter V2 compilation, Q/DQ operations are lowered to
`stablehlo.custom_call` annotations (`quant.quantize`, `quant.dequantize`, and
`quant.fake_quant`), which `LowerQuantAnnotationsPass` and `FuseQDQPass` in the
C++ MLIR converter lower and fuse into quantized TFLite ops.

Most quantization schemas (Weight-Only, Static Range Quantization, etc.) arise
naturally from the placement of `quant.quantize` and `quant.dequantize` in the
graph without pattern-matching linear/matmul ops in the PyTorch layer. Aside
from translating TorchAO's op signature (`block_size` tuple and constant
`scale`/`zero_point` tensor inputs) into MLIR `backend_config` attributes
(`quantization_dimension` and 1D attribute lists), this FX pass bridges two
specific gaps in the C++ MLIR converter (documented inline as Converter Gaps 1
and 2):
  1. Dynamic Range Quantization (DRQ): Collapsing TorchAO's 3-op dynamic
     `choose_qparams_affine -> quantize_affine -> dequantize_affine` sequence
     into a single `quant.fake_quant` custom call with an empty `scale`
     attribute, as expected by `IsDrqFakeQuant` in `fuse_qdq_pass.cc`.
  2. Pre-quantized blockwise `dequantize_affine`: Wrapping sub-channel blockwise
     weight dequantization in a 2D reshape `[total_blocks, group_size]` until
     `RewriteDequantizeCustomCallOp` in `lower_quant_annotations_pass.cc`
     supports blockwise annotations on pre-quantized integer weights.
"""

import json
from typing import Any, Optional, Union
import uuid
from litert_torch import fx_infra
from litert_torch.backend import _torch_library
from litert_torch.backend import lowerings
from litert_converter.mlir import ir
from litert_converter.mlir.dialects import stablehlo
import numpy as np
import torch

try:
  import torchao  # pylint: disable=unused-import,g-import-not-at-top

  # Retain high-level TorchAO Q/DQ ops (`quantize_affine`, `dequantize_affine`,
  # `choose_qparams_affine`) during pre-convert export instead of decomposing
  # them into primitive aten math ops (div, round, clamp, sub, mul).
  _TORCHAO_OPS = [
      getattr(torch.ops.torchao, name).default
      for name in dir(torch.ops.torchao)
      if hasattr(getattr(torch.ops.torchao, name, None), "default")
  ]
  for _op in _TORCHAO_OPS:
    fx_infra.decomp.remove_pre_convert_decomp(_op)
except ImportError:
  _TORCHAO_OPS = []


# Maps PyTorch dtypes or string names to LiteRT quantization dtype strings.
_DTYPE_MAP = {
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.uint8: "ui8",
    "i2": "i2",
    "i4": "i4",
    "ui4": "ui4",
    "i8": "i8",
    "ui8": "ui8",
    "i16": "i16",
    "i32": "i32",
}
if hasattr(torch, "int4"):
  _DTYPE_MAP[torch.int4] = "i4"
if hasattr(torch, "int2"):
  _DTYPE_MAP[torch.int2] = "i2"


def _infer_dtype_str(
    quant_min: int, quant_max: int, fallback_dtype: torch.dtype = torch.int8
) -> str:
  """Infers the MLIR quantization dtype string from quant min/max bounds."""
  if quant_min >= -2 and quant_max <= 1:
    return "i2"
  if quant_min >= -8 and quant_max <= 7:
    return "i4"
  if quant_min >= 0 and quant_max <= 15:
    return "ui4"
  if quant_min >= -128 and quant_max <= 127:
    return "i8"
  if quant_min >= 0 and quant_max <= 255:
    return "ui8"
  if quant_min >= -32768 and quant_max <= 32767:
    return "i16"
  return _DTYPE_MAP.get(fallback_dtype, "i8")


def _infer_quant_dimension(
    tensor_shape: Union[torch.Size, tuple[int, ...], list[int]],
    block_size: Union[tuple[int, ...], list[int]],
    scale_numel: int,
) -> Optional[int]:
  """Infers the quantization axis dimension from tensor shape and block_size.

  Translates TorchAO's per-dimension `block_size` tuple (e.g., `(1, K)` for
  per-channel quantization along axis 0 on an `(N, K)` tensor) into the
  `quantization_dimension` axis index expected by MLIR
  `UniformQuantizedPerAxisType`.

  Args:
    tensor_shape: Shape of the quantized tensor.
    block_size: Per-dimension block size tuple from TorchAO.
    scale_numel: Number of elements in the scale tensor.

  Returns:
    The inferred quantization dimension index, or None for per-tensor.
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
    scale: Optional[Union[float, list[float], np.ndarray, torch.Tensor]] = None,
    zero_point: Optional[
        Union[int, float, list[int], list[float], np.ndarray, torch.Tensor]
    ] = None,
    dtype: Union[torch.dtype, str] = "i8",
    quant_dimension: Optional[int] = None,
    narrow_range: bool = True,
    block_shape: Optional[Union[list[int], tuple[int, ...]]] = None,
    symmetric: Optional[bool] = None,
    act_scale_dtype: Optional[str] = None,
    range_dilation: Optional[float] = None,
) -> dict[str, Any]:
  """Constructs the attributes dictionary for `quant.*` custom calls."""
  attrs: dict[str, Any] = {
      "dtype": _DTYPE_MAP.get(dtype, str(dtype)),
      "narrow_range": bool(narrow_range),
  }

  if block_shape is not None:
    attrs["block_shape"] = [int(b) for b in block_shape]
  if symmetric is not None:
    attrs["symmetric"] = bool(symmetric)
  if act_scale_dtype is not None:
    attrs["act_scale_dtype"] = str(act_scale_dtype)
  if range_dilation is not None:
    attrs["range_dilation"] = float(range_dilation)

  if scale is not None:
    if isinstance(scale, torch.Tensor):
      scale_list = [float(s) for s in scale.detach().cpu().flatten().tolist()]
      if block_shape is not None and scale.ndim >= 1:
        attrs["scale_shape"] = list(scale.shape)
    elif isinstance(scale, (np.ndarray, list, tuple)):
      scale_arr = np.asarray(scale, dtype=np.float32)
      scale_list = [float(s) for s in scale_arr.flatten().tolist()]
      if block_shape is not None and scale_arr.ndim >= 1:
        attrs["scale_shape"] = list(scale_arr.shape)
    elif isinstance(scale, (int, float, np.number)):
      scale_list = [float(scale)]
    else:
      scale_list = []
    attrs["scale"] = scale_list
  else:
    attrs["scale"] = []

  if zero_point is not None:
    has_float_in_seq = isinstance(zero_point, (list, tuple)) and np.issubdtype(
        np.asarray(zero_point).dtype, np.floating
    )
    is_float_zp = block_shape is not None and (
        isinstance(zero_point, (float, np.floating))
        or (
            isinstance(zero_point, torch.Tensor)
            and zero_point.is_floating_point()
        )
        or (
            isinstance(zero_point, np.ndarray)
            and np.issubdtype(zero_point.dtype, np.floating)
        )
        or has_float_in_seq
    )
    if is_float_zp:
      attrs["zp_is_float"] = True
      if isinstance(zero_point, torch.Tensor):
        zp_list = [
            float(zp) for zp in zero_point.detach().cpu().flatten().tolist()
        ]
        if zero_point.ndim >= 1:
          attrs["zp_shape"] = list(zero_point.shape)
      elif isinstance(zero_point, (np.ndarray, list, tuple)):
        zp_arr = np.asarray(zero_point, dtype=np.float32)
        zp_list = [float(zp) for zp in zp_arr.flatten().tolist()]
        if zp_arr.ndim >= 1:
          attrs["zp_shape"] = list(zp_arr.shape)
      else:
        zp_list = [float(zero_point)]
    else:
      if isinstance(zero_point, torch.Tensor):
        zp_list = [
            int(zp) for zp in zero_point.detach().cpu().flatten().tolist()
        ]
        if block_shape is not None and zero_point.ndim >= 1:
          attrs["zp_shape"] = list(zero_point.shape)
      elif isinstance(zero_point, (np.ndarray, list, tuple)):
        zp_arr = np.asarray(zero_point, dtype=np.int64)
        zp_list = [int(zp) for zp in zp_arr.flatten().tolist()]
        if block_shape is not None and zp_arr.ndim >= 1:
          attrs["zp_shape"] = list(zp_arr.shape)
      else:
        zp_list = [int(zero_point)]
    attrs["zero_point"] = zp_list
  else:
    attrs["zero_point"] = []

  if quant_dimension is not None:
    attrs["quantization_dimension"] = int(quant_dimension)

  return attrs


def _is_torchao_op(node: Any, op_name: str) -> bool:
  """Returns True if `node` is a call_function to `torch.ops.torchao.<op_name>`."""
  if not isinstance(node, torch.fx.Node) or node.op != "call_function":
    return False
  target = node.target
  target_name = getattr(target, "_name", "") or str(target)
  if f"torchao::{op_name}" in target_name:
    return True
  torchao_ns = getattr(torch.ops, "torchao", None)
  if torchao_ns is not None:
    op_obj = getattr(torchao_ns, op_name, None)
    if target == op_obj or target == getattr(op_obj, "default", None):
      return True
  return False


def _is_dynamic_scale_from_choose_qparams(scale_node: Any) -> bool:
  """Returns True if `scale_node` is produced by `torchao.choose_qparams_affine`."""
  if not isinstance(scale_node, torch.fx.Node):
    return False
  if (
      scale_node.op == "call_function"
      and getattr(scale_node.target, "__name__", "") == "getitem"
      and scale_node.args
  ):
    return _is_torchao_op(scale_node.args[0], "choose_qparams_affine")
  return _is_torchao_op(scale_node, "choose_qparams_affine")


class LowerTorchAOPass(fx_infra.ExportedProgramPassBase):
  """Lowers TorchAO Q/DQ ops to `quant.*` annotations for the MLIR converter."""

  def call(
      self, exported_program: torch.export.ExportedProgram
  ) -> fx_infra.ExportedProgramPassResult:
    # NOTE: When Converter V2 exports with `run_decompositions=False`, TorchAO
    # tensor subclasses (e.g. `AffineQuantizedTensor`) remain wrapped in
    # `state_dict`. Running `safe_run_decompositions` unwraps them into explicit
    # `torchao.quantize_affine`, `torchao.dequantize_affine`, and
    # `torchao.choose_qparams_affine` FX nodes.
    if any(
        hasattr(v, "__tensor_flatten__")
        for v in exported_program.state_dict.values()
    ):
      exported_program = fx_infra.graph_utils.reset_from_node_meta(
          exported_program
      )
      exported_program = fx_infra.safe_run_decompositions(
          exported_program,
          fx_infra.decomp.pre_convert_decomp(),
          can_skip=False,
      )

    gm = exported_program.graph_module
    graph = gm.graph
    state_dict = exported_program.state_dict
    inputs_to_params = exported_program.graph_signature.inputs_to_parameters
    inputs_to_buffers = exported_program.graph_signature.inputs_to_buffers
    modified = False

    def get_constant_tensor(node: Any) -> Optional[torch.Tensor]:
      """Resolves constant `scale`, `zero_point`, or `weight` tensors from `state_dict`."""
      if isinstance(node, torch.Tensor):
        return node
      if not isinstance(node, torch.fx.Node):
        return None
      if node.op == "placeholder":
        name = inputs_to_params.get(node.target) or inputs_to_buffers.get(
            node.target
        )
        return state_dict.get(name) if name else None
      if node.op == "get_attr":
        val = getattr(gm, node.target, None)
        return val if isinstance(val, torch.Tensor) else None
      return None

    def get_tensor_shape(node: torch.fx.Node) -> Optional[tuple[int, ...]]:
      t = get_constant_tensor(node)
      if t is not None:
        return tuple(t.shape)
      meta = node.meta.get("tensor_meta") or node.meta.get("val")
      if meta is not None and hasattr(meta, "shape"):
        return tuple(int(d) for d in meta.shape)
      return None

    def _is_drq_act_node(act: Any) -> bool:
      stack = [act]
      visited = set()
      while stack:
        curr = stack.pop()
        if not isinstance(curr, torch.fx.Node) or curr in visited:
          continue
        visited.add(curr)
        if curr.op == "call_function":
          if getattr(curr.target, "__name__", "") == "quant_fake_quant":
            return True
          if _is_torchao_op(curr, "choose_qparams_affine"):
            return True
          stack.extend(a for a in curr.args if isinstance(a, torch.fx.Node))
      return False

    def _consumer_has_dynamic_act(weight_deq_node: torch.fx.Node) -> bool:
      """Checks if a consumer of `weight_deq_node` has a DRQ activation."""
      stack = list(weight_deq_node.users)
      visited = set()
      while stack:
        u = stack.pop()
        if u in visited or u.op != "call_function":
          continue
        visited.add(u)
        u_str = str(u.target)
        if "addmm" in u_str and len(u.args) >= 2:
          if _is_drq_act_node(u.args[1]):
            return True
        elif any(
            k in u_str for k in ("linear", "aten.mm", "aten.bmm", "aten.matmul")
        ):
          if u.args and _is_drq_act_node(u.args[0]):
            return True
        else:
          stack.extend(u.users)
      return False

    for node in list(graph.nodes):
      # ========================================================================
      # 1. Lower `torchao.dequantize_affine`
      # ========================================================================
      # Signature: (input, block_size, scale, zero_point, input_dtype,
      #             quant_min, quant_max, ...)
      if _is_torchao_op(node, "dequantize_affine"):
        args = node.args
        input_node = args[0]
        block_size = list(args[1]) if len(args) > 1 else [1, 1]
        scale_node = args[2] if len(args) > 2 else None
        zp_node = args[3] if len(args) > 3 else None
        input_dtype = args[4] if len(args) > 4 else torch.int8
        quant_min = args[5] if len(args) > 5 else -128
        quant_max = args[6] if len(args) > 6 else 127
        dtype_str = _infer_dtype_str(
            quant_min, quant_max, fallback_dtype=input_dtype
        )

        # ----------------------------------------------------------------------
        # TODO(b/422588785): Converter Gap 1 (Dynamic Range Quantization / DRQ).
        # TorchAO decomposes dynamic activation quantization into three FX ops:
        #   scale, zp = torchao.choose_qparams_affine(act, ...)
        #   q = torchao.quantize_affine(act, block_size, scale, zp, ...)
        #   dq = torchao.dequantize_affine(q, block_size, scale, zp, ...)
        # where `scale` and `zp` are runtime tensor values. However, in the C++
        # MLIR converter:
        #   1. `RewriteQuantizeCustomCallOp` and `RewriteDequantizeCustomCallOp`
        #      (`FillCompositeParams` in `lower_quant_annotations_helper.h`)
        #      require static `DenseFPElementsAttr` `scale` attributes and fail
        #      when `scale` is dynamic or empty.
        #   2. `FuseQDQPass` (`IsDrqFakeQuant` in `fuse_qdq_pass.cc`) only
        #      recognizes DRQ when the activation is produced by a single
        #      unlowered `quant.fake_quant` custom call with an empty `scale`
        #      attribute (`scale = []`).
        # Until the MLIR converter supports dynamic scale/zp tensor operands on
        # `quant.quantize`/`quant.dequantize`, we collapse this 3-op dynamic
        # sequence into `quant.fake_quant(act, scale=[])`.
        # ----------------------------------------------------------------------
        if _is_torchao_op(input_node, "quantize_affine") and (
            _is_dynamic_scale_from_choose_qparams(scale_node)
            or _is_dynamic_scale_from_choose_qparams(input_node.args[2])
        ):
          act_input = input_node.args[0]
          with graph.inserting_before(node):
            attrs = _get_quant_composite_attributes(
                scale=None,
                zero_point=None,
                dtype=dtype_str,
                quant_dimension=None,
                narrow_range=True,
            )
            fq_op = _create_fake_quant_function(attrs)
            fq_node = graph.call_function(fq_op, (act_input,))
            node.replace_all_uses_with(fq_node)
            modified = True
          continue

        # Static `dequantize_affine`: resolve constant `scale` and `zero_point`
        # into static custom_call attributes (`Converter Gap 3`).
        scale_tensor = get_constant_tensor(scale_node)
        zp_tensor = get_constant_tensor(zp_node)
        if scale_tensor is None:
          continue
        if zp_tensor is None:
          zp_tensor = torch.zeros_like(scale_tensor, dtype=torch.int32)

        weight_tensor = get_constant_tensor(input_node)
        tensor_shape = get_tensor_shape(input_node)
        orig_shape = list(tensor_shape) if tensor_shape is not None else []
        ic = orig_shape[-1] if orig_shape else 0
        is_blockwise = (
            weight_tensor is not None
            and len(orig_shape) >= 2
            and len(block_size) == len(orig_shape)
            and all(b == 1 for b in block_size[:-1])
            and 1 < block_size[-1] < ic
            and ic % block_size[-1] == 0
        )

        with graph.inserting_before(node):
          if is_blockwise:
            # ------------------------------------------------------------------
            # TODO(b/422588785): Converter Gap 2 (Blockwise `quant.dequantize`
            # on pre-quantized integer weights).
            # In `lower_quant_annotations_pass.cc`, `IsBlockwiseAnnotation`
            # (`LowerBlockwiseFakeQuant`) is only implemented on
            # `quant.fake_quant` (which takes a floating-point constant weight
            # and quantizes it at compile time into `tfl.blockwise_dequantize`),
            # and is NOT implemented in `RewriteDequantizeCustomCallOp`
            # (`quant.dequantize`) for pre-quantized integer weights.
            # Furthermore, `FuseQDQPass` (`fuse_qdq_pass.cc:503`) only fuses
            # `tfl.blockwise_dequantize` when the activation is also blockwise
            # DRQ, whereas weight-only blockwise requires `tfl.dequantize` +
            # `tfl.reshape`.
            # Until `RewriteDequantizeCustomCallOp` and `FuseQDQPass` handle
            # pre-quantized blockwise `quant.dequantize` directly, we reshape
            # the weight to 2D `[total_blocks, group_size]`, apply per-axis
            # `quant.dequantize` along axis 0, and reshape back to `orig_shape`.
            # ------------------------------------------------------------------
            if _consumer_has_dynamic_act(node):
              raise NotImplementedError(
                  "Per-tensor/per-token DRQ activation with blockwise weight"
                  " dequantize is not supported by TFLite FuseQDQPass."
              )
            group_size = block_size[-1]
            total_blocks = weight_tensor.numel() // group_size
            reshaped_weight = graph.call_function(
                torch.ops.aten.reshape.default,
                (input_node, [total_blocks, group_size]),
            )
            attrs = _get_quant_composite_attributes(
                scale=scale_tensor.flatten().tolist(),
                zero_point=zp_tensor.flatten().tolist(),
                dtype=dtype_str,
                quant_dimension=0,
                narrow_range=True,
            )
            dq_op = _create_dequantize_function(attrs)
            dq_node = graph.call_function(dq_op, (reshaped_weight,))
            out_node = graph.call_function(
                torch.ops.aten.reshape.default,
                (dq_node, orig_shape),
            )
          else:
            # Direct 1-to-1 lowering of `torchao.dequantize_affine` to
            # `quant.dequantize`.
            quant_dim = _infer_quant_dimension(
                orig_shape, block_size, scale_tensor.numel()
            )
            attrs = _get_quant_composite_attributes(
                scale=scale_tensor.flatten().tolist(),
                zero_point=zp_tensor.flatten().tolist(),
                dtype=dtype_str,
                quant_dimension=quant_dim,
                narrow_range=True,
            )
            dq_op = _create_dequantize_function(attrs)
            out_node = graph.call_function(dq_op, (input_node,))

          node.replace_all_uses_with(out_node)
          modified = True
        continue

      # ========================================================================
      # 2. Lower `torchao.quantize_affine` (1-to-1 to `quant.quantize`)
      # ========================================================================
      # Signature: (input, block_size, scale, zero_point, output_dtype,
      #             quant_min, quant_max, ...)
      if _is_torchao_op(node, "quantize_affine"):
        args = node.args
        input_node = args[0]
        block_size = list(args[1]) if len(args) > 1 else [1, 1]
        scale_node = args[2] if len(args) > 2 else None
        zp_node = args[3] if len(args) > 3 else None
        output_dtype = args[4] if len(args) > 4 else torch.int8
        quant_min = args[5] if len(args) > 5 else -128
        quant_max = args[6] if len(args) > 6 else 127

        # Dynamic `quantize_affine` (where scale comes from
        # `choose_qparams_affine`) is folded together with its consumer
        # `dequantize_affine` into `quant.fake_quant` above (Converter Gap 1).
        if _is_dynamic_scale_from_choose_qparams(scale_node):
          continue

        scale_tensor = get_constant_tensor(scale_node)
        zp_tensor = get_constant_tensor(zp_node)
        if scale_tensor is None:
          continue
        if zp_tensor is None:
          zp_tensor = torch.zeros_like(scale_tensor, dtype=torch.int32)

        dtype_str = _infer_dtype_str(
            quant_min, quant_max, fallback_dtype=output_dtype
        )
        tensor_shape = get_tensor_shape(input_node) or ()
        quant_dim = _infer_quant_dimension(
            tensor_shape, block_size, scale_tensor.numel()
        )
        with graph.inserting_before(node):
          attrs = _get_quant_composite_attributes(
              scale=scale_tensor.flatten().tolist(),
              zero_point=zp_tensor.flatten().tolist(),
              dtype=dtype_str,
              quant_dimension=quant_dim,
              narrow_range=True,
          )
          q_op = _create_quantize_function(attrs)
          q_node = graph.call_function(q_op, (input_node,))
          node.replace_all_uses_with(q_node)
          modified = True

    # Prune dead `choose_qparams_affine` / replaced `quantize_affine` /
    # `dequantize_affine` nodes with zero users.
    if modified:
      changed = True
      while changed:
        changed = False
        for n in list(reversed(graph.nodes)):
          if n.op not in ("placeholder", "output") and not n.users:
            n.args = ()
            n.kwargs = {}
            graph.erase_node(n)
            changed = True

    return fx_infra.ExportedProgramPassResult(exported_program, modified)


# ==============================================================================
# Custom PyTorch Op & StableHLO `custom_call` Lowering
# ==============================================================================
_torch_library.LITERT_TORCH_LIB.define(
    "quant_composite(Tensor x, str name, str attr_json) -> Tensor"
)
_quant_composite_op = torch.ops.litert_torch.quant_composite.default

# In-memory attribute cache indexed by unique string identifiers.
_QUANT_ATTR_CACHE: dict[str, dict[str, Any]] = {}


def _register_quant_attrs(attrs: dict[str, Any]) -> str:
  attr_id = f"quant_attr_{uuid.uuid4().hex}"
  _QUANT_ATTR_CACHE[attr_id] = attrs
  return attr_id


@torch.library.impl(
    _torch_library.LITERT_TORCH_LIB,
    "quant_composite",
    "CompositeExplicitAutograd",
)
def _quant_composite(
    x: torch.Tensor, name: str, attr_json: str
) -> torch.Tensor:
  """Functional eager-mode implementation of `quant_composite`."""
  attr = _QUANT_ATTR_CACHE.get(attr_json, {})
  def _param_tensor(values: list[Any], q_dim: Any) -> torch.Tensor:
    t = torch.tensor(values, dtype=torch.float32, device=x.device)
    if q_dim is not None and x.ndim > 1 and t.numel() > 1:
      shape = [1] * x.ndim
      shape[q_dim] = t.numel()
      t = t.view(shape)
    return t

  if name == "quant.dequantize":
    scale = attr.get("scale", [])
    if scale:
      q_dim = attr.get("quantization_dimension", 0)
      scale_t = _param_tensor(scale, q_dim)
      zp = attr.get("zero_point", [])
      x_f = x.to(torch.float32)
      if zp:
        x_f = x_f - _param_tensor(zp, q_dim)
      return x_f * scale_t
    return x.to(torch.float32)
  if name == "quant.fake_quant":
    scale = attr.get("scale", [])
    if scale:
      scale_t = torch.tensor(scale, dtype=torch.float32, device=x.device)
      dtype_str = attr.get("dtype", "i8")
      bounds = {
          "i2": (-2.0, 1.0),
          "i4": (-8.0, 7.0),
          "ui4": (0.0, 15.0),
          "ui8": (0.0, 255.0),
          "i16": (-32768.0, 32767.0),
      }
      min_v, max_v = bounds.get(dtype_str, (-128.0, 127.0))
      if "scale_shape" in attr and scale_t.numel() > 1:
        scale_t = scale_t.view(attr["scale_shape"])
      if scale_t.ndim == x.ndim and any(
          s != xs and s > 1 for s, xs in zip(scale_t.shape, x.shape)
      ):
        for dim, (s, xs) in enumerate(zip(scale_t.shape, x.shape)):
          if xs > s and xs % s == 0:
            scale_t = scale_t.repeat_interleave(xs // s, dim=dim)
      zp = attr.get("zero_point", [])
      zp_val = float(zp[0]) if zp else 0.0
      return (
          torch.clamp(torch.round(x / scale_t + zp_val), min_v, max_v) - zp_val
      ) * scale_t
  if name == "quant.quantize":
    scale = attr.get("scale", [])
    dtype_str = attr.get("dtype", "i8")
    torch_dtype = {
        "i8": torch.int8,
        "ui8": torch.uint8,
        "i16": torch.int16,
        "i32": torch.int32,
    }.get(dtype_str, torch.int8)
    if scale:
      q_dim = attr.get("quantization_dimension", 0)
      scale_t = _param_tensor(scale, q_dim)
      zp = attr.get("zero_point", [])
      q_val = x / scale_t
      if zp:
        q_val = q_val + _param_tensor(zp, q_dim)
      bounds = {"ui8": (0.0, 255.0), "i16": (-32768.0, 32767.0)}
      min_v, max_v = bounds.get(dtype_str, (-128.0, 127.0))
      return torch.clamp(torch.round(q_val), min_v, max_v).to(torch_dtype)
    return x.to(torch_dtype)
  return x


@torch.library.impl(_torch_library.LITERT_TORCH_LIB, "quant_composite", "Meta")
def _quant_composite_meta(
    x: torch.Tensor, name: str, attr_json: str
) -> torch.Tensor:
  """Meta-kernel for TorchDynamo and FakeTensor shape/dtype propagation."""
  if name == "quant.dequantize":
    return torch.empty(x.shape, dtype=torch.float32, device=x.device)
  if name == "quant.quantize":
    attr = _QUANT_ATTR_CACHE.get(attr_json, {})
    dtype_str = attr.get("dtype", "i8")
    torch_dtype = {
        "i8": torch.int8,
        "ui8": torch.uint8,
        "i16": torch.int16,
        "i32": torch.int32,
    }.get(dtype_str, torch.int8)
    return torch.empty(x.shape, dtype=torch_dtype, device=x.device)
  return torch.empty_like(x)


def _build_composite_attributes(
    attr: dict[str, Any], context: Optional[ir.Context] = None
) -> dict[str, ir.Attribute]:
  """Constructs an MLIR Attribute dictionary from Python quantization attributes."""
  if context is None:
    context = ir.Context()
  helper_keys = {"scale_len", "scale_shape", "zp_shape", "zp_is_float"}
  composite_attrs = {}
  with context, ir.Location.unknown(context):
    for k, v in attr.items():
      if k in helper_keys:
        continue
      if k == "scale":
        if not v and "block_shape" in attr:
          continue
        scale_shape = attr.get("scale_shape", [len(v)])
        arr = np.array(v, dtype=np.float32)
        if v and len(scale_shape) > 1:
          arr = arr.reshape(scale_shape)
        composite_attrs[k] = ir.DenseElementsAttr.get(
            arr,
            type=ir.F32Type.get(context),
            shape=scale_shape if v else [0],
        )
      elif k == "zero_point":
        if attr.get("zp_is_float", False):
          zp_vals = [v] if isinstance(v, (int, float, np.number)) else list(v)
          if not zp_vals:
            continue
          zp_shape = attr.get("zp_shape", [len(zp_vals)])
          if len(zp_vals) == 1 or all(x == zp_vals[0] for x in zp_vals):
            arr = np.array([zp_vals[0]], dtype=np.float32)
          else:
            arr = np.array(zp_vals, dtype=np.float32).reshape(zp_shape)
          composite_attrs[k] = ir.DenseElementsAttr.get(
              arr,
              type=ir.F32Type.get(context),
              shape=zp_shape,
          )
        elif isinstance(v, (int, np.integer)):
          zp_shape = attr.get("zp_shape", [1])
          arr = np.array([v], dtype=np.int64)
          composite_attrs[k] = ir.DenseElementsAttr.get(
              arr,
              type=ir.IntegerType.get_signless(64, context),
              shape=zp_shape,
          )
        elif isinstance(v, (list, tuple)):
          if not v:
            continue
          if "zp_shape" in attr:
            zp_shape = attr["zp_shape"]
            if len(v) == 1 or all(x == v[0] for x in v):
              arr = np.array([v[0]], dtype=np.int64)
            else:
              arr = np.array(v, dtype=np.int64).reshape(zp_shape)
            composite_attrs[k] = ir.DenseElementsAttr.get(
                arr,
                type=ir.IntegerType.get_signless(64, context),
                shape=zp_shape,
            )
          elif len(v) == 1 or all(x == v[0] for x in v):
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
      elif k == "block_shape":
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
      elif isinstance(v, float):
        composite_attrs[k] = ir.FloatAttr.get(ir.F32Type.get(context), v)
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
              arr,
              type=ir.IntegerType.get_signless(64, context),
              shape=[len(v)],
          )
  return composite_attrs


@lowerings.lower(torch.ops.litert_torch.quant_composite)
def _quant_composite_lowering(lctx, x: ir.Value, name: str, attr_json: str):
  """Lowers `litert_torch.quant_composite` to `stablehlo.custom_call`."""
  if attr_json in _QUANT_ATTR_CACHE:
    attr = _QUANT_ATTR_CACHE[attr_json]
  elif attr_json:
    attr = json.loads(attr_json)
  else:
    attr = {}
  custom_call_attrs = _build_composite_attributes(attr, lctx.ir_context)

  if name == "quant.dequantize":
    x_shaped = ir.ShapedType(x.type)
    out_type = ir.RankedTensorType.get(
        x_shaped.shape, ir.F32Type.get(lctx.ir_context)
    )
  elif name == "quant.quantize":
    x_shaped = ir.ShapedType(x.type)
    dtype_str = attr.get("dtype", "i8")
    bitwidth = {
        "i2": 8,
        "i4": 8,
        "ui4": 8,
        "i8": 8,
        "ui8": 8,
        "i16": 16,
        "i32": 32,
    }.get(dtype_str, 8)
    out_type = ir.RankedTensorType.get(
        x_shaped.shape, ir.IntegerType.get_signless(bitwidth, lctx.ir_context)
    )
  else:
    out_type = x.type

  custom_call = stablehlo.CustomCallOp(
      result=[out_type],
      inputs=[x],
      call_target_name=ir.StringAttr.get(name, lctx.ir_context),
  )
  for k, v in custom_call_attrs.items():
    custom_call.attributes[k] = v
  return custom_call.results[0]


def _create_fake_quant_function(attrs: dict[str, Any]):
  """Returns a callable for constructing a `quant.fake_quant` custom call in FX."""
  attr_key = _register_quant_attrs(attrs)

  def fake_quant_fn(x):
    return _quant_composite_op(x, "quant.fake_quant", attr_key)

  fake_quant_fn.__name__ = "quant_fake_quant"
  return fake_quant_fn


def _create_dequantize_function(attrs: dict[str, Any]):
  """Returns a callable for constructing a `quant.dequantize` custom call in FX."""
  attr_key = _register_quant_attrs(attrs)

  def dequantize_fn(x):
    return _quant_composite_op(x, "quant.dequantize", attr_key)

  dequantize_fn.__name__ = "quant_dequantize"
  return dequantize_fn


def _create_quantize_function(attrs: dict[str, Any]):
  """Returns a callable for constructing a `quant.quantize` custom call in FX."""
  attr_key = _register_quant_attrs(attrs)

  def quantize_fn(x):
    return _quant_composite_op(x, "quant.quantize", attr_key)

  quantize_fn.__name__ = "quant_quantize"
  return quantize_fn
