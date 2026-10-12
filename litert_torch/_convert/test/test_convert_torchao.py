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
"""Tests for TorchAO quantization conversion in litert_torch."""

import collections
import dataclasses
from typing import Any, Optional
from absl.testing import parameterized
import litert_torch
from litert_torch._convert.fx_passes import lower_torchao_pass
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torchao.quantization import quant_api
from absl.testing import absltest as googletest

Int8DynamicActivationIntxWeightConfig = (
    quant_api.Int8DynamicActivationIntxWeightConfig
)
IntxWeightOnlyConfig = quant_api.IntxWeightOnlyConfig
PerAxis = quant_api.PerAxis
PerGroup = quant_api.PerGroup
quantize_ = quant_api.quantize_


def get_diff(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
  """Calculates absolute and relative diff metrics between two arrays."""
  x_arr = np.asarray(x, dtype=np.float32)
  y_arr = np.asarray(y, dtype=np.float32)
  abs_diff = np.abs(x_arr - y_arr)
  rel_diff = np.nan_to_num(abs_diff / np.maximum(np.abs(x_arr), np.abs(y_arr)))
  return {
      "mean_abs_diff": float(abs_diff.mean()),
      "mean_rel_diff": float(rel_diff.mean()),
      "max_abs_diff": float(abs_diff.max()),
      "max_rel_diff": float(rel_diff.max()),
  }


@dataclasses.dataclass
class OpSummary:
  """Summary of an operation in the LiteRT graph."""

  index: int
  op_name: str
  inputs: np.ndarray
  outputs: np.ndarray
  operand_types: list[np.dtype]
  result_types: list[np.dtype]


class GraphSummary:
  """Provides a summary of the LiteRT graph operations and tensors."""

  def __init__(self, model):
    interpreter = model._get_interpreter()
    model_summary = interpreter._get_ops_details()
    self._ops = [OpSummary(**entry) for entry in model_summary]
    self._tensors = {t["index"]: t for t in interpreter.get_tensor_details()}
    self._graph_inputs, self._graph_outputs, self._defined_by = (
        self._cache_internals(self._ops)
    )

  def _cache_internals(self, ops: list[OpSummary]):
    """Caches graph inputs, graph outputs, and tensor producer map."""
    all_inputs, all_outputs = set(), set()
    defined_by: dict[int, OpSummary] = {}
    for op in ops:
      if op.op_name == "DELEGATE":
        continue
      for output_value in op.outputs:
        defined_by[output_value] = op

      all_inputs = all_inputs.union(op.inputs)
      all_outputs = all_outputs.union(op.outputs)
    graph_inputs = list(all_inputs - all_outputs)
    graph_outputs = list(all_outputs - all_inputs)
    return graph_inputs, graph_outputs, defined_by

  def ops(self):
    """Returns an iterator over non-delegate ops in the graph."""
    return (op for op in self._ops if op.op_name != "DELEGATE")

  def tensor(self, index: int) -> Optional[dict[str, Any]]:
    """Returns tensor details for the given tensor index."""
    return self._tensors.get(index)

  def defined_by(self, index: int) -> Optional[OpSummary]:
    """Returns the op that defines the given tensor index, if any."""
    return self._defined_by.get(index)

  def is_op_drq(
      self, op: OpSummary, weight_dtype: npt.DTypeLike = np.int8
  ) -> bool:
    """Returns true if the op is dynamically quantized (DRQ)."""
    if not all(res_type == np.float32 for res_type in op.result_types):
      return False
    operand_type_counter = collections.Counter(op.operand_types)
    return operand_type_counter.get(weight_dtype, 0) >= 1

  def is_op_weight_only(self, op: OpSummary) -> bool:
    """Returns true if the op is weight-only quantized."""
    if not all(
        elem_type == np.float32
        for elem_type in op.operand_types + op.result_types
    ):
      return False
    if len(op.inputs) < 2:
      return False
    def_op = self.defined_by(op.inputs[1])
    if def_op is None:
      return False
    if def_op.op_name == "RESHAPE" and len(def_op.inputs) > 0:
      def_op = self.defined_by(def_op.inputs[0]) or def_op
    return def_op.op_name == "DEQUANTIZE"


class TestConvertTorchAO(parameterized.TestCase):
  """Tests conversion, interpreter inspection, and numerical execution of TorchAO quantized models."""

  def setUp(self):
    super().setUp()
    if not litert_torch.converter_v2.is_supported():
      self.skipTest(
          "Converter V2 is not supported by the installed TensorFlow."
      )
    torch.manual_seed(42)

  def _ensure_output_structure(
      self,
      edge_model,
      target_op_name: str,
      is_drq: bool = False,
      expected_weight_dtype: Optional[np.dtype] = None,
      expected_num_scales: Optional[int] = None,
  ):
    """Validates ops and tensor details in the converted LiteRT model."""
    graph_summary = GraphSummary(edge_model)
    op_names = [op.op_name for op in graph_summary.ops()]

    self.assertIn(
        target_op_name,
        op_names,
        f"Expected '{target_op_name}' in LiteRT ops, found: {op_names}",
    )
    self.assertNotIn(
        "STABLEHLO_COMPOSITE",
        op_names,
        "Found unconverted STABLEHLO_COMPOSITE ops in final model.",
    )
    self.assertNotIn(
        "STABLEHLO_CUSTOM_CALL",
        op_names,
        "Found unconverted STABLEHLO_CUSTOM_CALL ops in final model.",
    )

    target_ops = [
        op for op in graph_summary.ops() if op.op_name == target_op_name
    ]

    if is_drq:
      self.assertNotIn(
          "DEQUANTIZE",
          op_names,
          f"In DRQ mode, found unexpected DEQUANTIZE ops: {op_names}",
      )
      self.assertTrue(
          any(graph_summary.is_op_drq(op) for op in target_ops),
          f"Expected at least one {target_op_name} op to satisfy DRQ operand"
          f" types, found: {[op.operand_types for op in target_ops]}",
      )
      for op in target_ops:
        if len(op.inputs) > 1:
          weight_t = graph_summary.tensor(op.inputs[1])
          if weight_t:
            if expected_weight_dtype is not None:
              self.assertEqual(weight_t["dtype"], expected_weight_dtype)
            qparams = weight_t.get("quantization_parameters") or {}
            scales = qparams.get("scales", np.array([]))
            if expected_num_scales is not None:
              self.assertLen(scales, expected_num_scales)
    else:
      weight_input_idx = 1
      for op in target_ops:
        if len(op.inputs) > weight_input_idx:
          def_op = graph_summary.defined_by(op.inputs[weight_input_idx])
          if def_op is not None:
            while (
                def_op.op_name == "RESHAPE"
                and len(def_op.inputs) > 0
                and graph_summary.defined_by(def_op.inputs[0]) is not None
            ):
              def_op = graph_summary.defined_by(def_op.inputs[0])
            self.assertEqual(
                def_op.op_name,
                "DEQUANTIZE",
                "Expected weight input to come from DEQUANTIZE, got"
                f" {def_op.op_name}",
            )
            if len(def_op.inputs) > 0:
              weight_t = graph_summary.tensor(def_op.inputs[0])
              if weight_t:
                if expected_weight_dtype is not None:
                  self.assertEqual(weight_t["dtype"], expected_weight_dtype)
                qparams = weight_t.get("quantization_parameters") or {}
                scales = qparams.get("scales", np.array([]))
                if expected_num_scales is not None:
                  self.assertLen(scales, expected_num_scales)

  def test_weight_only_int8_linear_per_channel(self):
    """Tests Linear module with Int8 per-channel weight-only quantization."""

    class LinearModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16, bias=True)

      def forward(self, x):
        return self.fc(x)

    torch_module = LinearModule().eval()
    quantize_(
        torch_module,
        IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
    )

    args = (torch.randn((2, 32)),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    edge_out = edge_model(*args)

    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 1e-2)

    self._ensure_output_structure(
        edge_model,
        target_op_name="FULLY_CONNECTED",
        is_drq=False,
        expected_weight_dtype=np.int8,
        expected_num_scales=16,
    )

  def test_weight_only_int4_linear_per_channel(self):
    """Tests Linear module with Int4 per-channel weight-only quantization."""

    class LinearModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16, bias=True)

      def forward(self, x):
        return self.fc(x)

    torch_module = LinearModule().eval()
    quantize_(
        torch_module,
        IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerAxis(0)),
    )

    args = (torch.randn((2, 32)),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    edge_out = edge_model(*args)

    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 5e-2)

    self._ensure_output_structure(
        edge_model,
        target_op_name="FULLY_CONNECTED",
        is_drq=False,
        expected_weight_dtype=np.int8,
        expected_num_scales=16,
    )

  def test_weight_only_int8_embedding_per_channel(self):
    """Tests Embedding module with Int8 per-channel weight-only quantization."""

    class EmbeddingModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 32)

      def forward(self, x):
        return self.emb(x)

    torch_module = EmbeddingModule().eval()
    quantize_(
        torch_module,
        IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerAxis(0)),
        filter_fn=lambda m, fqn: isinstance(m, nn.Embedding),
    )

    args = (torch.tensor([[1, 5, 10], [20, 25, 30]], dtype=torch.long),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    # enable_x64 defaults to False, so the converted model takes int32 ids.
    edge_out = edge_model(args[0].to(torch.int32))

    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 1e-2)

    self._ensure_output_structure(
        edge_model,
        target_op_name="EMBEDDING_LOOKUP",
        is_drq=False,
        expected_weight_dtype=np.int8,
        expected_num_scales=50,
    )

  def test_weight_only_int4_embedding_per_channel(self):
    """Tests Embedding module with Int4 per-channel weight-only quantization."""

    class EmbeddingModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 32)

      def forward(self, x):
        return self.emb(x)

    torch_module = EmbeddingModule().eval()
    quantize_(
        torch_module,
        IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerAxis(0)),
        filter_fn=lambda m, fqn: isinstance(m, nn.Embedding),
    )

    args = (torch.tensor([[2, 4, 6], [8, 10, 12]], dtype=torch.long),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    edge_out = edge_model(args[0].to(torch.int32))

    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 5e-2)

    self._ensure_output_structure(
        edge_model,
        target_op_name="EMBEDDING_LOOKUP",
        is_drq=False,
        expected_weight_dtype=np.int8,
        expected_num_scales=50,
    )

  def test_weight_only_int8_linear_blockwise(self):
    """Tests Linear module with Int8 blockwise weight-only quantization."""

    class LinearModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 16, bias=True)

      def forward(self, x):
        return self.fc(x)

    torch_module = LinearModule().eval()
    quantize_(
        torch_module,
        IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerGroup(32)),
    )

    args = (torch.randn((2, 64)),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    edge_out = edge_model(*args)

    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 1e-2)

    self._ensure_output_structure(
        edge_model,
        target_op_name="FULLY_CONNECTED",
        is_drq=False,
        expected_weight_dtype=np.int8,
        expected_num_scales=32,
    )

  def test_weight_only_int4_linear_blockwise(self):
    """Tests Linear module with Int4 blockwise weight-only quantization."""

    class LinearModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 16, bias=True)

      def forward(self, x):
        return self.fc(x)

    torch_module = LinearModule().eval()
    quantize_(
        torch_module,
        IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
    )

    args = (torch.randn((2, 64)),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    edge_out = edge_model(*args)

    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 5e-2)

    self._ensure_output_structure(
        edge_model,
        target_op_name="FULLY_CONNECTED",
        is_drq=False,
        expected_weight_dtype=np.int8,
        expected_num_scales=32,
    )

  def test_weight_only_int8_embedding_blockwise(self):
    """Tests Embedding module with Int8 blockwise weight-only quantization."""

    class EmbeddingModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 64)

      def forward(self, x):
        return self.emb(x)

    torch_module = EmbeddingModule().eval()
    quantize_(
        torch_module,
        IntxWeightOnlyConfig(weight_dtype=torch.int8, granularity=PerGroup(32)),
        filter_fn=lambda m, fqn: isinstance(m, nn.Embedding),
    )

    args = (torch.tensor([[1, 5, 10], [20, 25, 30]], dtype=torch.long),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    edge_out = edge_model(args[0].to(torch.int32))

    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 1e-2)

    self._ensure_output_structure(
        edge_model,
        target_op_name="EMBEDDING_LOOKUP",
        is_drq=False,
        expected_weight_dtype=np.int8,
        expected_num_scales=100,
    )

  def test_weight_only_int4_embedding_blockwise(self):
    """Tests Embedding module with Int4 blockwise weight-only quantization."""

    class EmbeddingModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 64)

      def forward(self, x):
        return self.emb(x)

    torch_module = EmbeddingModule().eval()
    quantize_(
        torch_module,
        IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)),
        filter_fn=lambda m, fqn: isinstance(m, nn.Embedding),
    )

    args = (torch.tensor([[2, 4, 6], [8, 10, 12]], dtype=torch.long),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    edge_out = edge_model(args[0].to(torch.int32))

    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 5e-2)

    self._ensure_output_structure(
        edge_model,
        target_op_name="EMBEDDING_LOOKUP",
        is_drq=False,
        expected_weight_dtype=np.int8,
        expected_num_scales=100,
    )

  def test_drq_int8_dynamic_act_int8_weight(self):
    """Tests Linear module with Dynamic Int8 activation + Int8 weight (DRQ a8w8)."""

    class LinearModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16, bias=True)

      def forward(self, x):
        return self.fc(x)

    torch_module = LinearModule().eval()
    quantize_(
        torch_module,
        Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8),
    )

    args = (torch.randn((2, 32)),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    edge_out = edge_model(*args)

    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 5e-2)

    self._ensure_output_structure(
        edge_model,
        target_op_name="FULLY_CONNECTED",
        is_drq=True,
        expected_weight_dtype=np.int8,
        expected_num_scales=16,
    )

  def test_drq_int8_dynamic_act_int4_weight(self):
    """Tests Linear module with Dynamic Int8 activation + Int4 weight (DRQ a8w4)."""

    class LinearModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 16, bias=True)

      def forward(self, x):
        return self.fc(x)

    torch_module = LinearModule().eval()
    quantize_(
        torch_module,
        Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4),
    )

    args = (torch.randn((2, 32)),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    edge_out = edge_model(*args)

    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 1e-1)

    self._ensure_output_structure(
        edge_model,
        target_op_name="FULLY_CONNECTED",
        is_drq=True,
        expected_weight_dtype=np.int8,
        expected_num_scales=16,
    )

  def test_drq_blockwise_raises_not_implemented(self):
    """Tests that blockwise dynamic range quantization raises NotImplementedError."""

    class LinearModule(nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 16, bias=True)

      def forward(self, x):
        return self.fc(x)

    torch_module = LinearModule().eval()
    quantize_(
        torch_module,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int8, weight_granularity=PerGroup(32)
        ),
    )

    args = (torch.randn((2, 64)),)
    with self.assertRaises(NotImplementedError):
      litert_torch.convert(torch_module, args, use_v2=True)

  def test_static_qdq_linear_lowering(self):
    """Tests 1-to-1 `quantize_affine` and `dequantize_affine` lowering on static QDQ Linear."""

    class StaticQDQLinear(nn.Module):
      """Model using explicit TorchAO quantize_affine and dequantize_affine ops."""

      def __init__(self):
        super().__init__()
        self.w_int = nn.Parameter(
            torch.randint(-127, 127, (16, 32), dtype=torch.int8),
            requires_grad=False,
        )
        self.w_scale = nn.Parameter(
            torch.full((16,), 0.02, dtype=torch.float32), requires_grad=False
        )
        self.w_zp = nn.Parameter(
            torch.zeros((16,), dtype=torch.int32), requires_grad=False
        )
        self.act_scale = nn.Parameter(
            torch.tensor([0.05], dtype=torch.float32), requires_grad=False
        )
        self.act_zp = nn.Parameter(
            torch.tensor([0], dtype=torch.int32), requires_grad=False
        )

      def forward(self, x):
        x_q = torch.ops.torchao.quantize_affine(
            x,
            list(x.shape),
            self.act_scale,
            self.act_zp,
            torch.int8,
            -128,
            127,
        )
        x_dq = torch.ops.torchao.dequantize_affine(
            x_q,
            list(x.shape),
            self.act_scale,
            self.act_zp,
            torch.int8,
            -128,
            127,
        )
        w_dq = torch.ops.torchao.dequantize_affine(
            self.w_int,
            [1, 32],
            self.w_scale,
            self.w_zp,
            torch.int8,
            -127,
            127,
        )
        out = torch.nn.functional.linear(x_dq, w_dq)
        out_q = torch.ops.torchao.quantize_affine(
            out,
            list(out.shape),
            self.act_scale,
            self.act_zp,
            torch.int8,
            -128,
            127,
        )
        return torch.ops.torchao.dequantize_affine(
            out_q,
            list(out.shape),
            self.act_scale,
            self.act_zp,
            torch.int8,
            -128,
            127,
        )

    torch_module = StaticQDQLinear().eval()
    args = (torch.randn((2, 32)),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args, use_v2=True)
    edge_out = edge_model(*args)
    diff = get_diff(edge_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 5e-2)

    graph_summary = GraphSummary(edge_model)
    op_names = [op.op_name for op in graph_summary.ops()]
    self.assertIn("FULLY_CONNECTED", op_names)
    self.assertNotIn("STABLEHLO_CUSTOM_CALL", op_names)

  def test_infer_dtype_str_uint8(self):
    """Tests that _infer_dtype_str properly handles uint8, int2, and fallback dtype."""
    infer_dtype = lower_torchao_pass._infer_dtype_str  # pylint: disable=protected-access

    self.assertEqual(infer_dtype(0, 255), "ui8")
    self.assertEqual(infer_dtype(0, 255, fallback_dtype=torch.uint8), "ui8")
    self.assertEqual(infer_dtype(10, 200), "ui8")
    self.assertEqual(infer_dtype(-128, 127), "i8")
    self.assertEqual(infer_dtype(0, 15), "ui4")
    self.assertEqual(infer_dtype(-8, 7), "i4")
    self.assertEqual(infer_dtype(-2, 1), "i2")
    self.assertEqual(infer_dtype(-32768, 32767), "i16")
    self.assertEqual(
        infer_dtype(-100000, 100000, fallback_dtype=torch.uint8), "ui8"
    )

  def test_blockwise_composite_attributes(self):
    """Tests that blockwise custom_call attributes match MLIR LowerQuantAnnotationsPass contract."""
    get_attrs = lower_torchao_pass._get_quant_composite_attributes  # pylint: disable=protected-access
    build_attrs = lower_torchao_pass._build_composite_attributes  # pylint: disable=protected-access

    attrs = get_attrs(
        dtype="i2",
        scale=[[0.1, 0.2], [0.3, 0.4]],
        zero_point=-0.5,
        block_shape=[1, 32],
        symmetric=False,
    )
    built = build_attrs(attrs)
    self.assertIn("dtype", built)
    self.assertIn("scale", built)
    self.assertIn("zero_point", built)
    self.assertIn("block_shape", built)
    self.assertIn("symmetric", built)
    self.assertEqual(list(built["scale"].type.shape), [2, 2])
    self.assertEqual(str(built["zero_point"].type.element_type), "f32")
    self.assertEqual(list(built["block_shape"].type.shape), [2])

    act_attrs = get_attrs(
        dtype="i4",
        block_shape=[1, 1, 32],
        symmetric=True,
        act_scale_dtype="e8m0",
        range_dilation=1.0,
    )
    built_act = build_attrs(act_attrs)
    self.assertIn("dtype", built_act)
    self.assertIn("block_shape", built_act)
    self.assertIn("symmetric", built_act)
    self.assertIn("act_scale_dtype", built_act)
    self.assertIn("range_dilation", built_act)
    self.assertNotIn("scale", built_act)

  def test_quantize_composite_lowering(self):
    """Tests that quant.quantize and quant.dequantize lower cleanly to StableHLO custom_call."""
    get_attrs = lower_torchao_pass._get_quant_composite_attributes  # pylint: disable=protected-access
    create_q = lower_torchao_pass._create_quantize_function  # pylint: disable=protected-access
    create_dq = lower_torchao_pass._create_dequantize_function  # pylint: disable=protected-access

    attrs = get_attrs(
        dtype="i8",
        scale=[0.1] * 8,
        zero_point=[0] * 8,
        quant_dimension=1,
    )
    q_fn = create_q(attrs)
    dq_fn = create_dq(attrs)

    class QuantizeModule(nn.Module):
      """Module wrapping quantize and dequantize custom calls."""

      def forward(self, x):
        q = q_fn(x)
        return dq_fn(q)

    model = QuantizeModule().eval()
    args = (torch.randn((2, 8)),)
    edge_model = litert_torch.convert(model, args, use_v2=True)
    edge_out = edge_model(*args)
    self.assertEqual(edge_out.shape, (2, 8))


if __name__ == "__main__":
  googletest.main()
