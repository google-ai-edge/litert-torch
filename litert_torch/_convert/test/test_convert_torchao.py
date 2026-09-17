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
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    PerAxis,
    PerGroup,
    quantize_,
)
from absl.testing import absltest as googletest


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
    all_inputs, all_outputs = set(), set()
    defined_by: dict[int, OpSummary] = {}
    for op in ops:
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

  def is_op_drq(
      self, op: OpSummary, weight_dtype: npt.DTypeLike = np.int8
  ) -> bool:
    """Returns true if the op is dynamically quantized (DRQ).

    In DRQ, activations and outputs are float32, and the weight operand is
    statically quantized to integer.
    """
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
    if op.inputs[1] not in self._defined_by:
      return False
    return self._defined_by[op.inputs[1]].op_name == "DEQUANTIZE"


class TestConvertTorchAO(parameterized.TestCase):
  """Tests conversion, interpreter inspection, and numerical execution of TorchAO quantized models."""

  def setUp(self):
    super().setUp()
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

    # Ensure target op is in the model
    self.assertIn(
        target_op_name,
        op_names,
        f"Expected '{target_op_name}' in LiteRT ops, found: {op_names}",
    )

    # Ensure no unconverted STABLEHLO_COMPOSITE ops remain
    self.assertNotIn(
        "STABLEHLO_COMPOSITE",
        op_names,
        "Found unconverted STABLEHLO_COMPOSITE ops in final model.",
    )

    target_ops = [
        op for op in graph_summary.ops() if op.op_name == target_op_name
    ]

    if is_drq:
      # In DRQ, there should be no DEQUANTIZE ops before the target op
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
              self.assertEqual(len(scales), expected_num_scales)
    else:
      # In Weight-Only, the weight input is driven by DEQUANTIZE
      for op in target_ops:
        if len(op.inputs) > 1 and op.inputs[1] in graph_summary._defined_by:
          def_op = graph_summary._defined_by[op.inputs[1]]
          if def_op.op_name == "DEQUANTIZE" and len(def_op.inputs) > 0:
            weight_t = graph_summary.tensor(def_op.inputs[0])
            if weight_t:
              if expected_weight_dtype is not None:
                self.assertEqual(weight_t["dtype"], expected_weight_dtype)
              qparams = weight_t.get("quantization_parameters") or {}
              scales = qparams.get("scales", np.array([]))
              if expected_num_scales is not None:
                self.assertEqual(len(scales), expected_num_scales)

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

    edge_model = litert_torch.convert(torch_module, args)
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

    edge_model = litert_torch.convert(torch_module, args)
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
    )

    args = (torch.tensor([[1, 5, 10], [20, 25, 30]], dtype=torch.long),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args)
    edge_out = edge_model(*args)

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
    )

    args = (torch.tensor([[2, 4, 6], [8, 10, 12]], dtype=torch.long),)
    with torch.no_grad():
      torch_out = torch_module(*args).detach().numpy()

    edge_model = litert_torch.convert(torch_module, args)
    edge_out = edge_model(*args)

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

    edge_model = litert_torch.convert(torch_module, args)
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

    edge_model = litert_torch.convert(torch_module, args)
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

    edge_model = litert_torch.convert(torch_module, args)
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

    edge_model = litert_torch.convert(torch_module, args)
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
      litert_torch.convert(torch_module, args)

  def test_force_weight_only_as_drq_int8(self):
    """Tests that weight-only int8 quantized linear is lowered as DRQ when flag is set."""

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

    # Without flag: lowered as Weight-Only (not DRQ)
    woq_model = litert_torch.convert(
        torch_module, args, _litert_converter_flags={"strict_qdq_mode": True}
    )
    self._ensure_output_structure(
        woq_model,
        target_op_name="FULLY_CONNECTED",
        is_drq=False,
        expected_weight_dtype=np.int8,
        expected_num_scales=16,
    )

    # With flag: lowered as DRQ
    drq_model = litert_torch.convert(
        torch_module,
        args,
        _litert_converter_flags={
            "strict_qdq_mode": True,
            "_experimental_weight_only_as_drq": True,
        },
    )
    drq_out = drq_model(*args)
    diff = get_diff(drq_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 5e-2)

    self._ensure_output_structure(
        drq_model,
        target_op_name="FULLY_CONNECTED",
        is_drq=True,
        expected_weight_dtype=np.int8,
        expected_num_scales=16,
    )

  def test_force_weight_only_as_drq_int4(self):
    """Tests that weight-only int4 quantized linear is lowered as DRQ when flag is set."""

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

    drq_model = litert_torch.convert(
        torch_module,
        args,
        _litert_converter_flags={
            "strict_qdq_mode": True,
            "_experimental_weight_only_as_drq": True,
        },
    )
    drq_out = drq_model(*args)
    diff = get_diff(drq_out, torch_out)
    self.assertLess(diff["mean_abs_diff"], 1e-1)

    self._ensure_output_structure(
        drq_model,
        target_op_name="FULLY_CONNECTED",
        is_drq=True,
        expected_weight_dtype=np.int8,
        expected_num_scales=16,
    )


if __name__ == "__main__":
  googletest.main()
