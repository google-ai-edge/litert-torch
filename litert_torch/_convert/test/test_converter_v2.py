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
"""Tests for LiteRT Torch Converter V2 bridge."""

import json
import os
import types
from unittest import mock

import litert_torch
from litert_torch import converter_v2
from litert_torch._convert import converter_v2 as converter_v2_lib
from litert_torch._convert.fx_passes import lower_torchao_pass
from litert_torch.backend import export as export_lib
from litert_torch.backend import export_utils
from litert_torch.generative.layers import moe
from litert_converter.mlir.dialects import stablehlo
import numpy as np
import torch
from torch import nn

from absl.testing import absltest as googletest


class SimpleLinearModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(4, 2)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(self.fc(x))


class MultiInputModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(3, 3))

  def forward(self, x, y):
    return (x - y) @ self.weight


class SharedWeightsModel1(nn.Module):

  def __init__(self, shared_weight, head_weight):
    super().__init__()
    self.shared = shared_weight
    self.head = head_weight

  def forward(self, x):
    return x @ self.shared + self.head


class SharedWeightsModel2(nn.Module):

  def __init__(self, shared_weight, head_weight):
    super().__init__()
    self.shared = shared_weight
    self.head = head_weight

  def forward(self, x):
    return (x @ self.shared) * self.head


class TestConverterV2(googletest.TestCase):
  """Unit tests for LiteRT Torch Converter V2 bridge."""

  def setUp(self):
    super().setUp()
    if not converter_v2_lib.is_supported():
      self.skipTest(
          "Converter V2 is not supported by the installed TensorFlow."
      )
    torch.manual_seed(42)

  def test_single_signature_conversion(self):
    """Test basic module conversion via use_v2=True."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)

    edge_model = litert_torch.convert(m, sample_input, use_v2=True)
    self.assertIsNotNone(edge_model)

    res = edge_model(sample_input[0].numpy())
    expected = m(*sample_input).detach().numpy()
    np.testing.assert_allclose(res, expected, rtol=1e-3, atol=1e-3)

  def test_multi_input_conversion(self):
    """Test multi-input conversion via use_v2=True."""
    m = MultiInputModel().eval()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    edge_model = litert_torch.convert(m, (x, y), use_v2=True)
    self.assertIsNotNone(edge_model)

    res = edge_model(x.numpy(), y.numpy())
    expected = m(x, y).detach().numpy()
    np.testing.assert_allclose(res, expected, rtol=1e-3, atol=1e-3)

  def test_export_dir_artifacts(self):
    """Test that export_dir creates valid .mlirbc, params.bin, and weights_metadata.json."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)
    export_dir = self.create_tempdir().full_path

    edge_model = litert_torch.convert(
        m, sample_input, use_v2=True, export_dir=export_dir
    )
    self.assertIsNotNone(edge_model)

    # Verify intermediate artifacts exist
    mlirbc_path = os.path.join(export_dir, "serving_default.mlirbc")
    params_path = os.path.join(export_dir, "params.bin")
    meta_path = os.path.join(export_dir, "weights_metadata.json")

    self.assertTrue(os.path.exists(mlirbc_path))
    self.assertTrue(os.path.exists(params_path))
    self.assertTrue(os.path.exists(meta_path))

    # Verify .mlirbc is a versioned VHLO portable artifact rather than raw
    # unversioned StableHLO bytecode.
    with open(mlirbc_path, "rb") as f:
      mlirbc_bytes = f.read()
    self.assertTrue(mlirbc_bytes.startswith(b"ML\xefR"))
    self.assertIn(b"StableHLO_v", mlirbc_bytes[:64])
    deserialized = stablehlo.deserialize_portable_artifact(
        export_utils.create_ir_context(), mlirbc_bytes
    )
    self.assertIn("stablehlo.dot_general", str(deserialized))

    with open(meta_path, "r") as f:
      meta = json.load(f)
    self.assertIn("signatures", meta)
    self.assertIn("serving_default", meta["signatures"])
    self.assertIn("signature_inputs", meta)
    self.assertIn("signature_outputs", meta)

    # Check that offsets and alignment in metadata are 64-byte aligned
    for entry in meta["signatures"]["serving_default"]:
      self.assertEqual(entry["offset"] % 64, 0)

    # Test output execution
    res = edge_model(sample_input[0].numpy())
    expected = m(*sample_input).detach().numpy()
    np.testing.assert_allclose(res, expected, rtol=1e-3, atol=1e-3)

  def test_multi_signature_shared_weights(self):
    """Test multi-signature conversion and weight deduplication in params.bin."""
    shared = nn.Parameter(torch.randn(4, 4))
    head1 = nn.Parameter(torch.randn(4))
    head2 = nn.Parameter(torch.randn(4))

    m1 = SharedWeightsModel1(shared, head1).eval()
    m2 = SharedWeightsModel2(shared, head2).eval()

    sample_input = (torch.randn(1, 4),)
    export_dir = self.create_tempdir().full_path

    edge_model = (
        litert_torch.signature("sig1", m1, sample_input)
        .add_signature("sig2", m2, sample_input)
        .convert(use_v2=True, export_dir=export_dir)
    )

    self.assertTrue(os.path.exists(os.path.join(export_dir, "sig1.mlirbc")))
    self.assertTrue(os.path.exists(os.path.join(export_dir, "sig2.mlirbc")))
    self.assertTrue(os.path.exists(os.path.join(export_dir, "params.bin")))
    self.assertTrue(
        os.path.exists(os.path.join(export_dir, "weights_metadata.json"))
    )

    with open(os.path.join(export_dir, "weights_metadata.json"), "r") as f:
      meta = json.load(f)

    # Verify both signatures exist in metadata
    self.assertIn("sig1", meta["signatures"])
    self.assertIn("sig2", meta["signatures"])

    # Verify shared weight has the same offset in both signatures
    sig1_offsets = {
        e["arg_index"]: e["offset"] for e in meta["signatures"]["sig1"]
    }
    sig2_offsets = {
        e["arg_index"]: e["offset"] for e in meta["signatures"]["sig2"]
    }
    common_offsets = set(sig1_offsets.values()).intersection(
        set(sig2_offsets.values())
    )
    self.assertNotEmpty(common_offsets)

    # Verify execution of both signatures
    res1 = edge_model(sample_input[0].numpy(), signature_name="sig1")
    expected1 = m1(*sample_input).detach().numpy()
    np.testing.assert_allclose(res1, expected1, rtol=1e-3, atol=1e-3)

    res2 = edge_model(sample_input[0].numpy(), signature_name="sig2")
    expected2 = m2(*sample_input).detach().numpy()
    np.testing.assert_allclose(res2, expected2, rtol=1e-3, atol=1e-3)

  def test_output_file_path(self):
    """Test output_file_path direct saving."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)
    out_file = self.create_tempfile("custom_model.tflite").full_path

    edge_model = litert_torch.convert(
        m, sample_input, use_v2=True, output_file_path=out_file
    )
    self.assertIsNotNone(edge_model)
    self.assertTrue(os.path.exists(out_file))
    self.assertGreater(os.path.getsize(out_file), 0)

  def test_standalone_converter_v2_api(self):
    """Test converter_v2 standalone API."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)

    edge_model = converter_v2.convert(m, sample_input)
    self.assertIsNotNone(edge_model)

    res = edge_model(sample_input[0].numpy())
    expected = m(*sample_input).detach().numpy()
    np.testing.assert_allclose(res, expected, rtol=1e-3, atol=1e-3)

  def test_standalone_converter_v2_fluent_signature_api(self):
    """Test the converter_v2.signature()/add_signature() fluent API."""
    m1 = SimpleLinearModel().eval()
    m2 = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)

    edge_model = (
        converter_v2.signature("sig1", m1, sample_input)
        .add_signature("sig2", m2, sample_input)
        .convert()
    )
    self.assertIsNotNone(edge_model)

    res1 = edge_model(sample_input[0].numpy(), signature_name="sig1")
    np.testing.assert_allclose(
        res1, m1(*sample_input).detach().numpy(), rtol=1e-3, atol=1e-3
    )

    res2 = edge_model(sample_input[0].numpy(), signature_name="sig2")
    np.testing.assert_allclose(
        res2, m2(*sample_input).detach().numpy(), rtol=1e-3, atol=1e-3
    )

  def test_standalone_converter_v2_duplicate_signature_name_raises(self):
    """Test that adding two signatures with the same name is rejected."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)

    converter = converter_v2.signature("sig1", m, sample_input)
    with self.assertRaises(ValueError):
      converter.add_signature("sig1", m, sample_input)

  def test_delete_in_memory_params(self):
    """Test conversion with delete_in_memory_params=True."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)

    edge_model = litert_torch.convert(
        m, sample_input, use_v2=True, delete_in_memory_params=True
    )
    self.assertIsNotNone(edge_model)
    self.assertEqual(m.fc.weight.untyped_storage().size(), 0)
    self.assertEqual(m.fc.bias.untyped_storage().size(), 0)
    res = edge_model(sample_input[0].numpy())
    self.assertIsNotNone(res)

  def test_reuse_intermediates_success(self):
    """Test successful reuse of existing complete intermediate artifacts."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)
    export_dir = self.create_tempdir().full_path

    # First export
    litert_torch.convert(m, sample_input, use_v2=True, export_dir=export_dir)

    # Re-conversion with reuse allowed should succeed without error
    edge_model = litert_torch.convert(
        m,
        sample_input,
        use_v2=True,
        export_dir=export_dir,
        allow_reuse_intermediates=True,
    )
    self.assertIsNotNone(edge_model)

  def test_existing_intermediates_raises_without_reuse_flag(self):
    """Test that existing intermediates raise ValueError if allow_reuse_intermediates=False."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)
    export_dir = self.create_tempdir().full_path

    litert_torch.convert(m, sample_input, use_v2=True, export_dir=export_dir)

    with self.assertRaises(ValueError):
      litert_torch.convert(
          m,
          sample_input,
          use_v2=True,
          export_dir=export_dir,
          allow_reuse_intermediates=False,
      )

  def test_weights_loader_shape_mismatch_raises(self):
    """A loader tensor with matching numel but wrong shape is rejected."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)
    # fc.weight is (2, 4); a transposed (4, 2) tensor has the same byte size.
    weights = {
        "fc.weight": torch.randn(4, 2),
        "fc.bias": torch.randn(2),
    }
    with self.assertRaisesRegex(ValueError, "fc.weight"):
      converter_v2.convert(m, sample_input, weights_loader=weights)

  def test_meta_lifted_constant_missing_from_loader_raises(self):
    """Meta lifted constants absent from the loader raise, not zero-fill."""
    for loader in ({}, lambda name: {}[name]):
      with self.subTest(loader_type=type(loader).__name__):
        registry = converter_v2_lib.ParameterRegistry()
        p = torch.empty(3, device="meta")
        registry.unique_params.append((("pid", 32), p, "lifted_tensor_0"))
        registry.id_to_offset[("pid", 32)] = 0
        out_dir = self.create_tempdir().full_path
        with self.assertRaisesRegex(KeyError, "lifted constant"):
          registry.save(
              os.path.join(out_dir, "params.bin"),
              os.path.join(out_dir, "weights_metadata.json"),
              weights_loader=loader,
          )

  def test_register_signature_missing_state_dict_tensor_raises(self):
    """A non-user input without a backing tensor must not be silently dropped."""
    lowered = types.SimpleNamespace(
        state_dict={},
        input_signature=[
            export_lib.VariableSignature(
                [2], "f32", input_spec=export_lib.InputSpec.parameter("w")
            ),
        ],
        output_signature=[],
    )
    registry = converter_v2_lib.ParameterRegistry()
    with self.assertRaisesRegex(ValueError, "'w'"):
      registry.register_signature("sig", lowered)

  def _fake_lowered(self, module):
    """Builds a minimal MlirLowered stand-in exposing `module`'s params."""
    state_dict = dict(module.named_parameters())
    return types.SimpleNamespace(
        state_dict=state_dict,
        input_signature=[
            export_lib.VariableSignature(
                list(t.shape),
                "f32",
                input_spec=export_lib.InputSpec.parameter(n),
            )
            for n, t in state_dict.items()
        ],
        output_signature=[],
    )

  def test_meta_distinct_params_same_fqn_not_deduped(self):
    """Distinct meta params sharing a relative FQN get separate offsets."""
    with torch.device("meta"):
      m1 = SimpleLinearModel()
      m2 = SimpleLinearModel()
    registry = converter_v2_lib.ParameterRegistry()
    registry.register_signature("sig1", self._fake_lowered(m1), module=m1)
    registry.register_signature("sig2", self._fake_lowered(m2), module=m2)
    offsets1 = {e["offset"] for e in registry.metadata["signatures"]["sig1"]}
    offsets2 = {e["offset"] for e in registry.metadata["signatures"]["sig2"]}
    self.assertEmpty(offsets1 & offsets2)
    self.assertLen(registry.unique_params, 4)

  def test_meta_shared_params_across_wrappers_deduped(self):
    """Meta params shared via distinct wrapper roots share one offset."""
    with torch.device("meta"):
      inner = SimpleLinearModel()
    w1 = nn.Sequential(inner)
    w2 = nn.Sequential(inner)
    registry = converter_v2_lib.ParameterRegistry()
    # Simulate export producing fresh meta tensors (identity not preserved).
    lowered1 = self._fake_lowered(w1)
    lowered2 = self._fake_lowered(w2)
    lowered2.state_dict = {
        n: torch.empty_like(t) for n, t in lowered2.state_dict.items()
    }
    registry.register_signature("sig1", lowered1, module=w1)
    registry.register_signature("sig2", lowered2, module=w2)
    self.assertEqual(
        registry.metadata["signatures"]["sig1"],
        registry.metadata["signatures"]["sig2"],
    )
    self.assertLen(registry.unique_params, 2)

  def test_weights_loader_ambiguous_fqn_across_signatures_raises(self):
    """One loader key feeding distinct params in two signatures is rejected."""
    m1 = SimpleLinearModel().eval()
    m2 = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)
    with self.assertRaisesRegex(ValueError, "ambiguous"):
      (
          converter_v2.signature("sig1", m1, sample_input)
          .add_signature("sig2", m2, sample_input)
          .convert(weights_loader=dict(m1.state_dict()))
      )

  def test_meta_shared_model_wrappers_convert_from_loader(self):
    """Meta model shared by two wrapper signatures converts from a loader."""

    class Passthrough(nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, x):
        return self.m(x)

    class Double(nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, x):
        return self.m(x) * 2

    ref = SimpleLinearModel().eval()
    with torch.device("meta"):
      inner = SimpleLinearModel().eval()
    weights = {f"m.{k}": v for k, v in ref.state_dict().items()}
    sample_input = (torch.empty(1, 4, device="meta"),)
    export_dir = self.create_tempdir().full_path

    edge_model = (
        converter_v2.signature("sig1", Passthrough(inner), sample_input)
        .add_signature("sig2", Double(inner), sample_input)
        .convert(export_dir=export_dir, weights_loader=weights)
    )

    with open(os.path.join(export_dir, "weights_metadata.json"), "r") as f:
      meta = json.load(f)
    self.assertEqual(
        {e["offset"] for e in meta["signatures"]["sig1"]},
        {e["offset"] for e in meta["signatures"]["sig2"]},
    )
    x = torch.randn(1, 4)
    np.testing.assert_allclose(
        edge_model(x.numpy(), signature_name="sig1"),
        ref(x).detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        edge_model(x.numpy(), signature_name="sig2"),
        (ref(x) * 2).detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_unused_submodule_params_excluded_from_params_bin(self):
    """Unused parameters on a wrapped root model are not serialized."""

    class TwoTowerModel(nn.Module):

      def __init__(self):
        super().__init__()
        self.encoder = SimpleLinearModel()
        self.unused_tower = nn.Linear(64, 64)

    class EncoderWrapper(nn.Module):

      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, x):
        return self.m.encoder(x)

    ref = TwoTowerModel().eval()
    with torch.device("meta"):
      meta_model = TwoTowerModel().eval()
    # Only provide encoder weights; querying unused_tower would raise KeyError.
    weights = {f"m.encoder.{k}": v for k, v in ref.encoder.state_dict().items()}
    sample_input = (torch.empty(1, 4, device="meta"),)
    export_dir = self.create_tempdir().full_path

    edge_model = converter_v2.convert(
        EncoderWrapper(meta_model),
        sample_input,
        export_dir=export_dir,
        weights_loader=weights,
    )

    with open(os.path.join(export_dir, "weights_metadata.json"), "r") as f:
      meta = json.load(f)
    self.assertLen(meta["signatures"]["serving_default"], 2)
    x = torch.randn(1, 4)
    np.testing.assert_allclose(
        edge_model(x.numpy()),
        ref.encoder(x).detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )

  def test_prune_unused_parameters_keeps_mutated_buffers(self):
    """_prune_unused_parameters drops unused params/buffers and keeps mutated buffers."""

    class StatefulWithUnusedModule(nn.Module):
      """Module with unused params/buffers and a mutated buffer."""

      def __init__(self):
        super().__init__()
        self.used = nn.Linear(4, 2, bias=False)
        self.unused = nn.Linear(8, 8)
        self.dead = nn.Linear(4, 2, bias=False)
        self.register_buffer("unused_buf", torch.ones(4))
        self.register_buffer("mutated_buf", torch.zeros(1, 2))

      def forward(self, x):
        _ = self.dead(x) + 1.0
        out = self.used(x)
        self.mutated_buf.copy_(out)
        return out

    m = StatefulWithUnusedModule().eval()
    ep = torch.export.export(m, (torch.randn(1, 4),), strict=False)
    self.assertIn("unused.weight", ep.state_dict)
    self.assertIn("dead.weight", ep.state_dict)
    self.assertIn("unused_buf", ep.state_dict)

    converter_v2_lib._prune_unused_parameters(ep)  # pylint: disable=protected-access
    ep.validate()
    self.assertNotIn("unused.weight", ep.state_dict)
    self.assertNotIn("unused.bias", ep.state_dict)
    self.assertNotIn("dead.weight", ep.state_dict)
    self.assertNotIn("unused_buf", ep.state_dict)
    self.assertIn("used.weight", ep.state_dict)
    self.assertIn("mutated_buf", ep.state_dict)
    self.assertEqual(list(ep.graph_signature.parameters), ["used.weight"])
    self.assertEqual(list(ep.graph_signature.buffers), ["mutated_buf"])

  def test_tensor_to_byte_view(self):
    """_tensor_to_byte_view supports contiguous, non-contiguous, and bfloat16 tensors."""
    for dtype in (torch.float32, torch.bfloat16, torch.int8, torch.uint8):
      with self.subTest(dtype=dtype):
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(dtype)
        for candidate in (t, t.t()):
          view = converter_v2_lib._tensor_to_byte_view(candidate)  # pylint: disable=protected-access
          self.assertIsInstance(view, memoryview)
          self.assertEqual(
              view.nbytes, candidate.numel() * candidate.element_size()
          )
          expected = candidate.contiguous().view(torch.uint8).numpy().tobytes()
          self.assertEqual(bytes(view), expected)

  def test_preserves_pytorch_module_hierarchy_in_op_scopes(self):
    """Test that Converter V2 lifts CallSiteLoc caller NameLocs onto Linear ops."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)
    export_dir = self.create_tempdir().full_path
    out_path = os.path.join(export_dir, "model.tflite")

    litert_torch.convert(
        m,
        sample_input,
        use_v2=True,
        export_dir=export_dir,
        output_file_path=out_path,
    )
    with open(out_path, "rb") as f:
      content = f.read()
    self.assertIn(b"torch.nn.modules.linear.Linear_fc;", content)

  def test_is_supported(self):
    self.assertTrue(converter_v2_lib.is_supported())

  def test_unsupported_tensorflow_raises_actionable_error(self):
    with mock.patch.object(
        converter_v2_lib, "is_supported", return_value=False
    ):
      with self.assertRaisesRegex(RuntimeError, "requires a newer TensorFlow"):
        converter_v2_lib._build_tfl_converter_flags()

  def test_pack_subbyte_array(self):
    pack = converter_v2_lib._pack_subbyte_array  # pylint: disable=protected-access
    np.testing.assert_array_equal(
        pack(np.array([-8, 7, -1, 0], dtype=np.int8), 4), [0x78, 0x0F]
    )
    np.testing.assert_array_equal(
        pack(np.array([1, -2, -1, 0, 1], dtype=np.int8), 2), [0x39, 0x01]
    )

  def _check_subbyte_linear(
      self,
      w: torch.Tensor,
      dtype: str,
      zero_point: int,
      expected_params_bytes: int,
      rows: int | None = None,
      check_outputs: bool = True,
      extra_loaders: dict[str, torch.Tensor] | None = None,
  ):
    """Converts `x @ dequantize(w[:rows]).T` and checks params.bin and outputs.

    Args:
      w: The integer weight.
      dtype: The `quant.dequantize` dtype.
      zero_point: The zero point of the dequantize.
      expected_params_bytes: The expected size of params.bin.
      rows: If set, slices the first `rows` rows of `w` before dequantizing.
      check_outputs: Whether to compare the outputs with the eager module.
      extra_loaders: More weights loader values for `weight` to stream from,
        by subtest name (e.g. `w` already packed).
    """

    class SubbyteLinear(nn.Module):

      def __init__(self, weight: torch.Tensor, dq_fn):
        super().__init__()
        self.register_buffer("weight", weight)
        self.dq_fn = dq_fn

      def forward(self, x):
        weight = self.weight if rows is None else self.weight[:rows]
        return torch.nn.functional.linear(x, self.dq_fn(weight))

    out_features = w.shape[0] if rows is None else rows
    dq_fn = lower_torchao_pass._create_dequantize_function(  # pylint: disable=protected-access
        lower_torchao_pass._get_quant_composite_attributes(  # pylint: disable=protected-access
            scale=[0.05] * out_features,
            zero_point=[zero_point],
            dtype=dtype,
            quant_dimension=0,
            narrow_range=dtype == "i4",
        )
    )
    x = torch.randn(2, w.shape[1])
    module = SubbyteLinear(w, dq_fn).eval()
    with torch.no_grad():
      expected = module(x).numpy()
    meta_module = SubbyteLinear(w.to("meta"), dq_fn).eval()
    cases = [
        ("in_memory", module, (x,), None),
        ("streaming", meta_module, (x.to("meta"),), {"weight": w}),
    ]
    for name, value in (extra_loaders or {}).items():
      cases.append((name, meta_module, (x.to("meta"),), {"weight": value}))
    for name, mod, args, loader in cases:
      with self.subTest(mode=name):
        export_dir = self.create_tempdir().full_path
        edge_model = converter_v2.convert(
            mod, args, export_dir=export_dir, weights_loader=loader
        )
        self.assertEqual(
            os.path.getsize(os.path.join(export_dir, "params.bin")),
            expected_params_bytes,
        )
        if check_outputs:
          np.testing.assert_allclose(
              edge_model(x.numpy()), expected, rtol=1e-3, atol=1e-3
          )

  def test_subbyte_quant_weights_are_packed(self):
    """Tests that i4/i2 dequantized weights are bit-packed in params.bin."""
    for dtype, low, high, bit_width in (("i4", -7, 8, 4), ("i2", -2, 2, 2)):
      with self.subTest(dtype=dtype):
        w = torch.randint(low, high, (16, 32), dtype=torch.int8)
        packed = torch.from_numpy(
            converter_v2_lib._pack_subbyte_array(w.numpy(), bit_width)  # pylint: disable=protected-access
        )
        self._check_subbyte_linear(
            w,
            dtype,
            0,
            expected_params_bytes=16 * 32 * bit_width // 8,
            extra_loaders={
                "prepacked": packed.reshape(16, -1),
                "prepacked_flat": packed,
            },
        )

  def test_row_padded_subbyte_weight_is_rejected(self):
    """A uint8 weight with padded rows is not taken as packed."""
    # (2, 35) i2 rows padded to 9 bytes have as many bytes as the packed weight.
    w = torch.randint(-2, 2, (2, 35), dtype=torch.int8)
    padded = np.stack([
        converter_v2_lib._pack_subbyte_array(row, 2)  # pylint: disable=protected-access
        for row in w.numpy()
    ])
    self.assertEqual(padded.shape, (2, 9))
    dq_fn = lower_torchao_pass._create_dequantize_function(  # pylint: disable=protected-access
        lower_torchao_pass._get_quant_composite_attributes(  # pylint: disable=protected-access
            scale=[0.05] * 2,
            zero_point=[0],
            dtype="i2",
            quant_dimension=0,
            narrow_range=False,
        )
    )

    class SubbyteLinear(nn.Module):

      def __init__(self):
        super().__init__()
        self.register_buffer("weight", w.to("meta"))

      def forward(self, x):
        return torch.nn.functional.linear(x, dq_fn(self.weight))

    with self.assertRaisesRegex(ValueError, "weights_loader returned shape"):
      converter_v2.convert(
          SubbyteLinear().eval(),
          (torch.randn(2, 35, device="meta"),),
          export_dir=self.create_tempdir().full_path,
          weights_loader={"weight": torch.from_numpy(padded)},
      )

  def test_ui4_quant_weights_are_packed(self):
    """Tests that ui8 parameters dequantized as ui4 are bit-packed."""
    # Only checks params.bin: ui4 `quant.dequantize` with a nonzero zero point
    # doesn't match the eager reference even for unpacked weights.
    self._check_subbyte_linear(
        torch.randint(0, 16, (16, 32), dtype=torch.uint8),
        "ui4",
        8,
        expected_params_bytes=16 * 32 // 2,
        check_outputs=False,
    )

  def test_small_subbyte_quant_weight_is_not_packed(self):
    """Weights of at most 64 elements stay one byte per element."""
    self._check_subbyte_linear(
        torch.randint(-7, 8, (4, 16), dtype=torch.int8),
        "i4",
        0,
        expected_params_bytes=4 * 16,
    )

  def test_sliced_subbyte_quant_weight_is_not_packed(self):
    """Weights reaching the dequantize through a slice stay one byte each."""
    self._check_subbyte_linear(
        torch.randint(-7, 8, (16, 32), dtype=torch.int8),
        "i4",
        0,
        expected_params_bytes=16 * 32,
        rows=8,
    )

  def test_mixed_dialect_tfl_custom_conversion(self):
    """Modules emitting raw tfl.custom ops serialize and convert via VHLO."""

    class MoeModule(nn.Module):

      def forward(
          self,
          src,
          top_weights,
          top_indices,
          ff_gate,
          ff1,
          linear,
          per_expert_scale,
      ):
        return moe.moe_experts(
            src,
            top_weights,
            top_indices,
            ff_gate,
            ff1,
            linear,
            per_expert_scale,
            num_experts=2,
            num_active_experts=2,
            model_dim=4,
            hidden_dim=6,
        )

    sample_inputs = (
        torch.randn(1, 2, 4),
        torch.softmax(torch.randn(1, 2, 2), dim=-1),
        torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int32),
        moe.flatten_expert_weight(torch.randn(2, 6, 4)),
        moe.flatten_expert_weight(torch.randn(2, 6, 4)),
        moe.flatten_expert_weight(torch.randn(2, 4, 6)),
        torch.ones(1, 1, 1, 2),
    )
    export_dir = self.create_tempdir().full_path

    edge_model = litert_torch.convert(
        MoeModule().eval(),
        sample_inputs,
        use_v2=True,
        export_dir=export_dir,
    )
    self.assertIsNotNone(edge_model)

    mlirbc_path = os.path.join(export_dir, "serving_default.mlirbc")
    with open(mlirbc_path, "rb") as f:
      mlirbc_bytes = f.read()
    self.assertTrue(mlirbc_bytes.startswith(b"ML\xefR"))
    self.assertIn(b"StableHLO_v", mlirbc_bytes[:64])
    deserialized = stablehlo.deserialize_portable_artifact(
        export_utils.create_ir_context(), mlirbc_bytes
    )
    self.assertIn("tfl.custom", str(deserialized))

  def test_serialization_failure_does_not_leave_empty_mlirbc(self):
    """A failed serialization leaves no 0-byte .mlirbc file in export_dir."""
    m = SimpleLinearModel().eval()
    sample_input = (torch.randn(1, 4),)
    export_dir = self.create_tempdir().full_path

    with mock.patch.object(
        converter_v2_lib,
        "_serialize_portable_artifact",
        autospec=True,
        side_effect=ValueError("serialization failed"),
    ):
      with self.assertRaisesRegex(ValueError, "serialization failed"):
        litert_torch.convert(
            m, sample_input, use_v2=True, export_dir=export_dir
        )

    self.assertFalse(
        os.path.exists(os.path.join(export_dir, "serving_default.mlirbc"))
    )
    self.assertFalse(
        converter_v2_lib._has_any_intermediates(export_dir, [])  # pylint: disable=protected-access
    )


if __name__ == "__main__":
  googletest.main()
