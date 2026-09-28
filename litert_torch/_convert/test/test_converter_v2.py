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

import litert_torch
from litert_torch import converter_v2
from litert_torch._convert import converter_v2 as converter_v2_lib
from litert_torch.backend import export as export_lib
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
        registry.unique_params.append(("pid", p, "lifted_tensor_0"))
        registry.id_to_offset["pid"] = 0
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


if __name__ == "__main__":
  googletest.main()
