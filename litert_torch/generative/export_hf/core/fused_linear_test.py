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
from collections.abc import Mapping

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.core import fused_linear
import torch


class _Scaled(torch.nn.Module):
  """A toy layer type that fuses itself when all scales match."""

  def __init__(self, in_features: int, out_features: int, scale: float):
    super().__init__()
    self.scale = scale
    self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, self.weight) * self.scale

  @classmethod
  def fuse_linear_modules(
      cls, parts: Mapping[str, torch.nn.Module]
  ) -> torch.nn.Module:
    scales = {p.scale for p in parts.values()}
    if len(scales) != 1:
      raise fused_linear.FusionNotSupportedError(
          f"different scales {sorted(scales)}"
      )
    first = next(iter(parts.values()))
    fused = cls(first.weight.shape[1], 1, first.scale)
    fused.weight = torch.nn.Parameter(
        torch.cat([p.weight for p in parts.values()])
    )
    return fused


class _NotFusible(torch.nn.Linear):
  pass


class FusedLinearTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_fuses_float_linears(self, bias):
    parts = {
        "q_proj": torch.nn.Linear(8, 4, bias=bias),
        "k_proj": torch.nn.Linear(8, 2, bias=bias),
        "v_proj": torch.nn.Linear(8, 2, bias=bias),
    }
    fused = fused_linear.fuse_linear_modules(
        parts, fused_name="qkv_proj", disable_flags="--fuse_qkv=False"
    )
    self.assertIsInstance(fused, torch.nn.Linear)
    x = torch.randn(3, 8)
    expected = torch.cat([p(x) for p in parts.values()], dim=-1)
    torch.testing.assert_close(fused(x), expected)

  def test_fuses_meta_linears_without_copy(self):
    with torch.device("meta"):
      parts = {
          "gate_proj": torch.nn.Linear(8, 16),
          "up_proj": torch.nn.Linear(8, 16),
      }
    fused = fused_linear.fuse_linear_modules(
        parts, fused_name="gate_up_proj", disable_flags="--fuse_gate_up=False"
    )
    self.assertTrue(fused.weight.is_meta)
    self.assertEqual(tuple(fused.weight.shape), (32, 8))

  def test_subclass_of_linear_uses_float_fusion(self):
    parts = {"gate_proj": _NotFusible(8, 4), "up_proj": _NotFusible(8, 4)}
    fused = fused_linear.fuse_linear_modules(
        parts, fused_name="gate_up_proj", disable_flags="--fuse_gate_up=False"
    )
    self.assertEqual(tuple(fused.weight.shape), (8, 8))

  def test_rejects_integer_linear_weights(self):
    q = torch.nn.Linear(8, 4, bias=False)
    q.weight = torch.nn.Parameter(
        torch.zeros(4, 8, dtype=torch.int8), requires_grad=False
    )
    with self.assertRaisesRegex(
        fused_linear.FusionNotSupportedError,
        r"(?s)`q_proj.weight` has non-floating-point dtype torch.int8.*"
        r"disable it with --fuse_qkv=False",
    ):
      fused_linear.fuse_linear_modules(
          {"q_proj": q, "k_proj": torch.nn.Linear(8, 4, bias=False)},
          fused_name="qkv_proj",
          disable_flags="--fuse_qkv=False",
      )

  def test_rejects_mismatched_bias(self):
    with self.assertRaisesRegex(
        fused_linear.FusionNotSupportedError, "bias"
    ):
      fused_linear.fuse_linear_modules(
          {
              "gate_proj": torch.nn.Linear(8, 4, bias=True),
              "up_proj": torch.nn.Linear(8, 4, bias=False),
          },
          fused_name="gate_up_proj",
          disable_flags="--fuse_gate_up=False",
      )

  def test_rejects_mismatched_in_features(self):
    with self.assertRaisesRegex(
        fused_linear.FusionNotSupportedError, "in_features=6, expected 8"
    ):
      fused_linear.fuse_linear_modules(
          {
              "gate_proj": torch.nn.Linear(8, 4),
              "up_proj": torch.nn.Linear(6, 4),
          },
          fused_name="gate_up_proj",
          disable_flags="--fuse_gate_up=False",
      )

  def test_delegates_to_fusible_layer_type(self):
    parts = {"a": _Scaled(8, 4, 2.0), "b": _Scaled(8, 3, 2.0)}
    fused = fused_linear.fuse_linear_modules(
        parts, fused_name="ab", disable_flags="--fuse=False"
    )
    x = torch.randn(2, 8)
    expected = torch.cat([p(x) for p in parts.values()], dim=-1)
    torch.testing.assert_close(fused(x), expected)

  def test_wraps_fusible_layer_error_with_flags(self):
    with self.assertRaisesRegex(
        fused_linear.FusionNotSupportedError,
        r"Cannot fuse \['a', 'b'\] \(.*_Scaled\) into `ab`: different scales"
        r" \[1.0, 2.0\]\. .*disable it with --fuse=False\.",
    ):
      fused_linear.fuse_linear_modules(
          {"a": _Scaled(8, 4, 1.0), "b": _Scaled(8, 4, 2.0)},
          fused_name="ab",
          disable_flags="--fuse=False",
      )

  def test_rejects_layer_type_without_fusion_support(self):
    with self.assertRaisesRegex(
        fused_linear.FusionNotSupportedError,
        r"(?s)don't implement `fuse_linear_modules`.*"
        + fused_linear.FUSE_GATE_UP_FLAGS,
    ):
      fused_linear.fuse_linear_modules(
          {"gate_proj": torch.nn.Identity(), "up_proj": torch.nn.Identity()},
          fused_name="gate_up_proj",
          disable_flags=fused_linear.FUSE_GATE_UP_FLAGS,
      )

  def test_rejects_mixed_layer_types(self):
    with self.assertRaisesRegex(
        fused_linear.FusionNotSupportedError, "fuse_linear_modules"
    ):
      fused_linear.fuse_linear_modules(
          {"a": _Scaled(8, 4, 1.0), "b": torch.nn.Linear(8, 4)},
          fused_name="ab",
          disable_flags="--fuse=False",
      )

  def test_check_float_linears(self):
    fused_linear.check_float_linears(
        {"q_proj": torch.nn.Linear(8, 4), "k_proj": torch.nn.Linear(8, 4)},
        fused_name="qkv_proj",
        disable_flags="--fuse_qkv=False",
    )
    with self.assertRaisesRegex(
        fused_linear.FusionNotSupportedError,
        r"(?s)`k_proj` is a _Scaled.*--fuse_qkv=False",
    ):
      fused_linear.check_float_linears(
          {"q_proj": torch.nn.Linear(8, 4), "k_proj": _Scaled(8, 4, 1.0)},
          fused_name="qkv_proj",
          disable_flags="--fuse_qkv=False",
      )


if __name__ == "__main__":
  absltest.main()
