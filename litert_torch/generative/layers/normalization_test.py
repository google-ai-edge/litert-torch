# Copyright 2025 The LiteRT Torch Authors.
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
"""Tests for normalization layers."""

from litert_torch.generative.layers import normalization
import torch
from absl.testing import absltest as googletest
from absl.testing import parameterized


class NormalizationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="rms_norm_test_1",
          model_dim=10,
          with_scale=False,
          scale_shift=1.0,
          enable_hlfb=False,
          expected_values=torch.ones((10,), dtype=torch.float32),
      ),
      dict(
          testcase_name="rms_norm_test_2",
          model_dim=10,
          with_scale=True,
          scale_shift=2.0,
          enable_hlfb=False,
          expected_values=torch.ones((10,), dtype=torch.float32) * 2.0,
      ),
      dict(
          testcase_name="rms_norm_test_3",
          model_dim=10,
          with_scale=True,
          scale_shift=2.0,
          enable_hlfb=True,
          expected_values=torch.ones((10,), dtype=torch.float32) * 2.0,
      ),
  )
  def test_rms_norm(
      self,
      model_dim: int,
      with_scale: bool,
      scale_shift: float,
      enable_hlfb: bool,
      expected_values: torch.Tensor,
  ):
    rms_norm = normalization.RMSNorm(
        dim=model_dim,
        with_scale=with_scale,
        scale_shift=scale_shift,
        enable_hlfb=enable_hlfb,
    )

    x = torch.ones((model_dim,), dtype=torch.float32)
    out = rms_norm(x)
    self.assertEqual(out.shape, (model_dim,))
    self.assertTrue(torch.allclose(out, expected_values))

  def test_rms_norm_with_hlfb_final_scale_optional(self):
    """`final_scale` omitted/None must be a true no-op vs. passing all-ones."""
    model_dim = 4
    x = torch.randn((model_dim,), dtype=torch.float32)
    w = torch.randn((model_dim,), dtype=torch.float32)
    eps = 1e-6
    ones_scale = torch.ones((model_dim,), dtype=torch.float32)

    out_omitted = normalization.rms_norm_with_hlfb(x, w, eps)
    out_none = normalization.rms_norm_with_hlfb(x, w, eps, None)
    out_with_ones = normalization.rms_norm_with_hlfb(x, w, eps, ones_scale)

    self.assertTrue(torch.allclose(out_omitted, out_with_ones))
    self.assertTrue(torch.allclose(out_none, out_with_ones))

  @parameterized.named_parameters(
      dict(
          testcase_name="layer_norm_with_bias_no_hlfb",
          use_bias=True,
          enable_hlfb=False,
      ),
      dict(
          testcase_name="layer_norm_with_bias_hlfb",
          use_bias=True,
          enable_hlfb=True,
      ),
      dict(
          testcase_name="layer_norm_no_bias_no_hlfb",
          use_bias=False,
          enable_hlfb=False,
      ),
      dict(
          testcase_name="layer_norm_no_bias_hlfb",
          use_bias=False,
          enable_hlfb=True,
      ),
  )
  def test_layer_norm(self, use_bias: bool, enable_hlfb: bool):
    model_dim = 10
    layer_norm = normalization.LayerNorm(
        dim=model_dim,
        use_bias=use_bias,
        enable_hlfb=enable_hlfb,
    )
    ref_layer_norm = torch.nn.LayerNorm(
        model_dim, elementwise_affine=True, bias=use_bias
    )
    with torch.no_grad():
      ref_layer_norm.weight.uniform_(0.5, 1.5)
      layer_norm.weight.copy_(ref_layer_norm.weight)
      if use_bias:
        ref_layer_norm.bias.uniform_(-0.5, 0.5)
        layer_norm.bias.copy_(ref_layer_norm.bias)

    x = torch.randn((2, 4, model_dim), dtype=torch.float32)
    out = layer_norm(x)
    expected = ref_layer_norm(x)
    self.assertEqual(out.shape, x.shape)
    self.assertTrue(torch.allclose(out, expected, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
  googletest.main()
