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
"""Tests for the inline-constants pre-lower pass.

Regression coverage for https://github.com/google-ai-edge/litert-torch/issues/1061:
two distinct constants must never be merged by the constant cache, even when one
constant's storage is freed and its address reused by another.
"""

import numpy as np
import torch
import torch.nn.functional as F

from litert_torch import backend
from litert_torch.backend import inline_consts
from litert_converter.mlir import ir

from absl.testing import absltest as googletest


class TestTensorFingerprint(googletest.TestCase):
  """The constant-cache key must depend on content, not on a storage address."""

  def test_recycled_storage_with_different_content_differs(self):
    # Same storage / same data_ptr(), different content: the fingerprint must
    # differ. The old data_ptr()-based key returned the same value here, which
    # is exactly how an address reused after free aliased two constants.
    t = torch.zeros(11, 1, 20)
    fp_zeros = inline_consts._ConstantFingerprint.from_tensor(t)
    t.fill_(1.0)
    fp_ones = inline_consts._ConstantFingerprint.from_tensor(t)
    self.assertNotEqual(fp_zeros, fp_ones)

  def test_equal_content_distinct_storage_matches(self):
    # Distinct storage, identical content: the fingerprint must match so the
    # cache still deduplicates genuinely identical constants.
    a = torch.arange(220, dtype=torch.float32).reshape(11, 1, 20)
    b = a.clone()
    self.assertNotEqual(a.untyped_storage().data_ptr(),
                        b.untyped_storage().data_ptr())
    self.assertEqual(
        inline_consts._ConstantFingerprint.from_tensor(a),
        inline_consts._ConstantFingerprint.from_tensor(b),
    )

  def test_different_content_same_shape_differs(self):
    torch.manual_seed(0)
    a = torch.randn(11, 1, 20)
    b = torch.randn(11, 1, 20)
    self.assertNotEqual(
        inline_consts._ConstantFingerprint.from_tensor(a),
        inline_consts._ConstantFingerprint.from_tensor(b),
    )


def _dft_inverse_bases(n_fft=20):
  """The cos/sin inverse-DFT bases, a normal real-valued inverse-STFT pair."""
  fb = n_fft // 2 + 1
  win = torch.hann_window(n_fft, periodic=True, dtype=torch.float32).numpy()
  n, k = np.arange(n_fft), np.arange(fb)
  ang = 2 * np.pi * np.outer(n, k) / n_fft
  inv_win = win * (1.0 / n_fft)
  w_cos = torch.from_numpy((np.cos(ang).T * inv_win)).float().unsqueeze(1)
  w_sin = torch.from_numpy((np.sin(ang).T * inv_win)).float().unsqueeze(1)
  return w_cos, w_sin


class TestInlineConstsLowering(googletest.TestCase):

  def test_distinct_conv_transpose_weights_not_aliased(self):
    """Two distinct conv_transpose1d weights must lower to distinct constants."""
    w_cos, w_sin = _dft_inverse_bases(20)
    self.assertFalse(torch.equal(w_cos, w_sin))

    class TwoConvT(torch.nn.Module):

      def __init__(self, w1, w2, stride):
        super().__init__()
        self.register_buffer("w1", w1)
        self.register_buffer("w2", w2)
        self.stride = stride

      def forward(self, a, b):
        return F.conv_transpose1d(
            a, self.w1, stride=self.stride
        ) - F.conv_transpose1d(b, self.w2, stride=self.stride)

    a = torch.randn(1, 11, 200)
    b = torch.randn(1, 11, 200)
    ep = torch.export.export(TwoConvT(w_cos, w_sin, 5).eval(), (a, b))
    lowered = backend.export.exported_program_to_mlir(ep)

    # Collect every [11, 1, 20] dense constant materialized into the module.
    kernels = []

    def _walk(op):
      for region in op.regions:
        for block in region.blocks:
          for inner in block.operations:
            if inner.operation.name in ("arith.constant", "stablehlo.constant"):
              if "11x1x20" in str(inner.results[0].type):
                kernels.append(
                    np.array(ir.DenseElementsAttr(inner.attributes["value"]))
                )
            _walk(inner)

    _walk(lowered.module.operation)

    self.assertLen(kernels, 2)
    # The two kernels must remain distinct (the bug made them byte-identical).
    self.assertFalse(np.array_equal(kernels[0], kernels[1]))


if __name__ == "__main__":
  googletest.main()
