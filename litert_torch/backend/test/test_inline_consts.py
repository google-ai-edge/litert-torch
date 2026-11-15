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


class TestTensorFingerprint(googletest.TestCase):
  """The constant-cache key must name the constant's own storage."""

  def test_distinct_non_contiguous_constants_differ(self):
    # The key is taken from the tensor, not from the temporary .contiguous()
    # returns, so it cannot follow a freed temporary's address.
    w_cos, w_sin = _dft_inverse_bases(20)
    self.assertFalse(w_cos.is_contiguous())
    self.assertNotEqual(
        inline_consts._ConstantFingerprint.from_tensor(w_cos),
        inline_consts._ConstantFingerprint.from_tensor(w_sin),
    )

  def test_same_view_matches(self):
    a = torch.arange(220, dtype=torch.float32).reshape(20, 1, 11)
    self.assertEqual(
        inline_consts._ConstantFingerprint.from_tensor(a.transpose(0, 2)),
        inline_consts._ConstantFingerprint.from_tensor(a.transpose(0, 2)),
    )

  def test_other_dtype_view_differs(self):
    a = torch.arange(220, dtype=torch.float32)
    self.assertNotEqual(
        inline_consts._ConstantFingerprint.from_tensor(a),
        inline_consts._ConstantFingerprint.from_tensor(a.view(torch.int32)),
    )


class _TwoConvT(torch.nn.Module):

  def __init__(self, w1, w2, stride):
    super().__init__()
    self.register_buffer("w1", w1)
    self.register_buffer("w2", w2)
    self.stride = stride

  def forward(self, a, b):
    return F.conv_transpose1d(
        a, self.w1, stride=self.stride
    ) - F.conv_transpose1d(b, self.w2, stride=self.stride)


def _kernels(lowered):
  """Every [11, 1, 20] dense constant materialized into the module."""
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
  return kernels


class TestInlineConstsLowering(googletest.TestCase):

  def test_distinct_conv_transpose_weights_not_aliased(self):
    """Distinct conv_transpose1d weights must lower to distinct constants."""
    w_cos, w_sin = _dft_inverse_bases(20)
    # #1061 needs non-contiguous constants: .contiguous() then copies each one
    # to a temporary, and the second temporary can reuse the first's address.
    self.assertFalse(w_cos.is_contiguous())
    self.assertFalse(w_sin.is_contiguous())
    self.assertFalse(torch.equal(w_cos, w_sin))

    a = torch.randn(1, 11, 200)
    b = torch.randn(1, 11, 200)
    ep = torch.export.export(_TwoConvT(w_cos, w_sin, 5).eval(), (a, b))
    kernels = _kernels(backend.export.exported_program_to_mlir(ep))

    self.assertLen(kernels, 2)
    # The two kernels must remain distinct (the bug made them byte-identical).
    self.assertFalse(np.array_equal(kernels[0], kernels[1]))

  def test_negated_views_not_aliased(self):
    """Views differing only in the negative bit lower to opposite constants."""
    z = torch.exp(1j * torch.linspace(0.1, 3.0, 220)).reshape(11, 1, 20)
    real = torch.arange(1.0, 221.0).reshape(11, 1, 20)
    a = torch.randn(1, 11, 200)
    b = torch.randn(1, 11, 200)
    # Non-contiguous, as in a complex DFT basis, and contiguous, which would
    # reach numpy() with the negative bit still set.
    for negated, plain in ((z.conj().imag, z.imag), (real._neg_view(), real)):
      with self.subTest(contiguous=negated.is_contiguous()):
        self.assertTrue(negated.is_neg())
        self.assertEqual(negated.data_ptr(), plain.data_ptr())
        self.assertEqual(negated.stride(), plain.stride())

        ep = torch.export.export(_TwoConvT(negated, plain, 5).eval(), (a, b))
        kernels = _kernels(backend.export.exported_program_to_mlir(ep))

        self.assertLen(kernels, 2)
        np.testing.assert_array_equal(kernels[0], -kernels[1])

  def test_cache_keeps_constants_alive(self):
    # The key holds a storage address, so the cache keeps each constant
    # alive: a freed constant's address could otherwise go to a later, distinct
    # constant, which would then hit the cache and get the first value.
    values = []
    with (
        backend.export_utils.create_ir_context() as context,
        ir.Location.unknown(),
    ):
      module = ir.Module.create()
      lctx = backend.export.LoweringContext(context, module)
      lctx.add_plugin(inline_consts.InlineConstsContext())
      with ir.InsertionPoint(module.body):
        for basis in _dft_inverse_bases(20):
          const = torch.empty_strided(basis.shape, basis.stride())
          const.copy_(basis)
          value = inline_consts.tensor_lowering_placeholder_lowering(
              lctx, const
          )
          values.append(
              np.array(ir.DenseElementsAttr(value.owner.attributes["value"]))
          )
          del const

    self.assertFalse(np.array_equal(values[0], values[1]))


class TestCacheAcrossPrograms(googletest.TestCase):
  """One InlineConstsContext serves all programs of a multi-signature model."""

  def _lower(self, ctx, ir_context, module, args):
    ep = torch.export.export(module.eval(), args)
    return backend.export.exported_program_to_mlir(
        ep, ir_context=ir_context, lowering_context_plugins=[ctx]
    )

  def test_shared_weights_lowered_once(self):
    # Two programs over one module (as prefill and decode are) share its
    # weights through the cache.
    ctx = inline_consts.InlineConstsContext()
    ir_context = backend.export_utils.create_ir_context()
    w_cos, w_sin = _dft_inverse_bases(20)
    module = _TwoConvT(w_cos, w_sin, 5)
    self._lower(ctx, ir_context, module, (torch.randn(1, 11, 200),) * 2)
    cached = len(ctx.constant_cache)
    self._lower(ctx, ir_context, module, (torch.randn(1, 11, 40),) * 2)

    self.assertLen(ctx.constant_cache, cached)


if __name__ == "__main__":
  googletest.main()
