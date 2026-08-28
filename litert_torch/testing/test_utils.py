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
"""Test utilities for LiteRT Torch tests, including V1/V2 parameterization."""

import litert_torch
from parameterized.parameterized import parameterized_class  # pylint: disable=unused-import
from absl.testing import absltest as googletest

V1_V2_PARAMETERS = [
    {"use_v2": False},
    {"use_v2": True},
]

_ORIG_CONVERT = litert_torch.convert
_ORIG_SIGNATURE = litert_torch.signature
_ORIG_CONVERTER_CLS = litert_torch.Converter


class V1V2TestCase(googletest.TestCase):
  """Base TestCase for conversion tests that parameterizes use_v2."""

  def setUp(self):
    super().setUp()
    self.addCleanup(setattr, litert_torch, "convert", _ORIG_CONVERT)
    self.addCleanup(setattr, litert_torch, "signature", _ORIG_SIGNATURE)
    self.addCleanup(setattr, litert_torch, "Converter", _ORIG_CONVERTER_CLS)

    use_v2 = getattr(self, "use_v2", getattr(self, "_use_v2", False))

    def patched_convert(*args, **kwargs):
      kwargs.setdefault("use_v2", use_v2)
      return _ORIG_CONVERT(*args, **kwargs)

    def patched_signature(*args, **kwargs):
      conv = _ORIG_SIGNATURE(*args, **kwargs)
      orig_convert = conv.convert

      def patched_conv_convert(*c_args, **c_kwargs):
        c_kwargs.setdefault("use_v2", use_v2)
        return orig_convert(*c_args, **c_kwargs)

      conv.convert = patched_conv_convert
      return conv

    class PatchedConverter(_ORIG_CONVERTER_CLS):

      def convert(self, *args, **kwargs):
        kwargs.setdefault("use_v2", use_v2)
        return super().convert(*args, **kwargs)

    litert_torch.convert = patched_convert
    litert_torch.signature = patched_signature
    litert_torch.Converter = PatchedConverter

  def tearDown(self):
    litert_torch.convert = _ORIG_CONVERT
    litert_torch.signature = _ORIG_SIGNATURE
    litert_torch.Converter = _ORIG_CONVERTER_CLS
    super().tearDown()
