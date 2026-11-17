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
"""Tests for `_litert_converter_flags` on the default conversion path."""

import types
from unittest import mock

import litert_torch
from litert_torch._convert import litert_converter
import torch

from absl.testing import absltest as googletest


class _Add(torch.nn.Module):

  def forward(self, x):
    return x + 1.0


class _AddInt(torch.nn.Module):

  def forward(self, x):
    return x + 1


class ApplyConverterFlagsTest(googletest.TestCase):

  def test_sets_fields_the_config_has(self):
    config = types.SimpleNamespace(enable_x64=False, strict_qdq_mode=False)
    ignored = litert_converter._apply_converter_flags(
        config, {"enable_x64": True, "strict_qdq_mode": True}
    )
    self.assertEmpty(ignored)
    self.assertTrue(config.enable_x64)
    self.assertTrue(config.strict_qdq_mode)

  def test_drops_experimental_prefix(self):
    config = types.SimpleNamespace(unsafe_fuse_dynamic_shaped_broadcast=True)
    ignored = litert_converter._apply_converter_flags(
        config, {"_experimental_unsafe_fuse_dynamic_shaped_broadcast": False}
    )
    self.assertEmpty(ignored)
    self.assertFalse(config.unsafe_fuse_dynamic_shaped_broadcast)

  def test_maps_tflite_converter_strict_qdq_name(self):
    # `tf.lite.TFLiteConverter` names the flag `_experimental_strict_qdq`.
    config = types.SimpleNamespace(strict_qdq_mode=False)
    ignored = litert_converter._apply_converter_flags(
        config, {"_experimental_strict_qdq": True}
    )
    self.assertEmpty(ignored)
    self.assertTrue(config.strict_qdq_mode)
    self.assertFalse(hasattr(config, "strict_qdq"))

  def test_returns_flags_the_config_lacks(self):
    config = types.SimpleNamespace(enable_x64=False)
    ignored = litert_converter._apply_converter_flags(
        config,
        {
            "serialize_debug_metadata": True,
            "debug_options": {"enable_timing": True},
            "enable_x64": True,
        },
    )
    self.assertCountEqual(
        ignored, ["serialize_debug_metadata", "debug_options"]
    )
    self.assertTrue(config.enable_x64)
    self.assertFalse(hasattr(config, "serialize_debug_metadata"))


class ConvertWithConverterFlagsTest(googletest.TestCase):

  def test_flag_reaches_the_converter_config(self):
    run_passes = litert_converter.converter_api_ext.run_convert_to_tfl_passes
    with mock.patch.object(
        litert_converter.converter_api_ext,
        "run_convert_to_tfl_passes",
        wraps=run_passes,
    ) as mock_run_passes:
      litert_torch.convert(
          _Add().eval(),
          (torch.randn(2, 3),),
          _litert_converter_flags={"_experimental_strict_qdq": True},
      )
    mock_run_passes.assert_called_once()
    config = mock_run_passes.call_args.args[2]
    self.assertTrue(config.strict_qdq_mode)

  def test_flag_overrides_built_in_default(self):
    run_passes = litert_converter.converter_api_ext.run_convert_to_tfl_passes
    with mock.patch.object(
        litert_converter.converter_api_ext,
        "run_convert_to_tfl_passes",
        wraps=run_passes,
    ) as mock_run_passes:
      litert_torch.convert(
          _Add().eval(),
          (torch.randn(2, 3),),
          _litert_converter_flags={
              "_experimental_unsafe_fuse_dynamic_shaped_broadcast": False
          },
      )
    config = mock_run_passes.call_args.args[2]
    self.assertFalse(config.unsafe_fuse_dynamic_shaped_broadcast)

  def test_enable_x64_flag_turns_off_downcast_warning(self):
    with mock.patch.object(
        litert_converter, "_warn_downcast_io"
    ) as mock_warn_downcast_io:
      litert_torch.convert(
          _AddInt().eval(),
          (torch.ones(2, 3, dtype=torch.int64),),
          enable_x64=False,
          _litert_converter_flags={"enable_x64": True},
      )
    mock_warn_downcast_io.assert_not_called()

  def test_enable_x64_flag_turns_on_downcast_warning(self):
    with mock.patch.object(
        litert_converter, "_warn_downcast_io"
    ) as mock_warn_downcast_io:
      litert_torch.convert(
          _AddInt().eval(),
          (torch.ones(2, 3, dtype=torch.int64),),
          enable_x64=True,
          _litert_converter_flags={"enable_x64": False},
      )
    mock_warn_downcast_io.assert_called_once()

  def test_flag_defaults_without_converter_flags(self):
    run_passes = litert_converter.converter_api_ext.run_convert_to_tfl_passes
    with mock.patch.object(
        litert_converter.converter_api_ext,
        "run_convert_to_tfl_passes",
        wraps=run_passes,
    ) as mock_run_passes:
      litert_torch.convert(_Add().eval(), (torch.randn(2, 3),))
    config = mock_run_passes.call_args.args[2]
    self.assertFalse(config.strict_qdq_mode)

  def test_unsupported_flag_warns_and_still_converts(self):
    x = torch.randn(2, 3)
    with self.assertLogs(level="WARNING") as logs:
      model = litert_torch.convert(
          _Add().eval(),
          (x,),
          _litert_converter_flags={"serialize_debug_metadata": True},
      )
    self.assertTrue(
        any("serialize_debug_metadata" in line for line in logs.output),
        msg=logs.output,
    )
    torch.testing.assert_close(torch.as_tensor(model(x)), x + 1.0)


if __name__ == "__main__":
  googletest.main()
