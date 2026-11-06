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
"""Tests for the short convolution helpers and composite."""

from typing import Optional

from absl.testing import parameterized
from litert_torch.generative.export_hf.experimental.composites import short_conv
import torch
import torch.nn.functional as F

from absl.testing import absltest as googletest


def _reference_short_conv(
    in_proj_out: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    num_valid: Optional[int] = None,
    is_gated: bool = True,
    use_silu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reference short conv via `F.conv1d` on a channels-first copy.

  Args:
    in_proj_out: [batch, seq_len, 3 * channels] or [batch, seq_len, channels].
    conv_state: Channels-last state [batch, L - 1, channels].
    conv_weight: Depthwise weight [channels, 1, L].
    conv_bias: Optional bias [channels].
    num_valid: Number of valid tokens; None means all.
    is_gated: Whether to compute `c * conv(b * x)`.
    use_silu: Whether to apply SiLU to the conv output.

  Returns:
    y [batch, seq_len, channels] and the channels-last next state.
  """
  if is_gated:
    b, c, x = in_proj_out.chunk(3, dim=-1)
    conv_input = b * x
  else:
    c = None
    conv_input = in_proj_out
  seq_len = conv_input.shape[1]
  padded = torch.cat([conv_state, conv_input], dim=1)  # [B, L - 1 + S, C]
  out = F.conv1d(
      padded.transpose(1, 2),
      conv_weight,
      conv_bias,
      groups=conv_weight.shape[0],
  ).transpose(1, 2)
  if use_silu:
    out = F.silu(out)
  y = c * out if c is not None else out
  n = seq_len if num_valid is None else num_valid
  state_len = conv_state.shape[1]
  return y, padded[:, n : n + state_len, :]


class ShortConvTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("weight_c1l_bias", 3, True),
      ("weight_cl_no_bias", 2, False),
  )
  def test_short_conv_step_accuracy(self, weight_rank, use_bias):
    batch_size = 1
    hidden_size = 64
    conv_L_cache = 3

    in_proj_out = torch.randn(batch_size, 1, 3 * hidden_size)
    conv_state = torch.randn(batch_size, conv_L_cache - 1, hidden_size)
    conv_weight = torch.randn(hidden_size, 1, conv_L_cache)
    conv_bias = torch.randn(hidden_size) if use_bias else None
    expected_y, expected_next_state = _reference_short_conv(
        in_proj_out, conv_state, conv_weight, conv_bias
    )

    actual_y, actual_next_state = short_conv.apply_short_conv_step(
        in_proj_out=in_proj_out,
        conv_state=conv_state,
        conv_weight=conv_weight if weight_rank == 3 else conv_weight.squeeze(1),
        conv_bias=conv_bias,
        conv_L_cache=conv_L_cache,
    )

    torch.testing.assert_close(actual_y, expected_y, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_next_state, expected_next_state)

  @parameterized.named_parameters(
      ("full", 6, True, 3),
      ("padded", 4, True, 3),
      ("shorter_than_state", 1, True, 3),
      ("full_no_bias", 6, False, 2),
      ("padded_no_bias", 4, False, 2),
      ("shorter_than_state_no_bias", 1, False, 2),
  )
  def test_short_conv_prefill_accuracy(self, num_valid, use_bias, weight_rank):
    batch_size = 1
    seq_len = 6
    hidden_size = 64
    conv_L_cache = 3

    in_proj_out = torch.randn(batch_size, seq_len, 3 * hidden_size)
    conv_state = torch.randn(batch_size, conv_L_cache - 1, hidden_size)
    conv_weight = torch.randn(hidden_size, 1, conv_L_cache)
    conv_bias = torch.randn(hidden_size) if use_bias else None
    expected_y, expected_next_state = _reference_short_conv(
        in_proj_out, conv_state, conv_weight, conv_bias, num_valid
    )

    actual_y, actual_next_state = short_conv.apply_short_conv_step(
        in_proj_out=in_proj_out,
        conv_state=conv_state,
        conv_weight=conv_weight if weight_rank == 3 else conv_weight.squeeze(1),
        conv_bias=conv_bias,
        num_valid_tokens=torch.tensor([num_valid], dtype=torch.int32),
        conv_L_cache=conv_L_cache,
    )

    torch.testing.assert_close(actual_y, expected_y, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_next_state, expected_next_state)

  def test_short_conv_prefill_matches_sequential_steps(self):
    batch_size = 1
    seq_len = 5
    hidden_size = 64
    conv_L_cache = 3

    in_proj_out = torch.randn(batch_size, seq_len, 3 * hidden_size)
    conv_state = torch.randn(batch_size, conv_L_cache - 1, hidden_size)
    conv_weight = torch.randn(hidden_size, 1, conv_L_cache)
    conv_bias = torch.randn(hidden_size)

    prefill_y, prefill_state = short_conv.apply_short_conv_step(
        in_proj_out=in_proj_out,
        conv_state=conv_state,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        num_valid_tokens=torch.tensor([seq_len], dtype=torch.int32),
        conv_L_cache=conv_L_cache,
    )

    state = conv_state
    step_ys = []
    for t in range(seq_len):
      step_y, state = short_conv.apply_short_conv_step(
          in_proj_out=in_proj_out[:, t : t + 1],
          conv_state=state,
          conv_weight=conv_weight,
          conv_bias=conv_bias,
          conv_L_cache=conv_L_cache,
      )
      step_ys.append(step_y)

    torch.testing.assert_close(
        prefill_y, torch.cat(step_ys, dim=1), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(prefill_state, state)

  @parameterized.named_parameters(
      ("with_bias", True, 5),
      ("no_bias", False, 4),
  )
  def test_short_conv_prefill_export_emits_composite(
      self, use_bias, num_inputs
  ):
    hidden_size = 64
    conv_L_cache = 3

    class Wrapper(torch.nn.Module):

      def forward(
          self, in_proj_out, conv_state, conv_weight, num_valid, conv_bias=None
      ):
        return short_conv.apply_short_conv_step(
            in_proj_out=in_proj_out,
            conv_state=conv_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            num_valid_tokens=num_valid,
            conv_L_cache=conv_L_cache,
        )

    args: tuple[torch.Tensor, ...] = (
        torch.randn(1, 8, 3 * hidden_size),
        torch.randn(1, conv_L_cache - 1, hidden_size),
        torch.randn(hidden_size, 1, conv_L_cache),
        torch.tensor([5], dtype=torch.int32),
    )
    if use_bias:
      args += (torch.randn(hidden_size),)
    exported = torch.export.export(Wrapper(), args)
    marks = [
        n
        for n in exported.graph.nodes
        if n.op == "call_function"
        and "mark_tensor" in str(n.target)
        and "odml.short_conv_step" in (str(n.args) + str(n.kwargs))
    ]
    # All inputs plus the two outputs (y, next_state) are marked.
    self.assertLen(marks, num_inputs + 2)

  @parameterized.named_parameters(
      ("all_valid", None, False, False),
      ("full", 6, True, False),
      ("padded", 4, True, False),
      ("none_valid", 0, False, False),
      ("scalar_num_valid", 4, False, True),
  )
  def test_causal_conv1d_with_state(
      self, num_valid, use_bias, scalar_num_valid
  ):
    channels = 16
    seq_len = 6
    kernel_size = 4

    x = torch.randn(1, seq_len, channels)
    conv_state = torch.randn(1, kernel_size - 1, channels)
    conv_weight = torch.randn(channels, 1, kernel_size)
    conv_bias = torch.randn(channels) if use_bias else None
    expected_out, expected_next_state = _reference_short_conv(
        x, conv_state, conv_weight, conv_bias, num_valid, is_gated=False
    )

    num_valid_tokens = None
    if num_valid is not None:
      num_valid_tokens = torch.tensor(
          num_valid if scalar_num_valid else [num_valid], dtype=torch.int32
      )
    actual_out, actual_next_state = short_conv.causal_conv1d_with_state(
        x, conv_state, conv_weight, conv_bias, num_valid_tokens
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_next_state, expected_next_state)

  def test_causal_conv1d_with_state_export_gathers_next_state(self):
    channels = 16
    kernel_size = 4

    class Wrapper(torch.nn.Module):

      def forward(self, x, conv_state, conv_weight, num_valid):
        return short_conv.causal_conv1d_with_state(
            x, conv_state, conv_weight, num_valid_tokens=num_valid
        )

    args = (
        torch.randn(1, 8, channels),
        torch.randn(1, kernel_size - 1, channels),
        torch.randn(channels, 1, kernel_size),
        torch.tensor([5], dtype=torch.int32),
    )
    targets = [
        str(n.target)
        for n in torch.export.export(Wrapper(), args).graph.nodes
        if n.op == "call_function"
    ]
    # The runtime state offset is a tfl.gather, which the GPU delegates
    # support, rather than a one-hot matmul.
    self.assertTrue(any("tfl.gather" in t for t in targets), targets)
    self.assertFalse(
        any(
            ("matmul" in t or "aten.mm" in t or "aten.bmm" in t)
            for t in targets
        ),
        targets,
    )

  @parameterized.named_parameters(
      ("decode_l2_gated", 1, None, 2, True, False, True),
      ("decode_l4_ungated_silu", 1, None, 4, False, True, False),
      ("prefill_full_l4", 6, 6, 4, False, True, True),
      ("prefill_padded_l4", 6, 4, 4, False, True, False),
      ("prefill_shorter_than_state_l3", 6, 1, 3, True, True, True),
  )
  def test_short_conv_matches_reference(
      self,
      seq_len,
      num_valid,
      conv_L_cache,
      is_gated,
      use_silu,
      use_bias,
  ):
    torch.manual_seed(0)
    hidden_size = 32
    in_channels = 3 * hidden_size if is_gated else hidden_size
    in_proj_out = torch.randn(1, seq_len, in_channels)
    conv_state = torch.randn(1, conv_L_cache - 1, hidden_size)
    conv_weight = torch.randn(hidden_size, 1, conv_L_cache)
    conv_bias = torch.randn(hidden_size) if use_bias else None
    num_valid_t = (
        torch.tensor([num_valid], dtype=torch.int32)
        if num_valid is not None
        else None
    )
    ref_y, ref_state = _reference_short_conv(
        in_proj_out,
        conv_state,
        conv_weight,
        conv_bias,
        num_valid,
        is_gated=is_gated,
        use_silu=use_silu,
    )

    for name, fn, kwargs in (
        (
            "composite",
            short_conv.apply_short_conv_step,
            {"conv_L_cache": conv_L_cache},
        ),
        ("decomposed", short_conv.short_conv, {}),
    ):
      with self.subTest(name):
        act_y, act_state = fn(
            in_proj_out=in_proj_out,
            conv_state=conv_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            num_valid_tokens=num_valid_t,
            is_gated=is_gated,
            use_silu=use_silu,
            **kwargs,
        )
        torch.testing.assert_close(act_y, ref_y, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(act_state, ref_state, rtol=1e-5, atol=1e-5)

  def test_short_conv_rejects_non_channels_last_state(self):
    hidden_size = 32
    conv_L_cache = 4
    in_proj_out = torch.randn(1, 1, hidden_size)
    conv_weight = torch.randn(hidden_size, 1, conv_L_cache)
    channels_first_state = torch.randn(1, hidden_size, conv_L_cache - 1)

    with self.assertRaisesRegex(ValueError, "channels-last"):
      short_conv.apply_short_conv_step(
          in_proj_out=in_proj_out,
          conv_state=channels_first_state,
          conv_weight=conv_weight,
          conv_L_cache=conv_L_cache,
          is_gated=False,
      )
    with self.assertRaisesRegex(ValueError, "channels-last"):
      short_conv.short_conv(
          in_proj_out=in_proj_out,
          conv_state=channels_first_state,
          conv_weight=conv_weight,
          is_gated=False,
      )
    with self.assertRaisesRegex(ValueError, "conv_L_cache"):
      short_conv.apply_short_conv_step(
          in_proj_out=in_proj_out,
          conv_state=torch.randn(1, conv_L_cache - 1, hidden_size),
          conv_weight=conv_weight,
          conv_L_cache=conv_L_cache + 1,
          is_gated=False,
      )

  def test_short_conv_export_folds_weight_to_channels_last(self):
    hidden_size = 64
    conv_L_cache = 4

    class Module(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(hidden_size, 1, conv_L_cache)
        )

      def forward(self, in_proj_out, conv_state):
        return short_conv.apply_short_conv_step(
            in_proj_out=in_proj_out,
            conv_state=conv_state,
            conv_weight=self.weight,
            conv_L_cache=conv_L_cache,
            is_gated=False,
            use_silu=True,
        )

    mod = Module().eval()
    args = (
        torch.randn(1, 1, hidden_size),
        torch.randn(1, conv_L_cache - 1, hidden_size),
    )
    exported = torch.export.export(mod, args)
    marks = [
        n
        for n in exported.graph.nodes
        if n.op == "call_function"
        and "mark_tensor" in str(n.target)
        and "odml.short_conv_step" in (str(n.args) + str(n.kwargs))
    ]
    self.assertLen(marks, 5)  # 3 inputs + 2 outputs
    # The layout is implied by the shapes; no `channels_last` attribute.
    self.assertFalse(any("channels_last" in str(n.args) for n in marks), marks)
    # Marked conv_weight (3rd marked input, pos=2) has channels-last [1, L, C].
    weight_marks = [
        n for n in marks if len(n.args) >= 4 and n.args[2] == 2 and n.args[3]
    ]
    self.assertLen(weight_marks, 1)
    self.assertEqual(
        tuple(weight_marks[0].meta["val"].shape),
        (1, conv_L_cache, hidden_size),
    )


if __name__ == "__main__":
  googletest.main()
