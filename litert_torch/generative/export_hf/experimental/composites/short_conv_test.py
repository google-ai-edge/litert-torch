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
"""Tests for short_conv composite op."""

from absl.testing import parameterized
from litert_torch.generative.export_hf.experimental.composites import short_conv
import torch

from absl.testing import absltest as googletest


class ShortConvTest(parameterized.TestCase):

  def test_short_conv_step_accuracy(self):
    batch_size = 1
    hidden_size = 64
    conv_L_cache = 3

    in_proj_out = torch.randn(batch_size, 1, 3 * hidden_size)
    conv_state = torch.randn(batch_size, hidden_size, conv_L_cache - 1)
    conv_weight = torch.randn(hidden_size, 1, conv_L_cache)
    conv_bias = torch.randn(hidden_size)

    # Reference computation:
    b, c, x_proj = in_proj_out.chunk(3, dim=-1)
    p = b * x_proj
    p_s = p.squeeze(1).unsqueeze(-1)
    padded = torch.cat([conv_state, p_s], dim=-1)
    expected_next_state = padded[:, :, -(conv_L_cache - 1) :]
    w = conv_weight.squeeze(1).unsqueeze(0)
    expected_out = (padded * w).sum(dim=-1) + conv_bias.unsqueeze(0)
    expected_y = c * expected_out.unsqueeze(1)

    actual_y, actual_next_state = short_conv.apply_short_conv_step(
        in_proj_out=in_proj_out,
        conv_state=conv_state,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        conv_L_cache=conv_L_cache,
    )

    self.assertTrue(torch.allclose(expected_y, actual_y, rtol=1e-5, atol=1e-5))
    self.assertTrue(
        torch.allclose(
            expected_next_state, actual_next_state, rtol=1e-5, atol=1e-5
        )
    )

  def test_short_conv_step_no_bias(self):
    batch_size = 1
    hidden_size = 64
    conv_L_cache = 3

    in_proj_out = torch.randn(batch_size, 1, 3 * hidden_size)
    conv_state = torch.randn(batch_size, hidden_size, conv_L_cache - 1)
    conv_weight = torch.randn(hidden_size, conv_L_cache)

    b, c, x_proj = in_proj_out.chunk(3, dim=-1)
    p = b * x_proj
    p_s = p.squeeze(1).unsqueeze(-1)
    padded = torch.cat([conv_state, p_s], dim=-1)
    expected_next_state = padded[:, :, -(conv_L_cache - 1) :]
    w = conv_weight.unsqueeze(0)
    expected_out = (padded * w).sum(dim=-1)
    expected_y = c * expected_out.unsqueeze(1)

    actual_y, actual_next_state = short_conv.apply_short_conv_step(
        in_proj_out=in_proj_out,
        conv_state=conv_state,
        conv_weight=conv_weight,
        conv_bias=None,
        conv_L_cache=conv_L_cache,
    )

    self.assertTrue(torch.allclose(expected_y, actual_y, rtol=1e-5, atol=1e-5))
    self.assertTrue(
        torch.allclose(
            expected_next_state, actual_next_state, rtol=1e-5, atol=1e-5
        )
    )

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
    conv_state = torch.randn(batch_size, hidden_size, conv_L_cache - 1)
    conv_weight = torch.randn(hidden_size, 1, conv_L_cache)
    conv_bias = torch.randn(hidden_size) if use_bias else None

    # Reference computation, one causal tap at a time:
    b, c, x_proj = in_proj_out.chunk(3, dim=-1)
    p = (b * x_proj).transpose(1, 2)
    padded = torch.cat([conv_state, p], dim=-1)
    w = conv_weight.squeeze(1)
    expected_out = torch.zeros(batch_size, hidden_size, seq_len)
    for t in range(seq_len):
      expected_out[:, :, t] = (padded[:, :, t : t + conv_L_cache] * w).sum(-1)
    if conv_bias is not None:
      expected_out = expected_out + conv_bias.view(1, -1, 1)
    expected_y = c * expected_out.transpose(1, 2)
    expected_next_state = padded[:, :, num_valid : num_valid + conv_L_cache - 1]

    actual_y, actual_next_state = short_conv.apply_short_conv_step(
        in_proj_out=in_proj_out,
        conv_state=conv_state,
        conv_weight=conv_weight if weight_rank == 3 else w,
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
    conv_state = torch.randn(batch_size, hidden_size, conv_L_cache - 1)
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
        torch.randn(1, hidden_size, conv_L_cache - 1),
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


if __name__ == "__main__":
  googletest.main()
