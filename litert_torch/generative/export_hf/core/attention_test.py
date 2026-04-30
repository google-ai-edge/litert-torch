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
"""Tests for attention layers."""

import itertools

from absl.testing import parameterized

# Not directly used but to register the attention implementation.
import litert_torch.generative.export_hf.core.attention  # pylint: disable=unused-import
import numpy as np
import torch
import transformers

from absl.testing import absltest as googletest


def _adapt_inputs(
    attention_implementation: str,
    key: torch.Tensor,
    value: torch.Tensor,
):
  """Adapt torch.F.scaled_dot_product_attention inputs."""
  if attention_implementation == 'lrt_transposed_attention':
    b, n, s, h = key.shape
    key = key.reshape(1, b * n, s, h)
    value = value.reshape(1, b * n, s, h).permute(0, 1, 3, 2)
  elif attention_implementation == 'sdpa':
    pass
  else:
    raise ValueError(
        f'Unsupported attention implementation: {attention_implementation}'
    )

  return key, value


class DummyAttentionModule(torch.nn.Module):
  """Dummy module for testing."""

  def __init__(
      self,
      attention_implementation: str,
      num_key_value_groups: int = 1,
      scaling: float | None = None,
      softcap: float | None = None,
  ):
    """Initializes the dummy attention module.

    Args:
      attention_implementation: The attention implementation to test, either
        'sdpa' or 'lrt_transposed_attention'.
      num_key_value_groups: The number of key value groups.
      scaling: The scaling factor to pass to the attention layer.
      softcap: The softcap factor to pass to the attention layer.
    """
    super().__init__()
    self.attention_implementation = attention_implementation
    self.num_key_value_groups = num_key_value_groups
    self.scaling = scaling
    self.softcap = softcap

  def forward(self, query, key, value, attention_mask, **kwargs):
    attention_interface = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS[
        self.attention_implementation
    ]
    key, value = _adapt_inputs(self.attention_implementation, key, value)
    return attention_interface(
        self,
        query,
        key,
        value,
        attention_mask,
        scaling=self.scaling,
        softcap=self.softcap,
        **kwargs,
    )[0]


class AttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='mha',
          num_query_heads=4,
          num_key_value_heads=4,
      ),
      dict(
          testcase_name='mqa',
          num_query_heads=4,
          num_key_value_heads=1,
      ),
      dict(
          testcase_name='gqa',
          num_query_heads=4,
          num_key_value_heads=2,
      ),
  )
  def test_attention(self, num_query_heads, num_key_value_heads):
    batch_size = [1, 3]
    seq_len = [1, 5]
    context_len = [10]
    head_dim = [8]
    scaling = [None, 0.5]
    softcap = [None, 42]

    for b, t, s, h, scl, scp in itertools.product(
        batch_size, seq_len, context_len, head_dim, scaling, softcap
    ):
      with self.subTest(name=f'Testcase: {b}, {t}, {s}, {h}, {scl}, {scp}'):
        query = torch.randn(b, num_query_heads, t, h, dtype=torch.float32)
        key = torch.randn(b, num_key_value_heads, s, h, dtype=torch.float32)
        value = torch.randn(b, num_key_value_heads, s, h, dtype=torch.float32)
        num_key_value_groups = num_query_heads // num_key_value_heads

        mask = np.arange(t)[:, None] >= np.arange(s)[None, :]
        mask = torch.tensor(mask[None, None, :, :], dtype=torch.bool)
        mask = mask.logical_not() * -400.0

        attn = DummyAttentionModule(
            attention_implementation='sdpa',
            num_key_value_groups=num_key_value_groups,
            scaling=scl,
            softcap=scp,
        )
        test_attn = DummyAttentionModule(
            attention_implementation='lrt_transposed_attention',
            num_key_value_groups=num_key_value_groups,
            scaling=scl,
            softcap=scp,
        )
        attention_kwargs = {
            'k_ts_idx': 2,
            'v_ts_idx': 3,
        }
        expected = attn(query, key, value, mask)
        actual = test_attn(query, key, value, mask, **attention_kwargs)
        self.assertTrue(
            torch.allclose(
                expected, actual, rtol=1e-2, atol=1e-2, equal_nan=True
            ),
            f'Expected: {expected},  Actual: {actual}',
        )


if __name__ == '__main__':
  googletest.main()
