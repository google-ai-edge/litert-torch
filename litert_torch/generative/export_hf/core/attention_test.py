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
from unittest import mock

from absl.testing import parameterized

# Not directly used but to register the attention implementation.
import litert_torch.generative.export_hf.core.attention  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import attention as attention_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
import numpy as np
import torch
from transformers import modeling_utils
from transformers.models.gemma2 import modeling_gemma2

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
  elif attention_implementation in ('sdpa', 'eager'):
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
        'sdpa', 'eager', or 'lrt_transposed_attention'.
      num_key_value_groups: The number of key value groups.
      scaling: The scaling factor to pass to the attention layer.
      softcap: The softcap factor to pass to the attention layer.
    """
    super().__init__()
    self.attention_implementation = attention_implementation
    self.num_key_value_groups = num_key_value_groups
    self.scaling = scaling
    self.softcap = softcap
    self.head_dim = None

  def forward(self, query, key, value, attention_mask, **kwargs):
    self.head_dim = query.shape[-1]
    if self.attention_implementation == 'eager' or (
        self.attention_implementation == 'sdpa' and self.softcap is not None
    ):
      attention_interface = modeling_gemma2.eager_attention_forward
    else:
      attention_interface = modeling_utils.ALL_ATTENTION_FUNCTIONS[
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


class TransposedAttentionIsCausalTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('text_only_global', True, True, True, True),
      ('not_opted_in', None, True, True, False),
      ('multimodal_export', False, True, True, False),
      ('sliding_layer', True, False, True, False),
      ('non_causal_module', True, True, False, False),
  )
  def test_is_causal_requires_opt_in(
      self, sdpa_is_causal, is_global, module_is_causal, expected
  ):
    module = torch.nn.Module()
    module.is_causal = module_is_causal
    query = torch.zeros(1, 2, 1, 8)
    kwargs = {
        'k_ts_idx': 2,
        'v_ts_idx': 3,
        'apply_gpu_composites': True,
        'is_global': is_global,
    }
    if sdpa_is_causal is not None:
      kwargs['sdpa_is_causal'] = sdpa_is_causal
    with mock.patch.object(
        attention_lib.gpu_sdpa,
        'scaled_dot_product_attention_transposed',
        return_value=query,
    ) as mock_sdpa:
      attention_lib.transposed_attention(
          module, query, query, query, None, 1.0, None, **kwargs
      )
    self.assertEqual(mock_sdpa.call_args.kwargs['is_causal'], expected)

  @parameterized.named_parameters(
      (
          'qwen3_text_generation',
          'qwen3',
          'dummy',
          exportable_module_config.ExportTask.TEXT_GENERATION,
          True,
      ),
      (
          'lfm2_text_generation',
          'lfm2',
          'dummy',
          exportable_module_config.ExportTask.TEXT_GENERATION,
          True,
      ),
      (
          'model_id_fallback_qwen3',
          None,
          'Qwen/Qwen3-0.6B',
          exportable_module_config.ExportTask.TEXT_GENERATION,
          True,
      ),
      (
          'unknown_model_type_text_generation',
          'bert',
          'Qwen/Qwen3-0.6B',
          exportable_module_config.ExportTask.TEXT_GENERATION,
          False,
      ),
      (
          'unknown_model_id_text_generation',
          None,
          'dummy_non_causal_model',
          exportable_module_config.ExportTask.TEXT_GENERATION,
          False,
      ),
      (
          'qwen3_multimodal_task',
          'qwen3',
          'Qwen/Qwen3-0.6B',
          exportable_module_config.ExportTask.IMAGE_TEXT_TO_TEXT,
          False,
      ),
  )
  def test_attention_kwargs_gates_on_task_and_model_key(
      self, model_type, model_id, task, expected
  ):
    model = torch.nn.Module()
    if model_type is not None:
      model.config = mock.MagicMock(model_type=model_type, text_config=None)
    config = exportable_module_config.ExportableModuleConfig(
        model=model_id,
        task=task,
    )
    exportable = (
        exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill(
            model=model,
            export_config=config,
        )
    )
    self.assertEqual(exportable.attention_kwargs()['sdpa_is_causal'], expected)


if __name__ == '__main__':
  googletest.main()
