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
from litert_torch.generative.export_hf.model_ext import extension as model_ext_extension
import numpy as np
import torch
import transformers
from transformers import modeling_utils

from absl.testing import absltest as googletest


def _adapt_inputs(
    attention_implementation: str,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Converts [B, K, S, H] key/value to the lrt_transposed_attention layout."""
  if attention_implementation == 'lrt_transposed_attention':
    b, n, s, h = key.shape
    key = key.reshape(1, b * n, s, h)
    value = value.reshape(1, b * n, s, h).permute(0, 1, 3, 2)
  else:
    raise ValueError(
        f'Unsupported attention implementation: {attention_implementation}'
    )

  return key, value


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    num_key_value_groups: int,
    scaling: float | None,
    softcap: float | None,
) -> torch.Tensor:
  """Eager attention reference with optional logit softcapping.

  HF's `sdpa` attention cannot be used as the reference: it ignores `softcap`
  in older transformers releases and rejects it in newer ones.

  Args:
    query: Query tensor of shape [B, N, T, H].
    key: Key tensor of shape [B, K, S, H].
    value: Value tensor of shape [B, K, S, H].
    attention_mask: Additive mask broadcastable to [B, N, T, S].
    num_key_value_groups: Number of query heads per key/value head (N // K).
    scaling: Logit scale; defaults to 1 / sqrt(H).
    softcap: Optional logit softcap.

  Returns:
    The attention output of shape [B, T, N, H].
  """
  if scaling is None:
    scaling = query.shape[-1] ** -0.5
  key = key.repeat_interleave(num_key_value_groups, dim=1)
  value = value.repeat_interleave(num_key_value_groups, dim=1)
  logits = torch.matmul(query, key.transpose(-2, -1)) * scaling
  if softcap is not None:
    logits = torch.tanh(logits / softcap) * softcap
  probs = torch.softmax(logits + attention_mask, dim=-1)
  return torch.matmul(probs, value).transpose(1, 2)


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
      attention_implementation: The attention implementation to test. Only
        'lrt_transposed_attention' is supported.
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
    torch.manual_seed(0)
    batch_size = [1, 3]
    seq_len = [1, 5]
    context_len = [10]
    head_dim = [8]
    scaling = [None, 0.5]
    # A small softcap visibly changes the outputs; a large one is near-identity.
    softcap = [None, 1.0, 42.0]

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
        expected = _reference_attention(
            query, key, value, mask, num_key_value_groups, scl, scp
        )
        actual = test_attn(query, key, value, mask, **attention_kwargs)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        if scp == 1.0 and t > 1:
          # Make sure this softcap meaningfully changes the result, so the
          # comparison above actually covers softcapping. With t == 1 only one
          # key is unmasked, so softcap cannot change the output.
          uncapped = _reference_attention(
              query, key, value, mask, num_key_value_groups, scl, None
          )
          self.assertFalse(
              torch.allclose(expected, uncapped, rtol=1e-2, atol=1e-2)
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
    exportable = exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill(
        model=model,
        export_config=config,
    )
    self.assertEqual(exportable.attention_kwargs()['sdpa_is_causal'], expected)

  def test_extension_controls_is_causal_text_model(self):
    class _OutOfTreeCausalExt(model_ext_extension.ModelExtension):
      model_types = ('unseen_causal_arch',)
      is_causal_text_model = True

    class _OutOfTreeNonCausalExt(model_ext_extension.ModelExtension):
      model_types = ('gemma3_text',)
      is_causal_text_model = False

    model_cfg = transformers.PretrainedConfig(
        model_type='unseen_causal_arch',
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_size=32,
    )
    model = torch.nn.Module()
    model.config = model_cfg

    # Without extension: unseen arch is not causal-opted-in.
    base_cfg = exportable_module_config.ExportableModuleConfig(model='dummy')
    base_prefill = (
        exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill(
            model=model,
            export_config=base_cfg,
        )
    )
    self.assertFalse(base_prefill.attention_kwargs()['sdpa_is_causal'])

    # With out-of-tree extension: sdpa_is_causal activates.
    ext_cfg = exportable_module_config.ExportableModuleConfig(
        model='dummy',
        model_extension=_OutOfTreeCausalExt(),
    )
    ext_prefill = (
        exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill(
            model=model,
            export_config=ext_cfg,
        )
    )
    self.assertTrue(ext_prefill.attention_kwargs()['sdpa_is_causal'])

    # Explicit False on an extension overrides core's built-in allowlist.
    override_model = torch.nn.Module()
    override_model.config = transformers.PretrainedConfig(
        model_type='gemma3_text'
    )
    override_prefill = (
        exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill(
            model=override_model,
            export_config=exportable_module_config.ExportableModuleConfig(
                model='dummy',
                model_extension=_OutOfTreeNonCausalExt(),
            ),
        )
    )
    self.assertFalse(override_prefill.attention_kwargs()['sdpa_is_causal'])


class _BidirectionalAttentionModule(torch.nn.Module):
  """Minimal stand-in for a `transformers` vision attention layer."""

  def __init__(self, num_heads: int, num_kv_heads: int, is_causal=False):
    super().__init__()
    self.num_key_value_groups = num_heads // num_kv_heads
    self.is_causal = is_causal
    self.training = False


class BidirectionalSdpaAttentionTest(parameterized.TestCase):

  def _inputs(self, num_heads=4, num_kv_heads=2, seq_len=12, head_dim=16):
    torch.manual_seed(0)
    query = torch.randn(1, num_heads, seq_len, head_dim)
    key = torch.randn(1, num_kv_heads, seq_len, head_dim)
    value = torch.randn(1, num_kv_heads, seq_len, head_dim)
    return query, key, value

  def test_is_registered(self):
    self.assertIs(
        modeling_utils.ALL_ATTENTION_FUNCTIONS[
            attention_lib.BIDIRECTIONAL_SDPA_ATTENTION
        ],
        attention_lib.bidirectional_sdpa_attention,
    )

  @parameterized.named_parameters(
      dict(testcase_name='mha', num_kv_heads=4, scaling=1.0),
      dict(testcase_name='gqa', num_kv_heads=2, scaling=1.0),
      dict(testcase_name='default_scaling', num_kv_heads=2, scaling=None),
  )
  def test_matches_eager_attention(self, num_kv_heads, scaling):
    module = _BidirectionalAttentionModule(4, num_kv_heads)
    query, key, value = self._inputs(num_kv_heads=num_kv_heads)
    eager_scaling = scaling if scaling is not None else 16**-0.5
    expected, _ = modeling_utils.ALL_ATTENTION_FUNCTIONS['sdpa'](
        module, query, key, value, None, scaling=eager_scaling
    )
    actual, weights = attention_lib.bidirectional_sdpa_attention(
        module, query, key, value, None, scaling=scaling
    )
    self.assertIsNone(weights)
    self.assertEqual(actual.shape, (1, 12, 4, 16))  # B, T, N, H
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name='mask', kwargs={}, use_mask=True),
      dict(testcase_name='causal_module', kwargs={}, module_is_causal=True),
      dict(testcase_name='causal_kwarg', kwargs={'is_causal': True}),
      dict(testcase_name='dropout', kwargs={'dropout': 0.1}),
      dict(testcase_name='softcap', kwargs={'softcap': 30.0}),
  )
  def test_rejects_unsupported(
      self, kwargs, use_mask=False, module_is_causal=False
  ):
    module = _BidirectionalAttentionModule(4, 2, is_causal=module_is_causal)
    query, key, value = self._inputs()
    mask = torch.zeros(1, 1, 12, 12) if use_mask else None
    with self.assertRaises(ValueError):
      attention_lib.bidirectional_sdpa_attention(
          module, query, key, value, mask, scaling=1.0, **kwargs
      )


if __name__ == '__main__':
  googletest.main()
