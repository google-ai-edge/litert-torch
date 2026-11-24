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
"""Tests for the GraniteMoeHybrid (Granite-4.0-H) export patch."""

import math
from typing import Any
from unittest import mock

from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.core import cache as cache_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.granitemoehybrid import extension
from litert_torch.generative.export_hf.model_ext.granitemoehybrid import patch
import numpy as np
import torch
from transformers.models.granitemoehybrid import configuration_granitemoehybrid
from transformers.models.granitemoehybrid import modeling_granitemoehybrid

from absl.testing import absltest as googletest

_MixerCls = modeling_granitemoehybrid.GraniteMoeHybridMambaLayer
# The value the LiteRT-LM runtime writes to masked positions of a float mask.
_MASKED = -0.7 * torch.finfo(torch.float32).max
_CACHE_LENGTH = 64


def _get_config(
    n_groups: int = 2,
    state_size: int = 16,
) -> configuration_granitemoehybrid.GraniteMoeHybridConfig:
  return configuration_granitemoehybrid.GraniteMoeHybridConfig(
      vocab_size=128,
      hidden_size=64,
      intermediate_size=128,
      shared_intermediate_size=128,
      num_hidden_layers=4,
      num_attention_heads=4,
      num_key_value_heads=2,
      layer_types=[
          "linear_attention",
          "linear_attention",
          "full_attention",
          "linear_attention",
      ],
      mamba_n_heads=8,
      mamba_d_head=16,
      mamba_d_state=state_size,
      mamba_expand=2,
      mamba_n_groups=n_groups,
      mamba_chunk_size=8,
      num_local_experts=0,
      position_embedding_type="nope",
      pad_token_id=0,
  )


def _randomize_ssm_parameters(module: torch.nn.Module):
  """Moves A, dt_bias, D and the biases away from their initial values."""
  for m in module.modules():
    if isinstance(m, _MixerCls):
      with torch.no_grad():
        m.A_log.uniform_(-1.0, 2.0)
        m.dt_bias.uniform_(-3.0, 1.0)
        m.D.uniform_(0.5, 1.5)
        # transformers initializes the conv bias of a model to zeros.
        conv_bias = m.conv1d.bias
        if conv_bias is not None:
          conv_bias.uniform_(-0.5, 0.5)
        for proj in (m.in_proj, m.out_proj):
          proj_bias = proj.bias
          if proj_bias is not None:
            proj_bias.uniform_(-0.5, 0.5)


def _get_model(
    n_groups: int = 2,
) -> modeling_granitemoehybrid.GraniteMoeHybridForCausalLM:
  model = modeling_granitemoehybrid.GraniteMoeHybridForCausalLM(
      _get_config(n_groups)
  )
  _randomize_ssm_parameters(model)
  return model.eval()


def _get_mixer(n_groups: int = 2, state_size: int = 16) -> torch.nn.Module:
  mixer = _MixerCls(_get_config(n_groups, state_size), layer_idx=0)
  _randomize_ssm_parameters(mixer)
  return mixer.eval()


def _causal_mask(input_pos: torch.Tensor) -> torch.Tensor:
  cache_pos = torch.arange(_CACHE_LENGTH).view(1, 1, 1, -1)
  visible = cache_pos <= input_pos.view(1, 1, -1, 1)
  return torch.where(visible, 0.0, _MASKED)


def _get_mamba_cache(config) -> cache_lib.LiteRTLMCache:
  """A one-layer cache for a mixer with `layer_idx` 0."""
  export_config = exportable_module_config.ExportableModuleConfig(
      model="dummy", cache_length=_CACHE_LENGTH
  )
  return cache_lib.LiteRTLMCache([
      cache_lib.LiteRTLMConvCacheLayer.create_from_config(
          config, 0, export_config
      )
  ])


class _MixerWrapper(torch.nn.Module):
  """Runs one mixer step on explicit state tensors."""

  def __init__(self, mixer: torch.nn.Module):
    super().__init__()
    self.mixer = mixer

  def forward(
      self, hidden_states, conv_state, recurrent_state, valid_mask=None
  ):
    layer = cache_lib.LiteRTLMConvCacheLayer(
        conv_state, recurrent_state, layer_type="linear_attention"
    )
    y = patch.mamba2_mixer_forward(
        self.mixer,
        hidden_states,
        cache_params=cache_lib.LiteRTLMCache([layer]),
        valid_mask=valid_mask,
    )
    return y, layer.conv_states, layer.recurrent_states


class _ChunkedScan(torch.nn.Module):
  """The cache-free chunked SSD scan of a mixer, for `torch.export`."""

  def __init__(self, mixer: Any):
    super().__init__()
    self.mixer = mixer

  def forward(self, x, b, c, dt):
    a = -torch.exp(self.mixer.A_log.float())
    # pylint: disable-next=protected-access
    return patch._ssd_chunked(self.mixer, x, b, c, dt, a, None, None)


def _scan_inputs(
    mixer: Any, seq_len: int, batch_size: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Random x, B, C and dt (before the softplus) for the SSD scan."""
  groups_size = mixer.n_groups * mixer.ssm_state_size
  return (
      torch.randn(batch_size, seq_len, mixer.num_heads * mixer.head_dim),
      torch.randn(batch_size, seq_len, groups_size),
      torch.randn(batch_size, seq_len, groups_size),
      torch.randn(batch_size, seq_len, mixer.num_heads),
  )


class PatchTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_check_mamba2_mixer_forward(self):
    rel = patch.check_mamba2_mixer_forward(_MixerCls, patch._check_config())
    self.assertLess(rel, 2e-4)
    # The check draws its mixer and inputs from a fixed seed, whatever the
    # random state of the caller.
    torch.manual_seed(1)
    self.assertEqual(
        patch.check_mamba2_mixer_forward(_MixerCls, patch._check_config()), rel
    )
    # A difference equal to the tolerance passes.
    self.assertEqual(
        patch.check_mamba2_mixer_forward(
            _MixerCls, patch._check_config(), tol=rel
        ),
        rel,
    )

  def test_check_reports_the_difference_relative_to_transformers(self):
    original_forward = _MixerCls.forward

    def doubled_forward(mixer, *args, **kwargs):
      return 2.0 * original_forward(mixer, *args, **kwargs)

    with mock.patch.object(_MixerCls, "forward", doubled_forward):
      rel = patch.check_mamba2_mixer_forward(
          _MixerCls, patch._check_config(), tol=1.0
      )
    # max |y - 2 y| / max |2 y| = 1 / 2.
    self.assertAlmostEqual(rel, 0.5, places=4)

  @parameterized.named_parameters(
      ("at_a_chunk_multiple", True),
      ("at_a_ragged_length", False),
  )
  def test_check_runs_a_chunk_multiple_and_a_ragged_length(
      self, off_at_chunk_multiples
  ):
    # A transformers mixer that changes at only one kind of length still stops
    # the check.
    original_forward = _MixerCls.forward

    def partly_shifted_forward(mixer, hidden_states, *args, **kwargs):
      out = original_forward(mixer, hidden_states, *args, **kwargs)
      at_multiple = hidden_states.shape[1] % mixer.chunk_size == 0
      return out + 1.0 if at_multiple == off_at_chunk_multiples else out

    with mock.patch.object(_MixerCls, "forward", partly_shifted_forward):
      with self.assertRaisesRegex(
          AssertionError, "disagrees with the installed transformers"
      ):
        patch.check_mamba2_mixer_forward(_MixerCls, patch._check_config())

  def test_model_patch_raises_when_the_transformers_mixer_changes(self):
    # The model patch first compares `mamba2_mixer_forward` with the installed
    # transformers mixer, and stops the export when the two disagree.
    original_forward = _MixerCls.forward

    def shifted_forward(mixer, *args, **kwargs):
      return original_forward(mixer, *args, **kwargs) + 1.0

    model = _get_model()
    with mock.patch.object(_MixerCls, "forward", shifted_forward):
      with self.assertRaisesRegex(
          AssertionError, "disagrees with the installed transformers"
      ):
        with patches_lib.patch_model(model, "granitemoehybrid", None):
          pass

  @parameterized.product(
      n_groups=[1, 2, 4, 8],
      seq_len=[16, 13, 7, 5, 3, 1],
  )
  def test_mixer_forward_matches_transformers(self, n_groups, seq_len):
    mixer = _get_mixer(n_groups)
    x = torch.randn(1, seq_len, 64)
    with torch.no_grad():
      expected = _MixerCls.forward(mixer, x)
      actual = patch.mamba2_mixer_forward(mixer, x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  def test_mixer_forward_matches_transformers_with_a_small_state(self):
    # A state smaller than the number of groups.
    mixer = _get_mixer(n_groups=4, state_size=2)
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
      expected = _MixerCls.forward(mixer, x)
      actual = patch.mamba2_mixer_forward(mixer, x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("floor", (0.5, math.inf)),
      ("ceiling", (0.0, 0.1)),
      ("floor_and_ceiling", (0.3, 0.6)),
  )
  def test_time_step_limit_matches_transformers(self, time_step_limit):
    # Granite's default limit (0, inf) clamps nothing. These limits clamp a
    # large part of the time steps in the chunked scan of both forwards.
    mixer = _get_mixer()
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
      unlimited = patch.mamba2_mixer_forward(mixer, x)
      mixer.time_step_limit = time_step_limit
      expected = _MixerCls.forward(mixer, x)
      actual = patch.mamba2_mixer_forward(mixer, x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    self.assertGreater((actual - unlimited).abs().max().item(), 1e-2)

  @parameterized.named_parameters(
      ("no_limit", (0.0, math.inf), []),
      ("floor", (0.5, math.inf), [(0.5, None)]),
      ("ceiling", (0.0, 0.1), [(None, 0.1)]),
      ("floor_and_ceiling", (0.3, 0.6), [(0.3, None), (None, 0.6)]),
  )
  def test_time_step_limit_exports_a_clamp_per_finite_bound(
      self, time_step_limit, expected_clamps
  ):
    # softplus is >= 0 and an infinite bound is no bound: the exported scan has
    # a clamp only for a positive floor and a finite ceiling.
    mixer = _get_mixer()
    mixer.time_step_limit = time_step_limit
    exported = torch.export.export(
        _ChunkedScan(mixer), _scan_inputs(mixer, 16)
    )
    clamps = []
    for node in exported.graph.nodes:
      if node.target == torch.ops.aten.clamp.default:
        bounds = list(node.args[1:]) + [None] * (3 - len(node.args))
        clamps.append(tuple(bounds))
    self.assertEqual(clamps, expected_clamps)

  @parameterized.named_parameters(
      ("three_chunks", 24),
      ("ragged", 13),
      ("two_sequences", 13, 2),
  )
  def test_chunked_scan_from_a_state_matches_the_recurrence(
      self, seq_len, batch_size=1
  ):
    mixer: Any = _get_mixer()
    with torch.no_grad():
      # Slow decays, so that the initial state and every chunk boundary reach
      # the outputs and the final state.
      mixer.A_log.uniform_(-2.0, -1.0)
      mixer.dt_bias.uniform_(-2.0, -1.0)
    x, b, c, dt = _scan_inputs(mixer, seq_len, batch_size)
    a = -torch.exp(mixer.A_log.float())
    state = torch.randn(
        batch_size, mixer.num_heads, mixer.head_dim, mixer.ssm_state_size
    )
    with torch.no_grad():
      # pylint: disable-next=protected-access
      y, final_state = patch._ssd_chunked(mixer, x, b, c, dt, a, state, None)
      expected_state = state
      expected_y = []
      for t in range(seq_len):
        # pylint: disable-next=protected-access
        y_t, expected_state = patch._ssd_step(
            mixer,
            x[:, t : t + 1],
            b[:, t : t + 1],
            c[:, t : t + 1],
            dt[:, t : t + 1],
            a,
            expected_state,
        )
        expected_y.append(y_t)
    torch.testing.assert_close(
        y, torch.cat(expected_y, dim=1), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        final_state, expected_state, rtol=1e-4, atol=1e-4
    )

  @parameterized.named_parameters(
      ("decode", 1, True, 1, 0),
      ("prefill", 16, True, 0, 1),
      ("single_token_without_a_cache", 1, False, 0, 1),
  )
  def test_only_a_cached_single_token_takes_the_recurrent_step(
      self, seq_len, with_cache, num_steps, num_scans
  ):
    # The recurrent step and the chunked scan give the same values, but the
    # ops they export into the decode graph differ.
    config = _get_config()
    mixer = _get_mixer()
    x = torch.randn(1, seq_len, 64)
    cache = _get_mamba_cache(config) if with_cache else None
    # pylint: disable=protected-access
    step = mock.patch.object(patch, "_ssd_step", wraps=patch._ssd_step)
    scan = mock.patch.object(patch, "_ssd_chunked", wraps=patch._ssd_chunked)
    # pylint: enable=protected-access
    with step as step_mock, scan as scan_mock, torch.no_grad():
      patch.mamba2_mixer_forward(mixer, x, cache_params=cache)
    self.assertEqual(step_mock.call_count, num_steps)
    self.assertEqual(scan_mock.call_count, num_scans)

  def test_mamba2_mixer_forward_accepts_cache_kwarg(self):
    torch.manual_seed(0)
    config = _get_config()
    mixer = _get_mixer()
    x = torch.randn(1, 1, 64)
    cache_new = _get_mamba_cache(config)
    cache_legacy = _get_mamba_cache(config)
    with torch.no_grad():
      out_new = patch.mamba2_mixer_forward(mixer, x, cache=cache_new)
      out_legacy = patch.mamba2_mixer_forward(
          mixer, x, cache_params=cache_legacy
      )
    torch.testing.assert_close(out_new, out_legacy)
    for name in ("conv_states", "recurrent_states"):
      torch.testing.assert_close(
          getattr(cache_new.layers[0], name),
          getattr(cache_legacy.layers[0], name),
      )

  @parameterized.named_parameters(
      ("decode", 1, True, 0),
      ("single_token_without_a_cache", 1, False, 0),
      ("prefill", 16, True, 1),
  )
  def test_only_a_multi_token_call_uses_conv1d(
      self, seq_len, with_cache, num_convs
  ):
    # The product sum over the window and F.conv1d give the same values, but
    # the ops they export into the decode graph differ.
    config = _get_config()
    mixer = _get_mixer()
    x = torch.randn(1, seq_len, 64)
    cache = _get_mamba_cache(config) if with_cache else None
    conv = mock.patch.object(patch.F, "conv1d", wraps=patch.F.conv1d)
    with conv as conv_mock, torch.no_grad():
      patch.mamba2_mixer_forward(mixer, x, cache_params=cache)
    self.assertEqual(conv_mock.call_count, num_convs)

  @parameterized.named_parameters(
      ("one_group", 1, 0),
      ("two_groups", 2, 2),
  )
  def test_single_token_step_repeats_groups_only_with_several_groups(
      self, n_groups, num_repeats
  ):
    # Broadcasting the one group over the heads and repeating it give the
    # same values, but the ops they export into the decode graph differ.
    config = _get_config(n_groups)
    mixer = _get_mixer(n_groups)
    x = torch.randn(1, 1, 64)
    # pylint: disable=protected-access
    repeat = mock.patch.object(
        patch, "_repeat_groups_step", wraps=patch._repeat_groups_step
    )
    # pylint: enable=protected-access
    with repeat as repeat_mock, torch.no_grad():
      patch.mamba2_mixer_forward(
          mixer, x, cache_params=_get_mamba_cache(config)
      )
    self.assertEqual(repeat_mock.call_count, num_repeats)

  def test_extension_registers_the_model_patch(self):
    ext = extension_lib.resolve_model_extension("granitemoehybrid")
    self.assertIsInstance(ext, extension.GraniteMoeHybridExtension)
    model = _get_model()
    mixers = [m for m in model.modules() if isinstance(m, _MixerCls)]
    with patches_lib.patch_model(model, "granitemoehybrid", None):
      for mixer in mixers:
        self.assertIn("forward", vars(mixer))
    for mixer in mixers:
      self.assertNotIn("forward", vars(mixer))

  def test_load_time_patch_builds_the_hlfb_rms_norms(self):
    original_norm = modeling_granitemoehybrid.GraniteMoeHybridRMSNorm
    with patches_lib.get_patch_context("granitemoehybrid"):
      model = _get_model()
    self.assertIs(
        modeling_granitemoehybrid.GraniteMoeHybridRMSNorm, original_norm
    )
    norms = [
        m
        for m in model.modules()
        if isinstance(m, patch.GraniteMoeHybridRMSNorm)
    ]
    # Two per decoder layer and the final norm.
    self.assertLen(norms, 2 * len(model.model.layers) + 1)

  def test_rms_norm_matches_the_transformers_formula(self):
    norm = patch.GraniteMoeHybridRMSNorm(64, eps=1e-5)
    with torch.no_grad():
      norm.weight.uniform_(0.5, 1.5)
      x = torch.randn(2, 5, 64)
      variance = x.pow(2).mean(-1, keepdim=True)
      expected = norm.weight * (x * torch.rsqrt(variance + 1e-5))
      torch.testing.assert_close(norm(x), expected)
      # A bfloat16 input is normalized in float32 and only the result is cast
      # back to bfloat16.
      x_bf16 = x.to(torch.bfloat16)
      torch.testing.assert_close(
          norm(x_bf16), norm(x_bf16.float()).to(torch.bfloat16), rtol=0, atol=0
      )
    self.assertEqual(norm.extra_repr(), "64, eps=1e-05")
    # The same default eps as the transformers RMSNorm.
    self.assertEqual(
        patch.GraniteMoeHybridRMSNorm(64).variance_epsilon,
        modeling_granitemoehybrid.GraniteMoeHybridRMSNorm(64).variance_epsilon,
    )

  def test_cache_layers_and_flatten_names(self):
    config = _get_config()
    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy", cache_length=_CACHE_LENGTH
    )
    kv_cache = cache_lib.LiteRTLMCache.create_from_config(
        config, export_config
    )
    conv_layer = kv_cache.layers[0]
    self.assertIsInstance(conv_layer, cache_lib.LiteRTLMConvCacheLayer)
    self.assertIsInstance(kv_cache.layers[2], cache_lib.LiteRTLMCacheLayer)
    # Channels-last (B, kernel - 1, conv_dim), with
    # conv_dim = heads * head_dim + 2 * groups * state_size.
    self.assertEqual(conv_layer.conv_states.shape, (1, 3, 8 * 16 + 2 * 2 * 16))
    self.assertEqual(conv_layer.recurrent_states.shape, (1, 8, 16, 16))
    self.assertEqual(conv_layer.recurrent_states.dtype, torch.float32)
    _, context = cache_lib._flatten_kvc_t(kv_cache)
    self.assertEqual(
        context[0], ["c_0", "r_0", "c_1", "r_1", "k_2", "v_2", "c_3", "r_3"]
    )

  @parameterized.named_parameters(
      ("full_chunk", 16, 16),
      ("padded", 11, 16),
      ("two_chunks_padded", 13, 24),
      ("one_group", 11, 16, 1),
      ("two_heads_per_group", 11, 16, 4),
      ("group_per_head", 11, 16, 8),
      ("pad_token_in_the_prompt", 11, 16, 2, 4),
  )
  def test_prefill_and_decode_match_transformers(
      self, prompt_len, prefill_len, n_groups=2, prompt_pad_at=None
  ):
    num_decode = 4
    model = _get_model(n_groups)
    config = model.config
    ids = torch.randint(1, config.vocab_size, (1, prompt_len + num_decode))
    # A decoded token may be the pad token (id 0): it is still a real token.
    ids[0, prompt_len + 1] = 0
    if prompt_pad_at is not None:
      # So is a pad token inside the prompt.
      ids[0, prompt_pad_at] = 0
    with torch.no_grad():
      expected = model(input_ids=ids).logits[0]

    model.config._attn_implementation = "lrt_transposed_attention"
    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        cache_length=_CACHE_LENGTH,
        prefill_lengths=[prefill_len],
    )
    with patch.patch_granitemoehybrid_model(model, export_config):
      prefill = exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill(
          model, export_config
      )
      decode = exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate(
          model, export_config
      )
      sample = prefill.get_sample_inputs(config)[f"prefill_{prefill_len}"][0]
      # The runtime pads the prefill chunk with the pad token (id 0 here) and
      # zero positions.
      tokens = torch.zeros((1, prefill_len), dtype=torch.int32)
      tokens[0, :prompt_len] = ids[0, :prompt_len]
      input_pos = torch.arange(prefill_len, dtype=torch.int32)
      input_pos[prompt_len:] = 0
      with torch.no_grad():
        kv_cache = prefill(
            tokens=tokens,
            input_pos=input_pos,
            kv_cache=sample["kv_cache"],
            mask=_causal_mask(input_pos),
        )["kv_cache"]
        for t in range(prompt_len, prompt_len + num_decode):
          pos = torch.tensor([t], dtype=torch.int32)
          out = decode(
              tokens=ids[:, t : t + 1].to(torch.int32),
              input_pos=pos,
              kv_cache=kv_cache,
              mask=_causal_mask(pos),
          )
          kv_cache = out["kv_cache"]
          torch.testing.assert_close(
              out["logits"][0, -1], expected[t], rtol=1e-5, atol=1e-5
          )

        # Decode only, from the zero state.
        kv_cache = prefill.get_sample_inputs(config)[f"prefill_{prefill_len}"][
            0
        ]["kv_cache"]
        for t in range(prompt_len + num_decode):
          pos = torch.tensor([t], dtype=torch.int32)
          out = decode(
              tokens=ids[:, t : t + 1].to(torch.int32),
              input_pos=pos,
              kv_cache=kv_cache,
              mask=_causal_mask(pos),
          )
          kv_cache = out["kv_cache"]
          torch.testing.assert_close(
              out["logits"][0, -1], expected[t], rtol=1e-5, atol=1e-5
          )

  @parameterized.named_parameters(
      ("prefill_of_16", 11, 16),
      ("prefill_of_2", 1, 2),
  )
  def test_pad_positions_keep_the_states(self, num_valid, prefill_len):
    config = _get_config()
    mixer = _get_mixer()
    x = torch.randn(1, prefill_len, 64)
    valid_mask = torch.arange(prefill_len)[None, :] < num_valid

    exact_cache = _get_mamba_cache(config)
    padded_cache = _get_mamba_cache(config)
    unmasked_cache = _get_mamba_cache(config)
    with torch.no_grad():
      exact = patch.mamba2_mixer_forward(
          mixer, x[:, :num_valid], cache_params=exact_cache
      )
      padded = patch.mamba2_mixer_forward(
          mixer, x, cache_params=padded_cache, valid_mask=valid_mask
      )
      patch.mamba2_mixer_forward(mixer, x, cache_params=unmasked_cache)

    torch.testing.assert_close(
        padded[:, :num_valid], exact, rtol=1e-5, atol=1e-5
    )
    # The pad positions add nothing: the output is zero there.
    self.assertEqual(torch.count_nonzero(padded[:, num_valid:]).item(), 0)
    for name in ("conv_states", "recurrent_states"):
      torch.testing.assert_close(
          getattr(padded_cache.layers[0], name),
          getattr(exact_cache.layers[0], name),
          rtol=1e-5,
          atol=1e-5,
      )
      # Without the mask the pad positions do change the state.
      self.assertFalse(
          torch.allclose(
              getattr(unmasked_cache.layers[0], name),
              getattr(exact_cache.layers[0], name),
              rtol=1e-3,
              atol=1e-3,
          )
      )

  @parameterized.named_parameters(
      ("without_projection_biases", False),
      ("with_projection_biases", True),
  )
  def test_padded_prefill_matches_transformers(self, proj_bias):
    # With a right-padded prefill the outputs agree at every position, pad
    # positions included: transformers also zeroes the pad positions of the
    # mixer input and of the convolution output. With the projection biases,
    # the gate of a pad position is not zero.
    config = _get_config()
    config.mamba_proj_bias = proj_bias
    mixer = _MixerCls(config, layer_idx=0)
    _randomize_ssm_parameters(mixer)
    mixer.eval()
    x = torch.randn(2, 16, 64)
    valid_mask = torch.arange(16)[None, :] < torch.tensor([[11], [16]])
    with torch.no_grad():
      expected = _MixerCls.forward(mixer, x, attention_mask=valid_mask)
      actual = patch.mamba2_mixer_forward(mixer, x, valid_mask=valid_mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("then_a_prefill", 5, True),
      ("then_a_decode", 15, True),
      ("then_a_decode_without_conv_bias", 15, False),
  )
  def test_prefill_continues_from_the_cached_states(self, first_len, conv_bias):
    # A prompt run in two calls gives the outputs and the states of one call:
    # the second call starts from the states the first one left.
    config = _get_config()
    config.mamba_conv_bias = conv_bias
    mixer = _MixerCls(config, layer_idx=0)
    _randomize_ssm_parameters(mixer)
    mixer.eval()
    x = torch.randn(1, 16, 64)
    one_call_cache = _get_mamba_cache(config)
    two_calls_cache = _get_mamba_cache(config)
    with torch.no_grad():
      expected = patch.mamba2_mixer_forward(
          mixer, x, cache_params=one_call_cache
      )
      first = patch.mamba2_mixer_forward(
          mixer, x[:, :first_len], cache_params=two_calls_cache
      )
      second = patch.mamba2_mixer_forward(
          mixer, x[:, first_len:], cache_params=two_calls_cache
      )
    torch.testing.assert_close(
        torch.cat([first, second], dim=1), expected, rtol=1e-5, atol=1e-5
    )
    for name in ("conv_states", "recurrent_states"):
      torch.testing.assert_close(
          getattr(two_calls_cache.layers[0], name),
          getattr(one_call_cache.layers[0], name),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_patch_binds_and_restores_mixer_forward(self):
    model = _get_model()
    mixers = [m for m in model.modules() if isinstance(m, _MixerCls)]
    self.assertLen(mixers, 3)
    modules = [model.base_model, *mixers]
    # pylint: disable=protected-access
    hooks_before = [
        (len(m._forward_pre_hooks), len(m._forward_hooks)) for m in modules
    ]
    with patch.patch_granitemoehybrid_model(model, None):
      for mixer in mixers:
        self.assertIn("forward", vars(mixer))
    for mixer in mixers:
      self.assertNotIn("forward", vars(mixer))
    # The patch leaves no hook behind.
    self.assertEqual(
        [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in modules],
        hooks_before,
    )
    # pylint: enable=protected-access

  def test_patch_drops_the_mask_after_each_model_call(self):
    # The prefill mask of a model call reaches the mixers only during that
    # call.
    model = _get_model()
    mixer = next(m for m in model.modules() if isinstance(m, _MixerCls))
    ids = torch.randint(1, model.config.vocab_size, (1, 16))
    x = torch.randn(1, 16, 64)
    valid_mask = torch.arange(16)[None, :] < 11
    with patch.patch_granitemoehybrid_model(model, None), torch.no_grad():
      before = mixer(x)
      model(input_ids=ids, valid_mask=valid_mask, use_cache=False)
      after = mixer(x)
      # Also after a call that fails (a token id outside the vocabulary).
      with self.assertRaises(IndexError):
        model(
            input_ids=torch.full_like(ids, model.config.vocab_size),
            valid_mask=valid_mask,
            use_cache=False,
        )
      after_a_failed_call = mixer(x)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(after_a_failed_call, before, rtol=0, atol=0)

  @parameterized.named_parameters(
      ("more_mamba_layers_in_the_config", ["linear_attention"] * 4),
      (
          "fewer_mamba_layers_in_the_config",
          ["linear_attention"] + ["full_attention"] * 3,
      ),
  )
  def test_patch_raises_on_mixer_count_mismatch(self, layer_types):
    model = _get_model()
    model.config.layer_types = layer_types
    with self.assertRaisesRegex(ValueError, "linear_attention layers"):
      with patch.patch_granitemoehybrid_model(model, None):
        pass

  @parameterized.named_parameters(
      ("prefill_padded", 16, 11),
      ("prefill_four_chunks", 16, 16, 4),
      ("decode", 1, None),
  )
  def test_converted_mixer_matches_and_is_gpu_friendly(
      self, seq_len, num_valid, chunk_size=8
  ):
    config = _get_config()
    config.mamba_chunk_size = chunk_size
    mixer = _MixerCls(config, layer_idx=0)
    _randomize_ssm_parameters(mixer)
    module = _MixerWrapper(mixer.eval()).eval()
    args = (
        torch.randn(1, seq_len, 64),
        torch.randn(1, 3, 8 * 16 + 2 * 2 * 16),
        torch.randn(1, 8, 16, 16),
    )
    if num_valid is not None:
      args += (torch.arange(seq_len)[None, :] < num_valid,)
    edge_model = litert_torch.convert(module, args)

    with torch.no_grad():
      expected = module(*args)
    actual = edge_model(*(arg.numpy() for arg in args))
    if not isinstance(actual, tuple):
      self.fail(f"Expected a tuple of outputs, got {type(actual)}.")
    for a, e in zip(actual, expected):
      np.testing.assert_allclose(np.asarray(a), e.numpy(), rtol=1e-4, atol=1e-4)

    # pylint: disable=protected-access
    interpreter = edge_model._get_interpreter()
    op_names = {op["op_name"] for op in interpreter._get_ops_details()}
    # pylint: enable=protected-access
    self.assertNotIn("BROADCAST_TO", op_names)
    max_rank = max(len(t["shape"]) for t in interpreter.get_tensor_details())
    self.assertLessEqual(max_rank, 4)


if __name__ == "__main__":
  googletest.main()
