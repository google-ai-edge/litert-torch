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

from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.core import cache as cache_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
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
      mamba_d_state=16,
      mamba_expand=2,
      mamba_n_groups=n_groups,
      mamba_chunk_size=8,
      num_local_experts=0,
      position_embedding_type="nope",
      pad_token_id=0,
  )


def _randomize_ssm_parameters(module: torch.nn.Module):
  """Moves A, dt_bias and D away from their initial values."""
  for m in module.modules():
    if isinstance(m, _MixerCls):
      with torch.no_grad():
        m.A_log.uniform_(-1.0, 2.0)
        m.dt_bias.uniform_(-3.0, 1.0)
        m.D.uniform_(0.5, 1.5)


def _get_model(
    n_groups: int = 2,
) -> modeling_granitemoehybrid.GraniteMoeHybridForCausalLM:
  model = modeling_granitemoehybrid.GraniteMoeHybridForCausalLM(
      _get_config(n_groups)
  )
  _randomize_ssm_parameters(model)
  return model.eval()


def _get_mixer(n_groups: int = 2) -> torch.nn.Module:
  mixer = _MixerCls(_get_config(n_groups), layer_idx=0)
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


class PatchTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def test_check_mamba2_mixer_forward(self):
    rel = patch.check_mamba2_mixer_forward(_MixerCls, patch._check_config())
    self.assertLess(rel, 2e-4)

  @parameterized.product(
      n_groups=[1, 2, 8],
      seq_len=[16, 13, 5],
  )
  def test_mixer_forward_matches_transformers(self, n_groups, seq_len):
    mixer = _get_mixer(n_groups)
    x = torch.randn(1, seq_len, 64)
    with torch.no_grad():
      expected = _MixerCls.forward(mixer, x)
      actual = patch.mamba2_mixer_forward(mixer, x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

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
    # conv_dim = heads * head_dim + 2 * groups * state_size, width kernel - 1.
    self.assertEqual(conv_layer.conv_states.shape, (1, 8 * 16 + 2 * 2 * 16, 3))
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
      ("group_per_head", 11, 16, 8),
  )
  def test_prefill_and_decode_match_transformers(
      self, prompt_len, prefill_len, n_groups=2
  ):
    num_decode = 4
    model = _get_model(n_groups)
    config = model.config
    ids = torch.randint(1, config.vocab_size, (1, prompt_len + num_decode))
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

  def test_pad_positions_keep_the_states(self):
    config = _get_config()
    mixer = _get_mixer()
    num_valid, prefill_len = 11, 16
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

  def test_patch_binds_and_restores_mixer_forward(self):
    model = _get_model()
    mixers = [m for m in model.modules() if isinstance(m, _MixerCls)]
    self.assertLen(mixers, 3)
    with patch.patch_granitemoehybrid_model(model, None):
      for mixer in mixers:
        self.assertIn("forward", vars(mixer))
    for mixer in mixers:
      self.assertNotIn("forward", vars(mixer))

  def test_patch_raises_on_mixer_count_mismatch(self):
    model = _get_model()
    model.config.layer_types = ["linear_attention"] * 4
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
        torch.randn(1, 8 * 16 + 2 * 2 * 16, 3),
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
