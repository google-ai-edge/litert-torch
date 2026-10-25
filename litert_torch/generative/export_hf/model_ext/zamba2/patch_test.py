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
"""Tests for the Zamba2 export patch."""

from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.core import cache as cache_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext.granitemoehybrid import patch as granitemoehybrid_patch
from litert_torch.generative.export_hf.model_ext.zamba2 import patch
import numpy as np
import torch
from transformers.models.zamba2 import configuration_zamba2
from transformers.models.zamba2 import modeling_zamba2

from absl.testing import absltest as googletest

_MixerCls = modeling_zamba2.Zamba2MambaMixer
# The value the LiteRT-LM runtime writes to masked positions of a float mask.
_MASKED = -0.7 * torch.finfo(torch.float32).max
_CACHE_LENGTH = 64
_LAYER_TYPES = ["linear_attention", "hybrid", "linear_attention", "hybrid"]
# Zamba2Model.get_layers numbers the shared blocks by layer index in
# transformers 5.16.1 and by hybrid layer in 5.20.1. With hybrid layers 0 and
# 3, both give the two hybrid layers blocks 0 and 1.
_TWO_MEM_BLOCKS_LAYER_TYPES = [
    "hybrid",
    "linear_attention",
    "linear_attention",
    "hybrid",
]


def _get_config(
    n_groups: int = 2,
    num_mem_blocks: int = 1,
    layer_types: list[str] | None = None,
) -> configuration_zamba2.Zamba2Config:
  layer_types = layer_types or _LAYER_TYPES
  return configuration_zamba2.Zamba2Config(
      vocab_size=128,
      hidden_size=64,
      intermediate_size=128,
      num_hidden_layers=len(layer_types),
      layers_block_type=layer_types,
      num_attention_heads=4,
      # The shared attention adapters need as many KV heads as heads, as in the
      # published models.
      num_key_value_heads=4,
      n_mamba_heads=8,
      mamba_d_state=16,
      mamba_ngroups=n_groups,
      mamba_d_conv=4,
      chunk_size=8,
      num_mem_blocks=num_mem_blocks,
      use_shared_attention_adapter=True,
      adapter_rank=8,
      use_mem_rope=True,
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


def _get_model(**config_kwargs) -> modeling_zamba2.Zamba2ForCausalLM:
  model = modeling_zamba2.Zamba2ForCausalLM(_get_config(**config_kwargs))
  _randomize_ssm_parameters(model)
  return model.eval()


def _get_mixer(n_groups: int = 2) -> modeling_zamba2.Zamba2MambaMixer:
  mixer = _MixerCls(_get_config(n_groups), layer_idx=0)
  _randomize_ssm_parameters(mixer)
  return mixer.eval()


def _causal_mask(input_pos: torch.Tensor) -> torch.Tensor:
  cache_pos = torch.arange(_CACHE_LENGTH).view(1, 1, 1, -1)
  visible = cache_pos <= input_pos.view(1, 1, -1, 1)
  return torch.where(visible, 0.0, _MASKED)


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
    y = granitemoehybrid_patch.mamba2_mixer_forward(
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
    rel = granitemoehybrid_patch.check_mamba2_mixer_forward(
        _MixerCls, patch._check_config()
    )
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
      actual = granitemoehybrid_patch.mamba2_mixer_forward(mixer, x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  def test_cache_layers_and_flatten_names(self):
    config = _get_config()
    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy", cache_length=_CACHE_LENGTH
    )
    kv_cache = cache_lib.LiteRTLMCache.create_from_config(
        config, export_config
    )
    self.assertEqual(
        [type(layer) for layer in kv_cache.layers],
        [
            cache_lib.LiteRTLMConvCacheLayer,
            cache_lib.LiteRTLMHybridCacheLayer,
            cache_lib.LiteRTLMConvCacheLayer,
            cache_lib.LiteRTLMHybridCacheLayer,
        ],
    )
    mamba2_layer_types = (
        cache_lib.LiteRTLMConvCacheLayer,
        cache_lib.LiteRTLMHybridCacheLayer,
    )
    for i, layer in enumerate(kv_cache.layers):
      if not isinstance(layer, mamba2_layer_types):
        self.fail(f"Layer {i} is a {type(layer).__name__}.")
      # conv_dim = heads * head_dim + 2 * groups * state_size, width kernel - 1.
      self.assertEqual(layer.conv_states.shape, (1, 8 * 16 + 2 * 2 * 16, 3))
      self.assertEqual(layer.recurrent_states.shape, (1, 8, 16, 16))
      self.assertEqual(layer.recurrent_states.dtype, torch.float32)
    layer_1, layer_3 = kv_cache.layers[1], kv_cache.layers[3]
    if not (
        isinstance(layer_1, cache_lib.LiteRTLMHybridCacheLayer)
        and isinstance(layer_3, cache_lib.LiteRTLMHybridCacheLayer)
    ):
      self.fail("Layers 1 and 3 are not hybrid cache layers.")
    keys, values = layer_1.keys, layer_3.values
    if keys is None or values is None:
      self.fail("The hybrid layers have no KV cache.")
    # The shared attention block attends over 2 * hidden_size channels with
    # head_dim = 2 * hidden_size // num_attention_heads.
    self.assertEqual(keys.shape, (1, 4, _CACHE_LENGTH, 32))
    self.assertEqual(values.shape, (1, 4, 32, _CACHE_LENGTH))
    flattened, context = cache_lib._flatten_kvc_t(kv_cache)
    self.assertEqual(
        context[0],
        [
            "c_0",
            "r_0",
            "k_1",
            "v_1",
            "c_1",
            "r_1",
            "c_2",
            "r_2",
            "k_3",
            "v_3",
            "c_3",
            "r_3",
        ],
    )
    unflattened = cache_lib._unflatten_kvc_t(flattened, context)
    self.assertEqual(
        [type(layer) for layer in unflattened.layers],
        [type(layer) for layer in kv_cache.layers],
    )

  def test_shared_block_weights_are_tied(self):
    model = _get_model()
    first, second = (
        layer.shared_transformer
        for layer in (model.model.layers[1], model.model.layers[3])
        if isinstance(layer, modeling_zamba2.Zamba2HybridLayer)
    )
    self.assertIsNot(first, second)
    self.assertIs(
        first.self_attn.q_proj.weight, second.self_attn.q_proj.weight
    )
    self.assertIs(
        first.feed_forward.down_proj.weight,
        second.feed_forward.down_proj.weight,
    )

  @parameterized.named_parameters(
      ("full_chunk", 16, 16),
      ("padded", 11, 16),
      ("two_chunks_padded", 13, 24),
      ("one_group", 11, 16, 1),
      ("two_mem_blocks", 11, 16, 2, 2, _TWO_MEM_BLOCKS_LAYER_TYPES),
  )
  def test_prefill_and_decode_match_transformers(
      self,
      prompt_len,
      prefill_len,
      n_groups=2,
      num_mem_blocks=1,
      layer_types=None,
  ):
    num_decode = 4
    model = _get_model(
        n_groups=n_groups,
        num_mem_blocks=num_mem_blocks,
        layer_types=layer_types,
    )
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
    with patch.patch_zamba2_model(model, export_config):
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

  def test_patch_binds_and_restores_mixer_forward(self):
    model = _get_model()
    mixers = [m for m in model.modules() if isinstance(m, _MixerCls)]
    # The Mamba2 layers and the hybrid layers each have a mixer.
    self.assertLen(mixers, 4)
    with patch.patch_zamba2_model(model, None):
      for mixer in mixers:
        self.assertIs(
            mixer.forward.__func__, granitemoehybrid_patch.mamba2_mixer_forward
        )
    for mixer in mixers:
      self.assertNotIn("forward", vars(mixer))

  @parameterized.named_parameters(
      ("prefill_padded", 16, 11),
      ("decode", 1, None),
  )
  def test_converted_mixer_matches_and_is_gpu_friendly(
      self, seq_len, num_valid
  ):
    module = _MixerWrapper(_get_mixer()).eval()
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
