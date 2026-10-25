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
"""Tests for the NemotronH export patch."""

from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.core import cache as cache_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext.granitemoehybrid import patch as granitemoehybrid_patch
from litert_torch.generative.export_hf.model_ext.nemotron_h import patch
import numpy as np
import torch
from transformers.models.nemotron_h import configuration_nemotron_h
from transformers.models.nemotron_h import modeling_nemotron_h

from absl.testing import absltest as googletest

_MixerCls = modeling_nemotron_h.NemotronHMamba2Mixer
# The value the LiteRT-LM runtime writes to masked positions of a float mask.
_MASKED = -0.7 * torch.finfo(torch.float32).max
_CACHE_LENGTH = 64
_LAYER_TYPES = [
    "linear_attention",
    "mlp",
    "full_attention",
    "moe",
    "linear_attention",
]


def _get_config(n_groups: int = 2) -> configuration_nemotron_h.NemotronHConfig:
  return configuration_nemotron_h.NemotronHConfig(
      vocab_size=128,
      hidden_size=64,
      intermediate_size=128,
      layers_block_type=_LAYER_TYPES,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=16,
      mamba_num_heads=8,
      mamba_head_dim=16,
      ssm_state_size=16,
      n_groups=n_groups,
      conv_kernel=4,
      chunk_size=8,
      n_routed_experts=4,
      num_experts_per_tok=2,
      moe_intermediate_size=32,
      moe_shared_expert_intermediate_size=32,
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


def _get_model(n_groups: int = 2) -> modeling_nemotron_h.NemotronHForCausalLM:
  model = modeling_nemotron_h.NemotronHForCausalLM(_get_config(n_groups))
  _randomize_ssm_parameters(model)
  return model.eval()


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
    mixer = _MixerCls(_get_config(n_groups), layer_idx=0)
    _randomize_ssm_parameters(mixer)
    x = torch.randn(1, seq_len, 64)
    with torch.no_grad():
      expected = _MixerCls.forward(mixer.eval(), x)
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
    self.assertLen(kv_cache.layers, len(_LAYER_TYPES))
    self.assertIsInstance(
        kv_cache.layers[0], cache_lib.LiteRTLMConvCacheLayer
    )
    for i, layer_type in ((1, "mlp"), (3, "moe")):
      layer = kv_cache.layers[i]
      if not isinstance(layer, cache_lib.LiteRTLMStatelessCacheLayer):
        self.fail(f"Layer {i} is a {type(layer).__name__}.")
      self.assertEqual(layer.layer_type, layer_type)
    self.assertEqual(
        kv_cache.layers[0].conv_states.shape, (1, 8 * 16 + 2 * 2 * 16, 3)
    )
    self.assertEqual(kv_cache.layers[0].recurrent_states.shape, (1, 8, 16, 16))

    flattened, context = cache_lib._flatten_kvc_t(kv_cache)
    # The stateless layers own no tensors, and the others keep their absolute
    # layer index in the names.
    self.assertEqual(
        context[0], ["c_0", "r_0", "k_2", "v_2", "c_4", "r_4"]
    )
    unflattened = cache_lib._unflatten_kvc_t(flattened, context)
    self.assertEqual(
        [type(layer) for layer in unflattened.layers],
        [type(layer) for layer in kv_cache.layers],
    )
    moe_layer = unflattened.layers[3]
    if not isinstance(moe_layer, cache_lib.LiteRTLMStatelessCacheLayer):
      self.fail(f"Layer 3 is a {type(moe_layer).__name__}.")
    self.assertEqual(moe_layer.layer_type, "moe")

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
    with patch.patch_nemotron_h_model(model, export_config):
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

  def test_patch_reaches_mixers_built_from_mixer_types(self):
    # NemotronHBlock instantiates its mixer from the module-level MIXER_TYPES
    # dict, so the patch must bind the forward on the instances.
    model = _get_model()
    mixers = [m for m in model.modules() if isinstance(m, _MixerCls)]
    self.assertLen(mixers, 2)
    with patch.patch_nemotron_h_model(model, None):
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
    mixer = _MixerCls(_get_config(), layer_idx=0)
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
