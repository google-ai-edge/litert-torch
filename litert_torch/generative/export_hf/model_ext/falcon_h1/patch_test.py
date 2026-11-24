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
"""Tests for the FalconH1 (Falcon-H1) export patch."""

import importlib
import sys

from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.core import cache as cache_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext import patches as patches_lib
from litert_torch.generative.export_hf.model_ext.falcon_h1 import extension
from litert_torch.generative.export_hf.model_ext.falcon_h1 import patch
from litert_torch.generative.export_hf.model_ext.granitemoehybrid import patch as granitemoehybrid_patch
import numpy as np
import torch
from transformers.models.falcon_h1 import configuration_falcon_h1
from transformers.models.falcon_h1 import modeling_falcon_h1

from absl.testing import absltest as googletest

_MixerCls = modeling_falcon_h1.FalconH1Mixer
# The value the LiteRT-LM runtime writes to masked positions of a float mask.
_MASKED = -0.7 * torch.finfo(torch.float32).max
_CACHE_LENGTH = 64
# (mamba_rms_norm, mamba_norm_before_gate): no norm, as in Falcon-H1-0.5B, and
# the gated norm after and before the gate.
_NORMS = ((False, False), (True, True), (True, False))


def _get_config(
    mamba_rms_norm: bool = False,
    mamba_norm_before_gate: bool = False,
    n_groups: int = 1,
) -> configuration_falcon_h1.FalconH1Config:
  config = configuration_falcon_h1.FalconH1Config(
      vocab_size=128,
      hidden_size=64,
      intermediate_size=128,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      mamba_d_ssm=128,
      mamba_n_heads=8,
      mamba_d_state=16,
      mamba_n_groups=n_groups,
      mamba_d_conv=4,
      mamba_chunk_size=8,
      mamba_rms_norm=mamba_rms_norm,
      mamba_norm_before_gate=mamba_norm_before_gate,
      embedding_multiplier=2.0,
      lm_head_multiplier=0.5,
      key_multiplier=0.7,
      attention_in_multiplier=1.1,
      attention_out_multiplier=0.9,
      mlp_multipliers=[0.8, 0.6],
      ssm_in_multiplier=1.25,
      ssm_multipliers=[0.35, 0.25, 0.35, 0.5, 0.35],
      ssm_out_multiplier=0.25,
      pad_token_id=0,
  )
  # Not hidden_size // num_attention_heads, as in the published models. It is
  # not a FalconH1Config field; the attention reads it with getattr.
  config.head_dim = 8
  return config


def _randomize_ssm_parameters(module: torch.nn.Module):
  """Moves A, dt_bias and D away from their initial values."""
  for m in module.modules():
    if isinstance(m, _MixerCls):
      with torch.no_grad():
        m.A_log.uniform_(-1.0, 2.0)
        m.dt_bias.uniform_(-3.0, 1.0)
        m.D.uniform_(0.5, 1.5)
        # The mixer has the gated norm only with mamba_rms_norm.
        if isinstance(m.norm, modeling_falcon_h1.FalconH1RMSNormGated):
          m.norm.weight.uniform_(0.5, 1.5)


def _get_model(**config_kwargs) -> modeling_falcon_h1.FalconH1ForCausalLM:
  model = modeling_falcon_h1.FalconH1ForCausalLM(_get_config(**config_kwargs))
  _randomize_ssm_parameters(model)
  return model.eval()


def _get_mixer(**config_kwargs) -> modeling_falcon_h1.FalconH1Mixer:
  # The model registers the `mup_vector` buffer on its mixers.
  layer = _get_model(**config_kwargs).model.layers[0]
  if not isinstance(layer, modeling_falcon_h1.FalconH1DecoderLayer):
    raise TypeError(f"Layer 0 is a {type(layer).__name__}.")
  return layer.mamba.eval()


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

  @parameterized.parameters(*_NORMS)
  def test_check_mamba2_mixer_forward(
      self, mamba_rms_norm, mamba_norm_before_gate
  ):
    config = _get_config(mamba_rms_norm, mamba_norm_before_gate)
    rel = granitemoehybrid_patch.check_mamba2_mixer_forward(
        patch._FalconH1Mixer, patch._check_config(config)
    )
    self.assertLess(rel, 2e-4)

  @parameterized.product(
      norm=_NORMS,
      n_groups=[1, 2],
      seq_len=[16, 13, 5],
  )
  def test_mixer_forward_matches_transformers(self, norm, n_groups, seq_len):
    mixer = _get_mixer(
        mamba_rms_norm=norm[0],
        mamba_norm_before_gate=norm[1],
        n_groups=n_groups,
    )
    x = torch.randn(1, seq_len, 64)
    with torch.no_grad():
      expected = _MixerCls.forward(mixer, x)
      actual = granitemoehybrid_patch.mamba2_mixer_forward(mixer, x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

  def test_extension_registers_the_model_patch(self):
    ext = extension_lib.resolve_model_extension("falcon_h1")
    self.assertIsInstance(ext, extension.FalconH1Extension)
    model = _get_model()
    mixers = [m for m in model.modules() if isinstance(m, _MixerCls)]
    with patches_lib.patch_model(model, "falcon_h1", None):
      for mixer in mixers:
        self.assertIs(
            mixer.forward.__func__, granitemoehybrid_patch.mamba2_mixer_forward
        )
    for mixer in mixers:
      self.assertNotIn("forward", vars(mixer))

  def test_extension_refuses_the_unsupported_export_options(self):
    ext = extension.FalconH1Extension()
    config = _get_config()
    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy"
    )
    self.assertIs(
        ext.update_export_config(export_config, config), export_config
    )
    for option, export_config in (
        (
            "split_cache",
            exportable_module_config.ExportableModuleConfig(
                model="dummy", split_cache=True
            ),
        ),
        (
            "externalize_embedder",
            exportable_module_config.ExportableModuleConfig(
                model="dummy", externalize_embedder=True
            ),
        ),
    ):
      with self.assertRaisesRegex(
          ValueError, f"Falcon-H1 does not support {option};"
      ):
        ext.update_export_config(export_config, config)

  def test_importing_the_extension_registers_the_patches(self):
    # The export imports only the extension module, so that import has to
    # register the patches of patch.py. Both modules are imported again here
    # to see it, and put back after.
    package = sys.modules[extension.__name__.rpartition(".")[0]]
    modules = {"extension": extension, "patch": patch}
    load_patch = patches_lib._CONTEXT_REGISTRY.pop("falcon_h1")
    model_patch = patches_lib._MODEL_PATCH_REGISTRY.pop("falcon_h1")
    for name, module in modules.items():
      del sys.modules[module.__name__]
      delattr(package, name)
    try:
      importlib.import_module(extension.__name__)
      self.assertIn("falcon_h1", patches_lib._CONTEXT_REGISTRY)
      self.assertIn("falcon_h1", patches_lib._MODEL_PATCH_REGISTRY)
    finally:
      for name, module in modules.items():
        sys.modules[module.__name__] = module
        setattr(package, name, module)
      patches_lib._CONTEXT_REGISTRY["falcon_h1"] = load_patch
      patches_lib._MODEL_PATCH_REGISTRY["falcon_h1"] = model_patch

  def test_load_time_patch_builds_the_hlfb_rms_norms(self):
    original_norm = modeling_falcon_h1.FalconH1RMSNorm
    with patches_lib.get_patch_context("falcon_h1"):
      model = _get_model(mamba_rms_norm=True)
    self.assertIs(modeling_falcon_h1.FalconH1RMSNorm, original_norm)
    norms = [m for m in model.modules() if isinstance(m, patch.FalconH1RMSNorm)]
    # Two per layer and the final norm; the gated norm of the mixer is another
    # class.
    self.assertLen(norms, 2 * len(model.model.layers) + 1)

  def test_rms_norm_matches_the_transformers_formula(self):
    norm = patch.FalconH1RMSNorm(64, eps=1e-5)
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
        patch.FalconH1RMSNorm(64).variance_epsilon,
        modeling_falcon_h1.FalconH1RMSNorm(64).variance_epsilon,
    )

  def test_cache_layers_and_flatten_names(self):
    config = _get_config()
    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy", cache_length=_CACHE_LENGTH
    )
    kv_cache = cache_lib.LiteRTLMCache.create_from_config(
        config, export_config
    )
    self.assertLen(kv_cache.layers, 2)
    for i, layer in enumerate(kv_cache.layers):
      if not isinstance(layer, cache_lib.LiteRTLMHybridCacheLayer):
        self.fail(f"Layer {i} is a {type(layer).__name__}.")
      self.assertEqual(layer.layer_type, "hybrid")
      keys, values = layer.keys, layer.values
      if keys is None or values is None:
        self.fail("The hybrid layer has no KV cache.")
      # The attention head dim is `head_dim`, not hidden_size // heads.
      self.assertEqual(keys.shape, (1, 2, _CACHE_LENGTH, 8))
      self.assertEqual(values.shape, (1, 2, 8, _CACHE_LENGTH))
      # Channels-last (B, kernel - 1, conv_dim), with
      # conv_dim = mamba_d_ssm + 2 * groups * state_size.
      self.assertEqual(layer.conv_states.shape, (1, 3, 128 + 2 * 16))
      self.assertEqual(layer.recurrent_states.shape, (1, 8, 16, 16))
      self.assertEqual(layer.recurrent_states.dtype, torch.float32)
    flattened, context = cache_lib._flatten_kvc_t(kv_cache)
    self.assertEqual(
        context[0], ["k_0", "v_0", "c_0", "r_0", "k_1", "v_1", "c_1", "r_1"]
    )
    unflattened = cache_lib._unflatten_kvc_t(flattened, context)
    for layer, tensors in zip(
        unflattened.layers, (flattened[:4], flattened[4:])
    ):
      if not isinstance(layer, cache_lib.LiteRTLMHybridCacheLayer):
        self.fail(f"An unflattened layer is a {type(layer).__name__}.")
      for actual, expected in zip(
          (layer.keys, layer.values, layer.conv_states, layer.recurrent_states),
          tensors,
      ):
        self.assertIs(actual, expected)

  @parameterized.named_parameters(
      ("full_chunk", 16, 16),
      ("padded", 11, 16),
      ("two_chunks_padded", 13, 24),
      ("two_groups", 11, 16, 2),
      ("rms_norm", 11, 16, 1, True, True),
      ("rms_norm_after_gate", 11, 16, 1, True, False),
  )
  def test_prefill_and_decode_match_transformers(
      self,
      prompt_len,
      prefill_len,
      n_groups=1,
      mamba_rms_norm=False,
      mamba_norm_before_gate=False,
  ):
    num_decode = 4
    model = _get_model(
        mamba_rms_norm=mamba_rms_norm,
        mamba_norm_before_gate=mamba_norm_before_gate,
        n_groups=n_groups,
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
    with patch.patch_falcon_h1_model(model, export_config):
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
    # Every FalconH1 layer is a hybrid layer with a Mamba2 mixer.
    self.assertLen(mixers, 2)
    with patch.patch_falcon_h1_model(model, None):
      for mixer in mixers:
        self.assertIs(
            mixer.forward.__func__, granitemoehybrid_patch.mamba2_mixer_forward
        )
    for mixer in mixers:
      self.assertNotIn("forward", vars(mixer))

  @parameterized.parameters("extra_kwarg", "kwargs")
  def test_layers_get_model_kwargs_the_base_model_drops(self, name):
    # `kwargs` is also the name of the base model forward's `**kwargs`
    # parameter, which takes a keyword argument of that name too.
    model = _get_model()
    layers = list(model.model.layers)
    seen = []

    # FalconH1Model of transformers 5.16.1 does not pass the keyword arguments
    # of the model call on to its decoder layers. This pre-hook drops one from
    # every layer call, as that version does.
    def drop_kwarg(module, args, kwargs):
      del module
      return args, {k: v for k, v in kwargs.items() if k != name}

    # The keyword argument, and whether the layer call has `input_ids`, which
    # the base model forward takes by name.
    def record_kwargs(module, args, kwargs):
      del module, args
      seen.append((kwargs.get(name), "input_ids" in kwargs))

    def run():
      seen.clear()
      with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3]]), **{name: "value"})
      return list(seen)

    handles = [
        layer.register_forward_pre_hook(drop_kwarg, with_kwargs=True)
        for layer in layers
    ]
    try:
      with patch._pass_model_kwargs_to_layers(model):
        # Registered after the patch's own pre-hooks, so it sees what the
        # layer forward gets.
        handles += [
            layer.register_forward_pre_hook(record_kwargs, with_kwargs=True)
            for layer in layers
        ]
        self.assertEqual(run(), [("value", False)] * len(layers))
      # Without the patch, the layers do not get it.
      self.assertEqual(run(), [(None, False)] * len(layers))
    finally:
      for handle in handles:
        handle.remove()

  def test_layers_keep_their_own_kwargs(self):
    model = _get_model()
    layers = list(model.model.layers)
    seen = []

    # Gives every layer call its own value of the keyword argument.
    def set_kwarg(module, args, kwargs):
      del module
      return args, {**kwargs, "extra_kwarg": "layer value"}

    def record_kwarg(module, args, kwargs):
      del module, args
      seen.append(kwargs.get("extra_kwarg"))

    handles = [
        layer.register_forward_pre_hook(set_kwarg, with_kwargs=True)
        for layer in layers
    ]
    try:
      with patch._pass_model_kwargs_to_layers(model):
        # Registered after the patch's own pre-hooks, so it sees what the
        # layer forward gets.
        handles += [
            layer.register_forward_pre_hook(record_kwarg, with_kwargs=True)
            for layer in layers
        ]
        with torch.no_grad():
          model(input_ids=torch.tensor([[1, 2, 3]]), extra_kwarg="model value")
    finally:
      for handle in handles:
        handle.remove()
    # The patch adds only the keyword arguments that a layer call lacks.
    self.assertEqual(seen, ["layer value"] * len(layers))

  def test_layers_do_not_get_the_kwargs_of_a_call_that_raised(self):
    model = _get_model()
    layers = list(model.model.layers)
    raised = []
    seen = []

    class _Stop(Exception):
      pass

    def stop_once(module, args, kwargs):
      del module, args, kwargs
      if not raised:
        raised.append(True)
        raise _Stop()

    def record_kwarg(module, args, kwargs):
      del module, args
      seen.append(kwargs.get("extra_kwarg"))

    with patch._pass_model_kwargs_to_layers(model):
      # Registered after the patch's own pre-hooks, so they see what the layer
      # forward gets.
      handles = [
          layers[0].register_forward_pre_hook(stop_once, with_kwargs=True)
      ]
      handles += [
          layer.register_forward_pre_hook(record_kwarg, with_kwargs=True)
          for layer in layers
      ]
      try:
        with torch.no_grad():
          with self.assertRaises(_Stop):
            model(input_ids=torch.tensor([[1, 2, 3]]), extra_kwarg="value")
          # The next call has no extra_kwarg, and the layers do not get the
          # one of the call that raised.
          model(input_ids=torch.tensor([[1, 2, 3]]))
      finally:
        for handle in handles:
          handle.remove()
    self.assertEqual(seen, [None] * len(layers))

  @parameterized.named_parameters(
      ("prefill_padded", 16, 11),
      ("decode", 1, None),
      ("rms_norm_decode", 1, None, True),
  )
  def test_converted_mixer_matches_and_is_gpu_friendly(
      self, seq_len, num_valid, mamba_rms_norm=False
  ):
    mixer = _get_mixer(mamba_rms_norm=mamba_rms_norm)
    module = _MixerWrapper(mixer).eval()
    args = (
        torch.randn(1, seq_len, 64),
        torch.randn(1, 3, 128 + 2 * 16),
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
