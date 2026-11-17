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
"""Tests for Gemma 4 exportable modules, extension hooks, and attention fixes."""

import dataclasses
import os
from unittest import mock

from absl.testing import parameterized
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import weights_loader
from litert_torch.generative.export_hf.core.external_rope import preprocess_model as external_rope_preprocess_model
from litert_torch.generative.export_hf.core.split_cache import attention as split_cache_attention
from litert_torch.generative.export_hf.core.split_cache import combined_layers
from litert_torch.generative.export_hf.core.split_cache import exportable_module as split_cache_exportable
from litert_torch.generative.export_hf.experimental.composites import qkv_norm_rope
import litert_torch.generative.export_hf.model_ext as _
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.gemma4 import exportable_module as gemma4_exportable
from litert_torch.generative.export_hf.model_ext.gemma4 import patch as gemma4_patch
from litert_torch.generative.export_hf.model_ext.gemma4 import split_cache_exportable_module as gemma4_split_cache
import torch
from transformers.models.gemma4 import configuration_gemma4
from transformers.models.gemma4 import modeling_gemma4

from absl.testing import absltest as googletest
from ai_edge_litert import interpreter as interpreter_lib  # pylint: disable=g-direct-tensorflow-import


def _make_small_text_config(
    *,
    num_hidden_layers: int = 3,
    num_kv_shared_layers: int = 1,
    hidden_size_per_layer_input: int = 8,
    final_logit_softcapping: float | None = 30.0,
    num_global_key_value_heads: int | None = None,
    attention_k_eq_v: bool = False,
    per_layer_config: dict[int, dict[str, int]] | None = None,
    attn_implementation: str = "lrt_transposed_attention",
) -> configuration_gemma4.Gemma4TextConfig:
  layer_types = [
      "full_attention" if i == 1 else "sliding_attention"
      for i in range(num_hidden_layers)
  ]
  extra_kwargs = {}
  if per_layer_config is not None:
    extra_kwargs["per_layer_config"] = per_layer_config
  cfg = configuration_gemma4.Gemma4TextConfig(
      vocab_size=64,
      vocab_size_per_layer_input=64,
      hidden_size=32,
      intermediate_size=64,
      hidden_size_per_layer_input=hidden_size_per_layer_input,
      num_hidden_layers=num_hidden_layers,
      num_kv_shared_layers=num_kv_shared_layers,
      num_attention_heads=4,
      num_key_value_heads=2,
      num_global_key_value_heads=num_global_key_value_heads,
      attention_k_eq_v=attention_k_eq_v,
      head_dim=16,
      global_head_dim=16,
      sliding_window=16,
      layer_types=layer_types,
      final_logit_softcapping=final_logit_softcapping,
      attention_dropout=0.0,
      hidden_activation="gelu_pytorch_tanh",
      **extra_kwargs,
  )
  cfg._attn_implementation = attn_implementation
  return cfg


def _make_small_multimodal_config(
    **text_kwargs,
) -> configuration_gemma4.Gemma4Config:
  text_cfg = _make_small_text_config(**text_kwargs)
  vision_cfg = configuration_gemma4.Gemma4VisionConfig(
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=1,
      num_attention_heads=2,
      num_key_value_heads=2,
      head_dim=8,
      patch_size=4,
      pooling_kernel_size=1,
      default_output_length=4,
  )
  cfg = configuration_gemma4.Gemma4Config(
      text_config=text_cfg,
      vision_config=vision_cfg,
      audio_config=None,
  )
  cfg._attn_implementation = text_cfg._attn_implementation
  return cfg


class Gemma4ExportableModuleTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("multimodal_fp32", False, None, False),
      ("causal_lm_fp32", True, None, False),
      ("multimodal_bf16_verifier", False, "bfloat16", True),
      ("causal_lm_bf16", True, "bfloat16", False),
  )
  def test_generate_applies_final_logit_softcapping_and_fp32_cast(
      self,
      use_causal_lm: bool,
      model_dtype: str | None,
      export_verifier: bool,
  ):
    torch.manual_seed(0)
    with gemma4_patch.gemma4_litert_patch():
      if use_causal_lm:
        cfg = _make_small_text_config(final_logit_softcapping=30.0)
        model = modeling_gemma4.Gemma4ForCausalLM(cfg).eval()
      else:
        cfg = _make_small_multimodal_config(final_logit_softcapping=30.0)
        model = modeling_gemma4.Gemma4ForConditionalGeneration(cfg).eval()
      model.set_attn_implementation("lrt_transposed_attention")

    with torch.no_grad():
      # Amplify lm_head so raw pre-cap logits exceed +-30.0.
      model.lm_head.weight.mul_(200.0)
      if model_dtype == "bfloat16":
        weights_loader.cast_params_(model, torch.bfloat16)

    text_cfg = cfg.get_text_config()
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        externalize_embedder=True,
        single_token_embedder=True,
        cache_length=32,
        prefill_lengths=[8],
        sliding_window_ring_buffer_size=32,
        apply_gpu_composites=True,
        model_dtype=model_dtype,
        assistant_model="dummy_draft" if export_verifier else None,
        mtp_verifier_step=3,
    )
    gen_mod = (
        gemma4_exportable.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder(
            model, export_cfg
        ).eval()
    )
    sample_inputs = gen_mod.get_sample_inputs(text_cfg)
    decode_inputs, _ = sample_inputs["decode"]
    with (
        torch.no_grad(),
        gemma4_patch.patch_gemma4_model(model, export_cfg),
    ):
      out = gen_mod(**decode_inputs)

      logits = out["logits"]
      self.assertEqual(logits.dtype, torch.float32)
      self.assertLessEqual(logits.abs().max().item(), 30.0 + 1e-4)

      # Verify exact parity against softcapped reference logits.
      kv_inputs, _ = gen_mod.get_sample_kv_cache(text_cfg)
      ref_inputs = dict(decode_inputs)
      ref_inputs["kv_cache"] = kv_inputs["kv_cache"]
      adapted = gen_mod.adapt_inputs(
          None,
          ref_inputs["embeddings"],
          ref_inputs["per_layer_embeddings"],
          ref_inputs["input_pos"],
          ref_inputs["kv_cache"],
          ref_inputs["mask"],
          use_bool_mask=True,
          local_mask=ref_inputs.get("local_mask"),
          param_tensor=ref_inputs.get("param_tensor"),
          apply_gpu_composites=True,
          use_sdpa_composite=True,
      )
      adapted["past_key_values"] = adapted[
          "past_key_values"
      ].insert_dummy_cache_layers(text_cfg)
      adapted |= gen_mod.attention_kwargs()
      lm_out = gemma4_exportable.get_language_model(model)(**adapted)
      raw_logits = model.lm_head(lm_out.last_hidden_state)
      self.assertGreater(raw_logits.float().abs().max().item(), 40.0)
      expected_logits = (torch.tanh(raw_logits / 30.0) * 30.0).to(torch.float32)
      torch.testing.assert_close(logits, expected_logits, rtol=1e-4, atol=1e-4)

      if export_verifier:
        self.assertIn("activations", out)
        self.assertIn("verify", sample_inputs)
        verify_inputs, _ = sample_inputs["verify"]
        if "param_tensor" not in verify_inputs:
          verify_inputs["param_tensor"] = decode_inputs["param_tensor"]
        verify_out = gen_mod(**verify_inputs)
        self.assertEqual(verify_out["activations"].shape, (1, 3, 32))
      else:
        self.assertNotIn("activations", out)

  @parameterized.named_parameters(
      ("multimodal", False),
      ("causal_lm", True),
  )
  def test_split_cache_applies_final_logit_softcapping_and_supports_causal_lm(
      self, use_causal_lm: bool
  ):
    torch.manual_seed(1)
    with gemma4_patch.gemma4_litert_patch():
      if use_causal_lm:
        cfg = _make_small_text_config(final_logit_softcapping=30.0)
        model = modeling_gemma4.Gemma4ForCausalLM(cfg).eval()
      else:
        cfg = _make_small_multimodal_config(final_logit_softcapping=30.0)
        model = modeling_gemma4.Gemma4ForConditionalGeneration(cfg).eval()

    with torch.no_grad():
      model.lm_head.weight.mul_(200.0)

    text_cfg = cfg.get_text_config()
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        split_cache=True,
        cache_length=32,
        prefill_lengths=[8],
    )
    model = external_rope_preprocess_model.inject_rotary_position_embedding(
        model
    )
    model.set_attn_implementation("lrt_split_cache_attention")
    model.config._attn_implementation = "lrt_split_cache_attention"
    with gemma4_patch.patch_gemma4_model(model, export_cfg):
      prefill_mod = (
          gemma4_split_cache.LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill(
              model, export_cfg
          ).eval()
      )
      gen_mod = (
          gemma4_split_cache.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate(
              model, export_cfg
          ).eval()
      )

      prefill_inputs, _ = prefill_mod.get_sample_inputs(text_cfg)["prefill_8"]
      decode_inputs, _ = gen_mod.get_sample_inputs(text_cfg)["decode"]
      with torch.no_grad():
        prefill_out = prefill_mod(**prefill_inputs)
        decode_out = gen_mod(**decode_inputs)

    self.assertTrue(prefill_out)
    logits = decode_out["logits"]
    self.assertEqual(logits.dtype, torch.float32)
    # Without softcapping, amplified lm_head weights produce > 40 magnitude.
    self.assertLessEqual(logits.abs().max().item(), 30.0 + 1e-4)
    self.assertGreater(logits.abs().max().item(), 25.0)

  def _make_split_cache_modules(self, model, export_cfg, text_cfg):
    # Selects the classes as the export does: Gemma 4's split-cache classes,
    # with the combined layout added when split_cache_combined_layers is set.
    prefill_cls, gen_cls = export_lib.get_prefill_decode_exportable_cls(
        text_cfg, export_cfg
    )
    for cls, gemma4_cls in (
        (
            prefill_cls,
            gemma4_split_cache.LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill,
        ),
        (
            gen_cls,
            gemma4_split_cache.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate,
        ),
    ):
      self.assertTrue(issubclass(cls, gemma4_cls), cls)
      self.assertEqual(
          issubclass(cls, combined_layers.CombinedLayersMixin),
          export_cfg.split_cache_combined_layers,
      )
    return (
        prefill_cls(model, export_cfg).eval(),
        gen_cls(model, export_cfg).eval(),
    )

  def _make_split_cache_model(self, **config_kwargs):
    with gemma4_patch.gemma4_litert_patch():
      cfg = _make_small_text_config(**config_kwargs)
      model = modeling_gemma4.Gemma4ForCausalLM(cfg).eval()
    model = external_rope_preprocess_model.inject_rotary_position_embedding(
        model
    )
    model.set_attn_implementation("lrt_split_cache_attention")
    model.config._attn_implementation = "lrt_split_cache_attention"
    return cfg, model

  @parameterized.named_parameters(
      ("prefill", "prefill_8"),
      ("decode", "decode"),
  )
  def test_split_cache_combined_layers_matches_per_layer(self, signature):
    # 4 cache layers (Gemma 4 makes the last layer global): sliding layers 0
    # and 2, global layers 1 and 3. Layers are grouped by layer type (and cache
    # shape), so the combined layout has two groups: [0, 2] and [1, 3].
    torch.manual_seed(2)
    cfg, model = self._make_split_cache_model(
        num_hidden_layers=4,
        num_kv_shared_layers=0,
    )
    text_cfg = cfg.get_text_config()
    outputs = {}
    inputs = None
    for combined in (False, True):
      export_cfg = exportable_module_config.ExportableModuleConfig(
          model="dummy",
          split_cache=True,
          split_cache_combined_layers=combined,
          cache_length=32,
          prefill_lengths=[8],
      )
      with gemma4_patch.patch_gemma4_model(model, export_cfg):
        prefill_mod, gen_mod = self._make_split_cache_modules(
            model, export_cfg, text_cfg
        )
        mod = prefill_mod if signature.startswith("prefill") else gen_mod
        if inputs is None:
          # Per-layer inputs with random cache contents and embeddings.
          inputs, _ = mod.get_sample_inputs(text_cfg)[signature]
          for layer in inputs["kv_cache"].layers:
            layer.keys[0].normal_()
            layer.values[0].normal_()
          inputs["embeddings"].normal_()
          per_layer_cache = inputs["kv_cache"]
        else:
          inputs = dict(inputs)
          inputs["kv_cache"] = combined_layers.combine_split_cache(
              per_layer_cache
          )
          self.assertCountEqual(
              inputs["kv_cache"], ["k_0", "v_0", "k_1", "v_1"]
          )
          self.assertEqual(inputs["kv_cache"]["k_0"].shape[0], 2)
          self.assertEqual(inputs["kv_cache"]["k_1"].shape[0], 2)
          layout, _ = mod.kv_cache_group_layout()
          self.assertEqual(layout, [[0, 2], [1, 3]])
        with torch.no_grad():
          outputs[combined] = mod(**inputs)

    per_layer, combined = outputs[False], outputs[True]
    expected_slices = combined_layers.combine_slices(
        per_layer["kv_slice_k"], per_layer["kv_slice_v"], [[0, 2], [1, 3]]
    )
    for key in ("kv_slice_k", "kv_slice_v"):
      self.assertLen(combined[key], 2)
      for got, want in zip(combined[key], expected_slices[key]):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    if "logits" in per_layer:
      torch.testing.assert_close(
          combined["logits"], per_layer["logits"], rtol=1e-5, atol=1e-5
      )

  @parameterized.named_parameters(
      # (signature, start position, number of valid entries)
      ("decode_start", "decode_cache_update", 0, 1),
      ("decode_ring_end", "decode_cache_update", 31, 1),
      ("decode_ring_wrap", "decode_cache_update", 32, 1),
      ("decode_last", "decode_cache_update", 63, 1),
      ("prefill_start", "prefill_cache_update_8", 0, 8),
      ("prefill_ends_at_ring_end", "prefill_cache_update_8", 24, 8),
      ("prefill_wraps", "prefill_cache_update_8", 28, 8),
      ("prefill_wraps_padded", "prefill_cache_update_8", 30, 5),
      ("prefill_second_lap", "prefill_cache_update_8", 52, 8),
  )
  def test_combined_cache_update_matches_per_layer(
      self, signature, start, num_valid
  ):
    # Builds the auxiliary cache-update module as export_lib does, with and
    # without split_cache_combined_layers, and checks that the combined update
    # equals the per-layer update stacked into groups. Sliding layers 0 and 2
    # use a 32-slot ring buffer, global layers 1 and 3 a 64-slot cache.
    torch.manual_seed(5)
    text_cfg = _make_small_text_config(
        num_hidden_layers=4, num_kv_shared_layers=0
    )

    def make_config(combined):
      return exportable_module_config.ExportableModuleConfig(
          model="dummy",
          split_cache=True,
          split_cache_combined_layers=combined,
          cache_length=64,
          sliding_window_ring_buffer_size=32,
          prefill_lengths=[8],
      )

    def make_module(export_cfg):
      # As in export_lib.export_auxiliary_model.
      if export_cfg.split_cache_combined_layers:
        return combined_layers.CombinedCacheUpdate.from_config(
            text_cfg, export_cfg
        )
      return split_cache_exportable.CacheUpdate()

    per_layer_cfg, combined_cfg = make_config(False), make_config(True)
    self.assertEqual(
        combined_layers.ring_buffer_groups_for(text_cfg, combined_cfg),
        (True, False),
    )
    self.assertEqual(
        make_module(combined_cfg).ring_buffer_groups, (True, False)
    )

    # The combined signature inputs: one K / V tensor per group.
    combined_inputs, _ = make_module(combined_cfg).get_sample_inputs(
        text_cfg, combined_cfg
    )[signature]
    t = 1 if signature.startswith("decode") else 8
    expected_shapes = {
        "k_0": (2, 2, 32, 16),
        "v_0": (2, 2, 16, 32),
        "k_1": (2, 2, 64, 16),
        "v_1": (2, 2, 16, 64),
    }
    self.assertEqual(
        {k: tuple(v.shape) for k, v in combined_inputs["kv_cache"].items()},
        expected_shapes,
    )
    self.assertEqual(
        {k: tuple(v.shape) for k, v in combined_inputs["kv_slice"].items()},
        {
            "k_0": (2, 2, t, 16),
            "v_0": (2, 2, 16, t),
            "k_1": (2, 2, t, 16),
            "v_1": (2, 2, 16, t),
        },
    )

    # Per-layer inputs with random contents, and the same data combined.
    inputs, _ = make_module(per_layer_cfg).get_sample_inputs(
        text_cfg, per_layer_cfg
    )[signature]
    for cache in (inputs["kv_cache"], inputs["kv_slice"]):
      for layer in cache.layers:
        layer.keys.normal_()
        layer.values.normal_()
    inputs["input_pos"] = torch.arange(start, start + t, dtype=torch.int32)
    inputs["valid_mask"] = torch.arange(t).unsqueeze(0) < num_valid

    def combine(cache):
      return combined_layers.combine_kv_layers(
          [layer.keys.clone() for layer in cache.layers],
          [layer.values.clone() for layer in cache.layers],
          [layer.layer_type for layer in cache.layers],
      )

    combined = dict(
        inputs,
        kv_cache=combine(inputs["kv_cache"]),
        kv_slice=combine(inputs["kv_slice"]),
    )
    with torch.no_grad():
      want = combine(make_module(per_layer_cfg)(**inputs)["kv_cache"])
      got = make_module(combined_cfg)(**combined)["kv_cache"]
    self.assertCountEqual(got, want)
    for name in want:
      torch.testing.assert_close(got[name], want[name], rtol=0, atol=0)

  def test_export_auxiliary_model_combined_layers(self):
    # Runs the real auxiliary-model export (RoPE, masks, cache update) with
    # split_cache_combined_layers and two context buckets, and checks the cache
    # update signatures of the converted model: one K / V cache input and one
    # K / V slice input per group, in every bucket.
    torch.manual_seed(6)
    text_cfg, model = self._make_split_cache_model(
        num_hidden_layers=4, num_kv_shared_layers=0
    )
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        split_cache=True,
        split_cache_combined_layers=True,
        cache_length=128,
        cache_lengths=[64, 128],
        sliding_window_ring_buffer_size=32,
        prefill_lengths=[8],
        work_dir=self.create_tempdir().full_path,
    )
    artifacts = export_lib.export_auxiliary_model(
        export_lib.SourceModelArtifacts(
            model=model,
            model_config=text_cfg,
            text_model_config=text_cfg,
            tokenizer=None,
        ),
        export_cfg,
        export_lib.ExportedModelArtifacts(),
    )

    interpreter = interpreter_lib.Interpreter(
        model_path=artifacts.auxiliary_model_path
    )
    signatures = interpreter.get_signature_list()
    for cache_len in (64, 128):
      for name, t in (
          (f"prefill_cache_update_8_cache_{cache_len}", 8),
          (f"decode_cache_update_cache_{cache_len}", 1),
      ):
        self.assertIn(name, signatures)
        runner = interpreter.get_signature_runner(name)
        in_shapes = {
            k: tuple(v["shape"])
            for k, v in runner.get_input_details().items()
            if k.startswith(("kv_cache", "kv_slice"))
        }
        self.assertEqual(
            in_shapes,
            {
                # Sliding layers 0 and 2: 32-slot ring buffer.
                "kv_cache_k_0": (2, 2, 32, 16),
                "kv_cache_v_0": (2, 2, 16, 32),
                # Global layers 1 and 3: the bucket's cache length.
                "kv_cache_k_1": (2, 2, cache_len, 16),
                "kv_cache_v_1": (2, 2, 16, cache_len),
                "kv_slice_k_0": (2, 2, t, 16),
                "kv_slice_v_0": (2, 2, 16, t),
                "kv_slice_k_1": (2, 2, t, 16),
                "kv_slice_v_1": (2, 2, 16, t),
            },
            name,
        )
        out_names = sorted(runner.get_output_details())
        self.assertEqual(
            out_names,
            ["kv_cache_k_0", "kv_cache_k_1", "kv_cache_v_0", "kv_cache_v_1"],
            name,
        )

  def test_split_cache_combined_layers_tflite_signature(self):
    torch.manual_seed(3)
    cfg, model = self._make_split_cache_model(
        num_hidden_layers=4,
        num_kv_shared_layers=0,
    )
    text_cfg = cfg.get_text_config()
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        split_cache=True,
        split_cache_combined_layers=True,
        cache_length=32,
        prefill_lengths=[8],
    )
    with gemma4_patch.patch_gemma4_model(model, export_cfg):
      _, gen_mod = self._make_split_cache_modules(model, export_cfg, text_cfg)
      out_tflite = os.path.join(
          self.create_tempdir().full_path, "decode.tflite"
      )
      export_lib._convert_and_export_module(  # pylint: disable=protected-access
          gen_mod,
          gen_mod.get_sample_inputs(text_cfg),
          out_tflite,
          export_cfg,
          weights_loader=None,
      )
    interpreter = interpreter_lib.Interpreter(model_path=out_tflite)
    runner = interpreter.get_signature_runner("decode")
    in_shapes = {
        k: tuple(v["shape"]) for k, v in runner.get_input_details().items()
    }
    out_shapes = {
        k: tuple(v["shape"]) for k, v in runner.get_output_details().items()
    }
    # One K / V cache input and one K / V slice output per group.
    self.assertEqual(
        sorted(k for k in in_shapes if k.startswith("kv_cache")),
        ["kv_cache_k_0", "kv_cache_k_1", "kv_cache_v_0", "kv_cache_v_1"],
    )
    self.assertEqual(in_shapes["kv_cache_k_0"], (2, 2, 32, 16))
    self.assertEqual(in_shapes["kv_cache_v_0"], (2, 2, 16, 32))
    self.assertEqual(in_shapes["kv_cache_k_1"], (2, 2, 32, 16))
    self.assertEqual(
        sorted(k for k in out_shapes if k.startswith("kv_slice")),
        ["kv_slice_k_0", "kv_slice_k_1", "kv_slice_v_0", "kv_slice_v_1"],
    )
    self.assertEqual(out_shapes["kv_slice_k_0"], (2, 2, 1, 16))
    self.assertEqual(out_shapes["kv_slice_v_1"], (2, 2, 16, 1))

  @parameterized.named_parameters(
      ("experimental_fp16", True, None, torch.float16),
      ("bfloat16_model", False, "bfloat16", torch.bfloat16),
  )
  def test_prefill_and_generate_cast_per_layer_embeddings_to_compute_dtype(
      self,
      experimental_use_fp16: bool,
      model_dtype: str | None,
      expected_dtype: torch.dtype,
  ):
    with gemma4_patch.gemma4_litert_patch():
      cfg = _make_small_multimodal_config()
      model = modeling_gemma4.Gemma4ForConditionalGeneration(cfg).eval()
    if model_dtype == "bfloat16":
      weights_loader.cast_params_(model, torch.bfloat16)

    text_cfg = cfg.get_text_config()
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        externalize_embedder=True,
        single_token_embedder=True,
        cache_length=32,
        prefill_lengths=[8],
        sliding_window_ring_buffer_size=32,
        apply_gpu_composites=True,
        experimental_use_fp16=experimental_use_fp16,
        model_dtype=model_dtype,
    )
    prefill_mod = (
        gemma4_exportable.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder(
            model, export_cfg
        ).eval()
    )
    gen_mod = (
        gemma4_exportable.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder(
            model, export_cfg
        ).eval()
    )

    prefill_inputs, _ = prefill_mod.get_sample_inputs(text_cfg)["prefill_8"]
    adapted_prefill = prefill_mod.adapt_inputs(
        None,
        prefill_inputs["embeddings"],
        prefill_inputs["per_layer_embeddings"],
        prefill_inputs["input_pos"],
        prefill_inputs["kv_cache"],
        prefill_inputs["mask"],
        use_bool_mask=True,
        local_mask=prefill_inputs["local_mask"],
        param_tensor=prefill_inputs["param_tensor"],
        apply_gpu_composites=True,
    )
    self.assertEqual(adapted_prefill["inputs_embeds"].dtype, expected_dtype)
    self.assertEqual(adapted_prefill["per_layer_inputs"].dtype, expected_dtype)

    decode_inputs, _ = gen_mod.get_sample_inputs(text_cfg)["decode"]
    adapted_decode = gen_mod.adapt_inputs(
        None,
        decode_inputs["embeddings"],
        decode_inputs["per_layer_embeddings"],
        decode_inputs["input_pos"],
        decode_inputs["kv_cache"],
        decode_inputs["mask"],
        use_bool_mask=True,
        local_mask=decode_inputs["local_mask"],
        param_tensor=decode_inputs["param_tensor"],
        apply_gpu_composites=True,
    )
    self.assertEqual(adapted_decode["inputs_embeds"].dtype, expected_dtype)
    self.assertEqual(adapted_decode["per_layer_inputs"].dtype, expected_dtype)

  def test_per_layer_embedder_clamps_negative_ids_casts_fp32_and_meta_device(
      self,
  ):
    with gemma4_patch.gemma4_litert_patch():
      cfg = _make_small_text_config()
      model = modeling_gemma4.Gemma4ForCausalLM(cfg).eval()
    weights_loader.cast_params_(model, torch.bfloat16)

    ple_mod = gemma4_exportable.LiteRTExportableModuleForPerLayerEmbedder(
        model
    ).eval()
    token_ids = torch.tensor([[-256, -1, 0, 5]], dtype=torch.int32)
    with torch.no_grad():
      out = ple_mod(token_ids)["embeddings"]

    self.assertEqual(out.dtype, torch.float32)
    torch.testing.assert_close(out[:, 0], out[:, 2])
    torch.testing.assert_close(out[:, 1], out[:, 2])

    # Verify meta-device parameter placement is respected in get_sample_inputs.
    with (
        gemma4_patch.gemma4_litert_patch(),
        weights_loader.init_params_on_meta(),
    ):
      meta_model = modeling_gemma4.Gemma4ForCausalLM(cfg).eval()
    meta_ple = gemma4_exportable.LiteRTExportableModuleForPerLayerEmbedder(
        meta_model
    )
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        externalize_embedder=True,
        single_token_embedder=False,
        prefill_lengths=[8],
    )
    sample_inputs = meta_ple.get_sample_inputs(cfg, export_cfg)
    self.assertEqual(
        sample_inputs["decode_per_layer_embedder"][0]["token_ids"].device.type,
        "meta",
    )
    self.assertEqual(
        sample_inputs["prefill_per_layer_embedder_8"][0][
            "token_ids"
        ].device.type,
        "meta",
    )

  def test_extension_update_export_config_auto_enables_external_embedder(self):
    ext = extension_lib.resolve_model_extension("gemma4")
    self.assertIsNotNone(ext)

    ple_cfg = _make_small_multimodal_config(hidden_size_per_layer_input=8)
    text_gen_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        task=exportable_module_config.ExportTask.TEXT_GENERATION,
        apply_gpu_composites=True,
    )
    self.assertFalse(text_gen_cfg.externalize_embedder)
    updated = ext.update_export_config(text_gen_cfg, ple_cfg)
    self.assertTrue(updated.externalize_embedder)
    self.assertTrue(updated.single_token_embedder)
    self.assertEqual(updated.cache_length, text_gen_cfg.cache_length)

    # Split-cache keeps multi-signature embedders
    # (`single_token_embedder=False`).
    split_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        split_cache=True,
    )
    updated_split = ext.update_export_config(split_cfg, ple_cfg)
    self.assertTrue(updated_split.externalize_embedder)
    self.assertFalse(updated_split.single_token_embedder)

    # Non-PLE Gemma 4 variants keep `externalize_embedder=False` unless
    # requested.
    no_ple_cfg = _make_small_multimodal_config(hidden_size_per_layer_input=0)
    updated_no_ple = ext.update_export_config(text_gen_cfg, no_ple_cfg)
    self.assertFalse(updated_no_ple.externalize_embedder)

  def test_fused_attention_per_layer_kv_heads_and_1d_position_ids(self):
    torch.manual_seed(2)
    cfg = _make_small_text_config(
        num_hidden_layers=3,
        num_kv_shared_layers=1,
        attention_k_eq_v=False,
        per_layer_config={
            1: {"head_dim": 16, "num_key_value_heads": 1},
            2: {"head_dim": 16, "num_key_value_heads": 1},
        },
        attn_implementation="eager",
    )
    # Layer 1 is full_attention with per-layer num_key_value_heads=1 != 2;
    # Layer 2 is KV-shared.
    with gemma4_patch.gemma4_litert_patch():
      ref_attn = modeling_gemma4.Gemma4TextAttention(cfg, layer_idx=1).eval()
      fused_attn = gemma4_patch.FusedGemma4TextAttention(
          modeling_gemma4.Gemma4TextAttention(cfg, layer_idx=1).eval(),
          fuse_qkv=True,
          use_rope_composite=True,
          use_qkv_norm_rope_composite=True,
      ).eval()
      fused_attn.q_proj.load_state_dict(ref_attn.q_proj.state_dict())
      fused_attn.k_proj.load_state_dict(ref_attn.k_proj.state_dict())
      fused_attn.v_proj.load_state_dict(ref_attn.v_proj.state_dict())
      fused_attn.q_norm.load_state_dict(ref_attn.q_norm.state_dict())
      fused_attn.k_norm.load_state_dict(ref_attn.k_norm.state_dict())
      fused_attn.o_proj.load_state_dict(ref_attn.o_proj.state_dict())
      with torch.no_grad():
        fused_attn.qkv_proj.weight.copy_(
            torch.cat(
                [
                    ref_attn.q_proj.weight,
                    ref_attn.k_proj.weight,
                    ref_attn.v_proj.weight,
                ],
                dim=0,
            )
        )

      shared_attn = gemma4_patch.FusedGemma4TextAttention(
          modeling_gemma4.Gemma4TextAttention(cfg, layer_idx=2).eval(),
          fuse_qkv=True,
          use_rope_composite=True,
          use_qkv_norm_rope_composite=True,
      ).eval()

      # Also verify native `num_global_key_value_heads` with `attention_k_eq_v`.
      k_eq_v_cfg = _make_small_text_config(
          num_hidden_layers=3,
          num_kv_shared_layers=1,
          num_global_key_value_heads=1,
          attention_k_eq_v=True,
          attn_implementation="eager",
      )
      k_eq_v_attn = gemma4_patch.FusedGemma4TextAttention(
          modeling_gemma4.Gemma4TextAttention(k_eq_v_cfg, layer_idx=1).eval(),
          fuse_qkv=True,
          use_rope_composite=True,
          use_qkv_norm_rope_composite=True,
      ).eval()

    self.assertEqual(fused_attn.num_heads, 4)
    self.assertEqual(fused_attn.num_key_value_heads, 1)
    self.assertEqual(k_eq_v_attn.num_key_value_heads, 1)

    hidden = torch.randn(1, 4, cfg.hidden_size)
    pos_1d = torch.arange(4, dtype=torch.int32)
    pos_2d = pos_1d.unsqueeze(0)
    rotary = modeling_gemma4.Gemma4TextRotaryEmbedding(cfg)
    pos_emb = rotary(hidden, pos_2d, layer_type="full_attention")

    with torch.no_grad():
      ref_out, _ = ref_attn(
          hidden, pos_emb, attention_mask=None, shared_kv_states={}
      )
      shared_1d = {}
      out_1d, _ = fused_attn(
          hidden,
          pos_emb,
          attention_mask=None,
          shared_kv_states=shared_1d,
          position_ids=pos_1d,
          prefill_logits=True,
      )
      shared_2d = {}
      out_2d, _ = fused_attn(
          hidden,
          pos_emb,
          attention_mask=None,
          shared_kv_states=shared_2d,
          position_ids=pos_2d,
          prefill_logits=True,
      )
    torch.testing.assert_close(out_1d, out_2d, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_1d, ref_out, rtol=1e-4, atol=1e-4)

    # Also verify 1D vs 2D position_ids on KV-shared attention (layer 2 shares
    # `full_attention` KV states from layer 1, with 1 KV head).
    kv_donor = (
        torch.randn(1, 1, 4, cfg.head_dim),
        torch.randn(1, 1, 4, cfg.head_dim),
    )
    with torch.no_grad():
      sh_out_1d, _ = shared_attn(
          hidden,
          pos_emb,
          attention_mask=None,
          shared_kv_states={"full_attention": kv_donor},
          position_ids=pos_1d,
      )
      sh_out_2d, _ = shared_attn(
          hidden,
          pos_emb,
          attention_mask=None,
          shared_kv_states={"full_attention": kv_donor},
          position_ids=pos_2d,
      )
    torch.testing.assert_close(sh_out_1d, sh_out_2d, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      ("with_kv_sharing_no_prefill_logits", 1, False, 1),
      ("no_kv_sharing_no_prefill_logits", 0, False, 1),
      ("with_kv_sharing_with_prefill_logits", 1, True, 2),
  )
  def test_dead_q_prefill_guard_and_tflite_conversion(
      self,
      num_kv_shared_layers: int,
      prefill_logits: bool,
      expected_qkv_composite_calls: int,
  ):
    num_layers = 2 + num_kv_shared_layers
    with gemma4_patch.gemma4_litert_patch():
      cfg = _make_small_multimodal_config(
          num_hidden_layers=num_layers,
          num_kv_shared_layers=num_kv_shared_layers,
      )
      model = modeling_gemma4.Gemma4ForConditionalGeneration(cfg).eval()
      model.set_attn_implementation("lrt_transposed_attention")

    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        externalize_embedder=True,
        single_token_embedder=True,
        cache_length=32,
        prefill_lengths=[8],
        sliding_window_ring_buffer_size=32,
        apply_gpu_composites=True,
        use_v2=False,
    )
    with gemma4_patch.patch_gemma4_model(model, export_cfg):
      text_cfg = cfg.get_text_config()
      prefill_mod = (
          gemma4_exportable.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder(
              model, export_cfg
          ).eval()
      )
      sample_inputs = prefill_mod.get_sample_inputs(text_cfg)
      prefill_inputs, _ = sample_inputs["prefill_8"]

      with mock.patch.object(
          qkv_norm_rope,
          "apply_qkv_norm_rope",
          wraps=qkv_norm_rope.apply_qkv_norm_rope,
      ) as spy:
        if prefill_logits:
          adapted = prefill_mod.adapt_inputs(
              None,
              prefill_inputs["embeddings"],
              prefill_inputs["per_layer_embeddings"],
              prefill_inputs["input_pos"],
              prefill_inputs["kv_cache"],
              prefill_inputs["mask"],
              use_bool_mask=True,
              local_mask=prefill_inputs["local_mask"],
              param_tensor=prefill_inputs["param_tensor"],
              apply_gpu_composites=True,
              use_sdpa_composite=True,
          )
          adapted["past_key_values"] = adapted[
              "past_key_values"
          ].insert_dummy_cache_layers(text_cfg)
          adapted |= prefill_mod.attention_kwargs()
          adapted["prefill_logits"] = True
          with torch.no_grad():
            gemma4_exportable.get_language_model(model)(**adapted)
        else:
          with torch.no_grad():
            prefill_mod(**prefill_inputs)
        self.assertEqual(spy.call_count, expected_qkv_composite_calls)

      if not prefill_logits:
        # Verify full TFLite MLIR lowering (`BuildStableHLOCompositePass`)
        # succeeds without crashing on dead `pos=0` of `odml.qkv_norm_rope`.
        out_tflite = os.path.join(
            self.create_tempdir().full_path, "prefill.tflite"
        )
        fresh_inputs = prefill_mod.get_sample_inputs(text_cfg)
        export_lib._convert_and_export_module(  # pylint: disable=protected-access
            prefill_mod,
            fresh_inputs,
            out_tflite,
            export_cfg,
            weights_loader=None,
        )
        self.assertTrue(os.path.exists(out_tflite))
        self.assertGreater(os.path.getsize(out_tflite), 0)

  @parameterized.product(
      signature=["prefill_8", "decode"],
      # (layers, KV-shared layers, ring buffer size, attention calls over a
      # cache with the full length). The last layer is forced to be global, so
      # 3 layers are [sliding, global, global] and 4 layers are [sliding,
      # global, sliding, global]. Without a ring buffer every cache has the
      # full 64 slots; with a 32-slot ring buffer only the global caches do.
      # With KV sharing (as in the E-series models) the last 2 layers attend
      # over the caches of earlier layers of the same type, so they count too.
      layers_shared_ring_blocked=[
          (3, 0, None, 3),
          (3, 0, 32, 2),
          (4, 2, None, 4),
          (4, 2, 32, 2),
      ],
  )
  def test_split_cache_blocked_attention_matches_plain(
      self, signature, layers_shared_ring_blocked
  ):
    # Layers whose cache has the full 64 slots attend over it in 4 blocks of 16
    # slots; ring-buffer layers keep plain attention.
    num_layers, num_shared, ring_size, blocked_layers = (
        layers_shared_ring_blocked
    )
    torch.manual_seed(4)
    with gemma4_patch.gemma4_litert_patch():
      cfg = _make_small_text_config(
          num_hidden_layers=num_layers, num_kv_shared_layers=num_shared
      )
      model = modeling_gemma4.Gemma4ForCausalLM(cfg).eval()
    model = external_rope_preprocess_model.inject_rotary_position_embedding(
        model
    )
    model.set_attn_implementation("lrt_split_cache_attention")
    model.config._attn_implementation = "lrt_split_cache_attention"
    text_cfg = cfg.get_text_config()
    outputs = {}
    inputs = None
    for block_size in (None, 16):
      export_cfg = exportable_module_config.ExportableModuleConfig(
          model="dummy",
          split_cache=True,
          attention_kv_block_size=block_size,
          cache_length=64,
          sliding_window_ring_buffer_size=ring_size,
          prefill_lengths=[8],
      )
      with gemma4_patch.patch_gemma4_model(model, export_cfg):
        if signature.startswith("prefill"):
          mod_cls = (
              gemma4_split_cache.LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill
          )
        else:
          mod_cls = (
              gemma4_split_cache.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate
          )
        mod = mod_cls(model, export_cfg).eval()
        kwargs = mod.attention_kwargs()
        if block_size is None:
          self.assertNotIn("attention_kv_block_size", kwargs)
        else:
          self.assertEqual(kwargs["attention_kv_block_size"], 16)
          self.assertEqual(kwargs["attention_kv_block_cache_length"], 64)
        if inputs is None:
          # Random cache contents and embeddings, shared by both builds.
          inputs, _ = mod.get_sample_inputs(text_cfg)[signature]
          for layer in inputs["kv_cache"].layers:
            layer.keys[0].normal_()
            layer.values[0].normal_()
          inputs["embeddings"].normal_()
        with (
            torch.no_grad(),
            mock.patch.object(
                split_cache_attention,
                "_blocked_attention",
                wraps=split_cache_attention._blocked_attention,
            ) as spy,
        ):
          outputs[block_size] = mod(**inputs)
        self.assertEqual(
            spy.call_count, 0 if block_size is None else blocked_layers
        )

    plain, blocked = outputs[None], outputs[16]
    for key in ("kv_slice_k", "kv_slice_v"):
      for got, want in zip(blocked[key], plain[key]):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    if "logits" in plain:
      torch.testing.assert_close(
          blocked["logits"], plain["logits"], rtol=1e-5, atol=1e-5
      )

  @parameterized.parameters(32, 64)
  def test_split_cache_blocked_attention_per_bucket(self, cache_length):
    # Mirrors export_lib, which builds one config per cache-length bucket with
    # dataclasses.replace: each bucket blocks the caches of its own length.
    torch.manual_seed(4)
    with gemma4_patch.gemma4_litert_patch():
      cfg = _make_small_text_config(num_hidden_layers=3, num_kv_shared_layers=0)
      model = modeling_gemma4.Gemma4ForCausalLM(cfg).eval()
    model = external_rope_preprocess_model.inject_rotary_position_embedding(
        model
    )
    model.set_attn_implementation("lrt_split_cache_attention")
    model.config._attn_implementation = "lrt_split_cache_attention"
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        split_cache=True,
        attention_kv_block_size=16,
        cache_length=64,
        cache_lengths=[32, 64],
        prefill_lengths=[8],
    )
    bucket_cfg = dataclasses.replace(export_cfg, cache_length=cache_length)
    with gemma4_patch.patch_gemma4_model(model, bucket_cfg):
      mod = gemma4_split_cache.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate(
          model, bucket_cfg
      ).eval()
      kwargs = mod.attention_kwargs()
      self.assertEqual(kwargs["attention_kv_block_size"], 16)
      self.assertEqual(kwargs["attention_kv_block_cache_length"], cache_length)
      inputs, _ = mod.get_sample_inputs(cfg.get_text_config())["decode"]
      with (
          torch.no_grad(),
          mock.patch.object(
              split_cache_attention,
              "_blocked_attention",
              wraps=split_cache_attention._blocked_attention,
          ) as spy,
      ):
        mod(**inputs)
    # No ring buffer: all 3 cache layers have the bucket's full length.
    self.assertEqual(spy.call_count, 3)
    # Positional args: (query, key_past, ..., k_ts_idx, ...).
    key_past, k_ts_idx = spy.call_args.args[1], spy.call_args.args[9]
    self.assertEqual(key_past.shape[k_ts_idx], cache_length)

if __name__ == "__main__":
  googletest.main()
