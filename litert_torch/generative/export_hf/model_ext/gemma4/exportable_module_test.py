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

import copy
import os
from unittest import mock

from absl.testing import parameterized
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import weights_loader
from litert_torch.generative.export_hf.core.external_rope import exportable_module as external_rope_exportable
from litert_torch.generative.export_hf.core.external_rope import preprocess_model as external_rope_preprocess_model
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


def _make_reference_and_patched_models(
    final_logit_softcapping: float | None,
) -> tuple[
    modeling_gemma4.Gemma4ForConditionalGeneration,
    modeling_gemma4.Gemma4ForConditionalGeneration,
]:
  """Returns an HF reference model and a patched model with equal weights."""
  cfg = _make_small_multimodal_config(
      final_logit_softcapping=final_logit_softcapping,
      attn_implementation="eager",
  )
  torch.manual_seed(0)
  reference_model = modeling_gemma4.Gemma4ForConditionalGeneration(cfg).eval()
  with torch.no_grad():
    # Amplify lm_head so raw pre-cap logits exceed the cap.
    reference_model.lm_head.weight.mul_(200.0)
  with gemma4_patch.gemma4_litert_patch():
    model = modeling_gemma4.Gemma4ForConditionalGeneration(
        copy.deepcopy(cfg)
    ).eval()
  model.load_state_dict(reference_model.state_dict())
  return reference_model, model


def _get_external_embeddings(
    model: modeling_gemma4.Gemma4ForConditionalGeneration,
    tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
  language_model = gemma4_exportable.get_language_model(model)
  return {
      "embeddings": language_model.embed_tokens(tokens),
      "per_layer_embeddings": language_model.get_per_layer_inputs(tokens, None),
  }


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

  # A cap other than the 30.0 of Gemma 4 checkpoints checks that the cap is read
  # from the config.
  @parameterized.named_parameters(
      ("without_softcap", None),
      ("with_softcap", 20.0),
  )
  def test_generate_logits_match_hf(
      self, final_logit_softcapping: float | None
  ):
    reference_model, model = _make_reference_and_patched_models(
        final_logit_softcapping
    )
    model.set_attn_implementation("lrt_transposed_attention")
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        externalize_embedder=True,
        cache_length=32,
    )
    gen_mod = (
        gemma4_exportable.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder(
            model, export_cfg
        ).eval()
    )
    decode_inputs, _ = gen_mod.get_sample_inputs(
        model.config.get_text_config()
    )["decode"]

    tokens = torch.tensor([[5]])
    # The first token is decoded into an empty cache, so only cache position 0
    # is visible.
    mask = torch.full((1, 1, 1, 32), float("-inf"))
    mask[..., 0] = 0
    with torch.no_grad():
      decode_inputs |= _get_external_embeddings(model, tokens)
      decode_inputs["input_pos"] = torch.tensor([0], dtype=torch.int32)
      decode_inputs["mask"] = mask
      with gemma4_patch.patch_gemma4_model(model, export_cfg):
        logits = gen_mod(**decode_inputs)["logits"]
      expected_logits = reference_model(input_ids=tokens).logits

    torch.testing.assert_close(logits, expected_logits)

  @parameterized.named_parameters(
      ("without_softcap", None),
      ("with_softcap", 20.0),
  )
  def test_split_cache_generate_logits_match_hf(
      self, final_logit_softcapping: float | None
  ):
    reference_model, model = _make_reference_and_patched_models(
        final_logit_softcapping
    )
    model = external_rope_preprocess_model.inject_rotary_position_embedding(
        model
    )
    model.set_attn_implementation("lrt_split_cache_attention")
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        split_cache=True,
        cache_length=32,
    )
    gen_mod = (
        gemma4_split_cache.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate(
            model, export_cfg
        ).eval()
    )
    decode_inputs, _ = gen_mod.get_sample_inputs(
        model.config.get_text_config()
    )["decode"]

    tokens = torch.tensor([[5]])
    # The split-cache mask covers the cache followed by the current slice. The
    # cache is empty, so only the slice is visible.
    mask = torch.full((1, 1, 1, 32 + 1), float("-inf"))
    mask[..., -1] = 0
    with torch.no_grad():
      pos_emb = external_rope_exportable.RoPEEmbedder(model)(
          torch.tensor([0], dtype=torch.int32)
      )
      decode_inputs |= _get_external_embeddings(model, tokens)
      decode_inputs["pos_emb"] = {
          "cos": pos_emb["pos_emb_cos"],
          "sin": pos_emb["pos_emb_sin"],
          "local_cos": pos_emb["pos_emb_local_cos"],
          "local_sin": pos_emb["pos_emb_local_sin"],
      }
      decode_inputs["mask"] = {"global": mask, "local": mask}
      with gemma4_patch.patch_gemma4_model(model, export_cfg):
        logits = gen_mod(**decode_inputs)["logits"]
      expected_logits = reference_model(input_ids=tokens).logits

    torch.testing.assert_close(logits, expected_logits)

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
            use_v2=False,
        )
        self.assertTrue(os.path.exists(out_tflite))
        self.assertGreater(os.path.getsize(out_tflite), 0)


if __name__ == "__main__":
  googletest.main()
