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
"""Unit tests for the Inkling-2-Mini looped transformer ModelExtension."""

import json
import math
import os
import tempfile

from absl.testing import absltest
from litert_torch.generative.export_hf import export as export_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.inkling2_loop import cache as inkling2_cache
from litert_torch.generative.export_hf.model_ext.inkling2_loop import exportable_module as inkling2_exportable
from litert_torch.generative.export_hf.model_ext.inkling2_loop import extension as inkling2_ext
from litert_torch.generative.export_hf.model_ext.inkling2_loop import metadata_builder as inkling2_metadata
from litert_torch.generative.export_hf.model_ext.inkling2_loop import modeling_inkling2_loop_static as inkling2_static
import safetensors.torch
import torch
import torch.nn.functional as F

from litert_quantizer import qtyping
from litert_quantizer.utils import tfl_flatbuffer_utils


def _tiny_config() -> inkling2_static.Inkling2LoopTextConfig:
  # 4 physical layers:
  #   layer 0 (sliding, prelude), layer 1 (full, prelude),
  #   layer 2 (sliding, core loop), layer 3 (full, core loop) -> 2 passes
  #   plus layers 4..5 coda (sliding, full) -> 6 physical layers, 2-layer block,
  #   core [2, 4), 3 passes => total_num_hidden_layers = 6 + 2*2 = 10.
  return inkling2_static.Inkling2LoopTextConfig(
      vocab_size=64,
      unpadded_vocab_size=60,
      hidden_size=32,
      num_hidden_layers=6,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=8,
      swa_num_attention_heads=4,
      swa_num_key_value_heads=2,
      swa_head_dim=8,
      sliding_window_size=8,
      d_rel=4,
      rel_extent=8,
      log_scaling_n_floor=4,
      log_scaling_alpha=0.1,
      n_layers_per_attn_res_block=2,
      conv_kernel_size=4,
      moe_intermediate_size=16,
      n_routed_experts=4,
      num_experts_per_tok=2,
      route_scale=2.0,
      mlp_gate_softcap=4.0,
      mlp_up_softcap=25.0,
      final_logit_softcapping=30.0,
      loop_layer_index_start=2,
      loop_layer_index_end=4,
      loop_num_passes=3,
      loop_gate_rank=8,
  )


def _init_random_weights(model: torch.nn.Module, seed: int = 42) -> None:
  gen = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in model.named_parameters():
      if p.is_floating_point():
        if (
            "norm.weight" in name
            or "layernorm.weight" in name
            or ".scale" in name
        ):
          noise = 0.05 * torch.randn(p.shape, generator=gen, dtype=p.dtype)
          p.copy_(torch.ones_like(p) + noise)
        else:
          p.copy_(0.1 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    for name, b in model.named_buffers():
      if "e_score_correction_bias" in name:
        b.copy_(0.05 * torch.randn(b.shape, generator=gen, dtype=b.dtype))


class Inkling2LoopExtensionTest(absltest.TestCase):

  def test_extension_registered(self):
    self.assertIsInstance(
        extension_lib.get_model_extension("inkling2_loop"),
        inkling2_ext.Inkling2LoopExtension,
    )
    self.assertIsInstance(
        extension_lib.get_model_extension("inkling2_loop_text"),
        inkling2_ext.Inkling2LoopExtension,
    )

  def test_config_total_layers_and_physical_mapping(self):
    cfg = _tiny_config()
    self.assertEqual(cfg.total_num_hidden_layers, 10)
    self.assertEqual(
        cfg.layer_types,
        [
            "hybrid_sliding",
            "hybrid",
            "hybrid_sliding",
            "hybrid",
            "hybrid_sliding",
            "hybrid",
        ],
    )
    # Pass 1: 0..5 -> 0..5, Pass 2: 6,7 -> 2,3, Pass 3: 8,9 -> 2,3
    self.assertEqual(
        [cfg.physical_layer_index_for_cache(i) for i in range(10)],
        [0, 1, 2, 3, 4, 5, 2, 3, 2, 3],
    )

  def test_relative_position_bias_matches_upstream_gather(self):
    torch.manual_seed(0)
    cfg = _tiny_config()
    rel = inkling2_static.Inkling2LoopRelativePositionBias(
        cfg, num_heads=4, rel_extent=8, log_scaling=True
    )
    with torch.no_grad():
      rel.proj_t.normal_(0.0, 0.2)
    h = torch.randn(1, 5, cfg.hidden_size)
    pos = torch.tensor([[3, 4, 5, 6, 7]], dtype=torch.int32)
    kv_pos = torch.arange(16, dtype=torch.int32)
    rel_dist = pos[0].unsqueeze(1) - kv_pos.unsqueeze(0)  # [5, 16]

    actual = rel(h, rel_dist, pos)

    # Upstream reference: (relative_states @ proj).transpose(1, 2).gather(-1,i)
    assert rel.linear is not None
    rel_states = rel.norm(rel.linear(h).view(1, 5, 4, cfg.d_rel), pos)
    rel_logits = (rel_states @ rel.proj_t.T).transpose(1, 2)
    gather_idx = rel_dist.clamp(0, 7).long().expand(1, 4, 5, 16)
    expected = rel_logits.gather(-1, gather_idx)
    expected = expected.masked_fill(
        ((rel_dist < 0) | (rel_dist >= 8)).view(1, 1, 5, 16), 0.0
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

  def test_dynamic_short_conv_rank4_matches_upstream_5d(self):
    torch.manual_seed(1)
    dconv = inkling2_static.Inkling2LoopDynamicShortConvolution(
        hidden_size=16,
        input_size=32,
        num_heads=2,
        conv_kernel_size=4,
        layer_idx=1,
        conv_idx=0,
    )
    with torch.no_grad():
      dconv.conv_weight.normal_(0.0, 0.2)
    x = torch.randn(1, 6, 16)
    dyn_in = torch.randn(1, 6, 32)
    actual = dconv(x, dyn_in)

    # Upstream 5D reference (conv_weight is oldest -> newest; flip to lag).
    assert dconv.weight_proj is not None
    dyn_w = dconv.weight_proj(dyn_in).float().view(1, 6, 2, 1, 4)
    static_w = dconv.conv_weight[0].float().flip(0)
    w = static_w.T.view(2, 8, 4) + dyn_w
    ctx = F.pad(x.float().transpose(1, 2), (3, 0))
    windows = (
        ctx.unfold(-1, 4, 1).flip(-1).transpose(1, 2).reshape(1, 6, 2, 8, 4)
    )
    expected = (
        (windows * w).sum(-1).reshape(1, 6, 16) + x.float()
    ) / math.sqrt(9)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

  def test_prefill_plus_decode_matches_full_sequence(self):
    cfg = _tiny_config()
    model = inkling2_static.Inkling2LoopStaticForCausalLM(cfg).eval()
    _init_random_weights(model, seed=7)

    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="inkling2_loop", cache_length=16, batch_size=1
    )
    tokens = torch.tensor([[3, 11, 27, 4, 19, 8]], dtype=torch.int32)

    # Full 6-token forward in one shot.
    cache_full = inkling2_cache.Inkling2LoopLiteRTLMCache.create_from_config(
        cfg, export_cfg
    )
    pos_full = torch.arange(6, dtype=torch.int32)
    cache_full.set_cache_runtime_args({"cache_position": pos_full})
    mask_full = inkling2_exportable.create_causal_mask(6, 16, pos_full)
    with torch.no_grad():
      logits_full, _ = model(
          tokens,
          pos_full.unsqueeze(0),
          past_key_values=cache_full,
          attention_mask=mask_full,
      )

    # Prefill first 4 tokens (padded to 6 with 2 pad tokens!), then decode 2.
    cache_step = inkling2_cache.Inkling2LoopLiteRTLMCache.create_from_config(
        cfg, export_cfg
    )
    prefill_toks = torch.tensor([[3, 11, 27, 4, 0, 0]], dtype=torch.int32)
    prefill_pos = torch.tensor([0, 1, 2, 3, 0, 0], dtype=torch.int32)
    prefill_valid = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.float32)
    cache_step.set_cache_runtime_args({"cache_position": prefill_pos})
    prefill_mask = inkling2_exportable.create_causal_mask(6, 16, prefill_pos)
    with torch.no_grad():
      logits_p, cache_step = model(
          prefill_toks,
          prefill_pos.unsqueeze(0),
          past_key_values=cache_step,
          attention_mask=prefill_mask,
          valid_mask=prefill_valid,
      )
      torch.testing.assert_close(
          logits_p[:, :4, :], logits_full[:, :4, :], atol=1e-4, rtol=1e-4
      )

      # Decode token 4 and token 5 one by one.
      for step_idx in (4, 5):
        step_tok = tokens[:, step_idx : step_idx + 1]
        step_pos = torch.tensor([step_idx], dtype=torch.int32)
        cache_step.set_cache_runtime_args({"cache_position": step_pos})
        step_mask = inkling2_exportable.create_causal_mask(1, 16, step_pos)
        logits_d, cache_step = model(
            step_tok,
            step_pos.unsqueeze(0),
            past_key_values=cache_step,
            attention_mask=step_mask,
        )
        torch.testing.assert_close(
            logits_d[:, 0, :], logits_full[:, step_idx, :], atol=1e-4, rtol=1e-4
        )

  def test_checkpoint_resolution_nvfp4_mxfp8_and_renames(self):
    ext = inkling2_ext.Inkling2LoopExtension()
    cfg = _tiny_config()

    # Synthetic raw checkpoint tensors.
    raw: dict[str, torch.Tensor] = {
        "model.llm.embed.weight": torch.randn(64, 32),
        "model.llm.embed_norm.weight": torch.randn(32),
        "model.llm.unembed.weight": torch.randn(64, 32),
        "model.llm.norm.weight": torch.randn(32),
        "model.llm.attn_res.output_query": torch.randn(32),
        "model.llm.middle_cycle_cross_pass_queries": (
            torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32)
        ),
        "model.llm.middle_cycle_cross_pass_kv_queries": (
            torch.arange(4 * 32, dtype=torch.float32).reshape(4, 32) + 1000.0
        ),
        "model.llm.attn_res.attn_queries.0": torch.randn(32),
        "model.llm.attn_res.mlp_queries.0": torch.randn(32),
        "model.llm.layers.0.attn.wq_du.weight": torch.randn(32, 32),
        "model.llm.layers.0.attn.gate_attn.weight": torch.randn(32, 32),
        "model.llm.layers.0.attn.wr_du.weight": torch.randn(16, 32),
        "model.llm.layers.0.attn.attn_sink_proj.weight": torch.randn(4, 32),
        "model.llm.layers.1.attn.wk_dv.weight": torch.randn(16, 32),
        "model.llm.layers.1.attn.k_sconv.weight_proj": torch.randn(8, 32),
        "model.llm.layers.2.controllers.attn.forget_weight": torch.randn(32, 8),
        "model.llm.layers.2.controllers.attn.gain_weight": torch.randn(32, 8),
        "model.llm.layers.0.attn.rel_logits_proj.proj": (
            torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
        ),
        "model.llm.layers.0.attn_sconv.weight": (
            torch.arange(32 * 4, dtype=torch.float32).reshape(32, 1, 4)
        ),
        "model.llm.layers.1.attn.k_sconv.static_w": (
            torch.arange(4 * 16, dtype=torch.float32).reshape(4, 16)
        ),
    }

    # NVFP4 expert at layer 0 (4 experts, 32 rows, 32 cols -> 16 packed bytes).
    raw["model.llm.layers.0.mlp.experts.w13_weight"] = torch.full(
        (4, 32, 16), 0x22, dtype=torch.uint8
    )  # low=2 (1.0), high=2 (1.0)
    raw["model.llm.layers.0.mlp.experts.w13_weight_scale"] = torch.ones(
        (4, 32, 2), dtype=torch.float8_e4m3fn
    )
    raw["model.llm.layers.0.mlp.experts.w13_weight_amax"] = torch.full(
        (4,), 2.0, dtype=torch.float32
    )
    raw["model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator"] = (
        torch.tensor([4.0], dtype=torch.float32)
    )

    # MXFP8 expert at layer 5 (4 experts, 32 rows, 32 cols, scale=128 -> 2^1=2).
    raw["model.llm.layers.5.mlp.experts.w2_weight"] = torch.full(
        (4, 32, 32), 1.5, dtype=torch.float8_e4m3fn
    )
    raw["model.llm.layers.5.mlp.experts.w2_weight_scale"] = torch.full(
        (4, 32, 1), 128, dtype=torch.uint8
    )

    def try_load(k: str) -> torch.Tensor | None:
      return raw.get(k)

    lm_head = ext.resolve_checkpoint_tensor("lm_head.weight", try_load, cfg)
    self.assertIsNotNone(lm_head)
    self.assertEqual(tuple(lm_head.shape), (60, 32))

    all_msg_q = ext.resolve_checkpoint_tensor(
        "model.message_input.all_queries", try_load, cfg
    )
    self.assertIsNotNone(all_msg_q)
    self.assertEqual(tuple(all_msg_q.shape), (8, 32))
    # Verify [Q, K, V, MLP] ordering for loop layer 0.
    torch.testing.assert_close(
        all_msg_q[0], raw["model.llm.middle_cycle_cross_pass_queries"][0]
    )
    torch.testing.assert_close(
        all_msg_q[1], raw["model.llm.middle_cycle_cross_pass_kv_queries"][0]
    )
    torch.testing.assert_close(
        all_msg_q[2], raw["model.llm.middle_cycle_cross_pass_kv_queries"][1]
    )
    torch.testing.assert_close(
        all_msg_q[3], raw["model.llm.middle_cycle_cross_pass_queries"][1]
    )

    fused_q = ext.resolve_checkpoint_tensor(
        "model.layers.0.self_attn.q_o_rel_sink_proj.weight", try_load, cfg
    )
    self.assertIsNotNone(fused_q)
    self.assertEqual(tuple(fused_q.shape), (32 + 32 + 16 + 4, 32))

    fused_k = ext.resolve_checkpoint_tensor(
        "model.layers.1.self_attn.k_proj.fused_linear.weight", try_load, cfg
    )
    self.assertIsNotNone(fused_k)
    self.assertEqual(tuple(fused_k.shape), (16 + 8, 32))

    fused_gate = ext.resolve_checkpoint_tensor(
        "model.layers.2.q_message_gate.gate_proj.weight", try_load, cfg
    )
    self.assertIsNotNone(fused_gate)
    self.assertEqual(tuple(fused_gate.shape), (64, 8))

    proj_t = ext.resolve_checkpoint_tensor(
        "model.layers.0.self_attn.relative_position_bias.proj_t", try_load, cfg
    )
    self.assertIsNotNone(proj_t)
    self.assertEqual(tuple(proj_t.shape), (8, 4))
    torch.testing.assert_close(
        proj_t, raw["model.llm.layers.0.attn.rel_logits_proj.proj"].T
    )

    conv_w = ext.resolve_checkpoint_tensor(
        "model.layers.0.post_attention_convolution.conv_weight", try_load, cfg
    )
    self.assertIsNotNone(conv_w)
    self.assertEqual(tuple(conv_w.shape), (1, 4, 32))
    torch.testing.assert_close(
        conv_w[0, 2],
        raw["model.llm.layers.0.attn_sconv.weight"][:, 0, 2],
    )

    static_w = ext.resolve_checkpoint_tensor(
        "model.layers.1.self_attn.k_proj.sconv.conv_weight", try_load, cfg
    )
    self.assertIsNotNone(static_w)
    self.assertEqual(tuple(static_w.shape), (1, 4, 16))
    # Oldest tap (index 0) corresponds to lag 3 in checkpoint static_w.
    torch.testing.assert_close(
        static_w[0, 0], raw["model.llm.layers.1.attn.k_sconv.static_w"][3]
    )

    # attn_queries.0 maps to layers.1.attention_input.query (+1 offset).
    self.assertIsNotNone(
        ext.resolve_checkpoint_tensor(
            "model.layers.1.attention_input.query", try_load, cfg
        )
    )
    self.assertIsNone(
        ext.resolve_checkpoint_tensor(
            "model.layers.0.attention_input.query", try_load, cfg
        )
    )

    # NVFP4 dequant: e2m1(2)=1.0 * scale(1.0) * (2.0 / 4.0) = 0.5, OHWI layout.
    nvfp4_gate = ext.resolve_checkpoint_tensor(
        "model.layers.0.mlp.experts.flattened_gate_weight", try_load, cfg
    )
    nvfp4_up = ext.resolve_checkpoint_tensor(
        "model.layers.0.mlp.experts.flattened_ff1_weight", try_load, cfg
    )
    self.assertIsNotNone(nvfp4_gate)
    self.assertIsNotNone(nvfp4_up)
    self.assertEqual(tuple(nvfp4_gate.shape), (16, 4, 1, 32))
    self.assertEqual(tuple(nvfp4_up.shape), (16, 4, 1, 32))
    torch.testing.assert_close(
        nvfp4_gate.float(),
        torch.full((16, 4, 1, 32), 0.5),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        nvfp4_up.float(), torch.full((16, 4, 1, 32), 0.5), atol=1e-2, rtol=1e-2
    )

    # MXFP8 dequant: 1.5 * 2^(128-127) = 3.0, OHWI layout.
    mxfp8_w = ext.resolve_checkpoint_tensor(
        "model.layers.5.mlp.experts.flattened_linear_weight", try_load, cfg
    )
    self.assertIsNotNone(mxfp8_w)
    self.assertEqual(tuple(mxfp8_w.shape), (32, 4, 1, 32))
    torch.testing.assert_close(
        mxfp8_w.float(), torch.full((32, 4, 1, 32), 3.0), atol=1e-2, rtol=1e-2
    )

  def test_torch_export_prefill_and_decode_rank_le_4(self):
    cfg = _tiny_config()
    model = inkling2_static.Inkling2LoopStaticForCausalLM(cfg).eval()
    _init_random_weights(model, seed=13)

    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="inkling2_loop",
        prefill_lengths=[4],
        cache_length=16,
        batch_size=1,
        cache_implementation="Inkling2LoopLiteRTLMCache",
        apply_gpu_composites=False,
    )
    prefill_mod = (
        inkling2_exportable.LiteRTExportableModuleForInkling2LoopPrefill(
            model, export_cfg
        )
    )
    decode_mod = (
        inkling2_exportable.LiteRTExportableModuleForInkling2LoopGenerate(
            model, export_cfg
        )
    )

    prefill_samples = prefill_mod.get_sample_inputs(cfg)
    decode_samples = decode_mod.get_sample_inputs(cfg)

    ep_prefill = torch.export.export(
        prefill_mod, args=(), kwargs=prefill_samples["prefill_4"][0]
    )
    ep_decode = torch.export.export(
        decode_mod, args=(), kwargs=decode_samples["decode"][0]
    )

    # Verify every tensor node in both exported graphs has rank <= 4 (ML Drift
    # GPU invariant).
    for ep in (ep_prefill, ep_decode):
      for node in ep.graph.nodes:
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
          self.assertLessEqual(
              val.ndim,
              4,
              f"Node {node.name} ({node.target}) has rank {val.ndim} > 4:"
              f" {tuple(val.shape)}",
          )

  def test_executor_metadata_buffers(self):
    cfg = _tiny_config()
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model="inkling2_loop", cache_length=16, batch_size=1
    )

    class _Artifacts:
      model_config = cfg

    exec_meta = inkling2_metadata.build_inkling2_loop_executor_metadata(
        _Artifacts(), export_cfg, None
    )
    # 10 logical layers * (2 KV + 4 conv) = 60 state buffers.
    self.assertLen(exec_meta.llm_executor_metadata.state_buffers, 60)
    self.assertEqual(exec_meta.llm_executor_metadata.max_history_size, 0)

  def test_tiny_v2_export_and_quantize_w4b32_moe(self):
    cfg = _tiny_config()
    # BLOCKWISE_32 requires hidden_size and moe_intermediate_size % 32 == 0.
    cfg.moe_intermediate_size = 32
    model = inkling2_static.Inkling2LoopStaticForCausalLM(cfg).eval()
    _init_random_weights(model, seed=19)

    recipe_path = os.path.join(
        os.path.dirname(os.path.abspath(inkling2_ext.__file__)),
        "recipes",
        "recipe_w4b32_moe.json",
    )
    self.assertTrue(os.path.isfile(recipe_path))

    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_dir = os.path.join(tmpdir, "ckpt")
      out_dir = os.path.join(tmpdir, "out")
      os.makedirs(ckpt_dir, exist_ok=True)
      with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f)
      vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
      for idx in range(4, cfg.vocab_size):
        vocab[f"tok_{idx}"] = idx
      with open(os.path.join(ckpt_dir, "tokenizer.json"), "w") as f:
        json.dump(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": vocab,
                    "unk_token": "<unk>",
                },
            },
            f,
        )
      with open(os.path.join(ckpt_dir, "tokenizer_config.json"), "w") as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
            },
            f,
        )
      sd = {
          k: v.detach().cpu().contiguous()
          for k, v in model.state_dict().items()
      }
      safetensors.torch.save_file(
          sd, os.path.join(ckpt_dir, "model.safetensors")
      )

      export_lib.export(
          model=ckpt_dir,
          output_dir=out_dir,
          prefill_lengths=[4],
          cache_length=16,
          quantization_recipe=recipe_path,
          use_v2=True,
          bundle_litert_lm=True,
          apply_gpu_composites=False,
          externalize_weights=False,
          delete_in_memory_params=True,
      )

      litertlm_files = [
          f for f in os.listdir(out_dir) if f.endswith(".litertlm")
      ]
      self.assertNotEmpty(litertlm_files)
      self.assertGreater(
          os.path.getsize(os.path.join(out_dir, litertlm_files[0])), 0
      )

  def test_tiny_v2_export_bfloat16_and_buffer_sharing(self):
    cfg = _tiny_config()
    cfg.moe_intermediate_size = 32
    model = inkling2_static.Inkling2LoopStaticForCausalLM(cfg).eval()
    _init_random_weights(model, seed=23)

    recipe_path = os.path.join(
        os.path.dirname(os.path.abspath(inkling2_ext.__file__)),
        "recipes",
        "recipe_mixed_iq4_inspired.json",
    )
    self.assertTrue(os.path.isfile(recipe_path))

    with tempfile.TemporaryDirectory() as tmpdir:
      ckpt_dir = os.path.join(tmpdir, "ckpt")
      out_dir = os.path.join(tmpdir, "out")
      os.makedirs(ckpt_dir, exist_ok=True)
      with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f)
      vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
      for idx in range(4, cfg.vocab_size):
        vocab[f"tok_{idx}"] = idx
      with open(os.path.join(ckpt_dir, "tokenizer.json"), "w") as f:
        json.dump(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": vocab,
                    "unk_token": "<unk>",
                },
            },
            f,
        )
      with open(os.path.join(ckpt_dir, "tokenizer_config.json"), "w") as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
            },
            f,
        )
      sd = {
          k: v.detach().cpu().to(torch.bfloat16).contiguous()
          for k, v in model.state_dict().items()
      }
      safetensors.torch.save_file(
          sd, os.path.join(ckpt_dir, "model.safetensors")
      )

      export_lib.export(
          model=ckpt_dir,
          output_dir=out_dir,
          prefill_lengths=[4],
          cache_length=16,
          quantization_recipe=recipe_path,
          model_dtype="bfloat16",
          experimental_use_fp16=False,
          use_v2=True,
          bundle_litert_lm=True,
          keep_temporary_files=True,
          apply_gpu_composites=True,
          externalize_weights=False,
          delete_in_memory_params=True,
      )

      litertlm_files = [
          f for f in os.listdir(out_dir) if f.endswith(".litertlm")
      ]
      self.assertNotEmpty(litertlm_files)
      quant_tflite = os.path.join(out_dir, "model_quantized.tflite")
      self.assertTrue(os.path.isfile(quant_tflite))

      fb_model = tfl_flatbuffer_utils.read_model(quant_tflite)
      moe_gate_buffers_per_sg = []
      moe_scopes = []
      for sg in fb_model.subgraphs:
        sg_gate_bufs = []
        for op in sg.operators:
          code = tfl_flatbuffer_utils.get_custom_code(
              fb_model.operatorCodes[op.opcodeIndex]
          )
          if code == "moe":
            # Quantized MoE op has 10 inputs with INT4/INT8 weights.
            self.assertLen(op.inputs, 10)
            gate_tensor = sg.tensors[op.inputs[3]]
            self.assertIn(
                gate_tensor.type,
                (qtyping.TensorType.INT4, qtyping.TensorType.INT8),
            )
            sg_gate_bufs.append(gate_tensor.buffer)
            moe_scopes.append(tfl_flatbuffer_utils.get_op_scope(op, sg.tensors))
        if sg_gate_bufs:
          moe_gate_buffers_per_sg.append(sg_gate_bufs)

      self.assertNotEmpty(moe_gate_buffers_per_sg)
      # Tiny model has 10 logical layers from 6 physical layers: 10 MoE ops per
      # signature, sharing only 6 unique gate weight buffers across loop passes.
      for sg_gate_bufs in moe_gate_buffers_per_sg:
        self.assertLen(sg_gate_bufs, 10)
        self.assertLen(set(sg_gate_bufs), 6)
      # All signatures (prefill & decode) share the exact same 6 buffers.
      all_gate_bufs = set()
      for bufs in moe_gate_buffers_per_sg:
        all_gate_bufs.update(bufs)
      self.assertLen(all_gate_bufs, 6)

      # Verify full ML Drift GPU delegate compatibility: zero GATHER_ND ops,
      # zero constant-input RESHAPE / SLICE / STRIDED_SLICE / TRANSPOSE ops,
      # zero BFLOAT16 tensors (avoids YNNPACK CPU fallbacks), and every constant
      # GATHER input has a dtype accepted by `ObjectReader::AddConstInput`.
      gather_nd_ops = []
      const_input_gpu_unsupported_ops = []
      const_gather_bad_dtype_ops = []
      bf16_tensors = []
      composite_counts: dict[str, int] = {}
      const_forbidden_builtins = {
          22: "RESHAPE",
          39: "TRANSPOSE",
          45: "STRIDED_SLICE",
          65: "SLICE",
      }
      allowed_const_gather_dtypes = {
          qtyping.TensorType.FLOAT32,
          qtyping.TensorType.FLOAT16,
          qtyping.TensorType.BFLOAT16,
          qtyping.TensorType.INT8,
          qtyping.TensorType.UINT8,
          qtyping.TensorType.INT4,
          qtyping.TensorType.INT2,
          qtyping.TensorType.BOOL,
          qtyping.TensorType.INT32,
      }
      for sg_idx, sg in enumerate(fb_model.subgraphs):
        for t in sg.tensors:
          if t.type == qtyping.TensorType.BFLOAT16:
            bf16_tensors.append((sg_idx, t.name))
        for op_idx, op in enumerate(sg.operators):
          op_code = fb_model.operatorCodes[op.opcodeIndex]
          builtin = max(op_code.deprecatedBuiltinCode, op_code.builtinCode)
          if (
              builtin == qtyping.BuiltinOperator.STABLEHLO_COMPOSITE
              and getattr(op, "builtinOptions2", None) is not None
          ):
            comp_name = getattr(op.builtinOptions2, "name", b"")
            if isinstance(comp_name, bytes):
              comp_name = comp_name.decode("utf-8", errors="ignore")
            composite_counts[comp_name] = composite_counts.get(comp_name, 0) + 1
          # BuiltinOperator.GATHER_ND == 107, GATHER == 36.
          if builtin == 107:
            gather_nd_ops.append(
                (sg_idx, op_idx, sg.tensors[op.outputs[0]].name)
            )
          elif builtin == 36:
            for inp_pos, inp_idx in enumerate(op.inputs):
              if inp_idx < 0:
                continue
              in_t = sg.tensors[inp_idx]
              buf = fb_model.buffers[in_t.buffer]
              is_const = (buf.data is not None and len(buf.data) > 0) or (
                  getattr(buf, "size", 0) or 0
              ) > 0
              if is_const and in_t.type not in allowed_const_gather_dtypes:
                const_gather_bad_dtype_ops.append((
                    sg_idx,
                    op_idx,
                    inp_pos,
                    in_t.type,
                    in_t.name,
                    sg.tensors[op.outputs[0]].name,
                ))
          elif builtin in const_forbidden_builtins:
            in0 = sg.tensors[op.inputs[0]]
            buf = fb_model.buffers[in0.buffer]
            is_const = (buf.data is not None and len(buf.data) > 0) or (
                getattr(buf, "size", 0) or 0
            ) > 0
            if is_const:
              const_input_gpu_unsupported_ops.append((
                  const_forbidden_builtins[builtin],
                  sg_idx,
                  op_idx,
                  in0.name,
                  sg.tensors[op.outputs[0]].name,
              ))
      self.assertEmpty(gather_nd_ops, f"Found GATHER_ND ops: {gather_nd_ops}")
      self.assertEmpty(
          const_input_gpu_unsupported_ops,
          "Found constant-input GPU-unsupported ops:"
          f" {const_input_gpu_unsupported_ops}",
      )
      self.assertEmpty(
          const_gather_bad_dtype_ops,
          "Found constant-input GATHER ops with unsupported ML Drift"
          f" ObjectReader::AddConstInput dtype: {const_gather_bad_dtype_ops}",
      )
      self.assertEmpty(bf16_tensors, f"Found BFLOAT16 tensors: {bf16_tensors}")
      # Verify sensitive small routing / position / AttnRes weights stay float
      # (MessageAttentionResidual emits F.linear only if loop_len > block_size)
      # and no INT8 FC weight has a zero per-channel scale (XNNPACK CPU req).
      float_fc_hits = {
          "Inkling2LoopTopKRouter": 0,
          "Inkling2LoopRelativePositionBias": 0,
          "Inkling2LoopBlockAttentionResidual": 0,
      }
      for sg in fb_model.subgraphs:
        for op in sg.operators:
          op_code = fb_model.operatorCodes[op.opcodeIndex]
          builtin = max(op_code.deprecatedBuiltinCode, op_code.builtinCode)
          if builtin == qtyping.BuiltinOperator.FULLY_CONNECTED:
            scope = tfl_flatbuffer_utils.get_op_scope(op, sg.tensors)
            w_tensor = sg.tensors[op.inputs[1]]
            for key in float_fc_hits:
              if key in scope:
                self.assertEqual(
                    w_tensor.type,
                    qtyping.TensorType.FLOAT32,
                    f"Expected float32 weight for {scope}, got {w_tensor.type}",
                )
                float_fc_hits[key] += 1
            if (
                w_tensor.type == qtyping.TensorType.INT8
                and w_tensor.quantization is not None
                and w_tensor.quantization.scale is not None
            ):
              self.assertTrue(
                  all(s > 0.0 for s in w_tensor.quantization.scale),
                  f"Zero INT8 scale found in {scope}",
              )
      for key, count in float_fc_hits.items():
        self.assertGreater(
            count, 0, f"Expected >= 1 float32 FC matching {key}, got 0"
        )
      for expected_comp in (
          "odml.rms_norm",
          "odml.cache_update",
          "odml.runtime_bmm",
          "odml.short_conv_step",
      ):
        self.assertGreater(
            composite_counts.get(expected_comp, 0),
            0,
            f"Expected {expected_comp} composites, got {composite_counts}",
        )


if __name__ == "__main__":
  absltest.main()
