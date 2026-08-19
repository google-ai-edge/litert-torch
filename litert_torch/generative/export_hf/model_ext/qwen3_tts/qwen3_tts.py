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
"""Qwen3-TTS model implementation for LiteRT export."""

import copy
import importlib
import importlib.util
import json
import os
import shutil
import sys

import huggingface_hub
import litert_torch
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.speech import tts_model
from litert_torch.generative.export_hf.model_ext.qwen3_tts import exportable_module
from litert_torch.generative.export_hf.model_ext.qwen3_tts import patch as qwen3_tts_patch
from litert_torch.generative.export_hf.model_ext.qwen3_tts import speaker_encoder as qwen3_tts_speaker_encoder
from litert_torch.generative.export_hf.model_ext.qwen3_tts.tokenizer_v2 import configuration_qwen3_tts_tokenizer_v2 as tok_config_mod
from litert_torch.generative.export_hf.model_ext.qwen3_tts.tokenizer_v2 import modeling_qwen3_tts_tokenizer_v2 as tok_model_mod
from litert_torch.generative.quantize import quant_recipes
import safetensors
import safetensors.torch
import torch

from ai_edge_quantizer import recipe as recipe_lib

_TALKER_SYNTH_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 2149,
    "eos_token_id": 2150,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 32768,
    "max_window_layers": 28,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "use_cache": True,
    "vocab_size": 4096,
}


def setup_talker_recipe():
  """Sets up blockwise-32 INT4 weight-only quantization with INT8 embedding lookup."""
  wo_recipe = recipe_lib.dynamic_wi4_afp32()[0]
  emb_recipe = copy.deepcopy(wo_recipe)
  emb_recipe["op_config"]["weight_tensor_config"]["num_bits"] = 8
  emb_recipe["operation"] = "EMBEDDING_LOOKUP"
  block_recipe = copy.deepcopy(wo_recipe)
  if hasattr(recipe_lib, "AlgorithmName") and hasattr(
      recipe_lib.AlgorithmName, "OCTAV"
  ):
    block_recipe["algorithm_key"] = recipe_lib.AlgorithmName.OCTAV
  block_recipe["op_config"]["weight_tensor_config"][
      "granularity"
  ] = "BLOCKWISE_32"
  setattr(recipe_lib, "BOCTAV4", lambda: [block_recipe, emb_recipe])


def get_mtp_quant_config():
  """Returns the LiteRT quantization recipe for MTP."""
  return quant_recipes.full_fp16_recipe()


class Qwen3Tts(tts_model.TtsModel):
  """Qwen3-TTS multi-stage text-to-speech model."""

  def __init__(
      self,
      model_path: str,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
      override_transformers: bool = False,
  ):
    super().__init__(model_path, export_config)
    del override_transformers
    if os.path.exists(model_path):
      self.model_dir = model_path
    else:
      self.model_dir = huggingface_hub.snapshot_download(model_path)

    safetensors_path = os.path.join(self.model_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
      self.reader = safetensors.safe_open(safetensors_path, framework="pt")
    else:
      self.reader = None

  def export_speaker_encoder(self, output_dir: str) -> str:
    """Authors and exports the ECAPA-TDNN speaker encoder."""
    print("Exporting speaker encoder...")
    model = qwen3_tts_speaker_encoder.Qwen3TTSSpeakerEncoder().eval()
    if self.reader is not None:
      state_dict = {}
      prefix = "speaker_encoder."
      for k in self.reader.keys():
        if k.startswith(prefix):
          state_dict[k[len(prefix) :]] = self.reader.get_tensor(k).to(
              torch.float32
          )
      model.load_state_dict(state_dict)

    out_path = os.path.join(output_dir, "speaker_encoder_fp32.tflite")
    sample_input = (torch.zeros(1, 300, 128, dtype=torch.float32),)
    litert_torch.convert(model, sample_input).export(out_path)
    print(f"Speaker encoder exported to: {out_path}")
    return out_path

  def export_embedding_and_projection_tables(
      self, output_dir: str
  ) -> dict[str, str]:
    """Exports embedding tables and text projection directly to TFLite models."""
    if self.reader is None:
      return {}

    artifacts = {}
    print("Converting text_embedding.tflite...")
    text_emb_weights = self.reader.get_tensor(
        "talker.model.text_embedding.weight"
    ).to(torch.float32)

    class TextEmbeddingModule(torch.nn.Module):
      """PyTorch module for text token embedding lookup."""

      def __init__(self, emb_matrix: torch.Tensor):
        super().__init__()
        self.emb_matrix = torch.nn.Parameter(emb_matrix, requires_grad=False)

      def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(token_ids, self.emb_matrix)

    mod_text_emb = TextEmbeddingModule(text_emb_weights).eval()
    text_emb_path = os.path.join(output_dir, "text_embedding.tflite")
    sample_token_ids = (torch.zeros(1, dtype=torch.int32),)
    litert_torch.convert(mod_text_emb, sample_token_ids).export(text_emb_path)
    artifacts["text_embedding"] = text_emb_path

    print("Converting codec_embedding.tflite...")
    codec_emb_weights = self.reader.get_tensor(
        "talker.model.codec_embedding.weight"
    ).to(torch.float32)

    class CodecEmbeddingModule(torch.nn.Module):
      """PyTorch module for audio codec embedding lookup."""

      def __init__(self, emb_matrix: torch.Tensor):
        super().__init__()
        self.emb_matrix = torch.nn.Parameter(emb_matrix, requires_grad=False)

      def forward(self, codec_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(codec_ids, self.emb_matrix)

    mod_codec_emb = CodecEmbeddingModule(codec_emb_weights).eval()
    codec_emb_path = os.path.join(output_dir, "codec_embedding.tflite")
    sample_codec_ids = (torch.zeros(1, dtype=torch.int32),)
    litert_torch.convert(mod_codec_emb, sample_codec_ids).export(codec_emb_path)
    artifacts["codec_embedding"] = codec_emb_path

    print("Converting mtp_embedding.tflite...")
    mtp_embs = [
        self.reader.get_tensor(
            f"talker.code_predictor.model.codec_embedding.{i}.weight"
        ).to(torch.float32)
        for i in range(15)
    ]
    mtp_embs_flat = torch.cat(mtp_embs, dim=0)

    class MtpEmbeddingModule(torch.nn.Module):
      """PyTorch module for multi-step prediction embedding lookup."""

      def __init__(self, emb_matrix: torch.Tensor):
        super().__init__()
        self.emb_matrix = torch.nn.Parameter(emb_matrix, requires_grad=False)

      def forward(self, mtp_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(mtp_ids, self.emb_matrix)

    mod_mtp_emb = MtpEmbeddingModule(mtp_embs_flat).eval()
    mtp_emb_path = os.path.join(output_dir, "mtp_embedding.tflite")
    sample_mtp_ids = (torch.zeros(1, dtype=torch.int32),)
    litert_torch.convert(mod_mtp_emb, sample_mtp_ids).export(mtp_emb_path)
    artifacts["mtp_embedding"] = mtp_emb_path

    print("Converting text_projection.tflite...")
    w1 = self.reader.get_tensor("talker.text_projection.linear_fc1.weight").to(
        torch.float32
    )
    b1 = self.reader.get_tensor("talker.text_projection.linear_fc1.bias").to(
        torch.float32
    )
    w2 = self.reader.get_tensor("talker.text_projection.linear_fc2.weight").to(
        torch.float32
    )
    b2 = self.reader.get_tensor("talker.text_projection.linear_fc2.bias").to(
        torch.float32
    )

    class TextProjectionModule(torch.nn.Module):
      """PyTorch module for MLP text embedding projection."""

      def __init__(
          self,
          w1_mat: torch.Tensor,
          b1_vec: torch.Tensor,
          w2_mat: torch.Tensor,
          b2_vec: torch.Tensor,
      ):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(w1_mat.shape[1], w1_mat.shape[0])
        self.linear_fc2 = torch.nn.Linear(w2_mat.shape[1], w2_mat.shape[0])
        with torch.no_grad():
          self.linear_fc1.weight.copy_(w1_mat)
          self.linear_fc1.bias.copy_(b1_vec)
          self.linear_fc2.weight.copy_(w2_mat)
          self.linear_fc2.bias.copy_(b2_vec)

      def forward(self, text_embeds: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.linear_fc1(text_embeds))
        return self.linear_fc2(h)

    mod_text_proj = TextProjectionModule(w1, b1, w2, b2).eval()
    text_proj_path = os.path.join(output_dir, "text_projection.tflite")
    sample_text_embeds = (torch.zeros(1, 2048, dtype=torch.float32),)
    litert_torch.convert(mod_text_proj, sample_text_embeds).export(
        text_proj_path
    )
    artifacts["text_projection"] = text_proj_path
    return artifacts

  def export_codec_decoder(
      self, output_dir: str, litert_samples_conversion_dir: str | None = None
  ) -> str | None:
    """Exports codec_decoder_fp32.tflite using speech tokenizer modeling."""
    print("Exporting codec_decoder_fp32.tflite...")
    speech_tok_dir = os.path.join(self.model_dir, "speech_tokenizer")
    if not os.path.exists(speech_tok_dir):
      try:
        downloaded = huggingface_hub.snapshot_download(
            self.model_path, allow_patterns=["speech_tokenizer/*"]
        )
        speech_tok_dir = os.path.join(downloaded, "speech_tokenizer")
      except Exception:  # pylint: disable=broad-exception-caught
        speech_tok_dir = None

    if not speech_tok_dir or not os.path.exists(speech_tok_dir):
      print(
          f"Notice: speech_tokenizer directory not found in {self.model_dir}."
      )
      return None

    if speech_tok_dir not in sys.path:
      sys.path.insert(0, speech_tok_dir)

    model = None
    try:
      cfg_cls = tok_config_mod.Qwen3TTSTokenizerV2Config
      model_cls = tok_model_mod.Qwen3TTSTokenizerV2Model
      config = cfg_cls.from_pretrained(speech_tok_dir)
      try:
        model = model_cls.from_pretrained(
            speech_tok_dir, config=config, torch_dtype=torch.float32
        ).eval()
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            f"Notice: from_pretrained failed ({e}), loading via direct"
            " state_dict..."
        )
        model = model_cls(config).eval()
        safetensors_path = os.path.join(speech_tok_dir, "model.safetensors")
        if os.path.exists(safetensors_path):
          state_dict = safetensors.torch.load_file(safetensors_path)
          model.load_state_dict(state_dict, strict=False)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Notice: tokenizer_v2 direct import failed: {e}")

    if model is None or not hasattr(model, "decoder"):
      print(
          "Notice: Could not load codec decoder model. Skipping codec decoder"
          " export."
      )
      return None

    decoder = model.decoder
    if hasattr(decoder, "pre_transformer") and hasattr(
        decoder.pre_transformer, "config"
    ):
      decoder.pre_transformer.config.use_cache = False

      rotary = getattr(decoder.pre_transformer, "rotary_emb", None)
      if rotary is not None:
        dim = getattr(decoder.pre_transformer.config, "head_dim", 64)
        theta = getattr(decoder.pre_transformer.config, "rope_theta", 10000)
        inv_freq = getattr(rotary, "inv_freq", None)
        if inv_freq is not None:
          with torch.no_grad():
            rotary.inv_freq.copy_(
                1.0
                / (
                    theta
                    ** (
                        torch.arange(
                            0,
                            dim,
                            2,
                            dtype=torch.float32,
                            device=rotary.inv_freq.device,
                        )
                        / dim
                    )
                )
            )
        rotary.attention_scaling = 1.0

    class CodecDecode(torch.nn.Module):
      """Wrapper module to export codec decoder forward pass."""

      def __init__(self, dec):
        super().__init__()
        self.dec = dec

      def forward(self, codes):
        return self.dec(codes)

    out_path = os.path.join(output_dir, "codec_decoder_fp32.tflite")
    sample_input = (torch.zeros(1, 16, 64, dtype=torch.int32),)
    litert_torch.convert(CodecDecode(decoder).eval(), sample_input).export(
        out_path
    )
    print(f"Codec decoder exported to: {out_path}")
    return out_path

  def export_mtp(self, output_dir: str) -> str:
    """Authors and exports the static rank-4 MTP step model."""
    print("Exporting MTP model...")
    weights = {}
    if self.reader is not None:
      prefix = "talker.code_predictor."
      for key in self.reader.keys():
        if (
            key.startswith(prefix + "model.layers.")
            or key == prefix + "model.norm.weight"
        ):
          weights[key[len(prefix + "model.") :]] = self.reader.get_tensor(
              key
          ).to(torch.float32)
      heads = [
          self.reader.get_tensor(f"{prefix}lm_head.{i}.weight").to(
              torch.float32
          )
          for i in range(15)
      ]
      weights["heads"] = torch.stack(heads)
    else:
      for i in range(exportable_module.LAYERS):
        weights[f"layers.{i}.input_layernorm.weight"] = torch.randn(1024)
        weights[f"layers.{i}.self_attn.q_proj.weight"] = (
            torch.randn(
                exportable_module.HEADS * exportable_module.HEAD_DIM, 1024
            )
            * 0.02
        )
        weights[f"layers.{i}.self_attn.k_proj.weight"] = (
            torch.randn(
                exportable_module.KV_HEADS * exportable_module.HEAD_DIM, 1024
            )
            * 0.02
        )
        weights[f"layers.{i}.self_attn.v_proj.weight"] = (
            torch.randn(
                exportable_module.KV_HEADS * exportable_module.HEAD_DIM, 1024
            )
            * 0.02
        )
        weights[f"layers.{i}.self_attn.q_norm.weight"] = torch.randn(
            exportable_module.HEAD_DIM
        )
        weights[f"layers.{i}.self_attn.k_norm.weight"] = torch.randn(
            exportable_module.HEAD_DIM
        )
        weights[f"layers.{i}.self_attn.o_proj.weight"] = (
            torch.randn(
                1024, exportable_module.HEADS * exportable_module.HEAD_DIM
            )
            * 0.02
        )
        weights[f"layers.{i}.post_attention_layernorm.weight"] = torch.randn(
            1024
        )
        weights[f"layers.{i}.mlp.gate_proj.weight"] = (
            torch.randn(3072, 1024) * 0.02
        )
        weights[f"layers.{i}.mlp.up_proj.weight"] = (
            torch.randn(3072, 1024) * 0.02
        )
        weights[f"layers.{i}.mlp.down_proj.weight"] = (
            torch.randn(1024, 3072) * 0.02
        )
      weights["norm.weight"] = torch.randn(1024)
      weights["heads"] = torch.randn(15, exportable_module.VOCAB, 1024) * 0.02

    mtp_module = exportable_module.MtpStep(weights).eval()
    sample_inputs = {
        "embeddings": torch.zeros(1, 1, 1024, dtype=torch.float32),
        "input_ids": torch.zeros(1, dtype=torch.int32),
        "mask": torch.zeros(1, 1, 1, 32, dtype=torch.float32),
        **{
            f"kv_cache_k_{i}": torch.zeros(1, 32, 8, 128, dtype=torch.float32)
            for i in range(5)
        },
        **{
            f"kv_cache_v_{i}": torch.zeros(1, 32, 8, 128, dtype=torch.float32)
            for i in range(5)
        },
    }

    out_path = os.path.join(output_dir, "mtp_fp32.tflite")
    quant_cfg = get_mtp_quant_config()
    litert_torch.convert(
        mtp_module, sample_kwargs=sample_inputs, quant_config=quant_cfg
    ).export(out_path)
    print(f"MTP model exported to: {out_path}")
    return out_path

  def export_talker(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str | None:
    """Synthesizes causal checkpoint and exports Talker LLM."""
    if self.reader is None:
      return None

    print("Exporting Talker model...")
    synth_dir = os.path.join(
        export_config.work_dir or output_dir, "synth_talker_ckpt"
    )
    os.makedirs(synth_dir, exist_ok=True)

    out_tensors = {}
    for key in self.reader.keys():
      if (
          not key.startswith("talker.model.")
          and key != "talker.codec_head.weight"
      ):
        continue
      if key.startswith("talker.model.text_embedding"):
        continue
      tensor = self.reader.get_tensor(key).to(torch.float32)
      if key == "talker.model.codec_embedding.weight":
        pad = torch.zeros(4096 - tensor.shape[0], tensor.shape[1])
        out_tensors["model.embed_tokens.weight"] = torch.cat([tensor, pad], 0)
      elif key == "talker.codec_head.weight":
        eye = torch.eye(1024)
        eye = eye + 1e-6 * (1.0 - eye)
        out_tensors["lm_head.weight"] = torch.cat([tensor, eye], 0)
      else:
        out_tensors[key.replace("talker.model.", "model.")] = tensor

    safetensors.torch.save_file(
        out_tensors,
        os.path.join(synth_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )
    with open(os.path.join(synth_dir, "config.json"), "w") as f:
      json.dump(_TALKER_SYNTH_CONFIG, f, indent=1)
    with open(os.path.join(synth_dir, "generation_config.json"), "w") as f:
      json.dump({"bos_token_id": 2149, "eos_token_id": 2150}, f)

    for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
      src_path = os.path.join(self.model_dir, name)
      if os.path.exists(src_path):
        shutil.copy(src_path, os.path.join(synth_dir, name))

    talker_out_dir = os.path.join(
        export_config.work_dir or output_dir, "talker_export"
    )
    os.makedirs(talker_out_dir, exist_ok=True)

    recipe_name = None
    if export_config.quantization_recipe in (
        "int4_weight_only",
        "BOCTAV4",
        "dynamic_wi4_afp32",
    ):
      setup_talker_recipe()
      recipe_name = "BOCTAV4"
    elif export_config.quantization_recipe:
      recipe_name = export_config.quantization_recipe

    export_mod = importlib.import_module(
        "litert_torch.generative.export_hf.export"
    )

    exported_artifacts = export_mod.export(
        model=synth_dir,
        output_dir=talker_out_dir,
        quantization_recipe=recipe_name,
        externalize_embedder=export_config.externalize_embedder,
        single_token_embedder=export_config.single_token_embedder,
        cache_length=export_config.cache_length
        if export_config.cache_length != 4096
        else 1024,
        prefill_lengths=export_config.prefill_lengths
        if export_config.prefill_lengths != [128]
        else [32, 128],
        bundle_litert_lm=export_config.bundle_litert_lm,
        keep_temporary_files=True,
        use_jinja_template=export_config.use_jinja_template,
        trust_remote_code=export_config.trust_remote_code,
    )

    src_model_path = None
    if (
        exported_artifacts
        and hasattr(exported_artifacts, "prefill_decode_model_path")
        and exported_artifacts.prefill_decode_model_path
        and os.path.exists(exported_artifacts.prefill_decode_model_path)
    ):
      src_model_path = exported_artifacts.prefill_decode_model_path
    else:
      for root, _, files in os.walk(talker_out_dir):
        if "model_quantized.tflite" in files:
          src_model_path = os.path.join(root, "model_quantized.tflite")
          break
        elif "model.tflite" in files:
          src_model_path = os.path.join(root, "model.tflite")
          break
        for f in files:
          if f.endswith(".tflite") and not f.startswith("embedder"):
            src_model_path = os.path.join(root, f)
            break

    dest_path = os.path.join(
        output_dir,
        "talker_int4.tflite"
        if recipe_name == "BOCTAV4"
        else "talker_fp32.tflite",
    )

    if src_model_path and os.path.exists(src_model_path):
      shutil.copy(src_model_path, dest_path)
      print(f"Talker model exported to: {dest_path}")
    else:
      print(f"Warning: Talker model source not found in {talker_out_dir}")

    # Copy tokenizer files to output directory if present
    for tok_file in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.json",
    ):
      for src_dir in (talker_out_dir, self.model_dir):
        src_path = os.path.join(src_dir, tok_file)
        if os.path.exists(src_path):
          shutil.copy(src_path, os.path.join(output_dir, tok_file))
          break

    return dest_path

  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports all Qwen3-TTS submodules to LiteRT TFLite files."""
    output_dir = export_config.output_dir or export_config.work_dir
    if not output_dir:
      raise ValueError("Either output_dir or work_dir must be specified.")
    os.makedirs(output_dir, exist_ok=True)
    artifacts = {}

    targets = set(export_config.extra_kwargs.get("targets", ["all"]))
    export_all = "all" in targets

    if export_all or "speaker_encoder" in targets:
      artifacts["speaker_encoder"] = self.export_speaker_encoder(output_dir)

    if export_all or "embeddings" in targets:
      emb_artifacts = self.export_embedding_and_projection_tables(output_dir)
      artifacts.update(emb_artifacts)

    if export_all or "codec" in targets:
      litert_samples_dir = export_config.extra_kwargs.get(
          "litert_samples_conversion_dir", None
      )
      codec_path = self.export_codec_decoder(output_dir, litert_samples_dir)
      if codec_path:
        artifacts["codec_decoder"] = codec_path

    with qwen3_tts_patch.qwen3_tts_litert_patch():
      if export_all or "mtp" in targets:
        artifacts["mtp"] = self.export_mtp(output_dir)
      if export_all or "talker" in targets:
        talker_path = self.export_talker(output_dir, export_config)
        if talker_path:
          artifacts["talker"] = talker_path

    return artifacts
