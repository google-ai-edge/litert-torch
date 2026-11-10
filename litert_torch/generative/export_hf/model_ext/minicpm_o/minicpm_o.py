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
"""MiniCPM-o 4.5 (openbmb/MiniCPM-o-4_5) multimodal Any2Any LiteRT export."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import gc
import json
import logging
import os
import shutil
from typing import Any

import huggingface_hub
import litert_torch
from litert_torch._convert import interface as converter_utils
from litert_torch.generative.export_hf.core import attention as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import cache as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import weights_loader as weights_loader_lib
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
from litert_torch.generative.export_hf.model_ext.minicpm_o import encoders
from litert_torch.generative.export_hf.model_ext.minicpm_o import tts
from litert_torch.generative.export_hf.model_ext.minicpm_o import vocoder
from litert_torch.generative.export_hf.model_ext.minicpm_o import weights_utils
from litert_torch.generative.export_hf.model_ext.qwen3 import patch as _  # pylint: disable=unused-import
import safetensors
import torch
import transformers

try:
  from litert_lm_builder.runtime.proto import any2any_metadata_pb2
  from litert_lm_builder.runtime.proto import any2any_model_type_pb2
except ImportError:
  any2any_metadata_pb2 = None
  any2any_model_type_pb2 = None

# Re-export public module classes and helpers for backward compatibility.
_LARGE_TENSOR_NUMEL_THRESHOLD = weights_utils.LARGE_TENSOR_NUMEL_THRESHOLD
_make_linear = weights_utils.make_linear
_make_embedding = weights_utils.make_embedding
_build_1d_sincos_pos_embed = encoders.build_1d_sincos_pos_embed
_build_2d_sincos_pos_embed = encoders.build_2d_sincos_pos_embed
_rotate_half = tts.rotate_half
# pylint: disable=protected-access
_build_slaney_mel_basis = vocoder._build_slaney_mel_basis
_extract_ref_mel_24k = vocoder._extract_ref_mel_24k
# pylint: enable=protected-access

MiniCpmOGeluTransformerBlock = encoders.MiniCpmOGeluTransformerBlock
MiniCpmOVisionEncoderModule = encoders.MiniCpmOVisionEncoderModule
MiniCpmOAudioEncoderModule = encoders.MiniCpmOAudioEncoderModule
MiniCpmOTextEmbedderModule = encoders.MiniCpmOTextEmbedderModule
MiniCpmORMSNorm = tts.MiniCpmORMSNorm
MiniCpmOTtsTransformerBlock = tts.MiniCpmOTtsTransformerBlock
MiniCpmOTtsProjectorModule = tts.MiniCpmOTtsProjectorModule
MiniCpmOTtsDecoderModule = tts.MiniCpmOTtsDecoderModule
MiniCpmOAudioVocoderModule = vocoder.MiniCpmOAudioVocoderModule

_MINICPMO_EXTRA_KWARGS = (
    "generate_audio",
    "include_decode_hidden_states",
    "min_tts_steps",
    "tts_max_cache_len",
    "tts_num_layers",
    "vocoder_cfg_rate",
    "vocoder_chunk_frames",
    "vocoder_flow_dim",
    "vocoder_flow_heads",
    "vocoder_flow_layers",
    "vocoder_hidden_channels",
    "vocoder_input_tokens",
    "vocoder_mel_bins",
    "vocoder_n_timesteps",
    "vocoder_prompt_tokens",
    "vocoder_up_layers",
)
exportable_module_config.register_extra_kwargs_keys(_MINICPMO_EXTRA_KWARGS)

_DEFAULT_MINICPMO_CONFIG: dict[str, Any] = {
    "model_type": "minicpmo",
    "version": "4.5",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151748,
    "max_position_embeddings": 40960,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "attention_bias": False,
    "scale_resolution": 448,
    "patch_size": 14,
    "max_slice_nums": 9,
    "query_num": 64,
    "audio_pool_step": 5,
    "vision_config": {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "patch_size": 14,
        "image_size": 980,
    },
    "audio_config": {
        "d_model": 1024,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "encoder_layers": 24,
        "num_mel_bins": 80,
    },
    "tts_config": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 20,
        "num_attention_heads": 12,
        "num_audio_tokens": 6562,
        "num_mel_bins": 100,
    },
}


def _resolve_targets(
    export_config: exportable_module_config.ExportableModuleConfig,
) -> set[str]:
  """Normalizes `extra_kwargs['targets']` into a set of target names."""
  raw_targets = export_config.extra_kwargs.get("targets", ["all"])
  if isinstance(raw_targets, str):
    return {t.strip() for t in raw_targets.split(",") if t.strip()}
  return set(raw_targets)


def _resolve_generate_audio(
    export_config: exportable_module_config.ExportableModuleConfig,
) -> bool:
  """Resolves and validates `generate_audio` against targets and flags."""
  targets = _resolve_targets(export_config)
  has_tts_targets = "all" in targets or {
      "tts_projector",
      "tts_decoder",
      "audio_vocoder",
  }.issubset(targets)
  include_hidden = bool(
      export_config.extra_kwargs.get("include_decode_hidden_states", True)
  )
  raw_gen_audio = export_config.extra_kwargs.get("generate_audio")
  if raw_gen_audio is None:
    return has_tts_targets and include_hidden
  generate_audio = bool(raw_gen_audio)
  if generate_audio:
    if not include_hidden:
      raise ValueError(
          "generate_audio=True requires include_decode_hidden_states=True."
      )
    if not has_tts_targets:
      raise ValueError(
          "generate_audio=True requires 'tts_projector', 'tts_decoder', and"
          " 'audio_vocoder' in targets."
      )
  return generate_audio


_BaseGenerateModule = (
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder
)
_BasePrefillModule = (
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder
)


class _MiniCpmOLlmGenerateModule(_BaseGenerateModule):
  """Generate module concatenating `[logits, last_hidden_state]` in `logits`.

  `MiniCpmOLlmStage::RunDecodeStep` inspects the length of the `decode` output
  buffer (`logits`): when its size equals `vocab_size + llm_hidden_size`, the
  runtime splits the trailing `llm_hidden_size` floats into
  `step_out.hidden_state` (required by `MiniCpmOTtsStage` when `generate_audio`
  is enabled) and resizes `logits` to `vocab_size`.
  """

  # pylint: disable=arguments-renamed
  def forward(
      self,
      embeddings,
      input_pos,
      kv_cache,
      mask,
      local_mask=None,
      **kwargs,
  ):
    include_hidden = bool(
        self.export_config.extra_kwargs.get(
            "include_decode_hidden_states", True
        )
    )
    if not include_hidden:
      return super().forward(
          embeddings, input_pos, kv_cache, mask, local_mask=local_mask, **kwargs
      )
    orig_verifier = self.export_verifier
    self.export_verifier = True
    try:
      result = super().forward(
          embeddings, input_pos, kv_cache, mask, local_mask=local_mask, **kwargs
      )
    finally:
      self.export_verifier = orig_verifier
    last_hidden = result["activations"].to(torch.float32)
    result["logits"] = torch.cat([result["logits"], last_hidden], dim=-1)
    if not orig_verifier:
      result.pop("activations", None)
    return result


class _MiniCpmOLlmWeightsLoader(weights_loader_lib.HFCheckpointWeightsLoader):
  """Streams `llm.*`-prefixed Qwen3/Qwen2 weights from MiniCPM-o checkpoints."""

  def _candidates(self, param_name: str) -> list[str]:
    base = super()._candidates(param_name)
    return base + [f"llm.{c}" for c in base]

  def __call__(self, param_name: str) -> torch.Tensor:
    try:
      return super().__call__(param_name)
    except KeyError:
      if self._tie_word_embeddings and param_name.endswith("lm_head.weight"):
        for emb_key in (
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
        ):
          tensor = self._try_load(emb_key)
          if tensor is not None:
            return tensor
      raise


class MiniCpmO:
  """openbmb/MiniCPM-o-4_5 multimodal Any-to-Any model exporter for LiteRT."""

  def __init__(
      self,
      model_path: str,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ):
    self.model_path = model_path
    self.model_dir = (
        model_path
        if os.path.exists(model_path)
        else huggingface_hub.snapshot_download(model_path)
    )
    self.export_config = export_config
    self._config: dict[str, Any] = copy.deepcopy(_DEFAULT_MINICPMO_CONFIG)
    self._load_config()
    self._state_dict: dict[str, torch.Tensor] = {}
    self._weight_map: dict[str, str] = {}
    if export_config is None or not export_config.use_random_weights:
      self._load_weights_if_available()

  def _load_config(self) -> None:
    """Loads and deep-merges config.json from `self.model_dir` if present."""
    cfg_path = os.path.join(self.model_dir, "config.json")
    if os.path.exists(cfg_path):
      with open(cfg_path, "r") as f:
        loaded = json.load(f)
      for k, v in loaded.items():
        if isinstance(v, dict) and isinstance(self._config.get(k), dict):
          self._config[k].update(v)
        else:
          self._config[k] = v
      slice_cfg = loaded.get("slice_config")
      if isinstance(slice_cfg, dict):
        for sk in ("max_slice_nums", "scale_resolution", "patch_size"):
          if sk in slice_cfg and sk not in loaded:
            self._config[sk] = slice_cfg[sk]

  def _load_weights_if_available(self) -> None:
    """Indexes safetensors weights from `self.model_dir` if present."""
    single_st = os.path.join(self.model_dir, "model.safetensors")
    index_st = os.path.join(self.model_dir, "model.safetensors.index.json")
    if os.path.exists(single_st):
      with safetensors.safe_open(single_st, framework="pt", device="cpu") as f:
        for k in f.keys():
          self._weight_map[k] = single_st
    elif os.path.exists(index_st):
      with open(index_st, "r") as f:
        index_data = json.load(f)
      for k, shard in index_data.get("weight_map", {}).items():
        shard_path = os.path.join(self.model_dir, shard)
        if os.path.exists(shard_path):
          self._weight_map[k] = shard_path

  def _require_weights_or_random(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> None:
    if (
        not export_config.use_random_weights
        and not self._state_dict
        and not self._weight_map
    ):
      raise FileNotFoundError(
          f"No safetensors weights found in {self.model_dir} and "
          "use_random_weights=False."
      )

  def _get_weight(self, key: str) -> torch.Tensor | None:
    """Retrieves a weight tensor by key from memory or sharded safetensors."""
    return weights_utils.get_weight(self._state_dict, self._weight_map, key)

  def _copy_weight(
      self,
      target: torch.Tensor,
      *keys: str,
      strict: bool = True,
      populated: set[int] | None = None,
  ) -> None:
    """Copies the first matching checkpoint tensor in `keys` into `target`."""
    weights_utils.copy_weight(
        target,
        self._state_dict,
        self._weight_map,
        *keys,
        strict=strict,
        populated=populated,
    )

  def export_vision_encoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports SigLIP + Perceiver resampler vision encoder to TFLite.

    Args:
      output_dir: Directory where the exported `.tflite` file is written.
      export_config: Export configuration controlling shapes and quantization.

    Returns:
      Path to the exported vision encoder `.tflite` file.
    """
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    v_cfg = self._config.get("vision_config", {})
    vision_hidden = int(v_cfg.get("hidden_size", 1152))
    vision_inter = int(v_cfg.get("intermediate_size", 4304))
    vision_heads = int(v_cfg.get("num_attention_heads", 16))
    default_vision_layers = (
        min(int(v_cfg.get("num_hidden_layers", 1)), 1)
        if export_config.use_random_weights
        else int(v_cfg.get("num_hidden_layers", 27))
    )
    vision_layers = int(
        export_config.extra_kwargs.get(
            "vision_num_layers", default_vision_layers
        )
    )
    patch_size = int(
        self._config.get("patch_size", v_cfg.get("patch_size", 14))
    )
    image_size = int(v_cfg.get("image_size", 980))
    max_patches = max(16, (image_size // max(1, patch_size)) ** 2)
    query_num = int(self._config.get("query_num", 64))
    llm_hidden = int(self._config.get("hidden_size", 4096))
    default_num_patches = 16 if export_config.use_random_weights else 1088
    num_patches = int(
        export_config.extra_kwargs.get(
            "vision_num_patches", default_num_patches
        )
    )

    mod = encoders.MiniCpmOVisionEncoderModule(
        vision_hidden_size=vision_hidden,
        vision_intermediate_size=vision_inter,
        vision_num_layers=vision_layers,
        vision_num_heads=vision_heads,
        patch_size=patch_size,
        max_patches=max_patches,
        query_num=query_num,
        llm_hidden_size=llm_hidden,
    ).eval()
    encoders.load_vision_encoder_weights(
        mod, self._state_dict, self._weight_map, strict=strict
    )

    sample_kwargs = {
        "pixel_values": torch.zeros(
            1, 3, patch_size, patch_size * num_patches, dtype=torch.float32
        ),
        "position_ids": (
            torch.arange(num_patches, dtype=torch.int32).unsqueeze(0)
        ),
    }
    out_path = os.path.join(output_dir, "vision_encoder_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    quant_recipe = (
        export_config.vision_encoder_quantization_recipe
        if export_config.vision_encoder_quantization_recipe is not None
        else export_config.quantization_recipe
    )
    return weights_utils.maybe_quantize(out_path, quant_recipe)

  def export_audio_encoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports Whisper-medium audio encoder + projection to TFLite.

    Args:
      output_dir: Directory where the exported `.tflite` file is written.
      export_config: Export configuration controlling shapes and quantization.

    Returns:
      Path to the exported audio encoder `.tflite` file.
    """
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    a_cfg = self._config.get("audio_config", {})
    num_mel_bins = int(a_cfg.get("num_mel_bins", 80))
    audio_hidden = int(a_cfg.get("d_model", 1024))
    audio_inter = int(a_cfg.get("encoder_ffn_dim", 4096))
    audio_heads = int(a_cfg.get("encoder_attention_heads", 16))
    default_audio_layers = (
        min(int(a_cfg.get("encoder_layers", 1)), 1)
        if export_config.use_random_weights
        else int(a_cfg.get("encoder_layers", 24))
    )
    audio_layers = int(
        export_config.extra_kwargs.get("audio_num_layers", default_audio_layers)
    )
    max_source_pos = int(a_cfg.get("max_source_positions", 1500))
    pool_step = int(self._config.get("audio_pool_step", 5))
    llm_hidden = int(self._config.get("hidden_size", 4096))
    mel_frames = int(export_config.extra_kwargs.get("audio_mel_frames", 100))

    mod = encoders.MiniCpmOAudioEncoderModule(
        num_mel_bins=num_mel_bins,
        audio_hidden_size=audio_hidden,
        audio_intermediate_size=audio_inter,
        audio_num_layers=audio_layers,
        audio_num_heads=audio_heads,
        audio_pool_step=pool_step,
        llm_hidden_size=llm_hidden,
        max_source_positions=max_source_pos,
    ).eval()
    encoders.load_audio_encoder_weights(
        mod, self._state_dict, self._weight_map, strict=strict
    )

    # TODO(b/568027544): Enable `valid_mel_frames` and `position_offset` inputs
    # by default once `MiniCpmOFrontendStage` passes per-chunk frame lengths and
    # position offsets at runtime.
    sample_kwargs: dict[str, torch.Tensor] = {
        "mel_features": torch.zeros(
            1, num_mel_bins, mel_frames, dtype=torch.float32
        ),
    }
    if bool(export_config.extra_kwargs.get("audio_include_mask_inputs", False)):
      sample_kwargs["valid_mel_frames"] = torch.tensor(
          [mel_frames], dtype=torch.int32
      )
      sample_kwargs["position_offset"] = torch.tensor([0], dtype=torch.int32)
    out_path = os.path.join(output_dir, "audio_encoder_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    quant_recipe = (
        export_config.audio_encoder_quantization_recipe
        if export_config.audio_encoder_quantization_recipe is not None
        else export_config.quantization_recipe
    )
    return weights_utils.maybe_quantize(out_path, quant_recipe)

  def export_text_embedder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports Qwen3/Qwen2 text token embedding lookup table to TFLite.

    Args:
      output_dir: Directory where the exported `.tflite` file is written.
      export_config: Export configuration controlling shapes and quantization.

    Returns:
      Path to the exported text embedder `.tflite` file.
    """
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    vocab_size = int(self._config.get("vocab_size", 151748))
    llm_hidden = int(self._config.get("hidden_size", 4096))
    mod = encoders.MiniCpmOTextEmbedderModule(
        vocab_size=vocab_size, llm_hidden_size=llm_hidden
    ).eval()
    encoders.load_text_embedder_weights(
        mod, self._state_dict, self._weight_map, strict=strict
    )

    sample_kwargs = {"token_ids": torch.zeros(1, 1, dtype=torch.int32)}
    out_path = os.path.join(output_dir, "text_embedder_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return weights_utils.maybe_quantize(
        out_path, export_config.quantization_recipe
    )

  def export_llm_backbone(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports Qwen3/Qwen2 LLM backbone (`prefill_*` + `decode`) to TFLite.

    Args:
      output_dir: Directory where the exported `.tflite` file is written.
      export_config: Export configuration controlling shapes and quantization.

    Returns:
      Path to the exported LLM backbone `.tflite` file.
    """
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    llm_hidden = int(self._config.get("hidden_size", 4096))
    default_inter = (
        min(int(self._config.get("intermediate_size", 12288)), 1024)
        if export_config.use_random_weights
        else int(self._config.get("intermediate_size", 12288))
    )
    llm_inter = int(
        export_config.extra_kwargs.get("llm_intermediate_size", default_inter)
    )
    default_layers = (
        min(int(self._config.get("num_hidden_layers", 1)), 1)
        if export_config.use_random_weights
        else int(self._config.get("num_hidden_layers", 36))
    )
    llm_layers = int(
        export_config.extra_kwargs.get("llm_num_layers", default_layers)
    )
    llm_heads = int(self._config.get("num_attention_heads", 32))
    llm_kv_heads = int(self._config.get("num_key_value_heads", llm_heads))
    vocab_size = int(self._config.get("vocab_size", 151748))
    rms_norm_eps = float(self._config.get("rms_norm_eps", 1e-6))
    rope_theta = float(self._config.get("rope_theta", 1000000.0))
    max_pos = int(self._config.get("max_position_embeddings", 40960))
    tie_word_embeddings = bool(self._config.get("tie_word_embeddings", False))
    use_sliding_window = bool(self._config.get("use_sliding_window", False))
    sliding_window = (
        int(self._config.get("sliding_window", 131072))
        if use_sliding_window
        else None
    )

    all_keys = set(self._weight_map) | set(self._state_dict)
    has_qkv_bias = any(k.endswith(".self_attn.q_proj.bias") for k in all_keys)
    has_qk_norm = any(k.endswith(".self_attn.q_norm.weight") for k in all_keys)
    if has_qk_norm:
      is_qwen3 = True
    elif has_qkv_bias:
      is_qwen3 = False
    else:
      is_qwen3 = str(self._config.get("version", "4.5")) == "4.5" and not bool(
          self._config.get("attention_bias", False)
      )

    head_dim = int(
        self._config.get("head_dim", max(1, llm_hidden // max(1, llm_heads)))
    )
    if is_qwen3:
      llm_config = transformers.Qwen3Config(
          vocab_size=vocab_size,
          hidden_size=llm_hidden,
          intermediate_size=llm_inter,
          num_hidden_layers=llm_layers,
          num_attention_heads=llm_heads,
          num_key_value_heads=llm_kv_heads,
          head_dim=head_dim,
          hidden_act=str(self._config.get("hidden_act", "silu")),
          max_position_embeddings=max_pos,
          rms_norm_eps=rms_norm_eps,
          rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
          tie_word_embeddings=tie_word_embeddings,
          attention_bias=bool(self._config.get("attention_bias", False)),
          use_sliding_window=use_sliding_window,
          sliding_window=sliding_window,
          bos_token_id=int(self._config.get("bos_token_id", 151643)),
          eos_token_id=int(self._config.get("eos_token_id", 151645)),
      )
      patch_name = "qwen3"
      model_cls = transformers.Qwen3ForCausalLM
    else:
      llm_config = transformers.Qwen2Config(
          vocab_size=vocab_size,
          hidden_size=llm_hidden,
          intermediate_size=llm_inter,
          num_hidden_layers=llm_layers,
          num_attention_heads=llm_heads,
          num_key_value_heads=llm_kv_heads,
          hidden_act=str(self._config.get("hidden_act", "silu")),
          max_position_embeddings=max_pos,
          rms_norm_eps=rms_norm_eps,
          rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
          tie_word_embeddings=tie_word_embeddings,
          use_sliding_window=use_sliding_window,
          sliding_window=sliding_window,
          bos_token_id=int(self._config.get("bos_token_id", 151643)),
          eos_token_id=int(self._config.get("eos_token_id", 151645)),
      )
      patch_name = "qwen2"
      model_cls = transformers.Qwen2ForCausalLM
    # pylint: disable=protected-access
    llm_config._attn_implementation = "lrt_transposed_attention"
    # pylint: enable=protected-access

    has_checkpoint_weights = "llm.lm_head.weight" in self._weight_map or (
        tie_word_embeddings
        and "llm.model.embed_tokens.weight" in self._weight_map
    )
    use_v2 = strict and bool(export_config.use_v2) and has_checkpoint_weights
    base_loader = None
    with model_ext_patches.get_patch_context(patch_name):
      if use_v2:
        with weights_loader_lib.init_params_on_meta():
          qwen_model = model_cls(llm_config)
        qwen_model.eval()
        base_loader = _MiniCpmOLlmWeightsLoader(
            self.model_dir, tie_word_embeddings=tie_word_embeddings
        )
        weights_loader_lib.load_persistent_buffers_(qwen_model, base_loader)
      else:
        qwen_model = model_cls(llm_config).eval().to(torch.float32)
        populated: set[int] = set()
        self._copy_weight(
            qwen_model.model.embed_tokens.weight,
            "llm.model.embed_tokens.weight",
            strict=strict,
            populated=populated,
        )
        for idx, layer in enumerate(qwen_model.model.layers):
          p = f"llm.model.layers.{idx}"
          self._copy_weight(
              layer.input_layernorm.weight,
              f"{p}.input_layernorm.weight",
              strict=strict,
              populated=populated,
          )
          self._copy_weight(
              layer.self_attn.q_proj.weight,
              f"{p}.self_attn.q_proj.weight",
              strict=strict,
              populated=populated,
          )
          if layer.self_attn.q_proj.bias is not None:
            self._copy_weight(
                layer.self_attn.q_proj.bias,
                f"{p}.self_attn.q_proj.bias",
                strict=strict,
                populated=populated,
            )
          self._copy_weight(
              layer.self_attn.k_proj.weight,
              f"{p}.self_attn.k_proj.weight",
              strict=strict,
              populated=populated,
          )
          if layer.self_attn.k_proj.bias is not None:
            self._copy_weight(
                layer.self_attn.k_proj.bias,
                f"{p}.self_attn.k_proj.bias",
                strict=strict,
                populated=populated,
            )
          self._copy_weight(
              layer.self_attn.v_proj.weight,
              f"{p}.self_attn.v_proj.weight",
              strict=strict,
              populated=populated,
          )
          if layer.self_attn.v_proj.bias is not None:
            self._copy_weight(
                layer.self_attn.v_proj.bias,
                f"{p}.self_attn.v_proj.bias",
                strict=strict,
                populated=populated,
            )
          if hasattr(layer.self_attn, "q_norm"):
            self._copy_weight(
                layer.self_attn.q_norm.weight,
                f"{p}.self_attn.q_norm.weight",
                strict=strict,
                populated=populated,
            )
          if hasattr(layer.self_attn, "k_norm"):
            self._copy_weight(
                layer.self_attn.k_norm.weight,
                f"{p}.self_attn.k_norm.weight",
                strict=strict,
                populated=populated,
            )
          self._copy_weight(
              layer.self_attn.o_proj.weight,
              f"{p}.self_attn.o_proj.weight",
              strict=strict,
              populated=populated,
          )
          self._copy_weight(
              layer.post_attention_layernorm.weight,
              f"{p}.post_attention_layernorm.weight",
              strict=strict,
              populated=populated,
          )
          self._copy_weight(
              layer.mlp.gate_proj.weight,
              f"{p}.mlp.gate_proj.weight",
              strict=strict,
              populated=populated,
          )
          self._copy_weight(
              layer.mlp.up_proj.weight,
              f"{p}.mlp.up_proj.weight",
              strict=strict,
              populated=populated,
          )
          self._copy_weight(
              layer.mlp.down_proj.weight,
              f"{p}.mlp.down_proj.weight",
              strict=strict,
              populated=populated,
          )
        self._copy_weight(
            qwen_model.model.norm.weight,
            "llm.model.norm.weight",
            strict=strict,
            populated=populated,
        )
        if tie_word_embeddings:
          self._copy_weight(
              qwen_model.lm_head.weight,
              "llm.lm_head.weight",
              "llm.model.embed_tokens.weight",
              strict=strict,
              populated=populated,
          )
        else:
          self._copy_weight(
              qwen_model.lm_head.weight,
              "llm.lm_head.weight",
              strict=strict,
              populated=populated,
          )
        if strict:
          weights_utils.verify_all_parameters_populated(
              qwen_model, populated, "llm_backbone"
          )

    cache_lengths = export_config.cache_lengths or [export_config.cache_length]
    prefill_lengths = export_config.prefill_lengths or [128]

    out_path = os.path.join(output_dir, "llm_backbone_fp32.tflite")
    intermediates_dir = os.path.join(output_dir, "llm_backbone_intermediates")

    with model_ext_patches.patch_model(qwen_model, patch_name, export_config):
      qwen_model.set_attn_implementation("lrt_transposed_attention")
      signatures = []
      for cache_len in cache_lengths:
        iter_config = dataclasses.replace(
            export_config,
            cache_length=cache_len,
            prefill_lengths=prefill_lengths,
            externalize_embedder=True,
        )
        prefill_module = _BasePrefillModule(qwen_model, iter_config)
        decode_module = _MiniCpmOLlmGenerateModule(qwen_model, iter_config)
        for module in (prefill_module, decode_module):
          for sig_name, (inputs, dynamic_shapes) in module.get_sample_inputs(
              llm_config
          ).items():
            if len(cache_lengths) > 1:
              sig_name = f"{sig_name}_cache_{cache_len}"
            signatures.append((sig_name, module, inputs, dynamic_shapes))

      signature_modules = list({id(s[1]): s[1] for s in signatures}.values())
      lightweight = export_config.experimental_lightweight_conversion
      with contextlib.ExitStack() as stack:
        weights_loader = base_loader
        if weights_loader is not None:
          buffer_values = stack.enter_context(
              weights_loader_lib.buffers_on_meta(*signature_modules)
          )
          weights_loader = weights_loader_lib.with_tensors(
              weights_loader, buffer_values
          )
        converter = converter_utils.Converter()
        for sig_name, module, inputs, _ in signatures:
          converter.add_signature(
              sig_name,
              module.eval(),
              sample_kwargs=inputs,
          )
        if use_v2:
          converter.convert(
              lightweight_conversion=lightweight,
              strict_export=False,
              use_v2=True,
              delete_in_memory_params=True,
              allow_reuse_intermediates=False,
              weights_loader=weights_loader,
              export_dir=intermediates_dir,
              output_file_path=out_path,
              _litert_converter_flags={
                  "fold_fp16_casts_into_fully_connected": True,
              },
          )
        else:
          lrt_model = converter.convert(
              lightweight_conversion=lightweight,
              strict_export=False,
          )
          lrt_model.export(out_path)
          del lrt_model

    if os.path.exists(intermediates_dir):
      shutil.rmtree(intermediates_dir, ignore_errors=True)
    del qwen_model
    gc.collect()

    quant_recipe = export_config.quantization_recipe
    if (
        not export_config.use_random_weights
        and llm_layers > 2
        and (
            not quant_recipe
            or str(quant_recipe).strip().lower()
            in ("none", "null", "false", "")
        )
    ):
      logging.warning(
          "Exporting %d-layer MiniCPM-o LLM backbone without quantization"
          " (quantization_recipe=%r); resulting FP32 TFLite file may exceed"
          " 30 GB.",
          llm_layers,
          quant_recipe,
      )

    return weights_utils.maybe_quantize(out_path, quant_recipe)

  def export_tts_projector(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports LLM-to-TTS semantic projector (`tf_lite_tts_projector`).

    Args:
      output_dir: Directory where the exported `.tflite` file is written.
      export_config: Export configuration controlling shapes and quantization.

    Returns:
      Path to the exported TTS projector `.tflite` file.
    """
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    llm_hidden = int(self._config.get("hidden_size", 4096))
    tts_cfg = self._config.get("tts_config", {})
    tts_hidden = int(tts_cfg.get("hidden_size", 768))
    emb_text_w = self._get_weight("tts.emb_text.weight")
    default_num_text_tokens = (
        int(emb_text_w.shape[0])
        if emb_text_w is not None
        else int(self._config.get("vocab_size", 152064))
    )
    num_text_tokens = int(
        tts_cfg.get("num_text_tokens", default_num_text_tokens)
    )
    mod = tts.MiniCpmOTtsProjectorModule(
        llm_hidden_size=llm_hidden,
        tts_hidden_size=tts_hidden,
        num_text_tokens=num_text_tokens,
    ).eval()
    tts.load_tts_projector_weights(
        mod, self._state_dict, self._weight_map, tts_cfg, strict=strict
    )

    sample_kwargs = {
        "llm_hidden": torch.zeros(1, 1, llm_hidden, dtype=torch.float32),
        "token_id": torch.zeros(1, 1, dtype=torch.int32),
    }
    out_path = os.path.join(output_dir, "tts_projector_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return weights_utils.maybe_quantize(
        out_path, export_config.quantization_recipe
    )

  def export_tts_decoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports KV-cached TTS decoder (`tf_lite_tts_decoder`) to TFLite.

    Args:
      output_dir: Directory where the exported `.tflite` file is written.
      export_config: Export configuration controlling shapes and quantization.

    Returns:
      Path to the exported TTS decoder `.tflite` file.
    """
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    tts_cfg = self._config.get("tts_config", {})
    tts_hidden = int(tts_cfg.get("hidden_size", 768))
    tts_inter = int(tts_cfg.get("intermediate_size", 3072))
    default_tts_layers = (
        min(int(tts_cfg.get("num_hidden_layers", 1)), 1)
        if export_config.use_random_weights
        else int(tts_cfg.get("num_hidden_layers", 20))
    )
    tts_layers = int(
        export_config.extra_kwargs.get("tts_num_layers", default_tts_layers)
    )
    tts_heads = int(tts_cfg.get("num_attention_heads", 12))
    tts_vocab = int(tts_cfg.get("num_audio_tokens", 6562))
    tts_mel_bins = int(tts_cfg.get("num_mel_bins", 100))
    default_max_cache_len = 32 if export_config.use_random_weights else 256
    tts_max_cache_len = int(
        export_config.extra_kwargs.get(
            "tts_max_cache_len", default_max_cache_len
        )
    )
    min_tts_steps = int(export_config.extra_kwargs.get("min_tts_steps", 0))

    mod = tts.MiniCpmOTtsDecoderModule(
        tts_hidden_size=tts_hidden,
        tts_intermediate_size=tts_inter,
        tts_num_layers=tts_layers,
        tts_num_heads=tts_heads,
        tts_vocab_size=tts_vocab,
        tts_num_mel_bins=tts_mel_bins,
        tts_max_cache_len=tts_max_cache_len,
        min_tts_steps=min_tts_steps,
    ).eval()
    tts.load_tts_decoder_weights(
        mod, self._state_dict, self._weight_map, strict=strict
    )

    sample_kwargs = {
        "cond_emb": torch.zeros(1, 1, tts_hidden, dtype=torch.float32),
        "prev_audio_code": torch.zeros(1, 1, dtype=torch.int32),
        "step_idx": torch.zeros(1, dtype=torch.int32),
        "kv_cache": torch.zeros(
            1,
            2 * tts_layers,
            tts_max_cache_len,
            tts_hidden,
            dtype=torch.float32,
        ),
    }
    out_path = os.path.join(output_dir, "tts_decoder_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return weights_utils.maybe_quantize(
        out_path, export_config.quantization_recipe
    )

  def export_audio_vocoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports CosyVoice2 Token2Wav vocoder (`tf_lite_audio_vocoder`).

    Args:
      output_dir: Directory where the exported `.tflite` file is written.
      export_config: Export configuration controlling shapes and quantization.

    Returns:
      Path to the exported audio vocoder `.tflite` file.
    """
    self._require_weights_or_random(export_config)
    strict = not export_config.use_random_weights
    tts_cfg = self._config.get("tts_config", {})

    flow_path = os.path.join(self.model_dir, "assets", "token2wav", "flow.pt")
    hift_path = os.path.join(self.model_dir, "assets", "token2wav", "hift.pt")
    flow_sd: dict[str, torch.Tensor] = {}
    hift_sd: dict[str, torch.Tensor] = {}
    if os.path.exists(flow_path):
      try:
        loaded_flow = torch.load(
            flow_path, map_location="cpu", weights_only=True
        )
      except TypeError:  # pragma: no cover
        loaded_flow = torch.load(flow_path, map_location="cpu")
      if isinstance(loaded_flow, dict):
        flow_sd = loaded_flow
    if os.path.exists(hift_path):
      try:
        loaded_hift = torch.load(
            hift_path, map_location="cpu", weights_only=True
        )
      except TypeError:  # pragma: no cover
        loaded_hift = torch.load(hift_path, map_location="cpu")
      if isinstance(loaded_hift, dict):
        hift_sd = loaded_hift

    if not flow_sd and not hift_sd:
      all_keys = set(self._weight_map) | set(self._state_dict)
      for k in all_keys:
        if k.startswith("vocoder.flow."):
          t = self._get_weight(k)
          if t is not None:
            flow_sd[k.removeprefix("vocoder.flow.")] = t
        elif k.startswith("vocoder.hift."):
          t = self._get_weight(k)
          if t is not None:
            hift_sd[k.removeprefix("vocoder.hift.")] = t

    emb_w = flow_sd.get("input_embedding.weight")
    default_vocab_size = (
        int(emb_w.shape[0])
        if emb_w is not None
        else max(1, int(tts_cfg.get("num_audio_tokens", 6562)) - 1)
    )
    default_flow_dim = (
        int(emb_w.shape[1])
        if emb_w is not None
        else (16 if export_config.use_random_weights else 512)
    )
    flow_dim = int(
        export_config.extra_kwargs.get("vocoder_flow_dim", default_flow_dim)
    )
    pos_u = flow_sd.get("encoder.encoders.0.self_attn.pos_bias_u")
    default_flow_heads = (
        int(pos_u.shape[0])
        if pos_u is not None
        else (2 if export_config.use_random_weights else 8)
    )
    flow_heads = int(
        export_config.extra_kwargs.get("vocoder_flow_heads", default_flow_heads)
    )
    ff_w1 = flow_sd.get("encoder.encoders.0.feed_forward.w_1.weight")
    flow_ffn_dim = int(ff_w1.shape[0]) if ff_w1 is not None else 4 * flow_dim

    enc_indices = {
        int(k.split(".")[2])
        for k in flow_sd
        if k.startswith("encoder.encoders.")
    }
    default_num_encoders = (
        len(enc_indices)
        if enc_indices
        else (1 if export_config.use_random_weights else 6)
    )
    flow_num_encoders = int(
        export_config.extra_kwargs.get(
            "vocoder_flow_layers", default_num_encoders
        )
    )
    up_indices = {
        int(k.split(".")[2])
        for k in flow_sd
        if k.startswith("encoder.up_encoders.")
    }
    default_num_up_encoders = (
        len(up_indices)
        if up_indices
        else (1 if export_config.use_random_weights else 4)
    )
    flow_num_up_encoders = int(
        export_config.extra_kwargs.get(
            "vocoder_up_layers", default_num_up_encoders
        )
    )
    dit_indices = {
        int(k.split(".")[3])
        for k in flow_sd
        if k.startswith("decoder.estimator.blocks.")
    }
    default_num_dit_blocks = (
        len(dit_indices)
        if dit_indices
        else (1 if export_config.use_random_weights else 16)
    )
    flow_num_dit_blocks = int(
        export_config.extra_kwargs.get(
            "vocoder_flow_layers", default_num_dit_blocks
        )
    )
    spk_w = flow_sd.get("spk_embed_affine_layer.weight")
    default_mel_bins = (
        int(spk_w.shape[0])
        if spk_w is not None
        else (16 if export_config.use_random_weights else 80)
    )
    mel_bins = int(
        export_config.extra_kwargs.get("vocoder_mel_bins", default_mel_bins)
    )
    spk_dim = (
        int(spk_w.shape[1])
        if spk_w is not None
        else (16 if export_config.use_random_weights else 192)
    )
    conv_pre_b = hift_sd.get("conv_pre.bias")
    default_hidden_ch = (
        int(conv_pre_b.shape[0])
        if conv_pre_b is not None
        else (32 if export_config.use_random_weights else 512)
    )
    hift_base_channels = int(
        export_config.extra_kwargs.get(
            "vocoder_hidden_channels", default_hidden_ch
        )
    )
    f0_indices = {
        int(k.split(".")[2])
        for k in hift_sd
        if k.startswith("f0_predictor.condnet.")
    }
    hift_f0_conv_layers = (
        len(f0_indices)
        if f0_indices
        else (1 if export_config.use_random_weights else 5)
    )
    default_input_tokens = (
        8
        if (export_config.use_random_weights or flow_num_dit_blocks < 4)
        else 32
    )
    input_tokens = int(
        export_config.extra_kwargs.get(
            "vocoder_input_tokens",
            export_config.extra_kwargs.get(
                "vocoder_chunk_frames", default_input_tokens
            ),
        )
    )
    default_n_timesteps = (
        1
        if (export_config.use_random_weights or flow_num_dit_blocks < 4)
        else 6
    )
    n_timesteps = int(
        export_config.extra_kwargs.get(
            "vocoder_n_timesteps", default_n_timesteps
        )
    )
    prompt_tokens_len = export_config.extra_kwargs.get(
        "vocoder_prompt_tokens", None
    )
    if prompt_tokens_len is not None:
      prompt_tokens_len = int(prompt_tokens_len)
    elif any(
        os.path.exists(os.path.join(self.model_dir, "assets", w))
        for w in ("system_ref_audio.wav", "HT_ref_audio.wav")
    ):
      prompt_tokens_len = (
          len(vocoder.DEFAULT_VOICE_PROMPT_TOKENS)
          if input_tokens >= 32
          else min(4, max(1, input_tokens // 2))
      )
    cfg_rate = export_config.extra_kwargs.get("vocoder_cfg_rate", None)
    if cfg_rate is not None:
      cfg_rate = float(cfg_rate)

    mod = vocoder.MiniCpmOAudioVocoderModule(
        vocab_size=default_vocab_size,
        input_tokens=input_tokens,
        flow_dim=flow_dim,
        flow_heads=flow_heads,
        flow_ffn_dim=flow_ffn_dim,
        flow_num_encoders=flow_num_encoders,
        flow_num_up_encoders=flow_num_up_encoders,
        flow_num_dit_blocks=flow_num_dit_blocks,
        mel_bins=mel_bins,
        spk_dim=spk_dim,
        n_timesteps=n_timesteps,
        hift_base_channels=hift_base_channels,
        hift_f0_conv_layers=hift_f0_conv_layers,
        prompt_tokens_len=prompt_tokens_len,
        cfg_rate=cfg_rate,
    ).eval()
    populated: set[int] = set()
    if flow_sd or hift_sd or strict:
      vocoder.load_token2wav_weights(
          mod, flow_sd, hift_sd, strict=strict, populated=populated
      )
      if strict:
        weights_utils.verify_all_parameters_populated(
            mod, populated, "audio_vocoder"
        )
    if mod.prompt_tokens_len > 0:
      for wav_name in ("system_ref_audio.wav", "HT_ref_audio.wav"):
        wav_path = os.path.join(self.model_dir, "assets", wav_name)
        if os.path.exists(wav_path):
          with torch.no_grad():
            ref_mel = _extract_ref_mel_24k(
                wav_path,
                num_frames=mod.flow_dit.prompt_mel_frames,
                n_mels=mel_bins,
            )
            mod.flow_dit.prompt_mel.copy_(ref_mel)
          break

    sample_kwargs = {
        "audio_codes": torch.zeros(1, input_tokens, dtype=torch.int32)
    }
    out_path = os.path.join(output_dir, "audio_vocoder_fp32.tflite")
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return weights_utils.maybe_quantize(
        out_path, export_config.quantization_recipe
    )

  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports MiniCPM-o 4.5 vision, audio, text-embedder, LLM, and TTS models.

    Args:
      export_config: Export configuration specifying output directory, targets,
        and quantization recipes.

    Returns:
      Dictionary mapping exported artifact names to `.tflite` file paths.
    """
    _resolve_generate_audio(export_config)
    output_dir = (
        (
            export_config.work_dir
            if export_config.bundle_litert_lm
            else export_config.output_dir
        )
        or export_config.output_dir
        or export_config.work_dir
    )
    if not output_dir:
      raise ValueError("Either output_dir or work_dir must be specified.")
    os.makedirs(output_dir, exist_ok=True)
    artifacts: dict[str, str] = {}

    targets = _resolve_targets(export_config)
    export_all = "all" in targets

    if (
        export_all and export_config.export_vision_encoder
    ) or "vision_encoder" in targets:
      artifacts["vision_encoder"] = self.export_vision_encoder(
          output_dir, export_config
      )
    if (
        export_all and export_config.export_audio_encoder
    ) or "audio_encoder" in targets:
      artifacts["audio_encoder"] = self.export_audio_encoder(
          output_dir, export_config
      )
    if (
        export_all and export_config.externalize_embedder
    ) or "text_embedder" in targets:
      artifacts["text_embedder"] = self.export_text_embedder(
          output_dir, export_config
      )
    if export_all or "llm_backbone" in targets:
      artifacts["llm_backbone"] = self.export_llm_backbone(
          output_dir, export_config
      )
    if export_all or "tts_projector" in targets:
      artifacts["tts_projector"] = self.export_tts_projector(
          output_dir, export_config
      )
    if export_all or "tts_decoder" in targets:
      artifacts["tts_decoder"] = self.export_tts_decoder(
          output_dir, export_config
      )
    if export_all or "audio_vocoder" in targets:
      artifacts["audio_vocoder"] = self.export_audio_vocoder(
          output_dir, export_config
      )

    return artifacts

  def get_any2any_metadata(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> any2any_metadata_pb2.Any2AnyMetadata:
    """Builds Any2AnyMetadata proto for the exported MiniCPM-o 4.5 model.

    Args:
      export_config: Export configuration containing extra_kwargs.

    Returns:
      Populated `Any2AnyMetadata` protobuf message.
    """
    v_cfg = self._config.get("vision_config", {})
    a_cfg = self._config.get("audio_config", {})
    tts_cfg = self._config.get("tts_config", {})
    generate_audio = _resolve_generate_audio(export_config)

    minicpm_proto = any2any_model_type_pb2.MiniCpmO(
        llm_hidden_size=int(self._config.get("hidden_size", 4096)),
        llm_vocab_size=int(self._config.get("vocab_size", 151748)),
        tts_hidden_size=int(tts_cfg.get("hidden_size", 768)),
        tts_vocab_size=int(tts_cfg.get("num_audio_tokens", 6562)),
        tts_num_mel_bins=int(tts_cfg.get("num_mel_bins", 100)),
        vision_scale_resolution=int(
            self._config.get("scale_resolution", v_cfg.get("image_size", 448))
        ),
        vision_patch_size=int(
            self._config.get("patch_size", v_cfg.get("patch_size", 14))
        ),
        vision_max_slice_nums=int(self._config.get("max_slice_nums", 9)),
        num_image_queries=int(self._config.get("query_num", 64)),
        audio_pool_step=int(
            self._config.get("audio_pool_step", a_cfg.get("pool_step", 5))
        ),
        generate_audio=generate_audio,
        max_new_tokens=512,
        max_tts_tokens=1024,
        voice="default",
        language="en-US",
    )
    return any2any_metadata_pb2.Any2AnyMetadata(
        input_sample_rate_hz=16000,
        output_sample_rate_hz=24000,
        supported_languages=["en-US", "zh-CN"],
        any2any_model_type=any2any_model_type_pb2.Any2AnyModelType(
            minicpm_o=minicpm_proto
        ),
    )
