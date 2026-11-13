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
"""Top-level Qwen2.5-Omni Any2Any LiteRT exporter.

Orchestrates HuggingFace `Qwen2_5OmniForConditionalGeneration` (or
`Qwen2_5OmniThinkerForConditionalGeneration` when `generate_audio=False`) export
into LiteRT submodels:
- `text_embedder` and Thinker LLM prefill reuse `export_hf/core/external_emb`
  directly on upstream HF `thinker.get_input_embeddings()` and `thinker.model`.
- Thinker LLM decode, vision/audio encoders, Talker projector/decoder, and
  Token2Wav vocoder use the single-purpose wrappers in sibling modules
  (`llm.py`, `vision_encoder.py`, `audio_encoder.py`, `tts_projector.py`,
  `tts_decoder.py`, `vocoder.py`).
"""

from __future__ import annotations

import dataclasses
import gc
import os
from typing import Any

import huggingface_hub
import litert_torch
from litert_torch._convert import interface as converter_utils
from litert_torch.generative.export_hf.core import attention as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import cache as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import quantization
from litert_torch.generative.export_hf.core.external_emb import exportable_module as ext_emb
from litert_torch.generative.export_hf.model_ext.qwen_omni import audio_encoder
from litert_torch.generative.export_hf.model_ext.qwen_omni import llm
from litert_torch.generative.export_hf.model_ext.qwen_omni import tts_decoder
from litert_torch.generative.export_hf.model_ext.qwen_omni import tts_projector
from litert_torch.generative.export_hf.model_ext.qwen_omni import vision_encoder
from litert_torch.generative.export_hf.model_ext.qwen_omni import vocoder
import torch
from torch import nn
import transformers

try:
  from litert_lm_builder.runtime.proto import any2any_metadata_pb2
  from litert_lm_builder.runtime.proto import any2any_model_type_pb2
except ImportError:
  any2any_metadata_pb2 = None
  any2any_model_type_pb2 = None


def _parse_bool(val: Any, flag_name: str = "flag") -> bool:
  """Parses booleans including lowercase CLI strings such as `'false'`."""
  if isinstance(val, bool):
    return val
  if isinstance(val, str):
    lower = val.strip().lower()
    if lower in ("false", "0", "no", "off", ""):
      return False
    if lower in ("true", "1", "yes", "on"):
      return True
    raise ValueError(f"Invalid boolean value for {flag_name}: {val!r}")
  return bool(val)


def _maybe_quantize(out_path: str, quant_recipe: str | None) -> str:
  """Applies comma-separated quantization recipes to `out_path` if specified."""
  final_path = out_path
  for recipe in quant_recipe.split(",") if quant_recipe else [None]:
    prev_path = final_path
    final_path = quantization.maybe_quantize_model(final_path, recipe)
    if final_path != prev_path and os.path.exists(prev_path):
      os.remove(prev_path)
    gc.collect()
  return final_path


def _export_tflite(
    mod: nn.Module,
    sample_kwargs: dict[str, torch.Tensor],
    out_dir: str,
    quant_recipe: str | None,
) -> str:
  """Converts `mod` to FP32 `.tflite` and applies optional quantization."""
  os.makedirs(out_dir, exist_ok=True)
  fp32_path = os.path.join(out_dir, "model_fp32.tflite")
  litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(fp32_path)
  return _maybe_quantize(fp32_path, quant_recipe)


def _patch_size(v_cfg: Any) -> int:
  patch = v_cfg.patch_size
  return int(patch[0] if isinstance(patch, (list, tuple)) else patch)


class QwenOmni:
  """Qwen2.5-Omni Any-to-Any model exporter for LiteRT."""

  def __init__(
      self,
      model_path: str = "",
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ):
    self.model_path = model_path
    self.export_config = export_config
    self._hf_config: Any = None
    self._speaker_text_bos_token_id: int = 151872
    use_random = bool(export_config and export_config.use_random_weights)
    self.model_dir = self._resolve_model_dir(model_path, use_random)

  @staticmethod
  def _resolve_model_dir(model_path: str, use_random_weights: bool) -> str:
    if model_path and os.path.isdir(model_path):
      return model_path
    if model_path and not use_random_weights:
      try:
        return huggingface_hub.snapshot_download(model_path)
      except Exception:  # pylint: disable=broad-exception-caught
        return model_path
    return ""

  def _load_hf_config(self, load_from: str, use_random_weights: bool) -> Any:
    if load_from and os.path.isfile(os.path.join(load_from, "config.json")):
      return transformers.Qwen2_5OmniConfig.from_pretrained(load_from)
    if use_random_weights:
      return transformers.Qwen2_5OmniConfig()
    return transformers.Qwen2_5OmniConfig.from_pretrained(load_from)

  def _load_hf_model(
      self,
      export_config: exportable_module_config.ExportableModuleConfig,
      generate_audio: bool,
  ) -> Any:
    """Loads HF full Qwen2.5-Omni or Thinker-only model in FP32 eval mode."""
    model_path = export_config.model or self.model_path
    if model_path and not self.model_dir:
      self.model_dir = self._resolve_model_dir(
          model_path, export_config.use_random_weights
      )
    load_from = self.model_dir or model_path
    hf_cfg = self._load_hf_config(load_from, export_config.use_random_weights)
    self._hf_config = hf_cfg
    model_cls: Any = (
        transformers.Qwen2_5OmniForConditionalGeneration
        if generate_audio
        else transformers.Qwen2_5OmniThinkerForConditionalGeneration
    )
    cfg = hf_cfg if generate_audio else hf_cfg.thinker_config
    if export_config.use_random_weights:
      llm.ensure_mrope_section(hf_cfg.thinker_config.text_config)
      llm.ensure_mrope_section(hf_cfg.talker_config)
      return model_cls(cfg).eval().float()
    return model_cls.from_pretrained(
        load_from,
        config=cfg,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()

  def _vision_image_size(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> int:
    """Returns validated square vision resolution."""
    assert self._hf_config is not None
    v_cfg = self._hf_config.thinker_config.vision_config
    override = export_config.extra_kwargs.get("vision_image_size")
    image_size = (
        int(override) if override else int(getattr(v_cfg, "image_size", 280))
    )
    stride = _patch_size(v_cfg) * int(v_cfg.spatial_merge_size)
    if image_size <= 0 or image_size % stride != 0:
      raise ValueError(
          f"vision_image_size ({image_size}) must be a positive multiple of "
          f"patch_size * spatial_merge_size ({stride})."
      )
    return image_size

  def _audio_chunk_frames(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> int:
    assert self._hf_config is not None
    a_cfg = self._hf_config.thinker_config.audio_config
    override = export_config.extra_kwargs.get("audio_mel_frames") or getattr(
        a_cfg, "export_mel_frames", None
    )
    if override:
      return int(override)
    return min(int(a_cfg.n_window) * 2, int(a_cfg.max_source_positions) * 2)

  def _export_llm(
      self,
      thinker: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
      base_dir: str,
      generate_audio: bool,
  ) -> str:
    """Exports Thinker LLM prefill (core) and decode (`llm.py`) to `.tflite`."""
    llm.ensure_mrope_section(thinker.model.config)
    thinker.model.set_attn_implementation("lrt_transposed_attention")
    cache_lengths = export_config.cache_lengths or [export_config.cache_length]
    prefill_lengths = export_config.prefill_lengths or [128]
    out_dir = os.path.join(base_dir, "model")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_fp32.tflite")

    converter = converter_utils.Converter()
    for cache_len in cache_lengths:
      iter_config = dataclasses.replace(
          export_config,
          cache_length=cache_len,
          prefill_lengths=prefill_lengths,
          externalize_embedder=True,
          prefill_logits=False,
      )
      prefill_mod = (
          ext_emb.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder(
              thinker.model, iter_config
          )
      )
      decode_mod = llm.QwenOmniThinkerDecodeModule(
          thinker, iter_config, include_hidden_states=generate_audio
      )
      for module in (prefill_mod, decode_mod):
        for sig_name, (inputs, _) in module.get_sample_inputs(
            thinker.model.config
        ).items():
          if len(cache_lengths) > 1:
            sig_name = f"{sig_name}_cache_{cache_len}"
          converter.add_signature(sig_name, module.eval(), sample_kwargs=inputs)

    converter.convert(
        lightweight_conversion=export_config.experimental_lightweight_conversion,
        strict_export=False,
    ).export(out_path)
    return _maybe_quantize(out_path, export_config.quantization_recipe)

  def _export_tts_and_vocoder(
      self,
      hf_model: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
      base_dir: str,
  ) -> dict[str, str]:
    """Exports Talker projector, Talker decoder, and Token2Wav vocoder."""
    extra = export_config.extra_kwargs
    thinker, talker, token2wav = (
        hf_model.thinker,
        hf_model.talker,
        hf_model.token2wav,
    )
    if extra.get("tts_num_layers") is not None:
      talker.model.layers = talker.model.layers[: int(extra["tts_num_layers"])]

    out: dict[str, str] = {}
    out["tts_projector"] = _export_tflite(
        tts_projector.QwenOmniTtsProjectorWrapper(
            thinker.get_input_embeddings(), talker.thinker_to_talker_proj
        ).eval(),
        {
            "a_llm_hidden": torch.zeros(
                (1, 1, int(thinker.config.text_config.hidden_size)),
                dtype=torch.float32,
            ),
            "b_token_id": torch.zeros((1, 1), dtype=torch.int32),
        },
        os.path.join(base_dir, "tts_projector"),
        export_config.quantization_recipe,
    )

    max_cache_len = int(extra.get("tts_max_cache_len", 1024))
    dec_mod = tts_decoder.QwenOmniTtsDecoderWrapper(
        talker, max_cache_len=max_cache_len
    ).eval()
    kv_shape = (
        1,
        2 * len(talker.model.layers),
        max_cache_len,
        dec_mod.num_kv_heads * dec_mod.head_dim,
    )
    out["tts_decoder"] = _export_tflite(
        dec_mod,
        {
            "a_cond_emb": torch.zeros(
                (1, 1, int(talker.config.hidden_size)), dtype=torch.float32
            ),
            "b_prev_audio_code": torch.zeros((1, 1), dtype=torch.int32),
            "c_step_idx": torch.zeros((1,), dtype=torch.int32),
            "d_kv_cache": torch.zeros(kv_shape, dtype=torch.float32),
        },
        os.path.join(base_dir, "tts_decoder"),
        export_config.quantization_recipe,
    )

    speaker_name = str(extra.get("speaker", "Chelsie"))
    speaker_map = getattr(hf_model, "speaker_map", {}) or {}
    speaker_entry = speaker_map.get(speaker_name)
    if speaker_entry is None and speaker_map:
      if not export_config.use_random_weights:
        raise ValueError(
            f"Speaker {speaker_name!r} not found in speaker_map: "
            f"{list(speaker_map.keys())}"
        )
      speaker_entry = next(iter(speaker_map.values()))

    chunk_frames = int(extra.get("vocoder_chunk_frames", 48))
    output_frames_raw = extra.get("vocoder_output_frames")
    dit_cfg = token2wav.config.dit_config
    ode_steps_raw = extra.get("ode_steps")
    ref_mel_raw = extra.get("ref_mel_frames") or getattr(
        dit_cfg, "ref_mel_frames", None
    )
    voc_mod = vocoder.QwenOmniVocoderWrapper(
        token2wav,
        chunk_frames=chunk_frames,
        output_frames=(
            int(output_frames_raw) if output_frames_raw is not None else None
        ),
        ode_steps=int(
            ode_steps_raw
            if ode_steps_raw is not None
            else getattr(dit_cfg, "ode_steps", 10)
        ),
        cfg_strength=float(getattr(dit_cfg, "cfg_strength", 0.5)),
        ref_mel_frames=int(ref_mel_raw) if ref_mel_raw is not None else None,
        speaker_entry=speaker_entry,
    ).eval()
    if voc_mod.speaker_text_bos_token_id is not None:
      self._speaker_text_bos_token_id = voc_mod.speaker_text_bos_token_id

    # Keep `audio_vocoder` in FP32 by default because TFLite CPU Conv2D rejects
    # hybrid FLOAT32 activations + INT8 weights under `dynamic_wi8_afp32`.
    voc_quant = extra.get("vocoder_quantization_recipe")
    out["audio_vocoder"] = _export_tflite(
        voc_mod,
        {"audio_codes": torch.zeros((1, chunk_frames), dtype=torch.int32)},
        os.path.join(base_dir, "audio_vocoder"),
        str(voc_quant) if voc_quant else None,
    )
    return out

  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports Qwen2.5-Omni submodels for `export.export(task='any_to_any')`."""
    self.export_config = export_config
    extra = export_config.extra_kwargs
    generate_audio = _parse_bool(
        extra.get("generate_audio", True), "generate_audio"
    )
    export_vision = _parse_bool(
        export_config.export_vision_encoder, "export_vision_encoder"
    )
    export_audio = export_config.export_audio_encoder is None or _parse_bool(
        export_config.export_audio_encoder, "export_audio_encoder"
    )
    output_dir = (
        (
            export_config.work_dir
            if export_config.bundle_litert_lm
            else export_config.output_dir
        )
        or export_config.output_dir
        or export_config.work_dir
        or "."
    )
    os.makedirs(output_dir, exist_ok=True)

    hf_model = self._load_hf_model(export_config, generate_audio)
    thinker = hf_model.thinker if hasattr(hf_model, "thinker") else hf_model
    sub_models: dict[str, str] = {}

    if export_vision:
      image_size = self._vision_image_size(export_config)
      sub_models["vision_encoder"] = _export_tflite(
          vision_encoder.QwenOmniVisionEncoderWrapper(
              thinker.visual, image_size=image_size
          ).eval(),
          {"pixel_values": torch.zeros((1, 3, image_size, image_size))},
          os.path.join(output_dir, "vision_encoder"),
          export_config.vision_encoder_quantization_recipe,
      )

    if export_audio:
      a_cfg = self._hf_config.thinker_config.audio_config
      chunk_frames = self._audio_chunk_frames(export_config)
      sub_models["audio_encoder"] = _export_tflite(
          audio_encoder.QwenOmniAudioEncoderWrapper(
              thinker.audio_tower, chunk_frames=chunk_frames
          ).eval(),
          {
              "mel_features": torch.zeros(
                  (1, int(a_cfg.num_mel_bins), chunk_frames),
                  dtype=torch.float32,
              )
          },
          os.path.join(output_dir, "audio_encoder"),
          export_config.audio_encoder_quantization_recipe,
      )

    sub_models["text_embedder"] = _export_tflite(
        ext_emb.LiteRTExportableModuleForEmbedder(
            thinker.get_input_embeddings()
        ).eval(),
        {"token_ids": torch.zeros((1, 1), dtype=torch.int32)},
        os.path.join(output_dir, "text_embedder"),
        export_config.quantization_recipe,
    )

    sub_models["llm_backbone"] = self._export_llm(
        thinker, export_config, output_dir, generate_audio
    )
    if generate_audio:
      sub_models.update(
          self._export_tts_and_vocoder(hf_model, export_config, output_dir)
      )

    del hf_model, thinker
    gc.collect()
    return sub_models

  def get_any2any_metadata(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> any2any_metadata_pb2.Any2AnyMetadata:
    """Builds `Any2AnyMetadata` proto (`qwen_omni`) for `.litertlm` bundling."""
    if self._hf_config is None:
      load_from = self.model_dir or export_config.model or self.model_path
      self._hf_config = self._load_hf_config(
          load_from, export_config.use_random_weights
      )
    txt_cfg = self._hf_config.thinker_config.text_config
    v_cfg = self._hf_config.thinker_config.vision_config
    a_cfg = self._hf_config.thinker_config.audio_config
    talker_cfg = self._hf_config.talker_config
    t2w_cfg = self._hf_config.token2wav_config
    extra = export_config.extra_kwargs

    qwen_proto = any2any_model_type_pb2.QwenOmni(
        llm_hidden_size=int(txt_cfg.hidden_size),
        llm_vocab_size=int(txt_cfg.vocab_size),
        tts_hidden_size=int(talker_cfg.hidden_size),
        tts_vocab_size=int(talker_cfg.vocab_size),
        tts_num_mel_bins=int(t2w_cfg.bigvgan_config.mel_dim),
        tts_num_codebooks=1,
        vision_image_size=self._vision_image_size(export_config),
        vision_patch_size=_patch_size(v_cfg),
        vision_spatial_merge_size=int(v_cfg.spatial_merge_size),
        audio_num_mel_bins=int(a_cfg.num_mel_bins),
        audio_chunk_frames=self._audio_chunk_frames(export_config),
        audio_pool_step=2,
        generate_audio=_parse_bool(
            extra.get("generate_audio", True), "generate_audio"
        ),
        max_new_tokens=512,
        max_tts_tokens=1024,
        voice=str(extra.get("speaker", "Chelsie")),
        language="en-US",
        is_moe=False,
        tts_text_bos_token_id=int(self._speaker_text_bos_token_id),
    )
    return any2any_metadata_pb2.Any2AnyMetadata(
        input_sample_rate_hz=16000,
        output_sample_rate_hz=24000,
        supported_languages=["en-US", "zh-CN"],
        any2any_model_type=any2any_model_type_pb2.Any2AnyModelType(
            qwen_omni=qwen_proto
        ),
    )
