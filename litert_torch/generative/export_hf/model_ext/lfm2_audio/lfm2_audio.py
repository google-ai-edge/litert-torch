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
"""LFM2.5-Audio-1.5B (`LiquidAI/LFM2.5-Audio-1.5B`) Any2Any LiteRT export."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import dataclasses
import gc
import json
import os

import huggingface_hub
import litert_torch
from litert_torch._convert import interface as converter_utils
from litert_torch.generative.export_hf.core import attention as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import cache as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import quantization
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
from litert_torch.generative.export_hf.model_ext.lfm2 import patch as _  # pylint: disable=unused-import
from litert_torch.generative.export_hf.model_ext.lfm2_audio import modules
import safetensors
import torch
from torch import nn
import transformers

try:
  from litert_lm_builder.runtime.proto import any2any_metadata_pb2
  from litert_lm_builder.runtime.proto import any2any_model_type_pb2
except ImportError:
  any2any_metadata_pb2 = None
  any2any_model_type_pb2 = None

_LFM2_AUDIO_EXTRA_KWARGS = (
    'audio_mel_frames',
    'detok_chunk_frames',
    'export_audio_encoder',
    'export_llm_backbone',
    'export_tts',
    'export_vocoder',
)
exportable_module_config.register_extra_kwargs_keys(_LFM2_AUDIO_EXTRA_KWARGS)

_DEFAULT_LFM2_LAYER_TYPES = (
    'conv',
    'conv',
    'full_attention',
    'conv',
    'conv',
    'full_attention',
    'conv',
    'conv',
    'full_attention',
    'conv',
    'full_attention',
    'conv',
    'full_attention',
    'conv',
    'full_attention',
    'conv',
)


def _list_safetensors_shards(checkpoint_dir: str) -> list[str]:
  """Returns ordered safetensors shard paths in `checkpoint_dir`."""
  index_path = os.path.join(checkpoint_dir, 'model.safetensors.index.json')
  if os.path.exists(index_path):
    with open(index_path, 'r') as f:
      index_data = json.load(f)
    weight_map = index_data.get('weight_map', {})
    shard_names = sorted(set(weight_map.values()))
    return [os.path.join(checkpoint_dir, name) for name in shard_names]
  single_path = os.path.join(checkpoint_dir, 'model.safetensors')
  if os.path.exists(single_path):
    return [single_path]
  shards = sorted(
      os.path.join(checkpoint_dir, name)
      for name in os.listdir(checkpoint_dir)
      if name.endswith('.safetensors')
  )
  if not shards:
    raise FileNotFoundError(
        f'No .safetensors files found in checkpoint directory {checkpoint_dir}'
    )
  return shards


def load_safetensors_subset(
    checkpoint_dir: str,
    prefixes: Sequence[str] | None = None,
    *,
    strip_prefix: bool = False,
    dtype: torch.dtype | None = torch.float32,
) -> dict[str, torch.Tensor]:
  """Loads matching keys from safetensors shards in `checkpoint_dir`."""
  state_dict: dict[str, torch.Tensor] = {}
  for shard_path in _list_safetensors_shards(checkpoint_dir):
    with safetensors.safe_open(shard_path, framework='pt', device='cpu') as f:
      for key in f.keys():
        matched_prefix = None
        if prefixes is None:
          matched_prefix = ''
        else:
          for prefix in prefixes:
            if key == prefix or key.startswith(prefix):
              matched_prefix = prefix
              break
        if matched_prefix is None:
          continue
        target_key = key[len(matched_prefix) :] if strip_prefix else key
        tensor = f.get_tensor(key)
        if dtype is not None and tensor.is_floating_point():
          tensor = tensor.to(dtype)
        state_dict[target_key] = tensor
  return state_dict


def _maybe_quantize(out_path: str, quant_recipe: str | None) -> str:
  """Applies comma-separated quantization recipes to `out_path` if specified."""
  quant_list = quant_recipe.split(',') if quant_recipe else [None]
  final_path = out_path
  for recipe in quant_list:
    prev_path = final_path
    final_path = quantization.maybe_quantize_model(final_path, recipe)
    if final_path != prev_path and os.path.exists(prev_path):
      os.remove(prev_path)
    gc.collect()
  return final_path


_BasePrefillModule = (
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder
)
_BaseGenerateModule = (
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder
)


class _Lfm2LlmGenerateModule(_BaseGenerateModule):
  """Decode module that concatenates `[logits, last_hidden_state]` in `logits`."""

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
    orig_verifier = self.export_verifier
    self.export_verifier = True
    try:
      result = super().forward(
          embeddings, input_pos, kv_cache, mask, local_mask=local_mask, **kwargs
      )
    finally:
      self.export_verifier = orig_verifier
    last_hidden = result['activations'].to(torch.float32)
    result['logits'] = torch.cat([result['logits'], last_hidden], dim=-1)
    if not orig_verifier:
      result.pop('activations', None)
    return result


class Lfm2Audio(nn.Module):
  """Top-level LFM2.5-Audio-1.5B Any2Any model wrapper for `export_hf`."""

  def __init__(
      self,
      model_path: str,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ):
    super().__init__()
    self.model_path = model_path
    self.export_config = export_config
    self.model_dir = (
        model_path
        if os.path.exists(model_path)
        else huggingface_hub.snapshot_download(model_path)
    )
    config_path = os.path.join(self.model_dir, 'config.json')
    if os.path.exists(config_path):
      with open(config_path, 'r') as f:
        self.raw_config = json.load(f)
    else:
      self.raw_config = {}

    self.detok_dir = os.path.join(self.model_dir, 'audio_detokenizer')
    detok_cfg_path = os.path.join(self.detok_dir, 'config.json')
    if os.path.exists(detok_cfg_path):
      with open(detok_cfg_path, 'r') as f:
        self.detok_config = json.load(f)
    else:
      self.detok_config = self.raw_config.get('audio_detokenizer', {})

    self._parse_configs()

  def _parse_configs(self) -> None:
    """Parses LFM2, FastConformer, Depthformer, and Detokenizer configs."""
    lfm_dict = dict(
        self.raw_config.get('lfm')
        or self.raw_config.get('text_config')
        or self.raw_config
    )
    lfm_dict.setdefault('model_type', 'lfm2')
    lfm_dict.setdefault('vocab_size', 65536)
    lfm_dict.setdefault('hidden_size', 2048)
    lfm_dict.setdefault('intermediate_size', 12288)
    lfm_dict.setdefault('num_hidden_layers', 16)
    lfm_dict.setdefault('num_attention_heads', 32)
    lfm_dict.setdefault('num_key_value_heads', 8)
    lfm_dict.setdefault('conv_L_cache', 3)
    lfm_dict.setdefault('conv_bias', False)
    lfm_dict.setdefault('block_auto_adjust_ff_dim', True)
    lfm_dict.setdefault('block_ffn_dim_multiplier', 1.0)
    lfm_dict.setdefault('block_multiple_of', 256)
    lfm_dict.setdefault('norm_eps', 1e-5)
    lfm_dict.setdefault('rope_theta', 1000000.0)
    lfm_dict.setdefault('tie_word_embeddings', True)
    self.lfm_rope_theta = float(
        lfm_dict.get('rope_theta', lfm_dict.get('theta', 1000000.0))
    )
    num_layers = int(lfm_dict['num_hidden_layers'])
    if 'layer_types' not in lfm_dict and 'full_attn_idxs' not in lfm_dict:
      if num_layers == len(_DEFAULT_LFM2_LAYER_TYPES):
        lfm_dict['layer_types'] = list(_DEFAULT_LFM2_LAYER_TYPES)
      else:
        lfm_dict['layer_types'] = [
            'full_attention' if i % 2 == 1 else 'conv'
            for i in range(num_layers)
        ]

    self.lfm_config = transformers.Lfm2Config(**lfm_dict)
    self.lfm_config._attn_implementation = 'lrt_transposed_attention'  # pylint: disable=protected-access
    self.lfm_config.dtype = torch.float32

    encoder_dict = dict(
        self.raw_config.get('encoder')
        or self.raw_config.get('audio_config')
        or {}
    )
    self.encoder_config = modules.ConformerEncoderConfig.from_dict(encoder_dict)

    depthformer_dict = dict(self.raw_config.get('depthformer') or {})
    self.depthformer_layers = int(depthformer_dict.get('layers', 6))
    self.depthformer_dim = int(depthformer_dict.get('dim', 1024))
    self.depthformer_ff_dim = int(
        depthformer_dict.get('ff_dim', 4 * self.depthformer_dim)
    )
    self.depthformer_num_heads = int(depthformer_dict.get('num_heads', 32))
    self.depthformer_gqa_dim = int(depthformer_dict.get('gqa_dim', 8))
    self.depthformer_tie = bool(depthformer_dict.get('tie', True))

    self.codebooks = int(self.raw_config.get('codebooks', 8))
    self.audio_vocab_size = int(self.raw_config.get('audio_vocab_size', 2049))
    self.tie_audio_embeddings = bool(
        self.raw_config.get('tie_audio_embeddings', False)
    )

    detok_dict = dict(self.detok_config)
    if 'lfm' in detok_dict and isinstance(detok_dict['lfm'], dict):
      detok_lfm = dict(detok_dict['lfm'])
      detok_audio_vocab = int(
          detok_dict.get(
              'audio_vocab_size',
              detok_dict.get('vocab_size', self.audio_vocab_size - 1),
          )
      )
    else:
      # Real `audio_detokenizer/config.json` (`Lfm25AudioTokenizer`) stores LFM2
      # fields at the top level (`vocab_size: 65536` is `lfm.embed_tokens`,
      # while `FusedEmbedding` uses `self.audio_vocab_size - 1 = 2048`).
      detok_lfm = dict(detok_dict)
      detok_audio_vocab = int(
          detok_dict.get('audio_vocab_size', self.audio_vocab_size - 1)
      )
    detok_lfm.setdefault('vocab_size', int(self.lfm_config.vocab_size))
    detok_lfm.setdefault('hidden_size', 512)
    detok_lfm.setdefault('intermediate_size', 3328)
    detok_lfm.setdefault('num_hidden_layers', 8)
    detok_lfm.setdefault('num_attention_heads', 16)
    detok_lfm.setdefault('num_key_value_heads', 8)
    detok_lfm.setdefault('conv_L_cache', 3)
    detok_lfm.setdefault('conv_bias', False)
    detok_lfm.setdefault('block_auto_adjust_ff_dim', True)
    detok_lfm.setdefault('block_ffn_dim_multiplier', 1.0)
    detok_lfm.setdefault('block_multiple_of', 256)
    detok_lfm.setdefault('norm_eps', 1e-5)
    detok_lfm.setdefault('sliding_window', 30)
    detok_num_layers = int(detok_lfm['num_hidden_layers'])
    if 'layer_types' not in detok_lfm:
      default_detok_layers = [
          'conv',
          'conv',
          'sliding_attention',
          'conv',
          'sliding_attention',
          'conv',
          'sliding_attention',
          'conv',
      ]
      if detok_num_layers <= len(default_detok_layers):
        detok_lfm['layer_types'] = default_detok_layers[:detok_num_layers]
      else:
        detok_lfm['layer_types'] = [
            'sliding_attention' if i % 2 == 1 else 'conv'
            for i in range(detok_num_layers)
        ]
    detok_dict['lfm'] = detok_lfm
    detok_dict['audio_vocab_size'] = detok_audio_vocab
    detok_dict.setdefault('codebooks', self.codebooks)
    detok_dict.setdefault('vocab_size', detok_audio_vocab)
    detok_dict.setdefault('upsample_factor', 6)
    if 'output_size' in detok_dict and 'n_fft' not in detok_dict:
      detok_dict['n_fft'] = int(detok_dict['output_size']) - 2
    detok_dict.setdefault('n_fft', 1280)
    detok_dict.setdefault('hop_length', 320)
    self.detok_config = detok_dict

  def build_audio_encoder_module(
      self, use_random_weights: bool = False
  ) -> modules.Lfm2AudioEncoderModule:
    """Instantiates `Lfm2AudioEncoderModule` and loads weights (`strict=True`)."""
    adapter_hidden = int(
        self.raw_config.get(
            'audio_adapter_hidden_size', self.lfm_config.hidden_size
        )
    )
    module = modules.Lfm2AudioEncoderModule(
        encoder_cfg=self.encoder_config,
        llm_hidden_size=int(self.lfm_config.hidden_size),
        adapter_hidden_dims=(adapter_hidden,),
        adapter_use_layer_norm=True,
    )
    if not use_random_weights:
      sd = load_safetensors_subset(
          self.model_dir,
          prefixes=('conformer.', 'audio_adapter.'),
          strip_prefix=False,
          dtype=torch.float32,
      )
      module.load_state_dict(sd, strict=True)
    return module.eval()

  def build_text_embedder_module(
      self, use_random_weights: bool = False
  ) -> modules.Lfm2AudioTextEmbedderModule:
    """Instantiates `Lfm2AudioTextEmbedderModule` and loads `lfm.embed_tokens.weight`."""
    module = modules.Lfm2AudioTextEmbedderModule(
        vocab_size=int(self.lfm_config.vocab_size),
        hidden_size=int(self.lfm_config.hidden_size),
    )
    if not use_random_weights:
      sd = load_safetensors_subset(
          self.model_dir,
          prefixes=('lfm.embed_tokens.',),
          strip_prefix=True,
          dtype=torch.float32,
      )
      module.embed_tokens.load_state_dict(sd, strict=True)
    return module.eval()

  def build_tts_projector_module(
      self, use_random_weights: bool = False
  ) -> modules.Lfm2AudioTtsProjectorModule:
    """Instantiates `Lfm2AudioTtsProjectorModule` and loads weights (`strict=True`)."""
    module = modules.Lfm2AudioTtsProjectorModule(
        llm_hidden_size=int(self.lfm_config.hidden_size),
        depthformer_dim=self.depthformer_dim,
        codebooks=self.codebooks,
        audio_vocab_size=self.audio_vocab_size,
        tie_audio_embeddings=self.tie_audio_embeddings,
    )
    if not use_random_weights:
      sd = load_safetensors_subset(
          self.model_dir,
          prefixes=(
              'depth_linear.',
              'audio_embedding.',
              'codebook_offsets',
              'audio_loss_weights',
          ),
          strip_prefix=False,
          dtype=torch.float32,
      )
      if (
          'audio_embedding.to_logits.weight' not in sd
          and 'audio_embedding.embedding.weight' in sd
      ):
        sd['audio_embedding.to_logits.weight'] = sd[
            'audio_embedding.embedding.weight'
        ]
      module.load_state_dict(sd, strict=True)
    return module.eval()

  def build_depthformer_module(
      self, use_random_weights: bool = False
  ) -> modules.Lfm2DepthformerModule:
    """Instantiates `Lfm2DepthformerModule` and loads weights (`strict=True`)."""
    module = modules.Lfm2DepthformerModule(
        layers=self.depthformer_layers,
        dim=self.depthformer_dim,
        ff_dim=self.depthformer_ff_dim,
        num_heads=self.depthformer_num_heads,
        gqa_dim=self.depthformer_gqa_dim,
        qk_layernorm=True,
        norm_eps=float(self.lfm_config.norm_eps),
        multiple_of=int(self.lfm_config.block_multiple_of),
        theta=self.lfm_rope_theta,
        codebooks=self.codebooks,
        vocab_size=self.audio_vocab_size,
        tie_depth_embeddings=self.depthformer_tie,
    )
    if not use_random_weights:
      sd = load_safetensors_subset(
          self.model_dir,
          prefixes=('depthformer.', 'depth_embeddings.'),
          strip_prefix=False,
          dtype=torch.float32,
      )
      for k in range(self.codebooks):
        emb_key = f'depth_embeddings.{k}.embedding.weight'
        logits_key = f'depth_embeddings.{k}.to_logits.weight'
        if logits_key not in sd and emb_key in sd:
          sd[logits_key] = sd[emb_key]
      module.load_state_dict(sd, strict=True)
    return module.eval()

  def build_detokenizer_module(
      self, use_random_weights: bool = False
  ) -> modules.Lfm2AudioDetokenizerModule:
    """Instantiates `Lfm2AudioDetokenizerModule` and loads weights (`strict=True`)."""
    module = modules.Lfm2AudioDetokenizerModule(self.detok_config)
    if not use_random_weights:
      if os.path.isdir(self.detok_dir) and any(
          name.endswith('.safetensors') for name in os.listdir(self.detok_dir)
      ):
        sd = load_safetensors_subset(
            self.detok_dir,
            prefixes=None,
            strip_prefix=False,
            dtype=torch.float32,
        )
      else:
        sd = load_safetensors_subset(
            self.model_dir,
            prefixes=('audio_detokenizer.',),
            strip_prefix=True,
            dtype=torch.float32,
        )
      module.load_state_dict(sd, strict=True)
    return module.eval()

  def verify_checkpoint_keys(self) -> None:
    """Verifies that every key in `model.safetensors` belongs to a submodel."""
    allowed_prefixes = (
        'conformer.',
        'audio_adapter.',
        'lfm.',
        'depth_linear.',
        'audio_embedding.',
        'codebook_offsets',
        'audio_loss_weights',
        'depthformer.',
        'depth_embeddings.',
    )
    unconsumed: list[str] = []
    for shard_path in _list_safetensors_shards(self.model_dir):
      with safetensors.safe_open(shard_path, framework='pt', device='cpu') as f:
        for key in f.keys():
          if not any(key == p or key.startswith(p) for p in allowed_prefixes):
            unconsumed.append(key)
    if unconsumed:
      raise ValueError(
          f'Unconsumed checkpoint keys in {self.model_dir}:'
          f' {sorted(unconsumed)}'
      )

  def export_audio_encoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports FastConformer + MLP audio encoder (`tf_lite_audio_encoder`)."""
    mel_frames = int(export_config.extra_kwargs.get('audio_mel_frames', 2048))
    mod = self.build_audio_encoder_module(
        use_random_weights=export_config.use_random_weights
    )
    sample_kwargs = {
        'mel_features': torch.zeros(
            (1, self.encoder_config.feat_in, mel_frames), dtype=torch.float32
        ),
        'mel_length': torch.tensor([mel_frames], dtype=torch.int32),
    }
    out_path = os.path.join(output_dir, 'audio_encoder_fp32.tflite')
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    quant_recipe = (
        export_config.audio_encoder_quantization_recipe
        if export_config.audio_encoder_quantization_recipe is not None
        else export_config.quantization_recipe
    )
    return _maybe_quantize(out_path, quant_recipe)

  def export_text_embedder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports LFM2 text token embedder (`tf_lite_text_embedder`)."""
    mod = self.build_text_embedder_module(
        use_random_weights=export_config.use_random_weights
    )
    sample_kwargs = {'token_ids': torch.zeros((1, 1), dtype=torch.int32)}
    out_path = os.path.join(output_dir, 'text_embedder_fp32.tflite')
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return _maybe_quantize(out_path, export_config.quantization_recipe)

  def export_llm_backbone(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports LFM2-1.2B hybrid short-conv + GQA LLM backbone (`tf_lite_llm_backbone`)."""
    llm_export_cfg = dataclasses.replace(
        export_config,
        externalize_embedder=True,
        single_token_embedder=True,
        split_cache=False,
    )
    with model_ext_patches.get_patch_context('lfm2', llm_export_cfg):
      llm_backbone = (
          transformers.Lfm2ForCausalLM(copy.deepcopy(self.lfm_config))
          .eval()
          .to(torch.float32)
      )
      if not export_config.use_random_weights:
        sd = load_safetensors_subset(
            self.model_dir,
            prefixes=('lfm.',),
            strip_prefix=True,
            dtype=torch.float32,
        )
        lm_head_w = sd.pop('lm_head.weight', None)
        llm_backbone.model.load_state_dict(sd, strict=True)
        with torch.no_grad():
          if lm_head_w is not None:
            llm_backbone.lm_head.weight.copy_(lm_head_w)
          else:
            llm_backbone.lm_head.weight.copy_(sd['embed_tokens.weight'])

    cache_lengths = export_config.cache_lengths or [export_config.cache_length]
    prefill_lengths = export_config.prefill_lengths or [128]
    out_path = os.path.join(output_dir, 'llm_backbone_fp32.tflite')

    with model_ext_patches.patch_model(llm_backbone, 'lfm2', llm_export_cfg):
      llm_backbone.set_attn_implementation('lrt_transposed_attention')
      converter = converter_utils.Converter()
      for cache_len in cache_lengths:
        iter_config = dataclasses.replace(
            llm_export_cfg,
            cache_length=cache_len,
            prefill_lengths=prefill_lengths,
            externalize_embedder=True,
        )
        prefill_module = _BasePrefillModule(llm_backbone, iter_config).eval()
        decode_module = _Lfm2LlmGenerateModule(llm_backbone, iter_config).eval()
        for module in (prefill_module, decode_module):
          for sig_name, (inputs, _) in module.get_sample_inputs(
              self.lfm_config
          ).items():
            if len(cache_lengths) > 1:
              sig_name = f'{sig_name}_cache_{cache_len}'
            converter.add_signature(sig_name, module, sample_kwargs=inputs)
      lrt_model = converter.convert(
          lightweight_conversion=export_config.experimental_lightweight_conversion,
          strict_export=False,
      )
      lrt_model.export(out_path)
      del lrt_model

    del llm_backbone
    gc.collect()
    return _maybe_quantize(out_path, export_config.quantization_recipe)

  def export_tts_projector(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports `depth_linear` + `audio_embedding` (`tf_lite_tts_projector`)."""
    mod = self.build_tts_projector_module(
        use_random_weights=export_config.use_random_weights
    )
    sample_kwargs = {
        'llm_hidden': torch.zeros(
            (1, 1, int(self.lfm_config.hidden_size)), dtype=torch.float32
        ),
        'audio_codes': torch.zeros((1, self.codebooks), dtype=torch.int32),
    }
    out_path = os.path.join(output_dir, 'tts_projector_fp32.tflite')
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return _maybe_quantize(out_path, export_config.quantization_recipe)

  def export_tts_decoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports 8-codebook RQ-Transformer Depthformer (`tf_lite_tts_decoder`)."""
    mod = self.build_depthformer_module(
        use_random_weights=export_config.use_random_weights
    )
    sample_kwargs = {
        'cond_emb': torch.zeros((1, 1, mod.dim), dtype=torch.float32),
        'prev_audio_code': torch.zeros((1, 1), dtype=torch.int32),
        'step_idx': torch.zeros((1,), dtype=torch.int32),
        'kv_cache': torch.zeros(
            (1, 2 * mod.layers, mod.codebooks, mod.kv_dim), dtype=torch.float32
        ),
    }
    out_path = os.path.join(output_dir, 'tts_decoder_fp32.tflite')
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return _maybe_quantize(out_path, export_config.quantization_recipe)

  def export_audio_vocoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports LFM2 neural audio detokenizer (`tf_lite_audio_vocoder`)."""
    chunk_frames = int(export_config.extra_kwargs.get('detok_chunk_frames', 64))
    mod = self.build_detokenizer_module(
        use_random_weights=export_config.use_random_weights
    )
    sample_kwargs = {
        'audio_codes': torch.zeros(
            (1, self.codebooks, chunk_frames), dtype=torch.int32
        ),
    }
    out_path = os.path.join(output_dir, 'audio_vocoder_fp32.tflite')
    litert_torch.convert(mod, sample_kwargs=sample_kwargs).export(out_path)
    del mod
    gc.collect()
    return _maybe_quantize(out_path, export_config.quantization_recipe)

  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports all enabled LFM2.5-Audio submodels and returns `{name: path}`."""
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
      raise ValueError('Either output_dir or work_dir must be specified.')
    os.makedirs(output_dir, exist_ok=True)
    if not export_config.use_random_weights:
      self.verify_checkpoint_keys()
    extra = export_config.extra_kwargs
    artifacts: dict[str, str] = {}

    if extra.get('export_audio_encoder', True):
      artifacts['audio_encoder'] = self.export_audio_encoder(
          output_dir, export_config
      )
    if extra.get('export_llm_backbone', True):
      artifacts['text_embedder'] = self.export_text_embedder(
          output_dir, export_config
      )
      artifacts['llm_backbone'] = self.export_llm_backbone(
          output_dir, export_config
      )
    if extra.get('export_tts', True):
      artifacts['tts_projector'] = self.export_tts_projector(
          output_dir, export_config
      )
      artifacts['tts_decoder'] = self.export_tts_decoder(
          output_dir, export_config
      )
    if extra.get('export_vocoder', True):
      artifacts['audio_vocoder'] = self.export_audio_vocoder(
          output_dir, export_config
      )
    return artifacts

  def export_submodels(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Alias for `export`."""
    return self.export(export_config)

  def get_any2any_metadata(
      self,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> any2any_metadata_pb2.Any2AnyMetadata:
    """Builds `Any2AnyMetadata` proto for the exported LFM2.5-Audio-1.5B model."""
    del export_config
    lfm2_proto = any2any_model_type_pb2.Lfm2Audio(
        llm_hidden_size=int(self.lfm_config.hidden_size),
        llm_vocab_size=int(self.lfm_config.vocab_size),
        tts_hidden_size=self.depthformer_dim,
        tts_vocab_size=self.audio_vocab_size,
        tts_num_codebooks=self.codebooks,
        audio_mel_bins=self.encoder_config.feat_in,
        audio_subsampling_factor=self.encoder_config.subsampling_factor,
        generate_audio=True,
        max_new_tokens=512,
        max_tts_tokens=1024,
        voice='default',
        language='en-US',
    )
    return any2any_metadata_pb2.Any2AnyMetadata(
        input_sample_rate_hz=16000,
        output_sample_rate_hz=24000,
        supported_languages=['en-US'],
        any2any_model_type=any2any_model_type_pb2.Any2AnyModelType(
            lfm2_audio=lfm2_proto
        ),
    )
