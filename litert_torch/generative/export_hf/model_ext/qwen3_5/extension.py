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
"""Built-in `ModelExtension` for Qwen3.5 (`qwen3_5`, `qwen3_5_text`).

Illustrates custom prefill/decode exportables and Converter V2 checkpoint
weight streaming for a hybrid full-attention + linear-attention architecture:
- Split-cache NPU export (`export_config.split_cache=True`).
- Externalized embedder GPU/CPU export (`export_config.externalize_embedder`).
- Bundled embedder prefill/decode export (`qwen3_5/exportable_module.py`).
- Converter V2 `resolve_checkpoint_tensor` for `Qwen3_5RMSNorm` `(1.0 + weight)`
  rescaling and `odml.qkv_norm_rope` `[Q, K, V, Gate]` de-interleaving /
  partial-rotary channel permutation.
"""

from collections.abc import Callable
from typing import Any

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.qwen3_5 import exportable_module as qwen3_5_exportable
# Registers qwen3_5 patches.
from litert_torch.generative.export_hf.model_ext.qwen3_5 import patch as _
import torch

_WRAPPER_PREFIXES = (
    'model.',
    'module.',
    'static_model.',
    'original_hf_model.',
    'language_model.',
)
_RMSNORM_SUFFIXES = (
    '.input_layernorm.weight',
    '.post_attention_layernorm.weight',
    '.self_attn.q_norm.weight',
    '.self_attn.k_norm.weight',
)


def _strip_wrapper_prefixes(param_name: str) -> str:
  """Repeatedly strips standard export wrapper prefixes from `param_name`."""
  stripped = param_name
  while True:
    for prefix in _WRAPPER_PREFIXES:
      if stripped.startswith(prefix):
        stripped = stripped.removeprefix(prefix)
        break
    else:
      break
  return stripped


def _get_partial_rope_perm(model_config: Any) -> torch.Tensor | None:
  """Returns the Qwen3.5 partial-rotary channel permutation for a head."""
  if model_config is None:
    return None
  cfg = getattr(model_config, 'text_config', None) or model_config
  head_dim = getattr(cfg, 'head_dim', None) or (
      cfg.hidden_size // cfg.num_attention_heads
  )
  rope_params = getattr(cfg, 'rope_parameters', None)
  partial_factor = 0.25
  if isinstance(rope_params, dict):
    partial_factor = float(
        rope_params.get('partial_rotary_factor', partial_factor)
    )
  elif rope_params is not None:
    partial_factor = float(
        getattr(rope_params, 'partial_rotary_factor', partial_factor)
    )
  else:
    partial_factor = float(
        getattr(cfg, 'partial_rotary_factor', partial_factor)
    )
  rotary_dim = int(head_dim * partial_factor)
  if not (0 < rotary_dim < head_dim):
    return None
  half_rot = rotary_dim // 2
  half_dim = head_dim // 2
  first_pass_len = half_dim - half_rot
  pass_indices = torch.arange(rotary_dim, head_dim)
  perm = torch.empty(head_dim, dtype=torch.long)
  perm[:half_rot] = torch.arange(half_rot)
  perm[half_rot:half_dim] = pass_indices[:first_pass_len]
  perm[half_dim : half_dim + half_rot] = torch.arange(half_rot, rotary_dim)
  perm[half_dim + half_rot :] = pass_indices[first_pass_len:]
  return perm


def _resolve_qkv_proj_composite(
    param_name: str,
    try_load: Callable[[str], Any],
    model_config: Any,
) -> torch.Tensor | None:
  """Builds `[Q, K, V, Gate]` `qkv_proj` tensor for `odml.qkv_norm_rope`."""
  head, sep, tail = param_name.rpartition('qkv_proj.')
  if not sep or tail not in ('weight', 'bias') or model_config is None:
    return None
  q_t = try_load(f'{head}q_proj.{tail}')
  k_t = try_load(f'{head}k_proj.{tail}')
  v_t = try_load(f'{head}v_proj.{tail}')
  if q_t is None or k_t is None or v_t is None:
    return None

  cfg = getattr(model_config, 'text_config', None) or model_config
  num_heads = cfg.num_attention_heads
  num_kv_heads = getattr(cfg, 'num_key_value_heads', num_heads)
  head_dim = getattr(cfg, 'head_dim', None) or (cfg.hidden_size // num_heads)
  q_only_size = num_heads * head_dim
  k_size = num_kv_heads * head_dim
  perm = _get_partial_rope_perm(model_config)
  if tail == 'weight':
    q_raw_w = q_t.view(num_heads, 2, head_dim, -1)
    q_only_w = q_raw_w[:, 0, :, :].clone()
    gate_w = q_raw_w[:, 1, :, :].reshape(q_only_size, -1)
    k_w = k_t.view(num_kv_heads, head_dim, -1).clone()
    if perm is not None:
      q_only_w = q_only_w[:, perm, :]
      k_w = k_w[:, perm, :]
    return torch.cat(
        [
            q_only_w.reshape(q_only_size, -1),
            k_w.reshape(k_size, -1),
            v_t,
            gate_w,
        ],
        dim=0,
    )

  q_raw_b = q_t.view(num_heads, 2, head_dim)
  q_only_b = q_raw_b[:, 0, :].clone()
  gate_b = q_raw_b[:, 1, :].reshape(q_only_size)
  k_b = k_t.view(num_kv_heads, head_dim).clone()
  if perm is not None:
    q_only_b = q_only_b[:, perm]
    k_b = k_b[:, perm]
  return torch.cat(
      [
          q_only_b.reshape(q_only_size),
          k_b.reshape(k_size),
          v_t,
          gate_b,
      ],
      dim=0,
  )


@extension_lib.register_model_extension(override=False)
class Qwen3_5Extension(extension_lib.ModelExtension):  # pylint: disable=invalid-name
  """Built-in extension for Qwen3.5 hybrid linear-attention LLMs."""

  model_types = ('qwen3_5', 'qwen3_5_text', 'qwen3_5_moe', 'qwen3_5_moe_text')
  fused_weight_specs = (
      extension_lib.FusedWeightSpec(
          'in_proj_zba', ('in_proj_z', 'in_proj_b', 'in_proj_a'), dim=0
      ),
  )

  def get_export_defaults(self, config_dict, task=None, target='cpu_gpu'):
    """See base class."""
    # int4 exports can fail to load on the CPU backend (all-zero
    # int4 block scales), so Qwen3.5 defaults to int8 weights.
    del config_dict, task, target
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.TEXT_GENERATION,
        export_kwargs={
            'quantization_recipe': 'dynamic_wi8_afp32',
            'use_jinja_template': True,
            'litert_lm_llm_metadata_override': 'qwen3_5',
        },
    )

  def get_prefill_decode_exportables(self, model_config, export_config):
    """See base class."""
    model_types = {
        getattr(model_config, 'model_type', None),
        getattr(getattr(model_config, 'text_config', None), 'model_type', None),
    }
    if model_types & {'qwen3_5_moe', 'qwen3_5_moe_text'}:
      return None
    print('Using Qwen3.5 exportables.')
    if export_config.split_cache:
      return (
          qwen3_5_exportable.LiteRTSplitCacheExportableModuleForQwen3_5Prefill,
          qwen3_5_exportable.LiteRTSplitCacheExportableModuleForQwen3_5Generate,
      )
    if export_config.externalize_embedder:
      return (
          qwen3_5_exportable.LiteRTExportableModuleForQwen3_5PrefillExternalEmbedder,
          qwen3_5_exportable.LiteRTExportableModuleForQwen3_5GenerateExternalEmbedder,
      )
    return (
        qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Prefill,
        qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Generate,
    )

  def resolve_checkpoint_tensor(
      self,
      param_name: str,
      try_load: Callable[[str], Any],
      model_config: Any = None,
      export_config: Any = None,
  ) -> torch.Tensor | None:
    """Streams Qwen3.5 RMSNorm `(1 + w)` and `qkv_norm_rope` tensors."""
    use_qkv_norm_rope = bool(
        getattr(export_config, 'use_qkv_norm_rope_composite', False)
    )
    stripped = _strip_wrapper_prefixes(param_name)
    if stripped == 'norm.weight' or stripped.endswith(_RMSNORM_SUFFIXES):
      raw = try_load(param_name)
      if raw is None:
        return None
      tensor = raw.to(torch.float32) + 1.0
      if use_qkv_norm_rope and stripped.endswith(
          ('.self_attn.q_norm.weight', '.self_attn.k_norm.weight')
      ):
        perm = _get_partial_rope_perm(model_config)
        if perm is not None:
          tensor = tensor[perm]
      return tensor

    if use_qkv_norm_rope and stripped.endswith(
        ('.self_attn.qkv_proj.weight', '.self_attn.qkv_proj.bias')
    ):
      return _resolve_qkv_proj_composite(param_name, try_load, model_config)
    return None
