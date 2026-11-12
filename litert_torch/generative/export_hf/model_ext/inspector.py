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
r"""Architecture inspector and dry-run diagnostic CLI for `export_hf`.

Given a checkpoint directory or Hugging Face repo ID (`--model=...`), inspects:
1. `config.json`: `model_type`, `architectures`, `auto_map`, composite
   sub-configs (`text_config`, `vision_config`, `audio_config`),
   `sliding_window`, `layer_types`, `rope_parameters` / `rope_scaling`, and
   `tie_word_embeddings`.
2. `.safetensors` checkpoint keys: leading key prefixes (and whether they match
   the built-in `HFCheckpointWeightsLoader` prefixes or require
   `checkpoint_key_prefixes`), plus projection naming conventions.
3. Tiny 2-layer model structure (`from_config` on CPU with
   `num_hidden_layers=2`):
   - Normalization classes (`RMSNorm`) and whether they already emit the
     `odml.rms_norm` HLFB composite or require `patch_model`.
   - MLP projection attributes (`gate_proj`/`up_proj`/`down_proj` vs
     `w1`/`w3`/`w2`).
   - Attention modules and whether they dispatch through
     `ALL_ATTENTION_FUNCTIONS` / `_attn_implementation`.
   - Rotary embedding modules (`rotary_emb`, `rotary_emb_full`,
     `rotary_emb_sliding`) and whether their `forward` signature accepts the
     3-argument `(x, position_ids, layer_type)` call used by `split_cache` /
     `externalize_rope`.
4. Dry-run `torch.export.export` checks for standard prefill/decode and
   (optionally) `split_cache` + `RoPEEmbedder`, emitting actionable findings and
   a ready-to-use `ModelExtension` Python skeleton.

Example CLI usage:
  bazel run //third_party/py/litert_torch/generative/export_hf/model_ext:inspector -- \
    --model=/path/to/checkpoint \
    --trust_remote_code \
    --split_cache
"""

from collections.abc import Sequence
import contextlib
import copy
import dataclasses
import glob
import inspect
import json
import os
import re
import traceback
from typing import Any

import fire
from litert_torch.generative.export_hf.core import attention as _
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import patches as core_patches
from litert_torch.generative.export_hf.core import weights_loader as weights_loader_lib
from litert_torch.generative.export_hf.core.split_cache import attention as _
from litert_torch.generative.export_hf.model_ext import builtin_extensions as _
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
import safetensors
import torch
import transformers


@dataclasses.dataclass
class DryRunCheckResult:
  """Result of a tiny-model `torch.export.export` dry-run check."""

  mode: str
  passed: bool
  signatures: list[str] = dataclasses.field(default_factory=list)
  error_type: str | None = None
  error_message: str | None = None


@dataclasses.dataclass
class ArchitectureInspectionReport:
  """Structured diagnostic report for a checkpoint or Hugging Face model."""

  model_path: str
  model_type: str | None
  architectures: list[str]
  has_auto_map: bool
  registered_extension_cls: str | None
  sub_config_attrs: list[str]
  text_config_attr: str | None
  has_sliding_window: bool
  sliding_window: int | None
  layer_types: list[str] | None
  has_dual_rope_parameters: bool
  tie_word_embeddings: bool
  safetensors_prefixes: list[str]
  unrecognized_safetensors_prefixes: list[str]
  rmsnorm_classes: list[str]
  unpatched_rmsnorm_classes: list[str]
  mlp_classes: list[str]
  mlp_proj_names: list[str]
  attention_classes: list[str]
  attention_uses_interface: bool
  rotary_attributes: list[str]
  rotary_accepts_layer_type: bool
  dry_run_results: list[DryRunCheckResult]
  recommendations: list[str]
  skeleton_code: str
  conv_classes: list[str] = dataclasses.field(default_factory=list)
  unpatched_conv_classes: list[str] = dataclasses.field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary representation of the report."""
    return dataclasses.asdict(self)

  def to_markdown(self) -> str:
    """Formats the diagnostic report and starter extension skeleton as Markdown."""
    lines = [
        f'# `export_hf` Architecture Inspection: `{self.model_path}`',
        '',
        '## 1. Config & Extension Summary',
        f'- **`model_type`**: `{self.model_type}`',
        f'- **`architectures`**: `{self.architectures}`',
        f'- **`auto_map` (`trust_remote_code`)**: `{self.has_auto_map}`',
        (
            '- **Active `ModelExtension`**:'
            f' `{self.registered_extension_cls or "None (using generic defaults)"}`'
        ),
        f'- **Composite sub-configs**: `{self.sub_config_attrs}`',
        (
            f'- **Sliding window**: `{self.sliding_window}`'
            f' (`has_sliding_window={self.has_sliding_window}`)'
        ),
        (
            '- **Dual RoPE (`sliding_attention` in `rope_parameters`)**:'
            f' `{self.has_dual_rope_parameters}`'
        ),
        '',
        '## 2. Submodule & Checkpoint Structure',
        (
            f'- **RMSNorm classes**: `{self.rmsnorm_classes}`'
            f' (unpatched: `{self.unpatched_rmsnorm_classes}`)'
        ),
        (
            f'- **MLP classes**: `{self.mlp_classes}`'
            f' (projections: `{self.mlp_proj_names}`)'
        ),
        (
            f'- **Attention classes**: `{self.attention_classes}` (uses'
            f' `ALL_ATTENTION_FUNCTIONS`: `{self.attention_uses_interface}`)'
        ),
        (
            f'- **ShortConv / Conv classes**: `{self.conv_classes}`'
            f' (unpatched: `{self.unpatched_conv_classes}`)'
        ),
        (
            f'- **Rotary embedding attributes**: `{self.rotary_attributes}`'
            f' (accepts 3-arg `layer_type`: `{self.rotary_accepts_layer_type}`)'
        ),
        (
            '- **Safetensors top-level prefixes**:'
            f' `{self.safetensors_prefixes}` (unrecognized:'
            f' `{self.unrecognized_safetensors_prefixes}`)'
        ),
        '',
        '## 3. Tiny-Model `torch.export` Dry-Run Checks',
    ]
    if not self.dry_run_results:
      lines.append('- *(Dry-run checks skipped)*')
    for res in self.dry_run_results:
      status = 'PASS' if res.passed else 'FAIL'
      if res.passed:
        lines.append(
            f'- **`{res.mode}`**: **{status}** (signatures: `{res.signatures}`)'
        )
      else:
        lines.append(
            f'- **`{res.mode}`**: **{status}**'
            f' (`{res.error_type}: {res.error_message}`)'
        )

    lines.extend([
        '',
        '## 4. Actionable Recommendations',
    ])
    if not self.recommendations:
      lines.append(
          '- No blocking issues detected! Model exports cleanly with current'
          ' configuration.'
      )
    else:
      for rec in self.recommendations:
        lines.append(f'- {rec}')

    lines.extend([
        '',
        '## 5. Starter `ModelExtension` Skeleton',
        '```python',
        self.skeleton_code.rstrip(),
        '```',
    ])
    return '\n'.join(lines)


def _forward_takes_only_hidden_states(cls: type[torch.nn.Module]) -> bool:
  """Returns whether `cls.forward` takes `hidden_states` and nothing else.

  The skeleton's `_rms_norm_with_hlfb_forward(self, hidden_states)` can only
  replace such a `.forward`; a gated RMSNorm's `.forward(hidden_states, gate)`
  would raise a `TypeError` once patched.
  """
  try:
    params = list(inspect.signature(cls.forward).parameters.values())[1:]
  except (ValueError, TypeError):
    return True
  return len(params) == 1 and params[0].kind in (
      inspect.Parameter.POSITIONAL_ONLY,
      inspect.Parameter.POSITIONAL_OR_KEYWORD,
  )


def _inspect_safetensors_keys(model_path: str) -> tuple[list[str], list[str]]:
  """Returns `(top_prefixes, unrecognized_prefixes)` from a local checkpoint."""
  if not os.path.isdir(model_path):
    return [], []
  keys: list[str] = []
  index_file = os.path.join(model_path, 'model.safetensors.index.json')
  if os.path.exists(index_file):
    try:
      with open(index_file, 'r') as f:
        data = json.load(f)
      keys = list(data.get('weight_map', {}).keys())
    except Exception:  # pylint: disable=broad-exception-caught
      keys = []
  if not keys:
    st_files = sorted(glob.glob(os.path.join(model_path, '*.safetensors')))
    for st_file in st_files[:2]:
      try:
        with safetensors.safe_open(st_file, framework='pt', device='cpu') as f:
          keys.extend(f.keys())
      except Exception:  # pylint: disable=broad-exception-caught
        continue

  if not keys:
    return [], []

  prefixes: set[str] = set()
  unrecognized: set[str] = set()
  known_roots = {
      p.rstrip('.') for p in weights_loader_lib._WRAPPER_PREFIXES  # pylint: disable=protected-access
  } | {'lm_head', 'embed_tokens', 'layers', 'norm', 'visual', 'vision_tower'}

  for k in keys:
    parts = k.split('.')
    if len(parts) >= 2:
      root = f'{parts[0]}.'
      prefixes.add(root)
      if parts[0] not in known_roots:
        unrecognized.add(root)
  return sorted(prefixes), sorted(unrecognized)


def _shrink_config_for_tiny_model(
    config: transformers.PretrainedConfig,
) -> transformers.PretrainedConfig:
  """Returns a deep copy of `config` shrunk to 2+ layers for fast CPU tracing."""
  tiny_cfg = copy.deepcopy(config)

  def _clamp_token_id(val: Any, vocab_size: int, fallback: int) -> Any:
    if isinstance(val, int):
      return fallback if (val < 0 or val >= vocab_size) else val
    if isinstance(val, list):
      return [
          fallback if (isinstance(v, int) and (v < 0 or v >= vocab_size)) else v
          for v in val
      ]
    return val

  def _shrink_single(cfg: Any):
    orig_num_layers = int(getattr(cfg, 'num_hidden_layers', 2) or 2)
    target_layers = min(orig_num_layers, 2)
    if hasattr(cfg, 'layer_types') and isinstance(cfg.layer_types, list):
      orig_types = list(cfg.layer_types)
      # Preserve at least one instance of every distinct layer type in order of
      # first appearance (e.g. ['conv', 'sliding_attention', 'full_attention']
      # for tri-hybrid models like LFM3 where layers 0-1 are ['conv', 'conv']).
      unique_types = list(dict.fromkeys(orig_types))
      target_layers = max(target_layers, len(unique_types))
      target_layers = min(
          target_layers, max(orig_num_layers, len(unique_types))
      )
      if len(orig_types) > target_layers:
        combined: list[str] = list(unique_types)
        for lt in orig_types:
          if len(combined) >= target_layers:
            break
          combined.append(lt)
        cfg.layer_types = combined[:target_layers]
    if hasattr(cfg, 'num_hidden_layers'):
      cfg.num_hidden_layers = target_layers
    if hasattr(cfg, 'vocab_size') and isinstance(cfg.vocab_size, int):
      vocab_size = int(cfg.vocab_size)
      for attr, fallback in (
          ('pad_token_id', 0),
          ('bos_token_id', min(1, max(vocab_size - 1, 0))),
          ('eos_token_id', min(2, max(vocab_size - 1, 0))),
      ):
        if hasattr(cfg, attr):
          setattr(
              cfg,
              attr,
              _clamp_token_id(getattr(cfg, attr), vocab_size, fallback),
          )
    if hasattr(cfg, 'dtype'):
      cfg.dtype = torch.float32
    if hasattr(cfg, 'torch_dtype'):
      cfg.torch_dtype = torch.float32

  _shrink_single(tiny_cfg)
  for attr in ('text_config', 'language_config', 'llm_config', 'vision_config'):
    sub = getattr(tiny_cfg, attr, None)
    if sub is not None:
      _shrink_single(sub)
  return tiny_cfg


def _generate_extension_skeleton(
    model_type: str,
    unrecognized_prefixes: Sequence[str],
    unpatched_rmsnorm_classes: Sequence[str],
    has_dual_rope_attrs: bool,
    rotary_accepts_layer_type: bool,
    has_dual_rope_parameters: bool,
    has_auto_map: bool = False,
    architectures: Sequence[str] = (),
    mlp_proj_names: Sequence[str] = (),
) -> str:
  """Generates a tailored `ModelExtension` Python skeleton from findings."""
  safe_model_type = re.sub(
      r'[^0-9a-zA-Z_]', '_', model_type or 'custom_arch'
  ).lower()
  if safe_model_type and safe_model_type[0].isdigit():
    safe_model_type = f'_{safe_model_type}'
  cls_prefix = ''.join(
      part.capitalize()
      for part in re.split(r'[^0-9a-zA-Z]+', model_type or 'CustomArch')
      if part
  )
  if not cls_prefix or cls_prefix[0].isdigit():
    cls_prefix = f'Model{cls_prefix}'
  ext_cls_name = f'{cls_prefix}Extension'
  needs_rmsnorm_patch = bool(unpatched_rmsnorm_classes)
  needs_dual_rope_patch = (
      has_dual_rope_parameters or has_dual_rope_attrs
  ) and not rotary_accepts_layer_type
  needs_w1_w3_fusion = 'w1' in mlp_proj_names and 'w3' in mlp_proj_names
  arch_cls_name = (
      architectures[0] if architectures else f'{cls_prefix}ForCausalLM'
  )

  lines = [
      'import contextlib',
  ]
  if has_auto_map:
    lines.append('import sys')
  lines.append(
      'from litert_torch.generative.export_hf.model_ext import extension'
  )
  if needs_rmsnorm_patch:
    lines.append('from litert_torch.generative.layers import normalization')
  if needs_rmsnorm_patch or needs_dual_rope_patch:
    lines.append('import torch')
  lines.append('')
  lines.append('')

  if has_auto_map:
    lines.extend([
        f'def _find_{safe_model_type}_modeling_module():',
        '  for mod_name, mod in list(sys.modules.items()):',
        (
            f'    if mod_name.endswith("modeling_{safe_model_type}") and'
            f' hasattr(mod, {arch_cls_name!r}):'
        ),
        '      return mod',
        '  return None',
        '',
        '',
    ])

  if needs_rmsnorm_patch:
    lines.extend([
        'def _rms_norm_with_hlfb_forward(self, hidden_states: torch.Tensor):',
        '  dtype = hidden_states.dtype',
        '  hidden_states = hidden_states.to(torch.float32)',
        '  eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))',
        '  hidden_states = normalization.rms_norm_with_hlfb(',
        '      hidden_states,',
        '      self.weight,',
        '      eps,',
        '      torch.ones(',
        '          (self.weight.shape[0],),',
        '          dtype=torch.float32,',
        '          device=hidden_states.device,',
        '      ),',
        '  )',
        '  return hidden_states.to(dtype)',
        '',
        '',
    ])

  if needs_dual_rope_patch:
    lines.extend([
        'class _DualRotaryEmbeddingWrapper(torch.nn.Module):',
        '',
        '  def __init__(self, rotary_emb_full, rotary_emb_sliding=None):',
        '    super().__init__()',
        '    self.rotary_emb_full = rotary_emb_full',
        '    self.rotary_emb_sliding = rotary_emb_sliding',
        '',
        '  def forward(self, x, position_ids, attention_type=None):',
        '    if (',
        '        attention_type == "sliding_attention"',
        '        and self.rotary_emb_sliding is not None',
        '    ):',
        '      return self.rotary_emb_sliding(x, position_ids)',
        '    return self.rotary_emb_full(x, position_ids)',
        '',
        '',
    ])

  lines.extend([
      '@extension.register_model_extension()',
      f'class {ext_cls_name}(extension.ModelExtension):',
      f'  model_types = ({model_type!r},)',
      '  is_causal_text_model = True',
  ])
  if unrecognized_prefixes:
    prefixes_tuple = ', '.join(repr(p) for p in unrecognized_prefixes)
    lines.append(f'  checkpoint_key_prefixes = ({prefixes_tuple},)')
  if needs_w1_w3_fusion:
    lines.extend([
        '  fused_weight_specs = (',
        "      extension.FusedWeightSpec('gate_up_proj', ('w1', 'w3'), dim=0),",
        '  )',
    ])
  lines.extend([
      '',
      '  def get_export_defaults(',
      '      self, config_dict, task=None, target="cpu_gpu"',
      '  ):',
      '    del config_dict, task',
      '    is_npu = str(target or "").startswith("npu")',
      '    return extension.ExportDefaults(',
      '        task=extension.ExportTask.TEXT_GENERATION,',
      '        export_kwargs={',
      '            "quantization_recipe": (',
      '                "dynamic_wi8_afp32"',
      '                if is_npu',
      '                else "dynamic_wi4b32_afp32"',
      '            ),',
      '        },',
      '    )',
  ])
  if has_auto_map:
    lines.extend([
        '',
        '  def get_hf_model_cls(self, task, config, export_config):',
        '    del task, config, export_config',
        f'    mod = _find_{safe_model_type}_modeling_module()',
        (
            f'    return getattr(mod, {arch_cls_name!r}, None) if mod is not'
            ' None else None'
        ),
    ])

  if needs_rmsnorm_patch or needs_dual_rope_patch:
    lines.extend([
        '',
        '  @contextlib.contextmanager',
        '  def patch_model(self, model, export_config):',
        '    del export_config',
    ])
    if needs_rmsnorm_patch:
      target_names = ', '.join(repr(n) for n in unpatched_rmsnorm_classes)
      lines.extend([
          '    patched_cls = {}',
          '    for module in model.modules():',
          '      cls = type(module)',
          (
              f'      if cls.__name__ in ({target_names},) and cls not in'
              ' patched_cls:'
          ),
          '        patched_cls[cls] = cls.forward',
          '        cls.forward = _rms_norm_with_hlfb_forward',
      ])
    if needs_dual_rope_patch:
      lines.extend([
          '    inner = getattr(model, "model", model)',
          '    orig_full = getattr(inner, "rotary_emb_full", None)',
          '    orig_sliding = getattr(inner, "rotary_emb_sliding", None)',
          '    orig_rotary = getattr(inner, "rotary_emb", None)',
          '    if isinstance(orig_full, torch.nn.Module):',
          '      delattr(inner, "rotary_emb_full")',
          '      if hasattr(inner, "rotary_emb_sliding"):',
          '        delattr(inner, "rotary_emb_sliding")',
          (
              '      inner.rotary_emb = _DualRotaryEmbeddingWrapper('
              'orig_full, orig_sliding)'
          ),
          '      inner.__dict__["rotary_emb_full"] = (',
          '          lambda x, pos: inner.rotary_emb(x, pos, "full_attention")',
          '      )',
          '      inner.__dict__["rotary_emb_sliding"] = (',
          (
              '          (lambda x, pos: inner.rotary_emb('
              'x, pos, "sliding_attention"))'
          ),
          '          if orig_sliding is not None',
          '          else None',
          '      )',
      ])
    lines.extend([
        '    try:',
        '      yield',
        '    finally:',
    ])
    if needs_rmsnorm_patch:
      lines.extend([
          '      for cls, orig_fn in patched_cls.items():',
          '        cls.forward = orig_fn',
      ])
    if needs_dual_rope_patch:
      lines.extend([
          '      if isinstance(orig_full, torch.nn.Module):',
          '        inner.__dict__.pop("rotary_emb_full", None)',
          '        inner.__dict__.pop("rotary_emb_sliding", None)',
          '        inner.rotary_emb_full = orig_full',
          '        inner.rotary_emb_sliding = orig_sliding',
          '        if not hasattr(inner, "original_rotary_emb"):',
          '          inner.rotary_emb = orig_rotary',
      ])
    if not (needs_rmsnorm_patch or needs_dual_rope_patch):
      lines.append('      pass')

  return '\n'.join(lines) + '\n'


def inspect_model_architecture(
    model: str,
    trust_remote_code: bool = False,
    extension_modules: list[str] | tuple[str, ...] | str | None = None,
    model_extension: extension.ModelExtension | None = None,
    split_cache: bool = False,
    run_dry_run_export: bool = True,
) -> ArchitectureInspectionReport:
  """Inspects a checkpoint or model config and runs tiny-model dry-run checks.

  Args:
    model: Local checkpoint directory path or Hugging Face repo ID.
    trust_remote_code: Whether to enable `trust_remote_code` when loading
      configs and modeling classes.
    extension_modules: Optional module paths or `.py` file paths to load via
      `load_extension_modules` before inspection.
    model_extension: Optional explicit `ModelExtension` instance.
    split_cache: Whether to also run a `split_cache=True` (`externalize_rope`)
      dry-run `torch.export.export` check.
    run_dry_run_export: Whether to instantiate a 2-layer tiny model and run
      `torch.export.export` dry-run checks.

  Returns:
    An `ArchitectureInspectionReport` containing config/module findings,
    dry-run results, actionable recommendations, and a starter skeleton.
  """
  if extension_modules:
    extension.load_extension_modules(extension_modules)

  export_cfg = exportable_module_config.ExportableModuleConfig(
      model=model,
      trust_remote_code=trust_remote_code,
      extension_modules=extension_modules,
      model_extension=model_extension,
      externalize_embedder=True,
  )

  raw_cfg_dict: dict[str, Any] = {}
  config_json_path = (
      os.path.join(model, 'config.json') if os.path.isdir(model) else ''
  )
  if config_json_path and os.path.exists(config_json_path):
    with open(config_json_path, 'r') as f:
      raw_cfg_dict = json.load(f)
  else:
    with contextlib.suppress(Exception):
      raw_cfg_dict, _ = transformers.PretrainedConfig.get_config_dict(
          model, trust_remote_code=trust_remote_code
      )

  model_type = raw_cfg_dict.get('model_type')
  resolved_ext = extension.resolve_model_extension(model_type, export_cfg)
  active_ext = resolved_ext or extension.ModelExtension()

  def _make_patch_cm():
    if model_type or resolved_ext is not None:
      return model_ext_patches.get_patch_context(model_type or '', export_cfg)
    return contextlib.nullcontext()

  with _make_patch_cm():
    config = None
    if resolved_ext is not None:
      config = resolved_ext.load_config(
          model, export_cfg, trust_remote_code=trust_remote_code
      )
    if config is None:
      try:
        config = transformers.AutoConfig.from_pretrained(
            model,
            dtype=torch.float32,
            trust_remote_code=trust_remote_code,
        )
      except Exception:  # pylint: disable=broad-exception-caught
        config = transformers.PretrainedConfig.from_dict(raw_cfg_dict)

  model_type = getattr(config, 'model_type', model_type)
  resolved_ext = extension.resolve_model_extension(model_type, export_cfg)
  active_ext = resolved_ext or extension.ModelExtension()

  architectures = list(getattr(config, 'architectures', None) or [])
  has_auto_map = bool(
      getattr(config, 'auto_map', None) or raw_cfg_dict.get('auto_map')
  )
  sub_config_attrs = [
      attr
      for attr in (
          'text_config',
          'language_config',
          'llm_config',
          'vision_config',
          'audio_config',
      )
      if getattr(config, attr, None) is not None
  ]
  detected_text_config_attr = None
  for candidate in ('text_config', 'language_config', 'llm_config'):
    if getattr(config, candidate, None) is not None:
      detected_text_config_attr = candidate
      break

  text_cfg = (
      getattr(config, detected_text_config_attr, config)
      if detected_text_config_attr
      else config
  )
  sliding_window = getattr(text_cfg, 'sliding_window', None)
  has_sliding_window = bool(sliding_window)
  layer_types = getattr(text_cfg, 'layer_types', None)
  if layer_types is not None:
    layer_types = list(layer_types)
  rope_params = getattr(text_cfg, 'rope_parameters', None)
  has_dual_rope_parameters = isinstance(rope_params, dict) and (
      'sliding_attention' in rope_params
  )
  tie_word_embeddings = bool(getattr(text_cfg, 'tie_word_embeddings', False))

  st_prefixes, unrecognized_st_prefixes = _inspect_safetensors_keys(model)
  if resolved_ext is not None and resolved_ext.checkpoint_key_prefixes:
    ext_roots = {
        p.rstrip('.') + '.' for p in resolved_ext.checkpoint_key_prefixes
    }
    unrecognized_st_prefixes = [
        p for p in unrecognized_st_prefixes if p not in ext_roots
    ]

  rmsnorm_classes: set[str] = set()
  unpatched_rmsnorm_classes: set[str] = set()
  multi_input_rmsnorm_classes: set[str] = set()
  mlp_classes: set[str] = set()
  mlp_proj_names: set[str] = set()
  attention_classes: set[str] = set()
  attention_uses_interface = True
  conv_classes: set[str] = set()
  unpatched_conv_classes: set[str] = set()
  rotary_attributes: list[str] = []
  rotary_accepts_layer_type = True
  dry_run_results: list[DryRunCheckResult] = []
  recommendations: list[str] = []

  tiny_model = None
  tiny_config = _shrink_config_for_tiny_model(config)

  with _make_patch_cm():
    try:
      ext_model_cls = None
      if resolved_ext is not None:
        ext_model_cls = resolved_ext.get_hf_model_cls(
            exportable_module_config.ExportTask.TEXT_GENERATION,
            tiny_config,
            export_cfg,
        )
      auto_cls = ext_model_cls or transformers.AutoModelForCausalLM
      if hasattr(auto_cls, 'from_config'):
        tiny_model = auto_cls.from_config(
            tiny_config,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
        )
      elif hasattr(auto_cls, '_from_config'):
        tiny_model = auto_cls._from_config(  # pylint: disable=protected-access
            tiny_config, torch_dtype=torch.float32
        )
      else:
        tiny_model = auto_cls(tiny_config)
      tiny_model = tiny_model.eval()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      recommendations.append(
          'Could not instantiate tiny model via `AutoModelForCausalLM`:'
          f' `{type(exc).__name__}: {exc}`. Override `get_hf_model_cls`'
          ' on your `ModelExtension`.'
      )

  if tiny_model is not None:
    text_backbone = (
        tiny_model.get_text_model()
        if hasattr(tiny_model, 'get_text_model')
        else tiny_model
    )
    with _make_patch_cm(), model_ext_patches.patch_model(
        text_backbone, model_type or '', export_cfg
    ):
      inner_model = getattr(text_backbone, 'model', text_backbone)
      if hasattr(inner_model, 'language_model'):
        inner_model = inner_model.language_model

      for attr_name in ('rotary_emb', 'rotary_emb_full', 'rotary_emb_sliding'):
        if hasattr(inner_model, attr_name):
          rotary_attributes.append(attr_name)

      rotary_mod = getattr(inner_model, 'rotary_emb', None)
      if rotary_mod is not None and callable(rotary_mod):
        try:
          rotary_fn = getattr(rotary_mod, 'forward', rotary_mod)
          sig = inspect.signature(rotary_fn)
          params = [
              p
              for p in sig.parameters.values()
              if p.name != 'self'
              and p.kind in (
                  inspect.Parameter.POSITIONAL_ONLY,
                  inspect.Parameter.POSITIONAL_OR_KEYWORD,
              )
          ]
          has_varargs = any(
              p.kind == inspect.Parameter.VAR_POSITIONAL
              for p in sig.parameters.values()
          )
          rotary_accepts_layer_type = has_varargs or len(params) >= 3
        except (ValueError, TypeError):
          rotary_accepts_layer_type = True
      else:
        rotary_accepts_layer_type = False

      for _, submod in text_backbone.named_modules():
        cls = type(submod)
        cls_name = cls.__name__
        if 'RMSNorm' in cls_name or 'RmsNorm' in cls_name:
          rmsnorm_classes.add(cls_name)
          # Check if patched to core.patches.RMSNorm or uses rms_norm_with_hlfb.
          if issubclass(cls, core_patches.RMSNorm):
            continue
          try:
            src = inspect.getsource(cls.forward)
          except (OSError, TypeError):
            src = ''
          if 'rms_norm_with_hlfb' not in src:
            unpatched_rmsnorm_classes.add(cls_name)
            if not _forward_takes_only_hidden_states(cls):
              multi_input_rmsnorm_classes.add(cls_name)

        if cls_name.endswith('MLP') or cls_name.endswith('FeedForward'):
          mlp_classes.add(cls_name)
          for child_name, child in submod.named_children():
            if isinstance(child, torch.nn.Linear):
              mlp_proj_names.add(child_name)

        if cls_name.endswith('Attention') and not cls_name.startswith('LiteRT'):
          attention_classes.add(cls_name)
          try:
            attn_src = inspect.getsource(cls.forward)
            if (
                'ALL_ATTENTION_FUNCTIONS' not in attn_src
                and '_attn_implementation' not in attn_src
            ):
              attention_uses_interface = False
          except (OSError, TypeError):
            pass

        if (
            cls_name.endswith('Conv')
            and not issubclass(cls, torch.nn.modules.conv._ConvNd)  # pylint: disable=protected-access
            and not cls_name.startswith('LiteRT')
        ):
          conv_classes.add(cls_name)
          try:
            conv_src = inspect.getsource(cls.forward)
          except (OSError, TypeError):
            conv_src = ''
          if (
              'apply_short_conv_step' not in conv_src
              and 'short_conv_composite' not in conv_src
              and 'short_conv(' not in conv_src
          ):
            unpatched_conv_classes.add(cls_name)

    if run_dry_run_export:
      modes_to_check = [('default_prefill_decode', False)]
      if split_cache or has_dual_rope_parameters:
        modes_to_check.append(('split_cache_prefill_decode', True))

      for mode_name, is_split in modes_to_check:
        mode_cfg = dataclasses.replace(
            export_cfg,
            prefill_lengths=[8],
            cache_length=16,
            cache_lengths=[16],
            split_cache=is_split,
            cache_implementation=None,
            externalize_rope=is_split,
            externalize_embedder=True,
            sliding_window_ring_buffer_size=(
                32 if (is_split and has_sliding_window) else None
            ),
        )
        try:
          eps = ext_testing.assert_tiny_prefill_decode_torch_export(
              ext=active_ext,
              model=tiny_model,
              model_config=tiny_config,
              export_config=mode_cfg,
          )
          dry_run_results.append(
              DryRunCheckResult(
                  mode=mode_name,
                  passed=True,
                  signatures=sorted(eps.keys()),
              )
          )
        except Exception as exc:  # pylint: disable=broad-exception-caught
          tb_lines = traceback.format_exception(
              type(exc), exc, exc.__traceback__
          )
          short_tb = ''.join(tb_lines[-3:]).strip()
          dry_run_results.append(
              DryRunCheckResult(
                  mode=mode_name,
                  passed=False,
                  error_type=type(exc).__name__,
                  error_message=f'{exc}\n{short_tb}',
              )
          )

  if unrecognized_st_prefixes:
    recommendations.append(
        'Safetensors checkpoint contains non-standard leading key prefixes'
        f' `{unrecognized_st_prefixes}`: set `checkpoint_key_prefixes ='
        f' {tuple(unrecognized_st_prefixes)!r}` on your `ModelExtension` for'
        ' Converter V2 (`--use_v2`).'
    )

  if unpatched_rmsnorm_classes:
    recommendations.append(
        f'RMSNorm class(es) `{sorted(unpatched_rmsnorm_classes)}` do not use'
        ' `@use_kernel_forward_from_hub("RMSNorm")` or'
        ' `normalization.rms_norm_with_hlfb`: patch `.forward` in'
        ' `ModelExtension.patch_model` so NPU/GPU delegates receive the'
        ' `odml.rms_norm` composite.'
    )

  if multi_input_rmsnorm_classes:
    recommendations.append(
        'The skeleton does not patch'
        f' `{sorted(multi_input_rmsnorm_classes)}`: their `.forward` takes more'
        ' inputs than `hidden_states` (e.g. a `gate`), so patch them by hand.'
    )

  if has_dual_rope_parameters and not rotary_accepts_layer_type:
    recommendations.append(
        '`config.rope_parameters` defines `sliding_attention` (dual RoPE), but'
        f' `rotary_emb.forward` (attributes: `{rotary_attributes}`) only'
        ' accepts 2 positional arguments `(x, position_ids)`. Wrap'
        ' `rotary_emb` in `ModelExtension.patch_model` to accept a 3rd'
        ' `layer_type` argument (`"full_attention"` / `"sliding_attention"`)'
        ' for `split_cache` / `RoPEEmbedder`.'
    )

  if not attention_uses_interface and attention_classes:
    recommendations.append(
        f'Attention class(es) `{sorted(attention_classes)}` do not dispatch via'
        ' `ALL_ATTENTION_FUNCTIONS` / `_attn_implementation`: patch attention'
        ' in `patch_model` to call the registered LiteRT attention interface.'
    )

  if unpatched_conv_classes:
    recommendations.append(
        f'ShortConv class(es) `{sorted(unpatched_conv_classes)}` do not use'
        ' `short_conv.apply_short_conv_step` / `short_conv.short_conv`: patch'
        ' `.forward` in `patch_model` to read/write'
        ' `past_key_values.layers[layer_idx].conv_states`, pass'
        ' `self.in_proj(hidden_states)` to `short_conv`, and ensure'
        ' `causal_mask_mapping.setdefault("conv", None)` and `valid_mask`'
        ' (`**kwargs`) are forwarded in the model/decoder layers.'
    )

  if 'w1' in mlp_proj_names and 'w3' in mlp_proj_names:
    recommendations.append(
        'MLP uses `w1`/`w3` projections: when fusing into `gate_up_proj`,'
        ' declare `fused_weight_specs ='
        ' (extension.FusedWeightSpec("gate_up_proj", ("w1", "w3"), dim=0),)` on'
        ' your `ModelExtension` so Converter V2 (`--use_v2`) concatenates them'
        ' from `.safetensors`.'
    )

  for res in dry_run_results:
    if not res.passed:
      recommendations.append(
          f'Dry-run `{res.mode}` failed with `{res.error_type}:'
          f' {res.error_message}`.'
      )

  skeleton_code = _generate_extension_skeleton(
      model_type=model_type or 'custom_arch',
      unrecognized_prefixes=unrecognized_st_prefixes,
      unpatched_rmsnorm_classes=sorted(
          unpatched_rmsnorm_classes - multi_input_rmsnorm_classes
      ),
      has_dual_rope_attrs='rotary_emb_full' in rotary_attributes,
      rotary_accepts_layer_type=rotary_accepts_layer_type,
      has_dual_rope_parameters=has_dual_rope_parameters,
      has_auto_map=has_auto_map,
      architectures=architectures,
      mlp_proj_names=sorted(mlp_proj_names),
  )

  return ArchitectureInspectionReport(
      model_path=model,
      model_type=model_type,
      architectures=architectures,
      has_auto_map=has_auto_map,
      registered_extension_cls=(
          type(resolved_ext).__name__ if resolved_ext is not None else None
      ),
      sub_config_attrs=sub_config_attrs,
      text_config_attr=detected_text_config_attr,
      has_sliding_window=has_sliding_window,
      sliding_window=sliding_window,
      layer_types=layer_types,
      has_dual_rope_parameters=has_dual_rope_parameters,
      tie_word_embeddings=tie_word_embeddings,
      safetensors_prefixes=st_prefixes,
      unrecognized_safetensors_prefixes=unrecognized_st_prefixes,
      rmsnorm_classes=sorted(rmsnorm_classes),
      unpatched_rmsnorm_classes=sorted(unpatched_rmsnorm_classes),
      mlp_classes=sorted(mlp_classes),
      mlp_proj_names=sorted(mlp_proj_names),
      attention_classes=sorted(attention_classes),
      attention_uses_interface=attention_uses_interface,
      rotary_attributes=rotary_attributes,
      rotary_accepts_layer_type=rotary_accepts_layer_type,
      dry_run_results=dry_run_results,
      recommendations=recommendations,
      skeleton_code=skeleton_code,
      conv_classes=sorted(conv_classes),
      unpatched_conv_classes=sorted(unpatched_conv_classes),
  )


def inspect_cli(
    model: str,
    trust_remote_code: bool = False,
    extension_modules: str | None = None,
    split_cache: bool = False,
    run_dry_run_export: bool = True,
    output_format: str = 'markdown',
) -> str:
  """CLI entry point for `inspector`."""
  report = inspect_model_architecture(
      model=model,
      trust_remote_code=trust_remote_code,
      extension_modules=extension_modules,
      split_cache=split_cache,
      run_dry_run_export=run_dry_run_export,
  )
  if output_format == 'json':
    return json.dumps(report.to_dict(), indent=2)
  return report.to_markdown()


def main(_=None):
  fire.Fire(inspect_cli)


if __name__ == '__main__':
  main()
