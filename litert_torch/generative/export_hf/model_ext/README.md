# Pluggable Model Extensions (`model_ext`)

This directory contains the pluggable **`ModelExtension`** framework and all
built-in architecture extensions for `litert_torch.generative.export_hf`.

A `ModelExtension` encapsulates every architecture-specific customization needed
to export a Hugging Face (or custom PyTorch/Diffusers) checkpoint to LiteRT
(`.tflite`) and LiteRT-LM (`.litertlm`) across text generation, multimodal
(vision/audio), speech recognition (ASR), text-to-speech (TTS), text-to-image
diffusion, and Any2Any multimodal tasks—**without modifying `litert_torch` core
code**.

---

## Directory Structure

```text
model_ext/
├── __init__.py            # Public API exports & registration of built-ins
├── builtin_extensions.py  # Imports & registers built-in <arch>/extension.py
├── extension.py           # ModelExtension contract, composition, and registry
├── exportables.py         # Exportable dispatchers delegating to ModelExtension
├── patches.py             # Context managers for class & instance patching
├── metadata_builder.py    # LlmMetadata & ExecutorMetadata builder dispatchers
├── testing.py             # Reusable ModelExtensionTestMixin contract harness
├── extension_test.py      # Unit tests for registry, composition, and built-ins
└── <arch>/                # Self-contained per-architecture packages
    ├── extension.py       # @register_model_extension(override=False) subclass
    ├── patch.py           # Class/instance runtime patches for export
    ├── exportable_module.py / vision_exportable.py / audio_exportable.py
    └── metadata_builder.py
```

### Built-in Architecture Extensions

All built-in extensions are registered **by default** when `model_ext` or
`export_lib` is imported (with `@register_model_extension(override=False)` so
explicit user registrations take precedence), and `get_model_extension()` also
supports lazy fallback imports for `model_ext.<model_type>.extension`:

| Category | Subdirectory | Extension Class(es) | Registered `model_types` |
| :--- | :--- | :--- | :--- |
| **LLM / VLM** | `gemma3/` | `Gemma3Extension` | `gemma3`, `gemma3_text` |
| **LLM / VLM** | `gemma3n/` | `Gemma3nExtension` | `gemma3n` |
| **LLM / VLM / Audio** | `gemma4/` | `Gemma4Extension` | `gemma4`, `gemma4_text` |
| **VLM** | `gemma4_unified/` | `Gemma4UnifiedExtension` | `gemma4_unified` |
| **LLM** | `lfm2/` | `Lfm2Extension` | `lfm2` |
| **VLM** | `lfm2_vl/` | `Lfm2VlExtension` | `lfm2_vl` |
| **LLM** | `qwen3/` | `Qwen3Extension` | `qwen3`, `qwen3_text` |
| **LLM** | `qwen3_5/` | `Qwen3_5Extension` | `qwen3_5`, `qwen3_5_text` |
| **ASR (Speech)** | `moonshine/` | `MoonshineExtension` | `moonshine` |
| **ASR (Speech)** | `parakeet/` | `ParakeetCtcExtension`, `ParakeetTdtExtension` | `parakeet_ctc`, `parakeet_tdt` |
| **ASR (Speech)** | `qwen3/` | `Qwen3AsrExtension` | `qwen3_asr` |
| **ASR (Speech)** | `whisper/` | `WhisperExtension` | `whisper` |
| **TTS** | `kokoro_tts/` | `KokoroTtsExtension` | `kokoro`, `kokoro_tts`, `kokoro-82m`, `kokoro_82m` |
| **TTS** | `qwen3_tts/` | `Qwen3TtsExtension` | `qwen3_tts`, `qwen3_tts_talker` |
| **Text-to-Image** | `bonsai_flux2/` | `BonsaiFlux2Extension` | `bonsai_flux2` |
| **Text-to-Image** | `flux2_klein/` | `Flux2KleinExtension` | `flux2_klein`, `Flux2KleinPipeline` |
| **Any2Any** | `minicpm_o/` | `MiniCpmOExtension` | `minicpmo`, `minicpm_o`, `minicpm-o`, `minicpm-o-4.5`, `minicpm-o-4_5`, `MiniCPM-o-4_5`, `MiniCPMO` |

---

## Quick Start: Writing an Out-of-Tree `ModelExtension`

Subclass `ModelExtension` in a standalone `.py` file (or Python package), set
`model_types` to match `model_type` in the checkpoint's `config.json`, and
override only the hooks your architecture needs:

```python
# my_arch_extension.py
import contextlib
import dataclasses
from litert_torch.generative.export_hf.model_ext import extension
from litert_torch.generative.layers import normalization
import torch


@extension.register_model_extension()
class MyArchExtension(extension.ModelExtension):
  model_types = ('my_arch',)
  is_causal_text_model = True
  extra_kwargs_keys = ('key_ts_idx',)

  def update_export_config(self, export_config, model_config):
    del model_config
    if 'key_ts_idx' in export_config.extra_kwargs:
      export_config = dataclasses.replace(
          export_config, k_ts_idx=int(export_config.extra_kwargs['key_ts_idx'])
      )
    return export_config

  @contextlib.contextmanager
  def patch_model(self, model, export_config):
    del model, export_config
    yield
```

Then run `export_main` pointing `--extension_modules` at your `.py` file (or
importable dotted module name):

```bash
python -m litert_torch.generative.export_hf.export_main \
  --model=/path/to/checkpoint \
  --output_dir=$HOME/export_out \
  --extension_modules=./my_arch_extension.py \
  --trust_remote_code \
  --split_cache \
  --key_ts_idx=3 \
  --v_ts_idx=2
```

---

## Authoring Guide & Examples by Feature

### 1. Custom CLI Flags (`extra_kwargs_keys`) & `update_export_config`

By default, `ExportableModuleConfig` rejects unrecognized CLI flags with a
`ValueError` so mistyped flags are caught immediately. To add custom CLI flags
or override export settings for your model:

* **`extra_kwargs_keys: Sequence[str]`** — Declares custom CLI / `export(...)`
  keyword argument names accepted in `export_config.extra_kwargs`.
* **`update_export_config(self, export_config, model_config)`** — Inspects
  `export_config` (including `export_config.extra_kwargs`) and `model_config`
  after config loading and returns an updated `ExportableModuleConfig` via
  `dataclasses.replace(...)`. If `update_export_config` toggles pipeline stage
  flags (`externalize_embedder`, `split_cache`, `export_vision_encoder`,
  `export_audio_encoder`, etc.), `run_export_tasks` automatically rebuilds the
  export stage list.

**Built-in examples:** `model_ext/gemma4/extension.py`,
`model_ext/lfm2_vl/extension.py`.

```python
@extension.register_model_extension()
class MyVlmExtension(extension.ModelExtension):
  model_types = ('my_vlm',)
  # Allows passing `--my_vision_tokens=256` on the CLI or to `export.export()`.
  extra_kwargs_keys = ('my_vision_tokens',)

  def update_export_config(self, export_config, model_config):
    has_vision = getattr(model_config, 'vision_config', None) is not None
    # Automatically enable externalized embedder when exporting vision encoder.
    if has_vision and export_config.export_vision_encoder:
      export_config = dataclasses.replace(
          export_config,
          externalize_embedder=True,
          single_token_embedder=True,
      )
    return export_config
```

### 2. Custom Config Loading, Model Class & Attention Semantics

For architectures with custom Hugging Face config/model classes, causal SDPA, or
uniform sliding-window attention, configure them on the `ModelExtension`
subclass using these attributes and hooks:

* **`is_causal_text_model: bool | None`** — Controls whether text-generation
  prefill/decode exportables set `sdpa_is_causal = True` when GPU SDPA
  composites (`use_sdpa_composite` / `apply_gpu_composites`) are enabled:
  * `True`: Always treat this architecture as a causal decoder.
  * `False`: Disable `sdpa_is_causal` (e.g. for bidirectional or custom mask
    architectures).
  * `None` *(default)*: Defer to `core/exportable_module.py`'s default check.
* **`sliding_window_layers: str | None`** — Controls sliding-window attention
  mask generation when `config.sliding_window` is set without a per-layer
  `config.layer_types` list:
  * `'all'`: Treat every decoder layer as using sliding-window attention (e.g.
    Mistral-style uniform sliding window).
  * `'none'`: Disable sliding-window masks even when `sliding_window` is
    present on `config`.
  * `None` *(default)*: Inspect `config.layer_types` (with built-in fallback).
* **`load_config(self, model_path, export_config, trust_remote_code=False)`** —
  Loads a custom `transformers.PretrainedConfig` when `transformers.AutoConfig`
  cannot resolve the checkpoint's config class (return `None` to fall back to
  `AutoConfig.from_pretrained`).
* **`get_hf_model_cls(self, task, model_config, export_config)`** — Returns a
  custom model class for `text_generation`, `image_text_to_text`, or
  `multimodal_lm` exports when not in `transformers` `AutoModel*` tables. Can be
  an `AutoModel*` class, a `PreTrainedModel` subclass, or a plain
  `torch.nn.Module(model_config)` class (weights are loaded from `.safetensors`
  via `HFCheckpointWeightsLoader`).

```python
@extension.register_model_extension()
class CustomMultimodalExtension(extension.ModelExtension):
  model_types = ('custom_vlm',)
  is_causal_text_model = True
  sliding_window_layers = 'all'

  def load_config(self, model_path, export_config, trust_remote_code=False):
    del export_config, trust_remote_code
    return CustomVlmConfig.from_pretrained(model_path)

  def get_hf_model_cls(self, task, model_config, export_config):
    del task, model_config, export_config
    return CustomVlmForConditionalGeneration
```

### 3. Runtime Model Patching (`patch_context` vs. `patch_model`)

`ModelExtension` provides two context-manager hooks so you never need to modify
modeling files inside a checkpoint directory:

* **`patch_context(self)`** — Entered **before** `AutoModel.from_pretrained` /
  `from_config`. Use this to monkey-patch classes or functions in `transformers`
  that run during model construction (`__init__`).
* **`patch_model(self, model, export_config)`** — Entered **after** the model is
  instantiated and **around** tracing/conversion (`export_embedder_model` and
  `export_text_prefill_decode_model`). Use this to replace submodule methods,
  emit LiteRT HLFB composite ops, or adapt rotary embeddings on `model`
  (including `trust_remote_code` classes loaded dynamically from the
  checkpoint).

**Built-in examples:** `model_ext/qwen3_5/patch.py`,
`model_ext/gemma3/patch.py`, `model_ext/lfm2/patch.py`.

#### Example: Patching `RMSNorm` Composite & Dual RoPE for `split_cache`

Many `trust_remote_code` checkpoints define their own `RMSNorm` class (without
`@use_kernel_forward_from_hub("RMSNorm")`) and separate 2-argument
`rotary_emb_full` / `rotary_emb_sliding` modules, whereas LiteRT's `split_cache`
(`externalize_rope`) pipeline expects `model.model.rotary_emb(x, pos, type)` to
accept 3 arguments (`'full_attention'` and `'sliding_attention'`). Both can be
adapted in `patch_model` without editing the checkpoint:

```python
def _rms_norm_with_hlfb_forward(self, hidden_states: torch.Tensor):
  dtype = hidden_states.dtype
  hidden_states = hidden_states.to(torch.float32)
  hidden_states = normalization.rms_norm_with_hlfb(
      hidden_states,
      self.weight,
      self.variance_epsilon,
      torch.ones(
          (self.weight.shape[0],),
          dtype=torch.float32,
          device=hidden_states.device,
      ),
  )
  return hidden_states.to(dtype)


class _DualRotaryWrapper(torch.nn.Module):

  def __init__(self, full_rope, sliding_rope=None):
    super().__init__()
    self.rotary_emb_full = full_rope
    self.rotary_emb_sliding = sliding_rope

  def forward(self, x, position_ids, attention_type=None):
    if (
        attention_type == 'sliding_attention'
        and self.rotary_emb_sliding is not None
    ):
      return self.rotary_emb_sliding(x, position_ids)
    return self.rotary_emb_full(x, position_ids)


@extension.register_model_extension()
class Kanana2Extension(extension.ModelExtension):
  model_types = ('kanana2_tiny',)
  is_causal_text_model = True

  @contextlib.contextmanager
  def patch_model(self, model, export_config):
    del export_config
    # 1. Patch custom RMSNorm to emit the `odml.rms_norm` HLFB composite.
    patched_cls = {}
    for module in model.modules():
      cls = type(module)
      if cls.__name__ == 'Kanana2TinyRMSNorm' and cls not in patched_cls:
        patched_cls[cls] = cls.forward
        cls.forward = _rms_norm_with_hlfb_forward

    # 2. Wrap `rotary_emb_full` and `rotary_emb_sliding` into a 3-arg wrapper
    # for `inject_rotary_position_embedding` and `RoPEEmbedder`.
    inner = getattr(model, 'model', model)
    orig_full = getattr(inner, 'rotary_emb_full', None)
    orig_sliding = getattr(inner, 'rotary_emb_sliding', None)
    orig_rotary_emb = getattr(inner, 'rotary_emb', None)
    if isinstance(orig_full, torch.nn.Module):
      delattr(inner, 'rotary_emb_full')
      if hasattr(inner, 'rotary_emb_sliding'):
        delattr(inner, 'rotary_emb_sliding')
      inner.rotary_emb = _DualRotaryWrapper(orig_full, orig_sliding)
      inner.__dict__['rotary_emb_full'] = (
          lambda x, pos: inner.rotary_emb(x, pos, 'full_attention')
      )
      inner.__dict__['rotary_emb_sliding'] = (
          (lambda x, pos: inner.rotary_emb(x, pos, 'sliding_attention'))
          if orig_sliding is not None
          else None
      )
    try:
      yield
    finally:
      for cls, orig_fn in patched_cls.items():
        cls.forward = orig_fn
      if isinstance(orig_full, torch.nn.Module):
        inner.__dict__.pop('rotary_emb_full', None)
        inner.__dict__.pop('rotary_emb_sliding', None)
        inner.rotary_emb_full = orig_full
        inner.rotary_emb_sliding = orig_sliding
        if not hasattr(inner, 'original_rotary_emb'):
          if hasattr(inner, 'rotary_emb'):
            delattr(inner, 'rotary_emb')
          inner.rotary_emb = orig_rotary_emb
```

#### Example: Fusing MLP (`gate_up_proj`) & QKV (`qkv_proj`) Projections + SwiGLU Composite

For NPU and GPU delegates, fusing `gate_proj` + `up_proj` into a single
`gate_up_proj` (and optionally emitting the `odml.swiglu` composite via
`swiglu_composite.apply_swiglu`) or fusing `q_proj` + `k_proj` + `v_proj` into
`qkv_proj` eliminates redundant input broadcasts and separate matmul dispatches.
When replacing submodules in `patch_model`, always record `(parent, child_name,
original_child)` and restore them in `finally`:

```python
import contextlib
from litert_torch.generative.export_hf.experimental.composites import swiglu as swiglu_composite
from litert_torch.generative.export_hf.model_ext import extension
import torch


class FusedMLP(torch.nn.Module):
  """Replaces separate `gate_proj` + `up_proj` with a fused `gate_up_proj`."""

  def __init__(self, orig_mlp: torch.nn.Module, use_swiglu: bool = False):
    super().__init__()
    self.act_fn = orig_mlp.act_fn
    self.down_proj = orig_mlp.down_proj
    self.use_swiglu = use_swiglu

    gate_w = orig_mlp.gate_proj.weight
    up_w = orig_mlp.up_proj.weight
    in_features = gate_w.shape[1]
    out_features = gate_w.shape[0] + up_w.shape[0]
    self.gate_up_proj = torch.nn.Linear(
        in_features,
        out_features,
        bias=False,
        device=gate_w.device,
        dtype=gate_w.dtype,
    )
    if not gate_w.is_meta:
      self.gate_up_proj.weight = torch.nn.Parameter(
          torch.cat([gate_w.data, up_w.data], dim=0)
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_up = self.gate_up_proj(x)
    if self.use_swiglu:
      return self.down_proj(swiglu_composite.apply_swiglu(gate_up))
    gate, up = gate_up.chunk(2, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


@extension.register_model_extension()
class MyFusedArchExtension(extension.ModelExtension):
  model_types = ('my_arch',)
  # `FusedMLP` above fuses the standard HF `gate_proj` / `up_proj` names, which
  # `HFCheckpointWeightsLoader` already maps to `gate_up_proj` (likewise
  # `qkv_proj` <- `q_proj` / `k_proj` / `v_proj`), so Converter V2 (`--use_v2`)
  # needs no extra declaration here. If your checkpoint uses other names for the
  # fused parts (e.g. LFM2's `w1` / `w3`), declare them so Converter V2 can
  # stream and concatenate the checkpoint tensors (see section 4):
  #
  #   fused_weight_specs = (
  #       extension.FusedWeightSpec('gate_up_proj', ('w1', 'w3'), dim=0),
  #   )

  @contextlib.contextmanager
  def patch_model(self, model, export_config):
    fuse_mlp = getattr(export_config, 'fuse_gate_up', False) or getattr(
        export_config, 'use_swiglu_composite', False
    )
    use_swiglu = getattr(export_config, 'use_swiglu_composite', False)
    replaced = []

    def _replace(parent: torch.nn.Module):
      for name, child in parent.named_children():
        if fuse_mlp and hasattr(child, 'gate_proj') and hasattr(child, 'up_proj'):
          setattr(parent, name, FusedMLP(child, use_swiglu=use_swiglu))
          replaced.append((parent, name, child))
        else:
          _replace(child)

    _replace(model)
    try:
      yield
    finally:
      for parent, name, orig_child in reversed(replaced):
        setattr(parent, name, orig_child)
```

#### LiteRT-Torch Composite Ops & Import Cheat-Sheet

When authoring `patch_model` or `patch_context`, use the exact import paths and
call contracts below:

| Operator | Import Path | Call Signature & Tensor Shape Contract |
| :--- | :--- | :--- |
| **RMSNorm (`odml.rms_norm`)** | `from litert_torch.generative.layers import normalization` | `normalization.rms_norm_with_hlfb(hidden_states, weight, eps, torch.ones((hidden_size,), dtype=torch.float32, device=hidden_states.device))` — takes **4 positional arguments** (including the 4th `ones` tensor). |
| **SwiGLU (`odml.swiglu`)** | `from litert_torch.generative.export_hf.experimental.composites import swiglu` | `swiglu.apply_swiglu(gate_up)` — takes concatenated `[gate, up]` (`[B, T, 2 * D_ff]`) from `gate_up_proj(x)` and returns `[B, T, D_ff]` **before** `down_proj` / `w2`. |
| **ShortConv (`odml.short_conv`)** | `from litert_torch.generative.export_hf.experimental.composites import short_conv` | `short_conv.apply_short_conv_step(in_proj_out, conv_state, conv_weight, conv_bias, num_valid_tokens, conv_L_cache)` (or `short_conv.short_conv(...)`) — takes `in_proj_out = self.in_proj(hidden_states)` (`[B, T, 3 * D]`) and returns `(y, new_conv_state)` **before** `self.out_proj(y)`. |
| **Attention (`ALL_ATTENTION_FUNCTIONS`)** | `from litert_torch.generative.export_hf.core import attention as _` | Dispatch via `modeling_utils.ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`. Note that `lrt_transposed_attention` returns `attn_output` as `[B, T, num_heads, head_dim]` on CPU/prefill, whereas `gpu_sdpa` decode returns `[B, 1, num_heads * head_dim]`. Always reshape via `attn_output.reshape(*input_shape, self.num_heads, self.head_dim)` before applying per-head post-SDPA gates (e.g. `LFM3`'s `gate_proj`). |

#### Hybrid Architectures (`conv` + `sliding_attention` + `full_attention`) & `trust_remote_code`

For hybrid models (such as `LiquidAI/LFM3-1.7B`) that combine short-convolution
(`"conv"`), sliding-window attention (`"sliding_attention"`), and full causal
attention (`"full_attention"`) in `config.layer_types`:

1. **`trust_remote_code` Dynamic Module Lookup**: When loaded with
   `trust_remote_code=True`, the modeling module resides in `sys.modules` under
   `transformers_modules.<repo>.modeling_<arch>`. Scan `sys.modules` in a
   helper (`_find_<arch>_modeling_module()`) and return the `ForCausalLM` class
   from `get_hf_model_cls` so Converter V2 (`use_v2=True`) and `patch_context`
   can locate it.
2. **Mask Mapping for `"conv"` Layers**: When `sliding_window` is set,
   `exportable_module.adapt_inputs` passes `attention_mask = {'full_attention':
   ..., 'sliding_attention': ...}`. Ensure your patched `Model.forward` calls
   `causal_mask_mapping.setdefault('conv', None)` (or `.get(layer_type)`) and
   forwards `**kwargs` (`valid_mask`) through `DecoderLayer.forward` into
   `ShortConv.forward`.
3. **Reasoning / Thinking Channels (`<think>...</think>`)**: Override
   `build_llm_metadata` to set `llm_metadata.supports_thinking = True` and add
   a `'thought'` channel (`start='<think>'`, `end='</think>'`).

### 4. Converter V2 Weights (`checkpoint_key_prefixes`, `fused_weight_specs` & `resolve_checkpoint_tensor`)

When exporting with Converter V2 (`--use_v2=True`) using real weights (i.e.
without `--use_random_weights`), the model is traced on the `meta` device and
every parameter is streamed from the checkpoint during serialization:
`HFCheckpointWeightsLoader` (`core/weights_loader.py`) maps each parameter name
of the *patched*, traced module to tensor names inside the checkpoint's
`.safetensors` files.

> [!IMPORTANT]
> `--use_random_weights=True` bypasses `HFCheckpointWeightsLoader` entirely, so
> it cannot validate Converter V2 weight mapping. Fused names (such as
> `gate_up_proj`) also only exist when the flag that enables the fusion
> (`--fuse_qkv`, `--fuse_gate_up`, `--use_swiglu_composite`,
> `--use_qkv_norm_rope_composite`, or `--apply_gpu_composites`, which implies
> them) is on. A mapping miss surfaces late, in `Save Params Bin`, as
> `KeyError: Parameter '...' not found in weights_loader` and only names the
> first missing parameter. Run `testing.assert_v2_weights_resolvable` (see
> *Contract & Fast-Inner-Loop Verification* below) to catch all misses in
> seconds.

**Naming rule:** every parameter of the patched model must be one of:

1. An unchanged checkpoint key (after prefix matching). Retained submodules in
   a patched module must keep their checkpoint attribute names (e.g. LFM2's
   `FusedLfm2MLP` keeps `self.w2`; renaming it to `down_proj` breaks loading).
2. A `fused_weight_specs` entry. Specs match `*.{fused_name}.{weight|bias}`
   and concatenate the sibling checkpoint tensors `*.{part}.{weight|bias}`, so
   the fused submodule must live under the same parent as the parts it
   replaces.
3. Reconstructed by `resolve_checkpoint_tensor`.

If your model uses non-standard checkpoint wrapper prefixes or custom fused
projections, use these three hooks:

* **`checkpoint_key_prefixes: Sequence[str]`** — Additional leading module/key
  prefixes stripped and prepended when `HFCheckpointWeightsLoader` generates
  candidate safetensors keys. Built-in handling already strips `model.`,
  `module.`, `static_model.`, `original_hf_model.`, and `language_model.`, and
  tries keys under `model.`, `model.language_model.`, `language_model.`, and
  `transformer.`.
* **`fused_weight_specs: Sequence[FusedWeightSpec]`** — Declarative specs
  `FusedWeightSpec(fused_name, part_names, dim=0)` telling
  `HFCheckpointWeightsLoader` how to concatenate separate checkpoint tensors
  into a fused `nn.Linear` parameter. Built-in defaults only cover the standard
  HF names `gate_up_proj <- (gate_proj, up_proj)` and
  `qkv_proj <- (q_proj, k_proj, v_proj)` / `(q_proj, k_proj)`; declare any
  other naming (e.g. LFM2's `gate_up_proj <- (w1, w3)`) on your extension.
  Extension specs are tried before the defaults.
* **`resolve_checkpoint_tensor(self, param_name, try_load) -> Tensor | None`** —
  Called by `HFCheckpointWeightsLoader` when a parameter needs arbitrary custom
  reconstruction (such as interleaving heads, slicing, or transposing).
  `try_load(candidate_name)` looks up any candidate parameter name through the
  loader's prefix-matching table and returns a `torch.Tensor` or `None`.

```python
@extension.register_model_extension()
class CustomWeightsExtension(extension.ModelExtension):
  model_types = ('my_custom_llm',)
  # Strips/adds 'backbone.' or 'decoder.' when matching safetensors keys.
  checkpoint_key_prefixes = ('backbone.', 'decoder.')
  # Declaratively concatenate `.w1.weight` + `.w3.weight` into `.w13.weight`:
  fused_weight_specs = (
      extension.FusedWeightSpec('w13', ('w1', 'w3'), dim=0),
  )

  def resolve_checkpoint_tensor(self, param_name, try_load):
    # Arbitrary tensor transform (e.g. splitting or transposing a weight):
    if param_name.endswith('.custom_proj.weight'):
      prefix = param_name[: -len('.custom_proj.weight')]
      raw = try_load(f'{prefix}.raw_proj.weight')
      if raw is not None:
        return raw.transpose(0, 1).contiguous()
    return None
```

### 5. Custom Exportables & `.litertlm` Packaging Metadata

To export custom subgraphs (vision/audio encoders, per-layer embedders, or
specialized prefill/decode signatures) and package them into `.litertlm`:

* **`get_prefill_decode_exportables(self, model_config, export_config)`** —
  Returns `(prefill_cls, decode_cls)` or `None` to use the standard decoder-only
  exportables.
* **`get_vision_exportables(self, model_config, export_config=None)`** — Returns
  `VisionExportablesSpec(encoder_cls, adapter_cls=..., eoi_cls=...)` for
  `export_vision_encoder_models`.
* **`get_audio_exportables(self, model_config, export_config=None)`** — Returns
  `AudioExportablesSpec(encoder_cls, adapter_cls=..., eoa_cls=...)` for
  `export_audio_encoder_models_for_llm`.
* **`get_additional_exportables(self, model_config, export_config=None)`** —
  Returns `{submodel_name: exportable_cls}` (e.g. `{'per_layer_embedder': ...}`)
  exported as `<submodel_name>.tflite` into
  `exported_model_artifacts.additional_model_paths`.
* **`get_litert_lm_section_types(self)`** — Maps keys from
  `additional_model_paths` to `.litertlm` section types (such as
  `litert_lm_builder.TFLiteModelType.TF_LITE_PER_LAYER_EMBEDDER`).
* **`get_llm_model_type_proto(self, model_config, export_config)`** — Returns a
  custom `llm_model_type_pb2.LlmModelType` protobuf message (or `None` to fall
  back to `GenericModel`).
* **`build_llm_metadata(self, ...)`** — Populates stop tokens, prompt templates,
  or modality tokens on `LlmMetadata` given `(source_model_artifacts,
  export_config, exported_model_artifacts, llm_metadata)`.
* **`update_executor_metadata(self, ...)`** — Returns an `ExecutorMetadata`
  protobuf given `(source_model_artifacts, export_config,
  exported_model_artifacts)` when packaging `.litertlm`.

**Built-in examples:** `model_ext/gemma3/extension.py`,
`model_ext/gemma3n/extension.py`, `model_ext/gemma4/extension.py`,
`model_ext/bonsai_flux2/extension.py`.

### 6. Non-Text Tasks (ASR, TTS, ImageGen, Any2Any) & `match_checkpoint`

Some checkpoints (such as Diffusers pipelines with `model_index.json` or
standalone TTS checkpoints like Kokoro) do not have a standard Hugging Face
`model_type` field in `config.json`.

* **`match_checkpoint(cls, model_path, task=None, config_dict=None) -> bool`** —
  Classmethod inspected by `infer_model_type_from_extensions` when `config.json`
  has no `model_type`. When it returns `True`, `cls.model_types[0]` is used as
  the resolved `model_type`.
* **Task model/exportable hooks:**
  * **ASR (`automatic_speech_recognition`)**: `get_speech_model_cls(model_type)`
    and `get_speech_exportables(model_config, export_config)` (see
    `model_ext/whisper/extension.py`, `model_ext/parakeet/extension.py`).
  * **TTS (`text_to_speech`)**: `get_tts_model_cls(model_type)` and
    `is_matching_tts_checkpoint(config_dict, model_path)` (see
    `model_ext/kokoro_tts/extension.py`, `model_ext/qwen3_tts/extension.py`).
  * **Text-to-Image (`text_to_image`)**: `get_image_gen_model_cls(model_type)`
    (see `model_ext/bonsai_flux2/extension.py`,
    `model_ext/flux2_klein/extension.py`).
  * **Any2Any (`any_to_any`)**: `get_any2any_model_cls(model_type)` (see
    `model_ext/minicpm_o/extension.py`).

```python
@extension.register_model_extension()
class MyDiffusionExtension(extension.ModelExtension):
  model_types = ('my_diffusion_pipeline',)

  @classmethod
  def match_checkpoint(cls, model_path, task=None, config_dict=None) -> bool:
    del model_path
    return (
        task == 'text_to_image'
        and isinstance(config_dict, dict)
        and config_dict.get('_class_name') == 'MyDiffusionPipeline'
    )

  def get_image_gen_model_cls(self, model_type: str):
    del model_type
    return MyDiffusionImageGenModel
```

---

## Using, Composing & Distributing Extensions

### 1. Pass an Extension Directly in Python (`model_extension=...`)

Prototype or override behavior inline without global registration:

```python
from litert_torch.generative.export_hf import export

export.export(
    model='/path/to/checkpoint',
    output_dir='/path/to/output',
    model_extension=MyArchExtension(),
)
```

### 2. Layer on Top of a Built-in Extension (`CompositeModelExtension`)

If `model_config.model_type` matches a registered extension (e.g. `'gemma3'`)
and you also pass `model_extension=MyOverrideExtension()`,
`resolve_model_extension` composes them as
`CompositeModelExtension((builtin_ext, user_ext))`:

* Context managers (`patch_context`, `patch_model`) and transforms
  (`update_export_config`, `build_llm_metadata`) run sequentially in order.
* Selector hooks (`get_*`, `resolve_checkpoint_tensor`) use last-non-`None`-wins
  priority, so `user_ext` overrides only the hooks it implements while
  inheriting everything else.
* Sequence attributes (`extra_kwargs_keys`, `checkpoint_key_prefixes`) and
  dictionary hooks (`get_additional_exportables`, `get_litert_lm_section_types`)
  are merged automatically.

### 3. Load via `--extension_modules` (CLI or Python)

Pass comma-separated dotted module names **or `.py` file paths** to
`--extension_modules`. Any `ModelExtension` subclass in the loaded module/file
with non-empty `model_types` is registered automatically before CLI flag
validation runs:

```bash
python -m litert_torch.generative.export_hf.export_main \
  --model=/path/to/checkpoint \
  --output_dir=/path/to/output \
  --extension_modules=/path/to/my_arch_extension.py
```

### 4. Wheel Entry-Point Discovery (`pyproject.toml`)

Third-party Python packages can register extensions automatically on import via
standard Python entry points without needing `--extension_modules`:

```toml
[project.entry-points."litert_torch.model_extensions"]
my_arch = "my_package.my_arch_extension:MyArchExtension"
```

---

## Inspecting a Checkpoint & Testing a `ModelExtension`

### 1. Architecture Inspector & Dry-Run Diagnostic CLI (`model_ext/inspector.py`)

Before writing a `ModelExtension` (or to verify one you just wrote), run the
Architecture Inspector on a checkpoint directory or model ID. It shrinks the
config to a tiny 2-layer CPU model with random weights, inspects RMSNorm/MLP/
Attention/RoPE classes and `.safetensors` key prefixes, runs `torch.export`
prefill/decode dry runs in `<5s`, and prints actionable recommendations plus a
ready-to-edit `ModelExtension` skeleton:

```bash
bazel run //third_party/py/litert_torch/generative/export_hf/model_ext:inspector -- \
  --model=/path/to/checkpoint \
  --trust_remote_code \
  --split_cache
```

Pass `--extension_modules=/path/to/my_arch_extension.py` (or `--format=json`) to
verify that your extension resolves all dry-run checks before running a full
multi-gigabyte export. You can also call `inspect_model_architecture(...)`
directly from Python (`from litert_torch.generative.export_hf.model_ext import
inspector`).

### 2. Contract & Fast-Inner-Loop Verification (`model_ext/testing.py`)

`model_ext/testing.py` provides both a reusable `unittest.TestCase` mixin and
standalone fast-inner-loop helpers that verify numerical parity,
`torch.export.export` graph capture, and Converter V2 weight mapping on tiny
2-layer models in `<2s` on CPU:

* **`ModelExtensionTestMixin`** — Tests all `ModelExtension` hook contracts
  (`model_types`, `update_export_config`, `patch_context` / `patch_model`
  re-entrancy, exportable selectors, and `LlmModelType` / `LlmMetadata` hooks).
* **`assert_patch_numerical_parity(ext, model, ...)`** — Runs eager inference
  before and during `patch_model`, asserts output tensors match within `(atol,
  rtol)`, and verifies that submodule classes and `.forward` methods are
  cleanly restored after exiting `patch_model`.
* **`assert_tiny_prefill_decode_torch_export(ext, model, ...)`** — Wraps a tiny
  2-layer causal LM in the resolved prefill/decode `ExportableModule` classes
  (and `RoPEEmbedder` when `split_cache=True` / `externalize_rope=True`) and
  runs `torch.export.export` across all sample signatures.
* **`assert_v2_weights_resolvable(ext, model, ...)`** — Saves the tiny
  *unpatched* model as a safetensors checkpoint, applies `patch_model` with all
  fusion / composite flags on (by default), builds `HFCheckpointWeightsLoader`
  with your extension's `checkpoint_key_prefixes`, `fused_weight_specs`, and
  `resolve_checkpoint_tensor` exactly as `export_lib` does, and checks that
  every patched parameter loads with the right shape and value. Reports all
  misses at once instead of the first `KeyError` a full `--use_v2` export would
  hit after minutes of tracing. If `patch_context` swaps classes in a way that
  renames parameters, build the tiny model inside
  `patches.get_patch_context(model_type)`.

```python
from absl.testing import absltest
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
import torch
import transformers


class MyArchExtensionTest(
    ext_testing.ModelExtensionTestMixin, absltest.TestCase
):
  extension_cls = MyArchExtension

  def test_patch_numerical_parity_and_tiny_export(self):
    ext = self.get_extension()
    tiny_cfg = transformers.AutoConfig.for_model(
        'qwen3',
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    tiny_model = transformers.AutoModelForCausalLM.from_config(tiny_cfg).eval()
    export_cfg = exportable_module_config.ExportableModuleConfig(
        model='dummy',
        prefill_lengths=[8],
        cache_length=16,
        cache_lengths=[16],
        fuse_gate_up=True,
        fuse_qkv=True,
    )
    # 1. Verify eager numerical parity & clean unpatching after exiting context:
    self.assert_patch_numerical_parity(
        tiny_model,
        sample_kwargs={
            'input_ids': torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            'use_cache': False,
        },
        export_config=export_cfg,
    )
    # 2. Verify prefill + decode torch.export graph capture in <2s on CPU:
    exported = self.assert_tiny_prefill_decode_torch_export(
        tiny_model,
        model_config=tiny_cfg,
        export_config=export_cfg,
    )
    self.assertIn('prefill_8', exported)
    self.assertIn('decode', exported)
    # 3. Verify Converter V2 (`--use_v2`) can stream every patched parameter
    # (fused projections included) from a safetensors checkpoint:
    self.assert_v2_weights_resolvable(tiny_model)


if __name__ == '__main__':
  absltest.main()
```
