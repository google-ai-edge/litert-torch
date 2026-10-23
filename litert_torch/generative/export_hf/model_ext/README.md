# Pluggable Model Extensions (`model_ext`)

This directory contains the pluggable **`ModelExtension`** framework and all
built-in architecture extensions for `litert_torch.generative.export_hf`.

A `ModelExtension` encapsulates every architecture-specific customization needed
to export a Hugging Face (or custom PyTorch/Diffusers) checkpoint to LiteRT
(`.tflite`) and LiteRT-LM (`.litertlm`) across text generation, multimodal
(vision/audio), speech recognition (ASR), text-to-speech (TTS), text-to-image
diffusion, and Any2Any multimodal tasks.

---

## Directory Structure

```text
model_ext/
├── __init__.py            # Public API exports & eager registration of built-ins
├── builtin_extensions.py  # Imports & registers all built-in <arch>/extension.py modules
├── extension.py           # ModelExtension base class, CompositeModelExtension, and registry
├── exportables.py         # Exportable dispatchers delegating to ModelExtensions
├── patches.py             # Context managers for class-level & instance-level patching
├── metadata_builder.py    # LlmMetadata & ExecutorMetadata builder dispatchers
├── testing.py             # Reusable ModelExtensionTestMixin contract test harness
├── extension_test.py      # Unit tests for registry, composition, and built-ins
└── <arch>/                # Self-contained per-architecture packages
    ├── extension.py       # @register_model_extension(override=False) subclass
    ├── patch.py           # Class/instance monkey-patches for export compatibility
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
| **LLM** | `qwen3/` | `Qwen3Extension` | `qwen3`, `qwen3_moe` |
| **LLM / VLM** | `qwen3_5/` | `Qwen3_5Extension` | `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text` |
| **ASR (Speech)** | `moonshine/` | `MoonshineExtension` | `moonshine`, `moonshine_streaming` |
| **ASR (Speech)** | `parakeet/` | `ParakeetCtcExtension`, `ParakeetTdtExtension` | `parakeet_ctc`, `parakeet_tdt` |
| **ASR (Speech)** | `qwen3/` | `Qwen3AsrExtension` | `qwen3_asr` |
| **ASR (Speech)** | `whisper/` | `WhisperExtension` | `whisper` |
| **TTS** | `kokoro_tts/` | `KokoroTtsExtension` | `kokoro` |
| **TTS** | `qwen3_tts/` | `Qwen3TtsExtension` | `qwen3_tts` |
| **Text-to-Image** | `bonsai_flux2/` | `BonsaiFlux2Extension` | `bonsai_flux2` |
| **Text-to-Image** | `flux2_klein/` | `Flux2KleinExtension` | `flux2_klein`, `Flux2KleinPipeline` |
| **Any2Any** | `minicpm_o/` | `MiniCpmOExtension` | `minicpmo`, `minicpm_o`, `minicpm-o`, `minicpm-o-4.5`, `minicpm-o-4_5`, `MiniCPM-o-4_5`, `MiniCPMO` |

---

## The `ModelExtension` Contract

Subclasses of `ModelExtension` (`model_ext/extension.py`) only need to override
the hooks relevant to their architecture. All unimplemented hooks default to a
no-op or `None`.

### 1. Configuration & Patching Lifecycle

* **`model_types: Sequence[str]`** — Tuple of HF `config.model_type` (or
  pipeline class) strings handled by this extension.
* **`update_export_config(export_config, model_config) -> ExportableModuleConfig`** —
  Validates or updates `ExportableModuleConfig` before model loading/export
  (e.g., enabling `externalize_embedder=True` for multimodal models).
* **`patch_context()`** *(context manager)* — Entered around
  `AutoModel.from_pretrained(...)` for class-level monkey patches on
  `transformers` modules.
* **`patch_model(model, export_config)`** *(context manager)* — Entered around
  `torch.export` / `litert_torch.convert` for instance-level module replacement
  or rotary embedding adjustments.

### 2. Text & Multimodal Exportable Modules

* **`get_prefill_decode_exportables(model_config, export_config)`** — Returns
  `(prefill_exportable_cls, decode_exportable_cls)` or `None` to use the
  standard decoder-only exportables.
* **`get_vision_exportables(model_config, export_config=None)`** — Returns a
  `VisionExportablesSpec(encoder_cls, adapter_cls, eoi_cls)` (or 3-tuple) for
  vision encoders.
* **`get_audio_exportables(model_config, export_config=None)`** — Returns an
  `AudioExportablesSpec(encoder_cls, adapter_cls, eoa_cls)` (or 3-tuple) for
  audio encoders.
* **`get_additional_exportables(model_config, export_config=None)`** — Returns
  `dict[str, type]` mapping auxiliary submodel names (e.g.,
  `'per_layer_embedder'`) to exportable module classes.

### 3. Non-Text-Generation Task Hooks (ASR, TTS, ImageGen, Any2Any)

* **`get_speech_exportables(model_config, export_config=None)`** — Returns ASR
  encoder/decoder exportable module classes.
* **`get_speech_model_cls(model_type)`** — Returns the ASR `SourceSpeechModel`
  subclass.
* **`get_tts_model_cls(model_type)`** — Returns the TTS model wrapper class.
* **`get_image_gen_model_cls(model_type)`** — Returns the Text-to-Image model
  wrapper class.
* **`get_any2any_model_cls(model_type)`** — Returns the Any2Any multimodal model
  wrapper class.
* **`match_checkpoint(model_path, task=None, config_dict=None) -> bool`** —
  Infers `model_type` for checkpoints that lack a standard Hugging Face
  `model_type` field in `config.json` (such as Kokoro TTS or Diffusers
  pipelines with `model_index.json`).

### 4. `.litertlm` Packaging & Metadata Hooks

* **`get_litert_lm_section_types() -> dict[str, Any]`** — Maps custom
  `additional_model_paths` keys to `.litertlm` TFLite section types.
* **`get_llm_model_type_proto(model_config, export_config)`** — Returns an
  `LlmModelType` protobuf message or `None` to fall back to `GenericModel`.
* **`build_llm_metadata(source_model_artifacts, export_config, exported_model_artifacts, llm_metadata)`** —
  Mutates and returns the `LlmMetadata` protobuf (stop tokens, prompt templates,
  channels, etc.).
* **`get_executor_metadata_builder(model_config)`** — Returns an optional
  callable `(source_model_artifacts, export_config, exported_model_artifacts) -> ExecutorMetadata`.

---

## Using or Registering a Custom `ModelExtension`

### 1. Pass an Extension Instance Directly in Python

Users and agents can prototype or override behavior inline without registering
anything globally:

```python
import dataclasses
from litert_torch.generative.export_hf import export
from litert_torch.generative.export_hf import model_ext


class MyCustomExtension(model_ext.ModelExtension):
  model_types = ('my_custom_arch',)

  def update_export_config(self, export_config, model_config):
    return dataclasses.replace(export_config, externalize_embedder=True)


export.export(
    model='/path/to/checkpoint',
    output_dir='/tmp/out',
    model_extension=MyCustomExtension(),
)
```

### 2. Compose on Top of a Built-in Extension

If `model_config.model_type` already matches a registered built-in extension
(e.g. `'gemma3'`) and you also pass `model_extension=MyOverrideExtension()`,
`resolve_model_extension` automatically wraps them in a
`CompositeModelExtension((builtin_ext, user_ext))`:

* Context managers (`patch_context`, `patch_model`) and transforms
  (`update_export_config`, `build_llm_metadata`) run sequentially in order.
* Selector hooks (`get_*`) use last-non-`None`-wins priority, so `user_ext`
  cleanly overrides only the hooks it implements while inheriting the rest.
* Dictionary hooks (`get_additional_exportables`,
  `get_litert_lm_section_types`) merge keys with later extensions taking
  precedence.

### 3. Load Extension Modules by Dotted Path (CLI)

For CLI runs (`export_main`), pass `--extension_modules` with comma-separated
Python module paths that execute `@register_model_extension` at import time:

```bash
python -m litert_torch.generative.export_hf.export_main \
  --model=/path/to/checkpoint \
  --output_dir=/tmp/out \
  --extension_modules=my_package.my_arch_extension
```

### 4. Wheel Entry-Point Discovery (`pyproject.toml`)

Third-party Python packages can register extensions automatically via standard
Python entry points without needing `--extension_modules`:

```toml
[project.entry-points."litert_torch.model_extensions"]
my_custom_arch = "my_package.my_arch_extension:MyCustomExtension"
```

---

## Testing a `ModelExtension`

Use `ModelExtensionTestMixin` from `model_ext/testing.py` to automatically test
the contract of any `ModelExtension` subclass (`model_types`,
`update_export_config`, `patch_context`, `patch_model`, and `LlmModelType` /
`LlmMetadata` hooks):

```python
from absl.testing import absltest
from litert_torch.generative.export_hf.model_ext import testing as ext_testing


class MyCustomExtensionTest(
    ext_testing.ModelExtensionTestMixin, absltest.TestCase
):
  extension_cls = MyCustomExtension


if __name__ == '__main__':
  absltest.main()
```
