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
"""Pluggable model extension contract and registry for `export_hf`.

This module defines the `ModelExtension` interface and discovery/registration
utilities that decouple model-specific customization from the core `export_hf`
pipeline. Both built-in architectures under `model_ext/<arch>/extension.py` and
zero-touch out-of-tree user plugins implement `ModelExtension` to customize
every stage of the export lifecycle:

1. **Checkpoint, Config & CLI Flag Resolution**:
   - `extra_kwargs_keys` registers architecture-specific CLI / `export(...)`
     keyword arguments (see `model_ext/gemma4/extension.py`).
   - `match_checkpoint` / `is_matching_tts_checkpoint` identify checkpoints
     that omit a standard HuggingFace `model_type` (e.g. Diffusers pipelines in
     `model_ext/bonsai_flux2/extension.py` or typeless TTS checkpoints in
     `model_ext/kokoro_tts/extension.py`).
   - `load_config` loads custom `PretrainedConfig` objects for architectures
     not registered in `transformers.AutoConfig`.
   - `update_export_config` validates or mutates `ExportableModuleConfig` flags
     once the model config is loaded (rebuilding default export tasks when
     flags change).
2. **Model Loading & Runtime Patching**:
   - `patch_context` applies class-level patches around config and model
     construction so custom layer classes are instantiated (see
     `model_ext/gemma3/patch.py` and `model_ext/qwen3/patch.py`).
   - `get_hf_model_cls` supplies custom `PreTrainedModel` or plain
     `torch.nn.Module(config)` classes for text/multimodal LLM export when not
     in `transformers` `AutoModel*` tables; task-specific selectors
     (`get_speech_model_cls`, `get_tts_model_cls`, `get_image_gen_model_cls`,
     `get_any2any_model_cls`) cover ASR, TTS, ImageGen, and Any2Any tasks.
   - `checkpoint_key_prefixes`, `fused_weight_specs`, and
     `resolve_checkpoint_tensor` customize Converter V2 (`use_v2=True`)
     safetensors streaming without editing `core/weights_loader.py`.
3. **Graph Tracing, Attention Semantics & Exportable Selection**:
   - `patch_model` applies instance-level module replacements around
     `torch.export` tracing (e.g. fusing QKV/MLP projections or injecting HLFB
     composites; see `model_ext/gemma3/patch.py` and `model_ext/lfm2/patch.py`).
   - `is_causal_text_model` opts causal decoders into or out of `sdpa_is_causal`
     GPU composite masks.
   - Exportable selectors (`get_prefill_decode_exportables`,
     `get_vision_exportables`, `get_audio_exportables`,
     `get_additional_exportables`, `get_speech_exportables`) supply the
     `ExportableModule` subclasses traced for each submodel.
4. **LiteRT-LM Metadata & Packaging**:
   - `get_litert_lm_section_types`, `get_llm_model_type_proto`,
     `build_llm_metadata`, and `get_executor_metadata_builder` customize how
     exported `.tflite` flatbuffers, tokenizers, and runtime state descriptors
     are bundled into a `.litertlm` package (see
     `model_ext/gemma4/extension.py` and `model_ext/bonsai_flux2/extension.py`).

Extensions can be activated in three ways:
- **In-process Python**: Pass `model_extension=MyExtension` (or a sequence of
  extensions) to `export.export(...)` or `ExportableModuleConfig`.
- **CLI / dynamic import**: Pass `--extension_modules=my_pkg.my_ext` or a
  standalone `.py` path `--extension_modules=/path/to/my_ext.py` (subclasses
  with non-empty `model_types` or `@register_model_extension` register
  automatically).
- **Installed wheel entry points**: Declare an entry point under group
  `litert_torch.model_extensions` in `pyproject.toml`.

Built-in reference implementations live in `model_ext/<arch>/extension.py` and
are registered via `model_ext/builtin_extensions.py` with `override=False` so
explicit user registrations take precedence.
"""

from collections.abc import Callable, Mapping, Sequence
import contextlib
import hashlib
import importlib
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
import os
import sys
import types
from typing import Any, ClassVar, NamedTuple

from litert_torch.generative.export_hf.core import exportable_module_config
import transformers


ENTRY_POINT_GROUP = 'litert_torch.model_extensions'


class FusedWeightSpec(NamedTuple):
  """Declares how a fused parameter is concatenated from checkpoint tensors.

  Used by `ModelExtension.fused_weight_specs` and
  `core.weights_loader.HFCheckpointWeightsLoader` during Converter V2
  (`use_v2=True`) weight streaming when a runtime patch replaces separate
  linear projections in a checkpoint with a single fused `nn.Linear`.

  Example:
    `FusedWeightSpec('w13_proj', ('w1_proj', 'w3_proj'), dim=0)` instructs
    `HFCheckpointWeightsLoader` to materialize `...w13_proj.weight` (and
    `.bias`) by concatenating `...w1_proj.weight` and `...w3_proj.weight` along
    dimension 0.

  Attributes:
    fused_name: Submodule or attribute name of the fused projection in the
      patched export model (matched as `*.{fused_name}.{suffix}`).
    part_names: Ordered tuple of unfused checkpoint submodule names whose
      tensors are concatenated to form `fused_name`.
    dim: Dimension along which `part_names` tensors are concatenated (defaults
      to `0`, the output-features axis of `torch.nn.Linear.weight`).
  """

  fused_name: str
  part_names: tuple[str, ...]
  dim: int = 0


class VisionExportablesSpec(NamedTuple):
  """Exportable module classes for a multimodal vision tower.

  Returned by `ModelExtension.get_vision_exportables` when `task` is
  `image_text_to_text` or `multimodal_lm` and `export_vision_encoder=True`.
  See `model_ext/gemma3/extension.py`, `model_ext/gemma4/extension.py`, and
  `model_ext/lfm2_vl/extension.py` for built-in examples.

  Attributes:
    encoder_cls: Required `ExportableModule` subclass that wraps the vision
      backbone (e.g. `LiteRTExportableModuleForGemma3VisionEncoder`).
    adapter_cls: Optional `ExportableModule` subclass that projects vision
      features into the text hidden dimension, or `None` when the encoder
      already includes projection (as in
      `model_ext/gemma4_unified/extension.py`).
    eoi_cls: Optional `ExportableModule` subclass for exporting learned
      end-of-image (EOI) token embeddings, or `None` if unused (see
      `model_ext/gemma4/extension.py` for a non-`None` example).
  """

  encoder_cls: Any
  adapter_cls: Any = None
  eoi_cls: Any = None


class AudioExportablesSpec(NamedTuple):
  """Exportable module classes for a multimodal audio tower.

  Returned by `ModelExtension.get_audio_exportables` when `task` is
  `multimodal_lm` and `export_audio_encoder=True`. See
  `model_ext/gemma4/extension.py` for a built-in example.

  Attributes:
    encoder_cls: Required `ExportableModule` subclass that wraps the audio
      encoder tower (e.g. `LiteRTExportableModuleForGemma4AudioEncoder`).
    adapter_cls: Optional `ExportableModule` subclass that projects encoded
      audio features into the text hidden dimension, or `None` if fused into
      `encoder_cls`.
    eoa_cls: Optional `ExportableModule` subclass for exporting learned
      end-of-audio (EOA) token embeddings, or `None` if unused.
  """

  encoder_cls: Any
  adapter_cls: Any = None
  eoa_cls: Any = None


# Re-exported so extensions can name tasks without depending on `core`.
ExportTask = exportable_module_config.ExportTask


class ExportDefaults(NamedTuple):
  """Recommended export settings for a checkpoint.

  Returned by `ModelExtension.get_export_defaults` and consumed by
  `auto_export.resolve` / `auto_export.export` (and through them by tools such
  as LiteRT CLI), so per-architecture knowledge lives with the extension rather
  than in every caller. Explicit caller arguments always override these values.

  Attributes:
    task: Default `ExportTask` for the checkpoint, or `None` to let the caller
      decide (`text_generation` when nothing else applies).
    export_kwargs: Recommended `export.export` keyword arguments for the
      resolved task (e.g. `quantization_recipe`, `input_sec`,
      `externalize_embedder`).
    litert_lm_model_family: Directory name under LiteRT-LM `models/` holding the
      canonical `LlmMetadataProto.pbtext` (chat template, stop tokens, thinking
      and tool-call config), e.g. `'qwen3'`.
    litert_lm_chat_template: Path of a canonical Jinja chat template relative to
      LiteRT-LM `models/`, e.g. `'gemma4/chat_template.jinja'`.
    runtime_model_name: LiteRT-LM `omni_runner --model_name` value for non-LLM
      tasks, e.g. `'whisper'` or `'bonsai-flux2'`.
  """

  task: str | None = None
  export_kwargs: Mapping[str, Any] = types.MappingProxyType({})
  litert_lm_model_family: str | None = None
  litert_lm_chat_template: str | None = None
  runtime_model_name: str | None = None


class Speech2TextBundleSpec(NamedTuple):
  """How an encoder-decoder speech-to-text model is packaged in a `.litertlm`.

  Returned by `ModelExtension.get_speech2text_bundle_spec`. When an extension
  returns one, `export_hf` exports the speech model as a single TFLite model
  (`export_asr_models`) and packages it with its `Speech2TextMetadata` and HF
  tokenizer, the layout LiteRT-LM's `omni_runner` loads.

  Attributes:
    asr_model_type: `Speech2TextModelType` field name, e.g. `'whisper'`.
    tflite_model_type: Section model type of the exported TFLite model:
      `'encoder_decoder'` (encoder and decoder signatures in one model) or
      `'audio_encoder'` (CTC models).
    mel_frames_per_second: Log-mel frames per second of audio for models that
      take a log-mel spectrogram (100 for a 10 ms hop), or `None` for models
      that take raw PCM.
  """

  asr_model_type: str
  tflite_model_type: str
  mel_frames_per_second: int | None = None


def text_config_dict(config_dict: Mapping[str, Any]) -> Mapping[str, Any]:
  """Returns `config_dict['text_config']` if present, else `config_dict`."""
  return config_dict.get('text_config') or config_dict


class ModelExtension:
  """Base contract for pluggable model architecture extensions in `export_hf`.

  Instances customize how a HuggingFace or custom PyTorch checkpoint is
  configured, patched, wrapped in `ExportableModule` subclasses, and packaged
  into `.litertlm` artifacts. Every hook defaults to `None` or a no-op so
  subclasses only override the hooks needed for their architecture, and
  multiple extensions can be layered via `CompositeModelExtension`.

  Typical built-in examples by task:
  - **Text-only / Multimodal LLM**: `model_ext/qwen3_5/extension.py`,
    `model_ext/gemma3/extension.py`, `model_ext/gemma3n/extension.py`,
    `model_ext/gemma4/extension.py`, `model_ext/lfm2_vl/extension.py`.
  - **Speech Recognition (ASR)**: `model_ext/whisper/extension.py`,
    `model_ext/parakeet/extension.py`, `model_ext/moonshine/extension.py`.
  - **Text-to-Speech (TTS)**: `model_ext/kokoro_tts/extension.py`,
    `model_ext/qwen3_tts/extension.py`.
  - **Text-to-Image**: `model_ext/bonsai_flux2/extension.py`,
    `model_ext/flux2_klein/extension.py`.
  - **Any-to-Any Multimodal**: `model_ext/minicpm_o/extension.py`.

  Example:
    ```python
    from litert_torch.generative.export_hf import model_ext

    @model_ext.register_model_extension
    class MyArchExtension(model_ext.ModelExtension):
      model_types = ('my_arch',)

      def get_prefill_decode_exportables(self, model_config, export_config):
        return (MyPrefillExportable, MyDecodeExportable)
    ```

  Attributes:
    model_types: Tuple or sequence of `config.json` `model_type` strings (or
      Diffusers `_class_name` identifiers) handled by this extension.
    extra_kwargs_keys: Architecture-specific flag names accepted in
      `ExportableModuleConfig.extra_kwargs` and `--<flag>` CLI arguments (see
      `model_ext/gemma4/extension.py` for an example).
    is_causal_text_model: Optional boolean controlling `sdpa_is_causal` on
      text-generation exports. Set to `True` on out-of-tree causal decoder-only
      models so GPU composite SDPA (`sdpa_is_causal`) is enabled automatically,
      or `False` for prefix/bidirectional text models; `None` defers to core.
    sliding_window_layers: Optional sliding-window attention layer policy when
      `config.sliding_window` is set without `config.layer_types`. Set to
      `'all'` when every decoder layer uses sliding-window attention (e.g.
      Mistral-style uniform sliding window), or `'none'` to disable sliding
      window masks even when `sliding_window` is present on `config`; `None`
      defers to `config.layer_types` inspection.
    checkpoint_key_prefixes: Additional key prefixes (e.g. `('backbone.',)`)
      probed by `HFCheckpointWeightsLoader` when streaming safetensors weights
      with `use_v2=True`.
    fused_weight_specs: Sequence of `FusedWeightSpec` entries (or `(fused_name,
      part_names)` 2-tuples) describing fused projections for Converter V2
      weight streaming.
  """

  model_types: ClassVar[Sequence[str]] = ()
  extra_kwargs_keys: ClassVar[Sequence[str]] = ()
  is_causal_text_model: ClassVar[bool | None] = None
  sliding_window_layers: ClassVar[str | None] = None
  checkpoint_key_prefixes: ClassVar[Sequence[str]] = ()
  fused_weight_specs: ClassVar[
      Sequence[FusedWeightSpec | tuple[str, Sequence[str]]]
  ] = ()

  def update_export_config(
      self,
      export_config: exportable_module_config.ExportableModuleConfig,
      model_config: transformers.PretrainedConfig,
  ) -> exportable_module_config.ExportableModuleConfig:
    """Validates or updates `export_config` once `model_config` is loaded.

    Invoked by the export pipeline after `load_model` (once `model_config` is
    available) and before constructing or tracing exportable modules. Override
    this hook to enforce architecture-specific invariants (e.g. requiring
    `externalize_embedder`) or adjust default export flags based on model
    hyperparameters.

    Args:
      export_config: Current `ExportableModuleConfig` for the export run.
      model_config: Loaded HuggingFace `PretrainedConfig` for the target model.

    Returns:
      The updated (or mutated in-place) `ExportableModuleConfig`.
    """
    del model_config
    return export_config

  @contextlib.contextmanager
  def patch_context(self):
    """Enters a class-level runtime patch context around model construction.

    Invoked around `AutoModel*.from_pretrained(...)` / `from_config(...)` so
    extensions can temporarily swap module classes or attention functions on a
    modeling module before submodules are instantiated, restoring originals in a
    `finally` block. For comparable built-in class-level patches, see
    `gemma3_litert_patch` in `model_ext/gemma3/patch.py` and
    `qwen3_litert_patch` in `model_ext/qwen3/patch.py`.

    Yields:
      Control to the model-loading block while class-level patches are active.
    """
    yield

  @contextlib.contextmanager
  def patch_model(
      self,
      model: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
  ):
    """Enters an instance-level patch context around model tracing and export.

    Invoked after `model` is instantiated and before `torch.export` traces its
    forward signatures. Override this hook to swap child modules on `model` in
    place (such as fusing QKV/gate-up projections or inserting HLFB composite
    wrappers conditioned on `export_config`) and restore them in a `finally`
    block. For comparable built-in instance patches, see `patch_gemma3_model`
    in `model_ext/gemma3/patch.py` and `patch_qwen3_model` in
    `model_ext/qwen3/patch.py`.

    Args:
      model: Instantiated PyTorch model or task wrapper about to be traced.
      export_config: Active `ExportableModuleConfig` controlling fusions and
        composites.

    Yields:
      Control to the tracing/conversion block while `model` is patched.
    """
    del model, export_config
    yield

  def get_prefill_decode_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> tuple[Any, Any] | None:
    """Selects custom `(prefill_cls, decode_cls)` exportable module classes.

    Invoked during text-generation and multimodal LLM export when building the
    prefill/decode `.tflite` graph. Override this when the model passes extra
    inputs/outputs (such as per-layer embeddings or hybrid linear-attention
    caches) that are not covered by the generic decoder-only exportables in
    `core/exportable_module.py` or `core/split_cache/exportable_module.py`.
    See `model_ext/qwen3_5/extension.py`, `model_ext/gemma3n/extension.py`, and
    `model_ext/gemma4/extension.py` for built-in examples.

    Args:
      model_config: Loaded `PretrainedConfig` for the checkpoint.
      export_config: Active `ExportableModuleConfig` (inspect `split_cache` and
        `externalize_embedder` to return matching prefill/decode classes).

    Returns:
      A 2-tuple `(prefill_exportable_cls, decode_exportable_cls)` of
      `ExportableModule` subclasses, or `None` to fall back to the core generic
      decoder-only prefill/decode exportables.
    """
    del model_config, export_config
    return None

  def get_vision_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> VisionExportablesSpec | tuple[Any, Any, Any] | None:
    """Selects vision encoder, adapter, and EOI exportable module classes.

    Invoked when `task` is `image_text_to_text` or `multimodal_lm` and
    `export_config.export_vision_encoder` is True. See
    `model_ext/gemma3/extension.py` (encoder + adapter),
    `model_ext/gemma4_unified/extension.py` (encoder + EOI, `adapter_cls=None`),
    and `model_ext/gemma4/extension.py` (encoder + adapter + EOI).

    Args:
      model_config: Loaded `PretrainedConfig` for the multimodal checkpoint.
      export_config: Optional `ExportableModuleConfig` for the export run.

    Returns:
      A `VisionExportablesSpec` (or equivalent 3-tuple
      `(encoder_cls, adapter_cls, eoi_cls)`), or `None` if this extension does
      not support vision export for `model_config`.
    """
    del model_config, export_config
    return None

  def get_audio_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> AudioExportablesSpec | tuple[Any, Any, Any] | None:
    """Selects audio encoder, adapter, and EOA exportable module classes.

    Invoked when `task` is `multimodal_lm` and
    `export_config.export_audio_encoder` is True. See
    `model_ext/gemma4/extension.py` for a built-in implementation returning
    encoder, adapter, and end-of-audio exportables.

    Args:
      model_config: Loaded `PretrainedConfig` for the multimodal checkpoint.
      export_config: Optional `ExportableModuleConfig` for the export run.

    Returns:
      An `AudioExportablesSpec` (or equivalent 3-tuple
      `(encoder_cls, adapter_cls, eoa_cls)`), or `None` if this extension does
      not support audio encoder export for `model_config`.
    """
    del model_config, export_config
    return None

  def get_additional_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> dict[str, Any]:
    """Returns auxiliary `{submodel_name: exportable_cls}` entries to export.

    Invoked during LLM/multimodal export to export extra submodels alongside
    prefill/decode, embedder, vision, and audio graphs. If `submodel_name` is a
    custom section name (anything other than `'per_layer_embedder'`), also
    override `get_litert_lm_section_types()` so `.litertlm` packaging knows
    which section type to assign. See `model_ext/gemma3n/extension.py` and
    `model_ext/gemma4/extension.py` (`'per_layer_embedder'`) for examples.

    Args:
      model_config: Loaded `PretrainedConfig` for the checkpoint.
      export_config: Optional `ExportableModuleConfig` for the export run.

    Returns:
      Dictionary mapping submodel artifact names to `ExportableModule`
      subclasses; empty dict `{}` when no auxiliary submodels are needed.
    """
    del model_config, export_config
    return {}

  def get_speech_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> tuple[Any, ...] | None:
    """Selects exportable module classes for `automatic_speech_recognition`.

    Invoked during ASR export to determine which subgraphs to trace. When
    `None` is returned, `model_ext/exportables.py` falls back to the standard
    encoder-decoder pair `(LiteRTExportableModuleForAsrEncode,
    LiteRTExportableModuleForAsrDecode)`. Override this for encoder-only CTC
    models; see `ParakeetCtcExtension` in `model_ext/parakeet/extension.py`.

    Args:
      model_config: Loaded `PretrainedConfig` for the speech model.
      export_config: Optional `ExportableModuleConfig` for the export run.

    Returns:
      Tuple of ASR `ExportableModule` subclasses, or `None` to use the default
      ASR encode + decode exportables.
    """
    del model_config, export_config
    return None

  def get_speech_model_cls(self, model_type: str) -> Any | None:
    """Returns the ASR model wrapper class for `automatic_speech_recognition`.

    Invoked during `load_model` when `task='automatic_speech_recognition'`.
    See `model_ext/whisper/extension.py`, `model_ext/moonshine/extension.py`,
    `model_ext/parakeet/extension.py`, and `Qwen3AsrExtension` in
    `model_ext/qwen3/extension.py` for built-in examples.

    Args:
      model_type: Resolved `model_type` string of the speech checkpoint.

    Returns:
      An ASR model wrapper class (subclass of `core.speech.asr_model.AsrModel`),
      or `None` if unsupported by this extension.
    """
    del model_type
    return None

  def get_tts_model_cls(self, model_type: str) -> Any | None:
    """Returns the TTS model wrapper class for `text_to_speech` export.

    Invoked during `load_model` when `task='text_to_speech'`. See
    `model_ext/kokoro_tts/extension.py` and `model_ext/qwen3_tts/extension.py`
    for built-in examples.

    Args:
      model_type: Resolved `model_type` string of the TTS checkpoint.

    Returns:
      A TTS model wrapper class (subclass of `core.tts.tts_model.TtsModel`), or
      `None` if unsupported by this extension.
    """
    del model_type
    return None

  @classmethod
  def is_matching_tts_checkpoint(
      cls, config_dict: dict[str, Any], model_path: str
  ) -> bool:
    """Identifies a `text_to_speech` checkpoint lacking `config.model_type`.

    Called by `match_checkpoint` when `task == 'text_to_speech'` and a
    checkpoint's `config.json` is missing or omits `model_type` (common for
    standalone TTS repositories). See `model_ext/kokoro_tts/extension.py` for a
    built-in example.

    Args:
      config_dict: Parsed `config.json` dictionary (empty `{}` if absent).
      model_path: HuggingFace repo ID or local checkpoint directory path.

    Returns:
      True if `(config_dict, model_path)` belongs to this TTS architecture.
    """
    del config_dict, model_path
    return False

  @classmethod
  def match_checkpoint(
      cls,
      model_path: str,
      task: Any = None,
      config_dict: dict[str, Any] | None = None,
  ) -> bool:
    """Identifies checkpoints that do not declare a standard HF `model_type`.

    Invoked during `load_model` for tasks whose checkpoints may use Diffusers
    `model_index.json` or non-Transformers `config.json` layouts (such as
    `text_to_image`, `text_to_speech`, and `any_to_any`). When True, the first
    entry in `cls.model_types` is used as the inferred `model_type`. By default,
    delegates `text_to_speech` checks to `is_matching_tts_checkpoint`. See
    `model_ext/bonsai_flux2/extension.py` and
    `model_ext/flux2_klein/extension.py` for built-in examples.

    Args:
      model_path: HuggingFace repo ID or local checkpoint directory path.
      task: Export task enum or string (e.g. `'text_to_image'`).
      config_dict: Parsed JSON from `model_index.json` or `config.json`, or
        `None` if neither file exists.

    Returns:
      True if this extension handles the given checkpoint.
    """
    task_name = getattr(task, 'value', task)
    if config_dict is not None and task_name == 'text_to_speech':
      return cls.is_matching_tts_checkpoint(config_dict, model_path)
    return False

  def matches_checkpoint(
      self,
      model_path: str,
      task: Any = None,
      config_dict: dict[str, Any] | None = None,
  ) -> bool:
    """Instance-method dispatcher for `match_checkpoint`.

    Used by `CompositeModelExtension` and `infer_model_type_from_extensions` so
    both class-level heuristics and stateful extension instances can match
    checkpoints uniformly.

    Args:
      model_path: HuggingFace repo ID or local checkpoint directory path.
      task: Export task enum or string.
      config_dict: Optional parsed config/index dictionary.

    Returns:
      True if this extension matches the checkpoint.
    """
    return self.match_checkpoint(model_path, task=task, config_dict=config_dict)

  def get_image_gen_model_cls(self, model_type: str) -> Any | None:
    """Returns the ImageGen wrapper class for `text_to_image` export.

    Invoked during `load_model` when `task='text_to_image'`. See
    `model_ext/bonsai_flux2/extension.py` and
    `model_ext/flux2_klein/extension.py` for built-in examples.

    Args:
      model_type: Resolved `model_type` or Diffusers `_class_name` string.

    Returns:
      An image generation wrapper class (subclass of
      `core.image_gen.image_gen_model.ImageGenModel`), or `None` if unsupported.
    """
    del model_type
    return None

  def get_any2any_model_cls(self, model_type: str) -> Any | None:
    """Returns the Any2Any wrapper class for `any_to_any` multimodal export.

    Invoked during `load_model` when `task='any_to_any'`. See
    `model_ext/minicpm_o/extension.py` for a built-in example.

    Args:
      model_type: Resolved `model_type` string of the checkpoint.

    Returns:
      An Any2Any model wrapper class, or `None` if unsupported.
    """
    del model_type
    return None

  def get_export_defaults(
      self,
      config_dict: Mapping[str, Any],
      task: str | None = None,
  ) -> ExportDefaults | None:
    """Returns recommended export settings for a checkpoint.

    Invoked by `auto_export.resolve` before export, using only the raw
    `config.json` (or Diffusers `model_index.json`) so no model is loaded.
    Override it to declare the default task and the export arguments known to
    work for the architecture; callers merge them under their own explicit
    arguments.

    Args:
      config_dict: Parsed `config.json` (or `model_index.json`) contents, or an
        empty dict if the checkpoint has none.
      task: Task requested by the caller, or `None` when the task is detected
        automatically. Return arguments that suit this task (for example, drop
        vision-encoder arguments when a VLM is exported as `text_generation`).

    Returns:
      An `ExportDefaults`, or `None` if the extension has no recommendation.
    """
    del config_dict, task
    return None

  def get_speech2text_bundle_spec(
      self, model_config: transformers.PretrainedConfig
  ) -> Speech2TextBundleSpec | None:
    """Returns how to package an encoder-decoder speech-to-text model.

    Invoked for `automatic_speech_recognition` exports with
    `bundle_litert_lm=True`. Return a spec for models exported as one TFLite
    model by `export_asr_models` (Whisper, Moonshine, Parakeet); return `None`
    (the default) for audio-encoder + LLM models such as Qwen3-ASR, which use
    the LLM packaging path.

    Args:
      model_config: Loaded model configuration.

    Returns:
      A `Speech2TextBundleSpec`, or `None`.
    """
    del model_config
    return None

  def get_litert_lm_section_types(self) -> dict[str, Any]:
    """Maps auxiliary submodel names to `.litertlm` container section types.

    Invoked by `core/litert_lm_builder.py` when packaging
    `additional_model_paths` into a `.litertlm` archive (for LLM, ImageGen, and
    Any2Any tasks). Values may be `LlmMetadata` section enum values or
    section-type strings accepted by `LitertLmBuilder.add_tflite_model`. See
    `model_ext/bonsai_flux2/extension.py` and
    `model_ext/flux2_klein/extension.py` for built-in examples.

    Returns:
      Dictionary `{submodel_name: section_type}`; empty dict `{}` by default.
    """
    return {}

  def get_llm_model_type_proto(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> Any | None:
    """Returns a custom `LlmModelType` protobuf message for `.litertlm` metadata.

    Invoked by `core/litert_lm_builder.py` when populating
    `llm_metadata.llm_model_type` (unless `litert_lm_model_type_override` is
    set on `export_config`). Returning `None` falls back to the built-in
    `model_type` switch (which defaults unknown architectures to
    `GenericModel`).

    Args:
      model_config: Loaded `PretrainedConfig` for the checkpoint.
      export_config: Active `ExportableModuleConfig`.

    Returns:
      An `llm_model_type_pb2.LlmModelType` message, or `None` to use core's
      default `LlmModelType` selection.
    """
    del model_config, export_config
    return None

  def build_llm_metadata(
      self,
      source_model_artifacts: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
      exported_model_artifacts: Any,
      llm_metadata: Any,
  ) -> Any:
    """Customizes the `LlmMetadata` protobuf during `.litertlm` packaging.

    Invoked by `core/litert_lm_builder.py` after base stop tokens, prompt
    templates, and sampler parameters are populated. Override this hook to set
    modality token IDs, vision/audio patch parameters, or architecture-specific
    prompt formatting. See `model_ext/gemma3/extension.py`,
    `model_ext/gemma3n/extension.py`, and `model_ext/gemma4/extension.py` for
    built-in examples.

    Args:
      source_model_artifacts: `SourceModelArtifacts` containing `model`,
        `model_config`, `tokenizer`, and processor objects.
      export_config: Active `ExportableModuleConfig`.
      exported_model_artifacts: `ExportedModelArtifacts` with paths to all
        exported `.tflite` and tokenizer files.
      llm_metadata: Pre-populated `llm_metadata_pb2.LlmMetadata` proto.

    Returns:
      The updated `LlmMetadata` protobuf message.
    """
    del source_model_artifacts, export_config, exported_model_artifacts
    return llm_metadata

  def get_executor_metadata_builder(
      self,
      model_config: transformers.PretrainedConfig,
  ) -> Callable[..., Any] | None:
    """Returns a custom builder for `ExecutorMetadata` state-buffer descriptors.

    Invoked by `model_ext/metadata_builder.py` when packaging a `.litertlm`
    bundle. Returning `None` uses the default `build_executor_metadata`
    implementation in `model_ext/metadata_builder.py`, which automatically
    derives global KV, sliding-window local KV, `linear_attention`, and `conv`
    `StateBuffer` entries from `model_config.layer_types`. Override only when a
    model uses custom state tensor names or buffer policies.

    Args:
      model_config: Loaded `PretrainedConfig` for the checkpoint.

    Returns:
      A callable `(source_model_artifacts, export_config,
      exported_model_artifacts, executor_metadata=None) -> ExecutorMetadata`,
      or `None` to use the standard state-buffer builder.
    """
    del model_config
    return None

  def resolve_checkpoint_tensor(
      self,
      param_name: str,
      try_load: Callable[[str], Any],
  ) -> Any | None:
    """Resolves a custom parameter tensor during Converter V2 weight streaming.

    Invoked by `core.weights_loader.HFCheckpointWeightsLoader` (`use_v2=True`
    with `delete_in_memory_params=True`) before standard direct-name and
    `fused_weight_specs` lookups. Use this hook when a patched module splits,
    slices, transposes, or rescales checkpoint tensors in ways not expressible
    via `checkpoint_key_prefixes` or `fused_weight_specs` (compare
    `_MiniCpmOLlmWeightsLoader` in `model_ext/minicpm_o/minicpm_o.py`).

    Args:
      param_name: Export module parameter name requested by the converter.
      try_load: Helper callable `(key_or_param_name) -> Tensor | None` that
        searches candidate checkpoint prefixes in safetensors shards.

    Returns:
      A materialized `torch.Tensor` if handled by this extension, or `None` to
      fall back to standard `HFCheckpointWeightsLoader` resolution.
    """
    del param_name, try_load
    return None

  def load_config(
      self,
      model_path: str,
      export_config: exportable_module_config.ExportableModuleConfig,
      trust_remote_code: bool = False,
  ) -> transformers.PretrainedConfig | None:
    """Loads a custom `PretrainedConfig` for out-of-tree architectures.

    Invoked by `core/export_lib.py` before `transformers.AutoConfig` when an
    active extension is resolved from `export_config` or `config.json`
    `model_type`. Override this when exporting a new or custom architecture
    whose config class is not registered in `transformers` and does not ship
    `auto_map` remote code.

    Args:
      model_path: Local checkpoint directory or HuggingFace repo ID.
      export_config: Active `ExportableModuleConfig`.
      trust_remote_code: Whether remote code execution is enabled.

    Returns:
      A `transformers.PretrainedConfig` instance, or `None` to use standard
      `AutoConfig.from_pretrained` loading.
    """
    del model_path, export_config, trust_remote_code
    return None

  def get_hf_model_cls(
      self,
      task: exportable_module_config.ExportTask | str,
      model_config: transformers.PretrainedConfig,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> Any | None:
    """Returns a custom PyTorch/HF model class for LLM or multimodal tasks.

    Invoked by `core/export_lib.py` during `load_model` for `text_generation`,
    `image_text_to_text`, and `multimodal_lm` exports before falling back to
    `transformers.AutoModelForCausalLM` / `AutoModelForImageTextToText`. The
    returned class may be:
    - An `AutoModel*`-style class exposing `.from_pretrained(...)` and
      `.from_config(...)`,
    - A `transformers.PreTrainedModel` subclass (loaded via `.from_pretrained`
      or `._from_config`), or
    - A plain `torch.nn.Module` subclass constructible as `cls(model_config)`
      (weights are loaded automatically from `.safetensors` checkpoints via
      `HFCheckpointWeightsLoader` when `delete_in_memory_params=False`, or
      streamed on meta device when `use_v2=True` and
      `delete_in_memory_params=True`).

    Args:
      task: Active export task (e.g. `ExportTask.TEXT_GENERATION`).
      model_config: Loaded `PretrainedConfig`.
      export_config: Active `ExportableModuleConfig`.

    Returns:
      A custom model class to instantiate, or `None` to use standard
      `transformers.AutoModelFor*` classes.
    """
    del task, model_config, export_config
    return None


class CompositeModelExtension(ModelExtension):
  """Chains multiple `ModelExtension` instances in priority order.

  Used when layering an explicit `export_config.model_extension` override on
  top of a registered built-in extension (or when passing a list of extensions).
  Composition semantics by hook category:
  - **Context managers** (`patch_context`, `patch_model`): Entered in forward
    order via `contextlib.ExitStack` (outermost = first extension, innermost =
    last extension).
  - **Sequential transforms** (`update_export_config`, `build_llm_metadata`):
    Piped in forward order so later extensions refine earlier results.
  - **Single-winner selectors** (`get_prefill_decode_exportables`,
    `get_vision_exportables`, `get_audio_exportables`, `get_*_model_cls`,
    `get_llm_model_type_proto`, `get_executor_metadata_builder`): Queried in
    reverse order so the **last** extension returning non-`None` wins.
  - **Map mergers** (`get_additional_exportables`,
    `get_litert_lm_section_types`): Merged in forward order (`dict.update`) so
    later extensions can add keys or override earlier keys.
  """

  def __init__(self, extensions: Sequence[ModelExtension]):
    """Initializes a composite over `extensions` in low-to-high priority order."""
    self._extensions = tuple(extensions)

  @property
  def extensions(self) -> tuple[ModelExtension, ...]:
    """Returns the ordered tuple of underlying `ModelExtension` instances."""
    return self._extensions

  @property
  def is_causal_text_model(self) -> bool | None:  # pyrefly: ignore[bad-override]
    """Returns the highest-priority non-`None` `is_causal_text_model` flag."""
    for ext in reversed(self._extensions):
      if ext.is_causal_text_model is not None:
        return ext.is_causal_text_model
    return None

  @property
  def sliding_window_layers(self) -> str | None:  # pyrefly: ignore[bad-override]
    """Returns the highest-priority non-`None` `sliding_window_layers` policy."""
    for ext in reversed(self._extensions):
      if ext.sliding_window_layers is not None:
        return ext.sliding_window_layers
    return None

  @property
  def extra_kwargs_keys(self) -> tuple[str, ...]:  # pyrefly: ignore[bad-override]
    """Returns the deduplicated union of `extra_kwargs_keys` across members."""
    keys: list[str] = []
    for ext in self._extensions:
      for key in ext.extra_kwargs_keys:
        if key and key not in keys:
          keys.append(key)
    return tuple(keys)

  @property
  def checkpoint_key_prefixes(self) -> tuple[str, ...]:  # pyrefly: ignore[bad-override]
    """Returns the deduplicated union of `checkpoint_key_prefixes`."""
    prefixes: list[str] = []
    for ext in self._extensions:
      for prefix in ext.checkpoint_key_prefixes:
        if prefix and prefix not in prefixes:
          prefixes.append(prefix)
    return tuple(prefixes)

  @property
  def fused_weight_specs(  # pyrefly: ignore[bad-override]
      self,
  ) -> tuple[FusedWeightSpec | tuple[str, Sequence[str]], ...]:
    """Returns `fused_weight_specs` ordered from highest to lowest priority."""
    specs: list[FusedWeightSpec | tuple[str, Sequence[str]]] = []
    for ext in reversed(self._extensions):
      specs.extend(ext.fused_weight_specs)
    return tuple(specs)

  def update_export_config(
      self,
      export_config: exportable_module_config.ExportableModuleConfig,
      model_config: transformers.PretrainedConfig,
  ) -> exportable_module_config.ExportableModuleConfig:
    """See base class."""
    for ext in self._extensions:
      export_config = ext.update_export_config(export_config, model_config)
    return export_config

  @contextlib.contextmanager
  def patch_context(self):
    """See base class."""
    with contextlib.ExitStack() as stack:
      for ext in self._extensions:
        stack.enter_context(ext.patch_context())
      yield

  @contextlib.contextmanager
  def patch_model(
      self,
      model: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
  ):
    """See base class."""
    with contextlib.ExitStack() as stack:
      for ext in self._extensions:
        stack.enter_context(ext.patch_model(model, export_config))
      yield

  def get_prefill_decode_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> tuple[Any, Any] | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_prefill_decode_exportables(model_config, export_config)
      if res is not None:
        return res
    return None

  def get_vision_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> VisionExportablesSpec | tuple[Any, Any, Any] | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_vision_exportables(model_config, export_config)
      if res is not None:
        return res
    return None

  def get_audio_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> AudioExportablesSpec | tuple[Any, Any, Any] | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_audio_exportables(model_config, export_config)
      if res is not None:
        return res
    return None

  def get_additional_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> dict[str, Any]:
    """See base class."""
    merged: dict[str, Any] = {}
    for ext in self._extensions:
      merged.update(ext.get_additional_exportables(model_config, export_config))
    return merged

  def get_speech_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ) -> tuple[Any, ...] | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_speech_exportables(model_config, export_config)
      if res is not None:
        return res
    return None

  def get_speech_model_cls(self, model_type: str) -> Any | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_speech_model_cls(model_type)
      if res is not None:
        return res
    return None

  def get_tts_model_cls(self, model_type: str) -> Any | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_tts_model_cls(model_type)
      if res is not None:
        return res
    return None

  def get_image_gen_model_cls(self, model_type: str) -> Any | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_image_gen_model_cls(model_type)
      if res is not None:
        return res
    return None

  def get_any2any_model_cls(self, model_type: str) -> Any | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_any2any_model_cls(model_type)
      if res is not None:
        return res
    return None

  def get_export_defaults(
      self,
      config_dict: Mapping[str, Any],
      task: str | None = None,
  ) -> ExportDefaults | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_export_defaults(config_dict, task)
      if res is not None:
        return res
    return None

  def get_speech2text_bundle_spec(
      self, model_config: transformers.PretrainedConfig
  ) -> Speech2TextBundleSpec | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_speech2text_bundle_spec(model_config)
      if res is not None:
        return res
    return None

  def matches_checkpoint(
      self,
      model_path: str,
      task: Any = None,
      config_dict: dict[str, Any] | None = None,
  ) -> bool:
    """See base class."""
    return any(
        ext.matches_checkpoint(model_path, task=task, config_dict=config_dict)
        for ext in self._extensions
    )

  def get_litert_lm_section_types(self) -> dict[str, Any]:
    """See base class."""
    merged: dict[str, Any] = {}
    for ext in self._extensions:
      merged.update(ext.get_litert_lm_section_types())
    return merged

  def get_llm_model_type_proto(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> Any | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_llm_model_type_proto(model_config, export_config)
      if res is not None:
        return res
    return None

  def build_llm_metadata(
      self,
      source_model_artifacts: Any,
      export_config: exportable_module_config.ExportableModuleConfig,
      exported_model_artifacts: Any,
      llm_metadata: Any,
  ) -> Any:
    """See base class."""
    for ext in self._extensions:
      llm_metadata = ext.build_llm_metadata(
          source_model_artifacts,
          export_config,
          exported_model_artifacts,
          llm_metadata,
      )
    return llm_metadata

  def get_executor_metadata_builder(
      self,
      model_config: transformers.PretrainedConfig,
  ) -> Callable[..., Any] | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_executor_metadata_builder(model_config)
      if res is not None:
        return res
    return None

  def resolve_checkpoint_tensor(
      self,
      param_name: str,
      try_load: Callable[[str], Any],
  ) -> Any | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.resolve_checkpoint_tensor(param_name, try_load)
      if res is not None:
        return res
    return None

  def load_config(
      self,
      model_path: str,
      export_config: exportable_module_config.ExportableModuleConfig,
      trust_remote_code: bool = False,
  ) -> transformers.PretrainedConfig | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.load_config(
          model_path, export_config, trust_remote_code=trust_remote_code
      )
      if res is not None:
        return res
    return None

  def get_hf_model_cls(
      self,
      task: exportable_module_config.ExportTask | str,
      model_config: transformers.PretrainedConfig,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> Any | None:
    """See base class."""
    for ext in reversed(self._extensions):
      res = ext.get_hf_model_cls(task, model_config, export_config)
      if res is not None:
        return res
    return None


def compose_extensions(
    *extensions: ModelExtension | type[ModelExtension] | None,
) -> ModelExtension:
  """Normalizes and composes one or more `ModelExtension` classes or instances.

  Instantiates any `ModelExtension` subclasses, flattens nested
  `CompositeModelExtension` instances, skips `None` entries, and returns either
  a single `ModelExtension` or a `CompositeModelExtension`.

  Args:
    *extensions: Zero or more `ModelExtension` subclasses, instances, or `None`
      ordered from lowest to highest override priority.

  Returns:
    A normalized `ModelExtension` (base no-op `ModelExtension()` when all
    inputs are `None`).

  Raises:
    TypeError: If any non-`None` argument is neither a `ModelExtension`
      subclass nor a `ModelExtension` instance.
  """
  normalized: list[ModelExtension] = []
  for ext in extensions:
    if ext is None:
      continue
    if isinstance(ext, type) and issubclass(ext, ModelExtension):
      ext = ext()
    if isinstance(ext, CompositeModelExtension):
      normalized.extend(ext.extensions)
    elif isinstance(ext, ModelExtension):
      normalized.append(ext)
    else:
      raise TypeError(
          'Expected ModelExtension subclass or instance, got'
          f' {type(ext).__name__}.'
      )
  if not normalized:
    return ModelExtension()
  if len(normalized) == 1:
    return normalized[0]
  return CompositeModelExtension(normalized)


# Map of model_type -> ModelExtension instance.
_EXTENSION_REGISTRY: dict[str, ModelExtension] = {}
_ENTRY_POINTS_LOADED = False


def _sync_registered_extra_kwargs() -> None:
  """Synchronizes `exportable_module_config` extra_kwargs keys with registry."""
  keys: list[str] = []
  for ext in _EXTENSION_REGISTRY.values():
    for key in getattr(ext, 'extra_kwargs_keys', ()):
      if key and key not in keys:
        keys.append(key)
  exportable_module_config.reset_registered_extra_kwargs(keys)


def _coerce_extension(
    ext: ModelExtension | type[ModelExtension],
) -> ModelExtension:
  """Returns an instantiated `ModelExtension` from a class or instance."""
  if isinstance(ext, type) and issubclass(ext, ModelExtension):
    return ext()
  if isinstance(ext, ModelExtension):
    return ext
  raise TypeError(
      f'Expected ModelExtension subclass or instance, got {type(ext).__name__}.'
  )


def register_model_extension(
    model_types: str | Sequence[str] | None = None,
    *,
    override: bool = True,
):
  """Decorator or function to register a `ModelExtension` class or instance.

  Built-in extensions in `model_ext/<arch>/extension.py` register with
  `override=False` so out-of-tree extensions registered before or after import
  can override them with the default `override=True`.

  Usage as a decorator:
    ```python
    @register_model_extension(['my_arch', 'my_arch_text'])
    class MyArchExtension(ModelExtension):
      ...
    ```

  Or using `model_types` defined on the class:
    ```python
    @register_model_extension()
    class MyArchExtension(ModelExtension):
      model_types = ('my_arch',)
    ```

  Args:
    model_types: Single `model_type` string or sequence of `model_type`
      strings. If `None`, reads `cls.model_types`.
    override: If True (default), overwrites any existing registration for the
      same `model_type`; if False, preserves existing registrations.

  Returns:
    A decorator that registers the extension and returns the original class or
    instance unchanged.
  """

  def wrapper(ext_cls_or_instance: ModelExtension | type[ModelExtension]):
    instance = _coerce_extension(ext_cls_or_instance)
    types_to_register = model_types
    if types_to_register is None:
      types_to_register = getattr(ext_cls_or_instance, 'model_types', ())
    if isinstance(types_to_register, str):
      types_to_register = (types_to_register,)
    if not types_to_register:
      raise ValueError(
          'register_model_extension requires non-empty model_types either via'
          ' argument or class attribute `model_types`.'
      )
    registered_any = False
    for mt in types_to_register:
      if not override and mt in _EXTENSION_REGISTRY:
        continue
      _EXTENSION_REGISTRY[mt] = instance
      registered_any = True
    if registered_any and getattr(instance, 'extra_kwargs_keys', ()):
      exportable_module_config.register_extra_kwargs_keys(
          instance.extra_kwargs_keys
      )
    return ext_cls_or_instance

  return wrapper


def unregister_model_extension(model_type: str) -> ModelExtension | None:
  """Removes and returns the registered `ModelExtension` for `model_type`.

  Also resynchronizes `ExportableModuleConfig`'s allowed `extra_kwargs_keys`
  when the removed extension declared custom flags. Primarily used in unit
  tests to clean up temporary registrations.

  Args:
    model_type: Registry key to remove.

  Returns:
    The removed `ModelExtension` instance, or `None` if `model_type` was not
    registered.
  """
  removed = _EXTENSION_REGISTRY.pop(model_type, None)
  if removed is not None and getattr(removed, 'extra_kwargs_keys', ()):
    _sync_registered_extra_kwargs()
  return removed


def registered_model_types() -> tuple[str, ...]:
  """Returns the tuple of `model_type` keys currently in the registry."""
  return tuple(_EXTENSION_REGISTRY.keys())


def is_extension_file_path(spec: str) -> bool:
  """Returns True if `spec` is a filesystem `.py` path rather than module name.

  Args:
    spec: Single entry from `extension_modules` (e.g. `'./my_ext.py'` or
      `'my_pkg.my_ext'`).

  Returns:
    True when `spec` should be loaded via `load_extension_file` instead of
    `importlib.import_module`.
  """
  if not spec:
    return False
  if spec.endswith('.py') or spec.startswith(('.', '~')):
    return True
  if os.sep in spec or (os.altsep and os.altsep in spec):
    return True
  return os.path.isfile(os.path.expanduser(spec))


def load_extension_file(file_path: str) -> types.ModuleType:
  """Loads a standalone Python extension `.py` file idempotently by path.

  Assigns a deterministic synthetic module name derived from the canonical
  absolute path so repeated calls return the cached module in `sys.modules`
  without re-executing top-level registration side effects.

  Args:
    file_path: Relative, `~`-prefixed, or absolute path to an extension `.py`
      file.

  Returns:
    The imported `types.ModuleType` object.

  Raises:
    FileNotFoundError: If `file_path` does not exist on disk.
    ImportError: If `importlib` cannot construct a module spec for `file_path`.
  """
  abs_path = os.path.abspath(os.path.expanduser(file_path))
  if not os.path.isfile(abs_path):
    raise FileNotFoundError(
        f'Extension module file not found: {file_path!r} (resolved to'
        f' {abs_path!r}).'
    )
  stem = os.path.splitext(os.path.basename(abs_path))[0]
  safe_stem = ''.join(
      ch if ch.isidentifier() or ch.isdigit() else '_' for ch in stem
  )
  safe_stem = safe_stem or 'ext'
  digest = hashlib.sha256(abs_path.encode('utf-8')).hexdigest()[:12]
  mod_name = f'_litert_torch_ext_file_{safe_stem}_{digest}'
  existing = sys.modules.get(mod_name)
  if existing is not None:
    return existing

  spec = importlib_util.spec_from_file_location(mod_name, abs_path)
  if spec is None or spec.loader is None:
    raise ImportError(
        f'Could not create module spec from extension file: {abs_path!r}.'
    )
  module = importlib_util.module_from_spec(spec)
  sys.modules[mod_name] = module
  try:
    spec.loader.exec_module(module)
  except Exception:
    sys.modules.pop(mod_name, None)
    raise
  return module


def _auto_register_module_extensions(mod: types.ModuleType) -> None:
  """Registers any `ModelExtension` subclasses with `model_types` in `mod`."""
  for obj in vars(mod).values():
    if (
        isinstance(obj, type)
        and issubclass(obj, ModelExtension)
        and obj is not ModelExtension
        and obj is not CompositeModelExtension
        and getattr(obj, '__module__', None) == mod.__name__
        and getattr(obj, 'model_types', ())
    ):
      register_model_extension(override=True)(obj)


def load_extension_modules(
    modules: Sequence[str] | str | None,
) -> None:
  """Imports dotted Python modules or `.py` files to register extensions.

  Invoked automatically when `--extension_modules` /
  `ExportableModuleConfig.extension_modules` is provided. After importing each
  module or standalone `.py` file, any `ModelExtension` subclass defined in
  that module with non-empty `model_types` is also auto-registered with
  `override=True` so user-supplied extension modules take precedence over
  built-in registrations.

  Args:
    modules: Comma-separated string or sequence of importable dotted Python
      module names or filesystem paths to `.py` files, or `None` for a no-op.
  """
  if not modules:
    return
  if isinstance(modules, str):
    module_list = [m.strip() for m in modules.split(',') if m.strip()]
  else:
    module_list = [m.strip() for m in modules if m and m.strip()]
  for module_spec in module_list:
    if is_extension_file_path(module_spec):
      mod = load_extension_file(module_spec)
    else:
      mod = importlib.import_module(module_spec)
    if isinstance(mod, types.ModuleType):
      _auto_register_module_extensions(mod)


def discover_entry_point_extensions(reload_entry_points: bool = False) -> None:
  """Discovers and loads wheel plugins from `importlib.metadata` entry points."""
  _load_entry_point_extensions(reload_entry_points=reload_entry_points)


def _load_entry_point_extensions(reload_entry_points: bool = False) -> None:
  """Discovers and loads wheel plugins from `importlib.metadata` entry points."""
  global _ENTRY_POINTS_LOADED
  if _ENTRY_POINTS_LOADED and not reload_entry_points:
    return
  _ENTRY_POINTS_LOADED = True
  try:
    eps = importlib_metadata.entry_points(group=ENTRY_POINT_GROUP)
  except Exception:  # pylint: disable=broad-exception-caught
    return

  for ep in eps:
    try:
      loaded = ep.load()
      if isinstance(loaded, type) and issubclass(loaded, ModelExtension):
        if ep.name and ep.name not in _EXTENSION_REGISTRY:
          register_model_extension(ep.name, override=False)(loaded)
        elif getattr(loaded, 'model_types', ()):
          register_model_extension(override=False)(loaded)
      elif isinstance(loaded, ModelExtension):
        if ep.name and ep.name not in _EXTENSION_REGISTRY:
          register_model_extension(ep.name, override=False)(loaded)
        elif getattr(loaded, 'model_types', ()):
          register_model_extension(override=False)(loaded)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Warning: failed to load model extension entry point {ep}: {e}')


def _try_lazy_import_builtin(model_type: str) -> None:
  """Attempts to lazily import built-in `model_ext.<model_type>` modules."""
  if not model_type or not model_type.isidentifier():
    return
  base_pkg = f'litert_torch.generative.export_hf.model_ext.{model_type}'
  for suffix in ('extension', 'patch', ''):
    mod_name = f'{base_pkg}.{suffix}' if suffix else base_pkg
    try:
      importlib.import_module(mod_name)
    except ImportError:
      continue


def get_model_extension(model_type: str | None) -> ModelExtension | None:
  """Looks up a registered `ModelExtension` by `model_type`.

  Checks the in-memory registry first, then attempts lazy built-in module
  import (`model_ext.<model_type>.extension`), and finally scans installed
  Python wheel entry points under `litert_torch.model_extensions`.

  Args:
    model_type: Target `model_type` string, or `None`.

  Returns:
    The registered `ModelExtension` instance, or `None` if no extension is
    registered for `model_type`.
  """
  if not model_type:
    return None
  if model_type in _EXTENSION_REGISTRY:
    return _EXTENSION_REGISTRY[model_type]
  _try_lazy_import_builtin(model_type)
  if model_type in _EXTENSION_REGISTRY:
    return _EXTENSION_REGISTRY[model_type]
  _load_entry_point_extensions()
  return _EXTENSION_REGISTRY.get(model_type)


def _extract_model_types(
    model_type_or_config: str | transformers.PretrainedConfig | None,
) -> list[str]:
  """Extracts candidate `model_type` strings from a config or string."""
  if model_type_or_config is None:
    return []
  if isinstance(model_type_or_config, str):
    return [model_type_or_config] if model_type_or_config else []
  candidates: list[str] = []
  top_mt = getattr(model_type_or_config, 'model_type', None)
  if isinstance(top_mt, str) and top_mt:
    candidates.append(top_mt)
  text_cfg = getattr(model_type_or_config, 'text_config', None)
  if text_cfg is not None:
    text_mt = getattr(text_cfg, 'model_type', None)
    if isinstance(text_mt, str) and text_mt and text_mt not in candidates:
      candidates.append(text_mt)
  return candidates


def resolve_model_extension(
    model_type_or_config: str | transformers.PretrainedConfig | None = None,
    export_config: (
        exportable_module_config.ExportableModuleConfig | None
    ) = None,
) -> ModelExtension | None:
  """Resolves the active `ModelExtension` for a model and export config.

  Resolution order:
  1. Loads any `export_config.extension_modules` if specified.
  2. Checks registered extensions for `model_config.model_type` (and
     `model_config.text_config.model_type` if present).
  3. If `export_config.model_extension` is provided:
     - If a registered extension was also found, composes
       `(registered_ext, explicit_ext)` so the explicit extension overrides any
       hooks it implements while inheriting others.
     - Otherwise returns the explicit `ModelExtension` (or
       `CompositeModelExtension` if a list/tuple was passed).

  Args:
    model_type_or_config: Either a `model_type` string or a `PretrainedConfig`.
    export_config: Optional `ExportableModuleConfig` carrying `model_extension`
      and/or `extension_modules`.

  Returns:
    A `ModelExtension` instance, or `None` if no extension applies.
  """
  if export_config is not None and getattr(
      export_config, 'extension_modules', None
  ):
    load_extension_modules(export_config.extension_modules)

  found_extensions: list[ModelExtension] = []
  for mt in _extract_model_types(model_type_or_config):
    ext = get_model_extension(mt)
    if ext is not None and ext not in found_extensions:
      found_extensions.append(ext)

  explicit_ext = (
      getattr(export_config, 'model_extension', None)
      if export_config is not None
      else None
  )
  if explicit_ext is not None:
    if isinstance(explicit_ext, (list, tuple)):
      explicit_composed = compose_extensions(*explicit_ext)
    else:
      explicit_composed = compose_extensions(explicit_ext)
    found_extensions.append(explicit_composed)

  if not found_extensions:
    return None
  return compose_extensions(*found_extensions)


def infer_model_type_from_extensions(
    model_path: str,
    task: Any = None,
    config_dict: dict[str, Any] | None = None,
    export_config: (
        exportable_module_config.ExportableModuleConfig | None
    ) = None,
) -> str | None:
  """Infers `model_type` for checkpoints without a standard HF `model_type`.

  Queries `matches_checkpoint` on any explicit `export_config.model_extension`
  first, then iterates over registered extensions (including wheel entry
  points). Used for Diffusers pipelines and standalone TTS/Any2Any checkpoints;
  see `model_ext/bonsai_flux2/extension.py`,
  `model_ext/flux2_klein/extension.py`, and
  `model_ext/kokoro_tts/extension.py`.

  Args:
    model_path: HuggingFace repo ID or local checkpoint directory path.
    task: Export task enum or string (e.g. `'text_to_image'`).
    config_dict: Optional parsed dictionary from `model_index.json` or
      `config.json`.
    export_config: Optional `ExportableModuleConfig` carrying `model_extension`
      or `extension_modules`.

  Returns:
    The inferred `model_type` string, or `None` if no extension matched.
  """
  task_val = getattr(task, 'value', task)
  if export_config is not None and getattr(
      export_config, 'extension_modules', None
  ):
    load_extension_modules(export_config.extension_modules)
  _load_entry_point_extensions()
  explicit_ext = (
      getattr(export_config, 'model_extension', None)
      if export_config is not None
      else None
  )
  if explicit_ext is not None:
    ext = compose_extensions(*(
        explicit_ext
        if isinstance(explicit_ext, (list, tuple))
        else (explicit_ext,)
    ))
    if ext.matches_checkpoint(
        model_path, task=task_val, config_dict=config_dict
    ):
      if ext.model_types:
        return ext.model_types[0]
  for model_type, ext in _EXTENSION_REGISTRY.items():
    if ext.matches_checkpoint(
        model_path, task=task_val, config_dict=config_dict
    ):
      return model_type
  return None


def update_export_config(
    export_config: exportable_module_config.ExportableModuleConfig,
    model_config: transformers.PretrainedConfig,
) -> exportable_module_config.ExportableModuleConfig:
  """Resolves the active `ModelExtension` and runs `update_export_config`.

  Args:
    export_config: Current `ExportableModuleConfig` for the export run.
    model_config: Loaded `PretrainedConfig` for the target checkpoint.

  Returns:
    The updated `ExportableModuleConfig` (unchanged if no extension applies).
  """
  ext = resolve_model_extension(model_config, export_config)
  if ext is not None:
    return ext.update_export_config(export_config, model_config)
  return export_config
