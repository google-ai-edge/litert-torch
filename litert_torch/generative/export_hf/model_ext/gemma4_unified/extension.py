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
"""Built-in `ModelExtension` for Gemma 4 Unified (`gemma4_unified`).

Illustrates an encoder-free multimodal variant:

- Vision: the vision encoder already incorporates feature projection
  (`adapter_cls=None`), plus a dedicated end-of-image (EOI) module
  (`gemma4_unified/vision_exportable.py`).
- Audio: raw waveform frames are projected straight into the language model
  space (`gemma4_unified/audio_exportable.py`, `adapter_cls=None`), plus Gemma
  4's end-of-audio (EOA) module.
- Converter V2 weight streaming maps HF module names back to the released
  checkpoint names (see `resolve_checkpoint_tensor`).
- `LlmMetadata` reuses Gemma 4's builder (`gemma4/metadata_builder.py`).
"""

from litert_torch.generative.export_hf.model_ext import extension as extension_lib
from litert_torch.generative.export_hf.model_ext.gemma4 import audio_exportable as gemma4_audio_exportable
from litert_torch.generative.export_hf.model_ext.gemma4 import metadata_builder as gemma4_metadata_builder
from litert_torch.generative.export_hf.model_ext.gemma4 import vision_budgets as gemma4_vision_budgets
from litert_torch.generative.export_hf.model_ext.gemma4_unified import audio_exportable as gemma4_unified_audio_exportable
# Registers gemma4_unified patches.
from litert_torch.generative.export_hf.model_ext.gemma4_unified import patch as _
from litert_torch.generative.export_hf.model_ext.gemma4_unified import vision_exportable as gemma4_unified_vision_exportable

# `(module name, checkpoint name)` fragments mirroring the `gemma4_unified`
# `WeightRenaming`s in transformers' `conversion_mapping.py`. HF applies them in
# `from_pretrained`; Converter V2 streams weights by module name, so the
# released checkpoint names must be restored here.
CHECKPOINT_RENAMES = (
    (
        'embed_vision.multimodal_embedder.embedding_projection.',
        'embed_vision.embedding_projection.',
    ),
    ('embed_vision.patch_ln1.', 'vision_embedder.patch_ln1.'),
    ('embed_vision.patch_dense.', 'vision_embedder.patch_dense.'),
    ('embed_vision.patch_ln2.', 'vision_embedder.patch_ln2.'),
    ('embed_vision.pos_embedding', 'vision_embedder.pos_embedding'),
    ('embed_vision.pos_norm.', 'vision_embedder.pos_norm.'),
)

# Extra local KV cache slots beyond `sliding_window` so the smallest (128-token)
# prefill fits next to a full window, as in the released 12B `.litertlm`
# (1024 + 128 = 1152).
RING_BUFFER_PREFILL_MARGIN = 128


def to_checkpoint_name(param_name: str) -> str | None:
  """Returns the released checkpoint name for `param_name`, if renamed."""
  for module_name, checkpoint_name in CHECKPOINT_RENAMES:
    if module_name in param_name:
      return param_name.replace(module_name, checkpoint_name)
  return None


@extension_lib.register_model_extension(override=False)
class Gemma4UnifiedExtension(extension_lib.ModelExtension):
  """Built-in extension for Gemma 4 Unified multimodal models."""

  model_types = ('gemma4_unified',)
  extra_kwargs_keys = gemma4_vision_budgets.EXTRA_KWARGS_KEYS

  def get_export_defaults(self, config_dict, task=None, target='cpu_gpu'):
    """See base class."""
    if task is None:
      if config_dict.get('audio_config'):
        resolved_task = extension_lib.ExportTask.MULTIMODAL_LM
      else:
        resolved_task = extension_lib.ExportTask.IMAGE_TEXT_TO_TEXT
    else:
      resolved_task = task
    export_vision = resolved_task in (
        extension_lib.ExportTask.IMAGE_TEXT_TO_TEXT,
        extension_lib.ExportTask.MULTIMODAL_LM,
    )
    export_audio = resolved_task == extension_lib.ExportTask.MULTIMODAL_LM
    export_kwargs = {
        'export_vision_encoder': export_vision,
        'export_audio_encoder': export_audio,
        'use_jinja_template': True,
        'litert_lm_llm_metadata_override': (
            gemma4_metadata_builder.GEMMA4_12B_CANONICAL_METADATA
        ),
    }
    sliding_window = extension_lib.text_config_dict(config_dict).get(
        'sliding_window'
    )
    if sliding_window and not str(target or '').startswith('npu'):
      # Sliding-window layers attend bidirectionally within each image
      # (`use_bidirectional_attention='vision'`). LiteRT-LM only applies its
      # `ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL` local mask to models with
      # local (ring buffer) KV caches and a `local_mask` input; without them,
      # image tokens attend causally and image understanding degrades.
      export_kwargs['sliding_window_ring_buffer_size'] = (
          int(sliding_window) + RING_BUFFER_PREFILL_MARGIN
      )
    return extension_lib.ExportDefaults(
        task=resolved_task, export_kwargs=export_kwargs
    )

  def update_export_config(self, export_config, model_config):
    """See base class.

    Exports every supported vision soft-token budget unless one is requested:
    the canonical `gemma4_12b` `LlmMetadata` sets `max_num_patches` for the
    largest (1120) budget, and LiteRT-LM fails to initialize vision when the
    vision encoder has no signature that large.

    Args:
      export_config: Current export config.
      model_config: Loaded model config.

    Returns:
      The updated export config.
    """
    del model_config
    extra_kwargs = export_config.extra_kwargs
    if (
        export_config.export_vision_encoder
        and gemma4_vision_budgets.VISION_TOKEN_BUDGETS_FLAG not in extra_kwargs
        and gemma4_vision_budgets.LEGACY_VISION_MAX_SOFT_TOKENS_FLAG
        not in extra_kwargs
    ):
      extra_kwargs[gemma4_vision_budgets.VISION_TOKEN_BUDGETS_FLAG] = (
          gemma4_vision_budgets.SUPPORTED_VISION_TOKEN_BUDGETS
      )
    return export_config

  def get_vision_exportables(self, model_config, export_config=None):
    """See base class."""
    del model_config, export_config
    return extension_lib.VisionExportablesSpec(
        encoder_cls=(
            gemma4_unified_vision_exportable.LiteRTExportableModuleForGemma4UnifiedVisionEncoder
        ),
        adapter_cls=None,
        eoi_cls=(
            gemma4_unified_vision_exportable.LiteRTExportableModuleForGemma4UnifiedEndOfImage
        ),
    )

  def get_audio_exportables(self, model_config, export_config=None):
    """See base class."""
    del model_config, export_config
    return extension_lib.AudioExportablesSpec(
        encoder_cls=(
            gemma4_unified_audio_exportable.LiteRTExportableModuleForGemma4UnifiedAudioEncoder
        ),
        adapter_cls=None,
        eoa_cls=(
            gemma4_audio_exportable.LiteRTExportableModuleForGemma4EndOfAudio
        ),
    )

  def resolve_checkpoint_tensor(
      self, param_name, try_load, model_config=None, export_config=None
  ):
    """See base class."""
    del model_config, export_config
    checkpoint_name = to_checkpoint_name(param_name)
    if checkpoint_name is None:
      return None
    return try_load(checkpoint_name)

  def build_llm_metadata(
      self,
      source_model_artifacts,
      export_config,
      exported_model_artifacts,
      llm_metadata,
  ):
    """See base class."""
    llm_metadata = gemma4_metadata_builder.build_llm_metadata(
        source_model_artifacts,
        export_config,
        exported_model_artifacts,
        llm_metadata,
    )
    gemma4 = llm_metadata.llm_model_type.gemma4
    if exported_model_artifacts.vision_encoder_model_path:
      # The vision embedder consumes pooling_kernel_size^2 merged patches.
      gemma4.merge_patches = True
    if exported_model_artifacts.audio_encoder_model_path:
      # The audio embedder consumes raw waveform frames, not mel spectrograms.
      gemma4.skip_mel_spectrogram_extraction = True
    return llm_metadata
