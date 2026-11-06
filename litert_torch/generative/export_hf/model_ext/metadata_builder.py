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
"""LiteRT-LM `LlmMetadata` and `ExecutorMetadata` builders for `export_hf`.

Provides two metadata hooks consumed by `core/litert_lm_builder.py` when
packaging `.litertlm` bundles:

1. `get_metadata_builder`: Resolves the active `ModelExtension` and returns a
   callable that runs `ModelExtension.build_llm_metadata` to populate
   architecture-specific `LlmMetadata` fields (such as multimodal token IDs and
   prompt templates; see `model_ext/gemma3/metadata_builder.py` and
   `model_ext/gemma4/metadata_builder.py`).
2. `get_executor_metadata_builder` / `build_executor_metadata`: Constructs
   `ExecutorMetadata` describing runtime `StateBuffer` entries (global KV cache,
   sliding-window local KV cache, `linear_attention`, and `conv` recurrent
   states) inferred from `model_config.layer_types`, unless overridden by
   `ModelExtension.get_executor_metadata_builder`.
"""

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
# Registers built-in extensions.
from litert_torch.generative.export_hf.model_ext import builtin_extensions as _
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
import transformers


def get_metadata_builder(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Returns an `LlmMetadata` builder delegating to the active `ModelExtension`.

  See `model_ext/gemma3/extension.py` and `model_ext/gemma4/extension.py` for
  built-in `ModelExtension.build_llm_metadata` implementations.

  Args:
    model_config: Loaded `PretrainedConfig` for the checkpoint.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    A callable `(source_model_artifacts, export_config,
    exported_model_artifacts, llm_metadata) -> llm_metadata` that applies the
    resolved extension's `build_llm_metadata` hook (or returns `llm_metadata`
    unchanged if no extension applies).
  """
  ext = extension_lib.resolve_model_extension(model_config, export_config)

  def _combined_builder(
      source_model_artifacts,
      cfg,
      exported_model_artifacts,
      llm_metadata,
  ):
    resolved_ext = ext or extension_lib.resolve_model_extension(
        model_config, cfg
    )
    if resolved_ext is not None:
      llm_metadata = resolved_ext.build_llm_metadata(
          source_model_artifacts,
          cfg,
          exported_model_artifacts,
          llm_metadata,
      )
    return llm_metadata

  return _combined_builder


# Names of the `LlmNpuExecutorMetadata` fields that can be set at export time.
NPU_EXECUTOR_SETTING_FIELDS = (
    'enable_neon_for_npu_greedy_sampling',
    'use_hw_masking_for_npu',
    'use_hw_cache_update_for_npu',
    'use_hw_ple_for_npu',
)

# Maps HF / LiteRT-LM model type strings to the canonical `LlmModelType` oneof
# name used to look up NPU executor defaults. Multimodal checkpoints report
# `<type>_text` on their text config, so both spellings are listed.
_CANONICAL_LITERT_LM_MODEL_TYPES = {
    'gemma3': 'gemma3',
    'gemma3_text': 'gemma3',
    'gemma3n': 'gemma3n',
    'gemma3n_text': 'gemma3n',
    'gemma4': 'gemma4',
    'gemma4_text': 'gemma4',
    'gemma4_unified': 'gemma4',
    'qwen3': 'qwen3',
    'qwen3_asr': 'qwen3',
    'minicpm5': 'minicpm5',
}

# All handwritten NPU components enabled.
_NPU_ALL_HW_COMPONENTS = {
    'enable_neon_for_npu_greedy_sampling': True,
    'use_hw_masking_for_npu': True,
    'use_hw_cache_update_for_npu': True,
    'use_hw_ple_for_npu': True,
}

# Default NPU executor settings keyed by canonical LiteRT-LM model type. These
# models are known to work on NPU with all handwritten components enabled.
_NPU_EXECUTOR_DEFAULTS = {
    'gemma3': _NPU_ALL_HW_COMPONENTS,
    'gemma3n': _NPU_ALL_HW_COMPONENTS,
    'gemma4': _NPU_ALL_HW_COMPONENTS,
    'qwen3': _NPU_ALL_HW_COMPONENTS,
    'minicpm5': _NPU_ALL_HW_COMPONENTS,
}


def _canonicalize_litert_lm_model_type(model_type: str | None) -> str | None:
  """Returns the canonical LiteRT-LM model type, or None if unsupported.

  Matching is exact, like the `LlmModelType` match in `litert_lm_builder`.

  Args:
    model_type: A HF or LiteRT-LM model type string.
  """
  if not model_type:
    return None
  return _CANONICAL_LITERT_LM_MODEL_TYPES.get(model_type)


def resolve_litert_lm_model_type(
    source_model_artifacts: export_lib.SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
) -> str | None:
  """Resolves the canonical LiteRT-LM model type of the exported model.

  `litert_lm_model_type_override` is authoritative when set: it replaces the HF
  model type (as in `litert_lm_builder`), so an unsupported override yields
  None rather than falling back to the HF config. Otherwise the top-level HF
  config is checked first (e.g. `gemma3n` for a multimodal checkpoint), then
  the text config (e.g. `gemma3_text` for text-only Gemma3).

  Args:
    source_model_artifacts: The loaded source model artifacts.
    export_config: The export config.

  Returns:
    The canonical model type (e.g. 'gemma3', 'qwen3'), or None if the model
    type is not supported.
  """
  if export_config.litert_lm_model_type_override:
    return _canonicalize_litert_lm_model_type(
        export_config.litert_lm_model_type_override
    )
  for config in (
      source_model_artifacts.model_config,
      source_model_artifacts.text_model_config,
  ):
    model_type = _canonicalize_litert_lm_model_type(
        getattr(config, 'model_type', None)
    )
    if model_type:
      return model_type
  return None


def get_npu_executor_settings_for_model_type(
    model_type: str | None,
) -> dict[str, bool]:
  """Returns default NPU executor settings for a given model type.

  Args:
    model_type: A HF or LiteRT-LM model type string (e.g. 'gemma3',
      'gemma3n_text', 'qwen3'). Matching is exact.

  Returns:
    A dictionary mapping NPU executor settings attribute names to bool values,
    or an empty dict if the model type is not recognized.
  """
  canonical_type = _canonicalize_litert_lm_model_type(model_type)
  if canonical_type is None:
    return {}
  return dict(_NPU_EXECUTOR_DEFAULTS.get(canonical_type, {}))


try:
  from litert_lm_builder.runtime.proto import executor_metadata_pb2

  # TODO(b/542876170): Remove once litert-torch requires a litert-lm-builder
  # release that includes `LlmNpuExecutorMetadata`. Until then, OSS installs
  # may ship an older executor_metadata proto without the field.
  HAS_NPU_EXECUTOR_METADATA = (
      'llm_npu_executor_metadata'
      in executor_metadata_pb2.ExecutorMetadata.DESCRIPTOR.fields_by_name
  )

  def _add_state_buffer(
      llm_metadata: executor_metadata_pb2.LlmExecutorMetadata,
      name: str,
      buffer_type: executor_metadata_pb2.StateBuffer.Type,
      sequence_axis: int | None = None,
      maximum_sequence_length: int | None = None,
      minimum_sequence_length: int | None = None,
  ) -> None:
    """Adds a `StateBuffer` entry to `LlmExecutorMetadata`."""
    buffer = llm_metadata.state_buffers.add()
    buffer.prefill_input_name = name
    buffer.prefill_output_name = name
    buffer.decode_input_name = name
    buffer.decode_output_name = name
    buffer.policy = executor_metadata_pb2.StateBuffer.POLICY_DEFAULT
    buffer.type = buffer_type
    if sequence_axis is not None:
      buffer.sequence_axis = sequence_axis
    if maximum_sequence_length is not None:
      buffer.maximum_sequence_length = maximum_sequence_length
    if minimum_sequence_length is not None:
      buffer.minimum_sequence_length = minimum_sequence_length

  def build_executor_metadata(
      source_model_artifacts: export_lib.SourceModelArtifacts,
      export_config: exportable_module.ExportableModuleConfig,
      exported_model_artifacts: export_lib.ExportedModelArtifacts,
      executor_metadata: executor_metadata_pb2.ExecutorMetadata | None = None,
  ) -> executor_metadata_pb2.ExecutorMetadata:
    """Populates default `ExecutorMetadata` state buffers from `layer_types`.

    Emits per-layer `StateBuffer` descriptors matching the standard cache tensor
    naming convention (`kv_cache_k_{i}`, `kv_cache_v_{i}`, `kv_cache_c_{i}`,
    `kv_cache_r_{i}`) for `full_attention`, `sliding_attention`,
    `linear_attention` (e.g. `qwen3_5`), and `conv` (e.g. `lfm2`) layers, along
    with sliding-window attention mask settings when ring-buffer cache is used.

    Args:
      source_model_artifacts: Loaded `SourceModelArtifacts` with `model_config`.
      export_config: Active `ExportableModuleConfig` specifying cache lengths
        and KV time-step axes.
      exported_model_artifacts: Unused `ExportedModelArtifacts` placeholder for
        builder signature compatibility.
      executor_metadata: Optional existing `ExecutorMetadata` message to
        populate; a new message is allocated when `None`.

    Returns:
      Populated `executor_metadata_pb2.ExecutorMetadata` protobuf message.

    Raises:
      ValueError: If `text_config.layer_types` contains an unrecognized layer
        type string.
    """
    del exported_model_artifacts
    if executor_metadata is None:
      executor_metadata = executor_metadata_pb2.ExecutorMetadata()
    text_config = source_model_artifacts.model_config
    if hasattr(text_config, 'text_config'):
      text_config = text_config.text_config

    llm_metadata = executor_metadata.llm_executor_metadata
    sliding_window_size = getattr(text_config, 'sliding_window', None)
    max_cache_length = export_config.cache_length

    layer_types = getattr(text_config, 'layer_types', None)
    num_layers = text_config.num_hidden_layers
    num_shared_layers = getattr(text_config, 'num_kv_shared_layers', 0)
    num_layers -= num_shared_layers
    if layer_types is None:
      ext = extension_lib.resolve_model_extension(
          source_model_artifacts.model_config, export_config
      )
      use_sliding_window = (
          ext is not None
          and ext.sliding_window_layers == 'all'
          and bool(sliding_window_size)
      )
      default_layer_type = (
          'sliding_attention' if use_sliding_window else 'full_attention'
      )
      layer_types = [default_layer_type] * num_layers

    apply_gpu_composites = (
        getattr(export_config, 'apply_gpu_composites', False)
        or getattr(export_config, 'extra_kwargs', {}).get(
            'apply_gpu_composites', False
        )
    )
    del apply_gpu_composites
    enable_ring_buffer = (
        getattr(export_config, 'sliding_window_ring_buffer_size', None)
        is not None
    )
    for i, layer_type in enumerate(layer_types):
      if i >= num_layers:
        break
      if layer_type == 'sliding_attention' and not enable_ring_buffer:
        layer_type = 'full_attention'

      if layer_type == 'linear_attention':
        _add_state_buffer(
            llm_metadata,
            f'kv_cache_c_{i}',
            executor_metadata_pb2.StateBuffer.TYPE_LINEAR_ATTENTION,
        )
        _add_state_buffer(
            llm_metadata,
            f'kv_cache_r_{i}',
            executor_metadata_pb2.StateBuffer.TYPE_LINEAR_ATTENTION,
        )
      elif layer_type == 'conv':
        _add_state_buffer(
            llm_metadata,
            f'kv_cache_c_{i}',
            executor_metadata_pb2.StateBuffer.TYPE_LINEAR_ATTENTION,
        )
      elif layer_type == 'full_attention':
        _add_state_buffer(
            llm_metadata,
            f'kv_cache_k_{i}',
            executor_metadata_pb2.StateBuffer.TYPE_GLOBAL_KEY_CACHE,
            sequence_axis=export_config.k_ts_idx,
            maximum_sequence_length=max_cache_length,
        )
        _add_state_buffer(
            llm_metadata,
            f'kv_cache_v_{i}',
            executor_metadata_pb2.StateBuffer.TYPE_GLOBAL_VALUE_CACHE,
            sequence_axis=export_config.v_ts_idx,
            maximum_sequence_length=max_cache_length,
        )
      elif layer_type == 'sliding_attention':
        local_cache_length = (
            export_config.sliding_window_ring_buffer_size
            if export_config.sliding_window_ring_buffer_size is not None
            else max_cache_length
        )
        _add_state_buffer(
            llm_metadata,
            f'kv_cache_k_{i}',
            executor_metadata_pb2.StateBuffer.TYPE_LOCAL_KEY_CACHE,
            sequence_axis=export_config.k_ts_idx,
            maximum_sequence_length=local_cache_length,
            minimum_sequence_length=sliding_window_size,
        )
        _add_state_buffer(
            llm_metadata,
            f'kv_cache_v_{i}',
            executor_metadata_pb2.StateBuffer.TYPE_LOCAL_VALUE_CACHE,
            sequence_axis=export_config.v_ts_idx,
            maximum_sequence_length=local_cache_length,
            minimum_sequence_length=sliding_window_size,
        )
      else:
        raise ValueError(f'Unsupported layer type: {layer_type}')

    # Because linear attention states are fixed-size and not easily invertible,
    # rolling back is unsupported.
    if any(
        layer_type == 'linear_attention' or layer_type == 'conv'
        for layer_type in layer_types
    ):
      llm_metadata.max_history_size = 0
    else:
      llm_metadata.max_history_size = max_cache_length

    if sliding_window_size is not None and enable_ring_buffer:
      local_mask_type = executor_metadata_pb2.ATTENTION_MASK_TYPE_CAUSAL
      if (
          getattr(source_model_artifacts.model_config, 'vision_config', None)
          is not None
      ):
        local_mask_type = (
            executor_metadata_pb2.ATTENTION_MASK_TYPE_VISION_BIDIRECTIONAL
        )
      llm_metadata.attention_mask_settings.attention_mask_type = (
          executor_metadata_pb2.ATTENTION_MASK_TYPE_CAUSAL
      )
      llm_metadata.attention_mask_settings.local_attention_mask_type = (
          local_mask_type
      )
      llm_metadata.attention_mask_settings.sliding_window_size = (
          sliding_window_size
      )

    # A ringbuffer only exists when sliding_attention layers get local caches,
    # i.e. exactly when TYPE_LOCAL_*_CACHE buffers are emitted above. The
    # runtime uses the same signal to detect ringbuffer models.
    uses_ring_buffer = enable_ring_buffer and any(
        layer_type == 'sliding_attention'
        for layer_type in layer_types[:num_layers]
    )
    _populate_npu_executor_metadata(
        source_model_artifacts,
        export_config,
        executor_metadata,
        uses_ring_buffer=uses_ring_buffer,
    )
    return executor_metadata

  def _populate_npu_executor_metadata(
      source_model_artifacts: export_lib.SourceModelArtifacts,
      export_config: exportable_module.ExportableModuleConfig,
      executor_metadata: executor_metadata_pb2.ExecutorMetadata,
      uses_ring_buffer: bool,
  ) -> None:
    """Populates `llm_npu_executor_metadata` for split-cache exports.

    NPU exports always use split cache (`aot_backend` and the NPU export
    pipeline force it), so nothing is written for other exports. Precedence,
    highest first:
      1. Explicit user setting (config field).
      2. Ringbuffer requirement (`use_hw_cache_update_for_npu=True`).
      3. Defaults derived from the LiteRT-LM model type.
    Fields left unset defer to the runtime defaults.

    Args:
      source_model_artifacts: The loaded source model artifacts.
      export_config: The export config.
      executor_metadata: The executor metadata to populate.
      uses_ring_buffer: Whether any exported layer uses a sliding window
        ringbuffer KV cache.
    """
    # TODO(b/542876170): Remove with HAS_NPU_EXECUTOR_METADATA.
    if not HAS_NPU_EXECUTOR_METADATA:
      return
    if not export_config.split_cache:
      return

    model_type = resolve_litert_lm_model_type(
        source_model_artifacts, export_config
    )
    npu_fields = get_npu_executor_settings_for_model_type(model_type)

    # Models with a ringbuffer must use the handwritten KV cache update path
    # because they do not use dynamic_update_slice and reuse the same KV cache
    # buffers on model input and output.
    if uses_ring_buffer:
      npu_fields['use_hw_cache_update_for_npu'] = True

    for attr in NPU_EXECUTOR_SETTING_FIELDS:
      val = getattr(export_config, attr)
      if val is not None:
        npu_fields[attr] = val

    if not npu_fields:
      return
    npu_metadata = executor_metadata.llm_npu_executor_metadata
    for attr, val in npu_fields.items():
      setattr(npu_metadata, attr, val)


except ImportError:
  # LiteRT executor metadata is not available.
  build_executor_metadata = None
  HAS_NPU_EXECUTOR_METADATA = False


def get_executor_metadata_builder(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Returns the `ExecutorMetadata` builder for `model_config`.

  Uses `ModelExtension.get_executor_metadata_builder(model_config)` when the
  resolved extension defines one; otherwise returns the default
  `build_executor_metadata` function (or `None` when `executor_metadata_pb2` is
  unavailable in the environment).

  Args:
    model_config: Loaded `PretrainedConfig` for the checkpoint.
    export_config: Optional `ExportableModuleConfig`.

  Returns:
    An executor metadata builder callable, or `None` if executor metadata proto
    support is unavailable.
  """
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    ext_builder = ext.get_executor_metadata_builder(model_config)
    if ext_builder is not None:
      return ext_builder
  return build_executor_metadata
