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
"""Exportable modules for extended modules."""

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.model_ext import builtin_extensions as _  # Registers built-in extensions.
from litert_torch.generative.export_hf.model_ext import extension as extension_lib
import transformers


def get_metadata_builder(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Gets metadata builder."""
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


try:
  from litert_lm_builder.runtime.proto import executor_metadata_pb2

  def _add_state_buffer(
      llm_metadata: executor_metadata_pb2.LlmExecutorMetadata,
      name: str,
      buffer_type: executor_metadata_pb2.StateBuffer.Type,
      sequence_axis: int | None = None,
      maximum_sequence_length: int | None = None,
      minimum_sequence_length: int | None = None,
  ) -> None:
    """Adds a state buffer to the LLM executor metadata."""
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
    """Builds executor metadata."""
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
      layer_types = ['full_attention'] * num_layers

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

    return executor_metadata

except ImportError:
  # LiteRT executor metadata is not available.
  build_executor_metadata = None


def get_executor_metadata_builder(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig | None = None,
):
  """Gets executor metadata builder."""
  ext = extension_lib.resolve_model_extension(model_config, export_config)
  if ext is not None:
    ext_builder = ext.get_executor_metadata_builder(model_config)
    if ext_builder is not None:
      return ext_builder
  return build_executor_metadata
