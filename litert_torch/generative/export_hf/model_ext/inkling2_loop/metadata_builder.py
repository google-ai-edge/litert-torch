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
"""LiteRT-LM `LlmMetadata` and `ExecutorMetadata` builders for Inkling-2-Mini."""

from typing import Any

from litert_lm_builder.runtime.proto import executor_metadata_pb2
from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

INKLING2_EOS_TOKEN_ID = 200006  # <|content_model_end_sampling|>

# MiniJinja-compatible Inkling-2 chat template (avoids Jinja2 `namespace()`,
# `raise_exception()`, and `{% generation %}` unsupported by MiniJinja).
# Supports both plain string content and LiteRT-LM list-of-parts arrays.
INKLING2_MINIJINJA_TEMPLATE = (
    "{%- set role_map = {"
    "'user': '<|message_user|>',"
    " 'assistant': '<|message_model|>',"
    " 'system': '<|message_system|>',"
    " 'tool': '<|message_tool|>'"
    "} -%}\n"
    "{%- macro format_content(content) -%}\n"
    "    {%- if content is string -%}\n"
    "        {{- content -}}\n"
    "    {%- elif content is iterable and content is not mapping -%}\n"
    "        {%- for part in content -%}\n"
    "            {%- if part is string -%}\n"
    "                {{- part -}}\n"
    "            {%- elif 'text' in part -%}\n"
    "                {{- part['text'] -}}\n"
    "            {%- endif -%}\n"
    "        {%- endfor -%}\n"
    "    {%- endif -%}\n"
    "{%- endmacro -%}\n"
    "{%- set has_leading_system = messages and messages[0]['role'] == 'system'"
    " -%}\n"
    "{%- if has_leading_system -%}\n"
    "    {{- '<|message_system|><|content_text|>' +"
    " format_content(messages[0]['content']) + '<|end_message|>' -}}\n"
    "{%- endif -%}\n"
    "{%- if enable_thinking is defined and enable_thinking is false -%}\n"
    "    {{- '<|message_system|><|content_text|>Thinking effort level: 0"
    "<|end_message|>' -}}\n"
    "{%- else -%}\n"
    "    {{- '<|message_system|><|content_text|>Thinking effort level: 0.7"
    "<|end_message|>' -}}\n"
    "{%- endif -%}\n"
    "{%- set rest_messages = messages[1:] if has_leading_system else messages"
    " -%}\n"
    "{%- for message in rest_messages -%}\n"
    "    {%- set rtok = role_map[message['role']] -%}\n"
    "    {{- rtok + '<|content_text|>' + format_content(message['content']) +"
    " '<|end_message|>' -}}\n"
    "    {%- if message['role'] == 'assistant' -%}\n"
    "        {{- '<|content_model_end_sampling|>' -}}\n"
    "    {%- endif -%}\n"
    "{%- endfor -%}\n"
)


def build_inkling2_loop_llm_metadata(
    source_model_artifacts: Any,
    export_config: Any,
    exported_model_artifacts: Any,
    llm_metadata: llm_metadata_pb2.LlmMetadata | Any,
) -> Any:
  """Populates Inkling-2-Mini `LlmMetadata` (stop tokens, template, thought)."""
  del exported_model_artifacts
  text_cfg = getattr(source_model_artifacts, "text_model_config", None)
  if text_cfg is None:
    text_cfg = getattr(source_model_artifacts, "model_config", None)
    if hasattr(text_cfg, "text_config") and text_cfg.text_config is not None:
      text_cfg = text_cfg.text_config

  eos_id = int(getattr(text_cfg, "eos_token_id", None) or INKLING2_EOS_TOKEN_ID)
  del llm_metadata.stop_tokens[:]
  entry = llm_metadata.stop_tokens.add()
  entry.token_ids.ids.append(eos_id)

  override_tpl = getattr(export_config, "jinja_chat_template_override", None)
  if not override_tpl:
    llm_metadata.jinja_prompt_template = INKLING2_MINIJINJA_TEMPLATE

  if hasattr(llm_metadata, "supports_thinking"):
    llm_metadata.supports_thinking = True
  if hasattr(llm_metadata, "channels") and not llm_metadata.channels:
    channel = llm_metadata.channels.add()
    channel.channel_name = "thought"
    channel.start = "<|message_model|><|content_thinking|>"
    channel.end = "<|end_message|>"

  if hasattr(llm_metadata, "llm_model_type") and (
      not llm_metadata.llm_model_type.WhichOneof("model_type")
  ):
    llm_metadata.llm_model_type.generic_model.CopyFrom(
        llm_model_type_pb2.GenericModel()
    )

  if hasattr(llm_metadata, "max_num_tokens"):
    cache_lengths = getattr(export_config, "cache_lengths", None) or [
        getattr(export_config, "cache_length", 0)
    ]
    raw_cache_len = max((int(c or 0) for c in cache_lengths), default=0)
    if raw_cache_len > 0 and (
        not llm_metadata.max_num_tokens
        or llm_metadata.max_num_tokens > raw_cache_len
    ):
      llm_metadata.max_num_tokens = raw_cache_len

  return llm_metadata


def build_inkling2_loop_executor_metadata(
    source_model_artifacts: Any,
    export_config: Any,
    exported_model_artifacts: Any,
    executor_metadata: Any = None,
) -> Any:
  """Populates `ExecutorMetadata` for all 48 KV caches and 192 conv states."""
  del exported_model_artifacts
  if executor_metadata is None:
    executor_metadata = executor_metadata_pb2.ExecutorMetadata()

  text_cfg: Any = getattr(source_model_artifacts, "model_config", None)
  if text_cfg is None:
    return executor_metadata
  if hasattr(text_cfg, "text_config") and text_cfg.text_config is not None:
    text_cfg = text_cfg.text_config

  llm_exec = executor_metadata.llm_executor_metadata
  max_cache_len = int(export_config.cache_length)
  total_layers = int(text_cfg.total_num_hidden_layers)
  num_convs = int(getattr(text_cfg, "number_of_conv_states", 4) or 4)
  k_ts = (
      int(export_config.k_ts_idx) if export_config.k_ts_idx is not None else 2
  )
  v_ts = (
      int(export_config.v_ts_idx) if export_config.v_ts_idx is not None else 3
  )

  policy_default = executor_metadata_pb2.StateBuffer.POLICY_DEFAULT
  type_global_k = executor_metadata_pb2.StateBuffer.TYPE_GLOBAL_KEY_CACHE
  type_global_v = executor_metadata_pb2.StateBuffer.TYPE_GLOBAL_VALUE_CACHE
  type_linear_attn = executor_metadata_pb2.StateBuffer.TYPE_LINEAR_ATTENTION

  def _add_buf(
      name: str,
      buf_type: Any,
      seq_axis: int | None = None,
      max_seq_len: int | None = None,
  ) -> None:
    buf = llm_exec.state_buffers.add()
    buf.prefill_input_name = name
    buf.prefill_output_name = name
    buf.decode_input_name = name
    buf.decode_output_name = name
    buf.policy = policy_default
    buf.type = buf_type
    if seq_axis is not None:
      buf.sequence_axis = seq_axis
    if max_seq_len is not None:
      buf.maximum_sequence_length = max_seq_len

  for i in range(total_layers):
    _add_buf(
        f"kv_cache_k_{i}",
        type_global_k,
        seq_axis=k_ts,
        max_seq_len=max_cache_len,
    )
    _add_buf(
        f"kv_cache_v_{i}",
        type_global_v,
        seq_axis=v_ts,
        max_seq_len=max_cache_len,
    )
    for c_idx in range(num_convs):
      _add_buf(f"kv_cache_c_{i}_{c_idx}", type_linear_attn)

  # Short convolution states cannot be rolled back by truncating KV history.
  llm_exec.max_history_size = 0
  return executor_metadata
