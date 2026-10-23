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
"""Metadata builder for Flint."""

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module

from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2
from litert_lm_builder.runtime.proto import sampler_params_pb2

FLINT_MINIJINJA_CHAT_TEMPLATE = """\
{%- macro flint_text(t) -%}
{%- if t is string -%}
{{- t | replace("\\x00", "") | replace("<think>", "") | replace("</think>", "") | replace("<|begin_of_text|>", "&lt;|begin_of_text|&gt;") | replace("<|end_of_text|>", "&lt;|end_of_text|&gt;") | replace("<|system|>", "&lt;|system|&gt;") | replace("<|developer|>", "&lt;|developer|&gt;") | replace("<|user|>", "&lt;|user|&gt;") | replace("<|assistant|>", "&lt;|assistant|&gt;") | replace("<|observation|>", "&lt;|observation|&gt;") -}}
{%- else -%}
{{- t -}}
{%- endif -%}
{%- endmacro -%}
{%- macro format_content(content) -%}
{%- if content is string -%}
{{- flint_text(content) -}}
{%- elif content is sequence -%}
{%- for item in content -%}
{%- if item is mapping and item['type'] == 'text' -%}
{{- flint_text(item['text']) -}}
{%- elif item is mapping and item['type'] == 'tool_response' -%}
{%- if item['response'] is mapping or (item['response'] is sequence and item['response'] is not string) -%}
{{- flint_text(item['response'] | tojson) -}}
{%- else -%}
{{- flint_text(item['response'] | string) -}}
{%- endif -%}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- endmacro -%}
{%- macro format_assistant_content(content) -%}
{%- if content is string -%}
{{- content -}}
{%- elif content is sequence -%}
{%- for item in content -%}
{%- if item is mapping and item['type'] == 'text' -%}
{{- item['text'] -}}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- endmacro -%}
{%- set tool_preamble -%}
{%- if tools %}<|system|>
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{%- set fn = tool.function if tool.function is defined else tool %}
{{ fn | tojson }}
{%- endfor %}
</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>
{% endif -%}
{%- endset -%}
{%- set ns = namespace(tools_pending=true) -%}
{%- for m in messages -%}
{%- if ns.tools_pending and m.role != 'system' -%}
{{ tool_preamble }}
{%- set ns.tools_pending = false -%}
{%- endif -%}
{%- if m.role == 'system' -%}
<|system|>{{ format_content(m.content) }}
{%- elif m.role == 'developer' -%}
<|developer|>{{ format_content(m.content) }}
{%- elif m.role == 'user' -%}
<|user|>{{ format_content(m.content) }}
{%- elif m.role == 'assistant' -%}
<|assistant|>
{%- if m.reasoning_content is defined and m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif -%}
{%- if m.content is defined and m.content %}{{ format_assistant_content(m.content) }}{% endif -%}
{%- if m.tool_calls is defined and m.tool_calls -%}
{%- for tc in m.tool_calls -%}
{%- set fn = tc.function if tc.function is defined else tc -%}
<tool_call>{{ fn.name }}
{%- if fn.arguments is mapping -%}
{%- for k, v in fn.arguments | items -%}
<arg_key>{{ k }}</arg_key><arg_value>{{ v | tojson if v is not string else v }}</arg_value>
{%- endfor -%}
{%- endif -%}
</tool_call>
{%- endfor -%}
{%- endif -%}
{%- elif m.role == 'tool' -%}
{%- if m.content is sequence and m.content is not string -%}
{%- for item in m.content -%}
{%- if item is mapping and item['type'] == 'tool_response' -%}
{%- set resp = item['response'] | tojson if (item['response'] is mapping or (item['response'] is sequence and item['response'] is not string)) else item['response'] | string -%}
<|observation|><tool_response>{{ flint_text(resp) }}</tool_response>
{%- elif item is mapping and item['type'] == 'text' -%}
<|observation|><tool_response>{{ flint_text(item['text']) }}</tool_response>
{%- endif -%}
{%- endfor -%}
{%- else -%}
<|observation|><tool_response>{{ format_content(m.content) }}</tool_response>
{%- endif -%}
{%- endif -%}
{%- endfor -%}
{%- if ns.tools_pending -%}
{{ tool_preamble }}
{%- endif -%}
{%- if add_generation_prompt -%}
<|assistant|>
{%- endif -%}
"""


def build_llm_metadata(
    source_model_artifacts: export_lib.SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: export_lib.ExportedModelArtifacts,
    llm_metadata: llm_metadata_pb2.LlmMetadata,
) -> llm_metadata_pb2.LlmMetadata:
  """Builds LLM metadata for Flint."""
  del source_model_artifacts, exported_model_artifacts
  llm_metadata.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(
          generic_model=llm_model_type_pb2.GenericModel()
      )
  )
  llm_metadata.supports_thinking = True
  llm_metadata.supports_function_calling = True

  llm_metadata.tool_call_config.fence_start = '<tool_call>'
  llm_metadata.tool_call_config.fence_end = '</tool_call>'
  llm_metadata.tool_call_config.regex_syntax.tool_call_regex = (
      r'(?s)\s*([^<\s]+)\s*(.*)'
  )
  llm_metadata.tool_call_config.regex_syntax.argument_regex = (
      r'(?s)<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>'
  )

  if (
      export_config.use_jinja_template
      and not export_config.jinja_chat_template_override
  ):
    llm_metadata.jinja_prompt_template = FLINT_MINIJINJA_CHAT_TEMPLATE

  if not llm_metadata.channels:
    think_channel = llm_metadata.channels.add()
    think_channel.channel_name = 'thought'
    think_channel.start = '<think>'
    think_channel.end = '</think>'

  if not llm_metadata.HasField('sampler_params'):
    llm_metadata.sampler_params.type = (
        sampler_params_pb2.SamplerParameters.TOP_P
    )
    llm_metadata.sampler_params.k = 20
    llm_metadata.sampler_params.p = 0.95
    llm_metadata.sampler_params.temperature = 1.0

  existing_stop_ids = {
      t.token_ids.ids[0]
      for t in llm_metadata.stop_tokens
      if t.HasField('token_ids') and t.token_ids.ids
  }
  for stop_id in (150001, 150003, 150005, 150018):
    if stop_id not in existing_stop_ids:
      tu = llm_metadata.stop_tokens.add()
      tu.token_ids.ids.append(stop_id)

  return llm_metadata
