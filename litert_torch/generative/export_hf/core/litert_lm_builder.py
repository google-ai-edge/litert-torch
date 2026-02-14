# Copyright 2025 The LiteRT Torch Authors.
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
"""Litert LM builder for packing exported models."""

import os

from ai_edge_litert.internal import litertlm_builder
from ai_edge_litert.internal import llm_metadata_pb2
from ai_edge_litert.internal import llm_model_type_pb2
from ai_edge_litert.internal import sampler_params_pb2

_PH = 'KIMAIRA'


def parse_chat_template(tokenizer):
  """Parses chat template."""
  if tokenizer.chat_template is None:
    return None
  try:
    messages = [
        {'role': 'system', 'content': _PH},
    ]
    sys_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=False,
    )
    sys_prompt_parts = sys_prompt.split(_PH)
    no_sys_prompt = False
    if len(sys_prompt_parts) == 1:
      sys_prompt_parts = [sys_prompt_parts[0], '']
      no_sys_prompt = True
    if len(sys_prompt_parts) != 2:
      raise ValueError(
          f'System prompt {_PH} not found in chat template: {sys_prompt}'
      )
    if sys_prompt_parts[0].startswith(str(tokenizer.bos_token)):
      sys_prompt_parts[0] = sys_prompt_parts[0][len(tokenizer.bos_token) :]

    if no_sys_prompt:
      messages = [{'role': 'user', 'content': _PH}]
    else:
      messages.append({'role': 'user', 'content': _PH})
    user_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=False,
    )
    if not user_prompt.startswith(sys_prompt):
      raise ValueError('Cannot guess user prompt from prompt template.')
    user_prompt_substr = user_prompt[len(sys_prompt) :]
    user_prompt_parts = user_prompt_substr.split(_PH)
    if len(user_prompt_parts) != 2:
      raise ValueError(
          f'User prompt {_PH} not found in chat template: {user_prompt_substr}'
      )
    messages.append({'role': 'assistant', 'content': _PH})
    model_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=False,
    )
    if not model_prompt.startswith(user_prompt):
      raise ValueError('Cannot guess model prompt from prompt template.')
    model_prompt_substr = model_prompt[len(user_prompt) :]
    model_prompt_parts = model_prompt_substr.split(_PH)
    if len(model_prompt_parts) != 2:
      raise ValueError(
          f'Model prompt {_PH} not found in chat template:'
          f' {model_prompt_substr}'
      )
    return sys_prompt_parts, user_prompt_parts, model_prompt_parts
  except ValueError as e:
    print(f'Failed to parse chat template: {e}')
    return (None, None), (None, None), (None, None)
  except Exception as e:  # pylint: disable=broad-except
    print(f'Failed to parse chat template: {e}')
    return (None, None), (None, None), (None, None)


def build_llm_metadata(
    model,
    tokenizer,
    chat_templates: tuple[tuple, tuple, tuple] | str,  # pylint: disable=g-bare-generic,
    context_length: int,
    litert_lm_model_type_override: str | None = None,
):
  """Builds LLM metadata."""

  llm_metadata = llm_metadata_pb2.LlmMetadata()

  if getattr(tokenizer, 'add_bos_token', True):
    if isinstance(tokenizer.bos_token, int):
      llm_metadata.start_token.token_ids.ids.append(tokenizer.bos_token)
    elif isinstance(tokenizer.bos_token, str):
      llm_metadata.start_token.token_str = tokenizer.bos_token
    else:
      llm_metadata.start_token.token_str = str(tokenizer.bos_token)

  gen_config = getattr(model, 'generation_config', None)
  if gen_config:
    stop_tokens = set()
    if hasattr(gen_config, 'eos_token_id'):
      if isinstance(gen_config.eos_token_id, int):
        stop_tokens.add(gen_config.eos_token_id)
      elif isinstance(gen_config.eos_token_id, list):
        for token_id in gen_config.eos_token_id:
          stop_tokens.add(token_id)
    elif hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
      stop_tokens.add(tokenizer.eos_token)
    for stop_token in stop_tokens:
      if isinstance(stop_token, int):
        tu = llm_metadata.stop_tokens.add()
        tu.token_ids.ids.append(stop_token)
      elif isinstance(stop_token, str):
        tu = llm_metadata.stop_tokens.add()
        tu.token_str = stop_token

    if gen_config and gen_config.do_sample:
      sampler_params = llm_metadata.sampler_params
      if gen_config.top_k:
        sampler_params.type = sampler_params_pb2.SamplerParameters.TOP_K
        sampler_params.k = gen_config.top_k
      if gen_config.top_p:
        sampler_params.type = sampler_params_pb2.SamplerParameters.TOP_P
        sampler_params.p = gen_config.top_p
      if gen_config.temperature:
        sampler_params.temperature = gen_config.temperature

  if chat_templates is not None:
    if isinstance(chat_templates, str):
      llm_metadata.jinja_prompt_template = chat_templates
    else:
      sys_prompt_parts, user_prompt_parts, model_prompt_parts = chat_templates
      pairs = []
      if sys_prompt_parts[0] is not None:
        pairs.append((sys_prompt_parts, llm_metadata.prompt_templates.system))
      if user_prompt_parts[0] is not None:
        pairs.append((user_prompt_parts, llm_metadata.prompt_templates.user))
      if model_prompt_parts[0] is not None:
        pairs.append((model_prompt_parts, llm_metadata.prompt_templates.model))
      for pts, fld in pairs:
        fld.prefix = pts[0]
        fld.suffix = pts[1]

  llm_metadata.max_num_tokens = context_length

  model_type = litert_lm_model_type_override or model.config.model_type

  match (model_type):
    case 'qwen3':
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(
              qwen3=llm_model_type_pb2.Qwen3()
          )
      )
    case ('qwen2' | 'qwen2p5'):
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(
              qwen2p5=llm_model_type_pb2.Qwen2p5()
          )
      )
    case 'gemma3':
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(
              gemma3=llm_model_type_pb2.Gemma3()
          )
      )
    case 'function_gemma':
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(
              function_gemma=llm_model_type_pb2.FunctionGemma()
          )
      )
    case 'gemma3n':
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(
              gemma3n=llm_model_type_pb2.Gemma3N()
          )
      )
    case _:
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(
              generic_model=llm_model_type_pb2.GenericModel()
          )
      )

  return llm_metadata


def pack_to_litert_lm(
    model,
    tokenizer,
    tflite_model_path: str,
    tokenizer_model_path: str,
    cache_length: int,
    work_dir: str,
    output_dir: str,
    use_jinja_template: bool = False,
    litert_lm_model_type_override: str | None = None,
):
  """Packs models to LiteRT LM."""
  if use_jinja_template:
    chat_templates = getattr(tokenizer, 'chat_template', '')
  else:
    chat_templates = parse_chat_template(tokenizer)
  if not chat_templates:
    print('WARNING: Chat template is not found. Using empty template.')
  llm_metadata = build_llm_metadata(
      model, tokenizer, chat_templates, cache_length,
      litert_lm_model_type_override,
  )
  llm_metadata_path = os.path.join(work_dir, 'llm_metadata.pb')
  with open(llm_metadata_path, 'wb') as f:
    f.write(llm_metadata.SerializeToString())

  builder = litertlm_builder.LitertLmFileBuilder()
  builder.add_system_metadata(
      litertlm_builder.Metadata(
          key='Authors',
          value='ODML',
          dtype=litertlm_builder.DType.STRING,
      )
  )
  builder.add_llm_metadata(llm_metadata_path)
  if tokenizer_model_path.endswith('.json'):
    builder.add_hf_tokenizer(tokenizer_model_path)
  else:
    builder.add_sentencepiece_tokenizer(tokenizer_model_path)
  builder.add_tflite_model(
      tflite_model_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE
  )
  with open(os.path.join(output_dir, 'model.litertlm'), 'wb') as f:
    builder.build(f)


def package_model(
    model,
    tokenizer,
    tflite_model_path: str,
    tokenizer_model_path: str,
    cache_length: int,
    work_dir: str,
    output_dir: str,
    use_jinja_template: bool,
    litert_lm_model_type_override: str | None = None,
):
  """Packs models."""
  pack_to_litert_lm(
      model,
      tokenizer,
      tflite_model_path,
      tokenizer_model_path,
      cache_length,
      work_dir,
      output_dir,
      use_jinja_template,
      litert_lm_model_type_override,
  )
