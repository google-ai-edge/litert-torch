# Copyright 2024 The LiteRT Torch Authors.
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

"""Utilities for building LiteRT-LM files."""

import os
import pathlib
from google.protobuf import text_format

try:
  # pylint: disable=g-import-not-at-top
  import litert_lm_builder as litertlm_builder
  from litert_lm_builder.runtime.proto import llm_metadata_pb2
  from litert_lm_builder.runtime.proto import llm_model_type_pb2
  # pylint: enable=g-import-not-at-top

  _litertlm_builder_available = True
except ImportError:
  llm_metadata_pb2 = None
  llm_model_type_pb2 = None
  litertlm_builder = None
  _litertlm_builder_available = False


def is_litertlm_builder_available() -> bool:
  return _litertlm_builder_available


def build_litertlm(
    tflite_model_path: str,
    workdir: str,
    output_path: str,
    context_length: int,
    model_prompt_prefix: str | None = None,
    model_prompt_suffix: str | None = None,
    user_prompt_prefix: str | None = None,
    user_prompt_suffix: str | None = None,
    tokenizer_model_path: str | None = None,
    hf_tokenizer_model_path: str | None = None,
    start_token: str | None = None,
    start_token_id: int | None = None,
    stop_tokens: str | list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    llm_model_type: str = 'generic',
    jinja_prompt_template: str | None = None,
    base_llm_metadata_path: str | None = None,
    **kwargs,
):
  """Builds a LiteRT-LM file from a TFlite model."""
  del kwargs

  if not is_litertlm_builder_available():
    raise ValueError('LiteRT-LM builder is not available.')
  assert litertlm_builder is not None
  assert llm_metadata_pb2 is not None
  assert llm_model_type_pb2 is not None

  llm_metadata = llm_metadata_pb2.LlmMetadata()
  if base_llm_metadata_path:
    if base_llm_metadata_path.endswith('.pb'):
      with open(base_llm_metadata_path, 'rb') as f:
        llm_metadata.ParseFromString(f.read())
    elif base_llm_metadata_path.endswith('.textproto'):
      with open(base_llm_metadata_path, 'r') as f:
        text_format.Parse(f.read(), llm_metadata, allow_unknown_field=True)
    else:
      raise ValueError(
          'Base LLM metadata path must be a binary or text proto file.'
      )

  if start_token_id is not None:
    llm_metadata.start_token.token_ids.ids.append(start_token_id)
  elif start_token is not None:
    llm_metadata.start_token.token_str = start_token

  stop_tokens_list = []
  if stop_tokens is not None:
    if isinstance(stop_tokens, str):
      stop_tokens_list.append(stop_tokens)
    else:
      stop_tokens_list.extend(stop_tokens)
  if stop_token_ids is not None:
    stop_tokens_list.extend(stop_token_ids)
  for stop_token in stop_tokens_list:
    if isinstance(stop_token, str):
      tu = llm_metadata.stop_tokens.add()
      tu.token_str = stop_token
    else:
      assert isinstance(stop_token, int)
      tu = llm_metadata.stop_tokens.add()
      tu.token_ids.ids.append(stop_token)

  if model_prompt_prefix is not None:
    llm_metadata.prompt_templates.model.prefix = model_prompt_prefix
  if model_prompt_suffix is not None:
    llm_metadata.prompt_templates.model.suffix = model_prompt_suffix
  if user_prompt_prefix is not None:
    llm_metadata.prompt_templates.user.prefix = user_prompt_prefix
  if user_prompt_suffix is not None:
    llm_metadata.prompt_templates.user.suffix = user_prompt_suffix

  llm_metadata.max_num_tokens = context_length

  mdl_type = llm_metadata.llm_model_type.WhichOneof('model_type')
  if not mdl_type or mdl_type == 'generic_model':
    match llm_model_type:
      case 'generic':
        llm_metadata.llm_model_type.CopyFrom(
            llm_model_type_pb2.LlmModelType(
                generic_model=llm_model_type_pb2.GenericModel()
            )
        )
      case 'gemma3n':
        llm_metadata.llm_model_type.CopyFrom(
            llm_model_type_pb2.LlmModelType(
                gemma3n=llm_model_type_pb2.Gemma3N()
            )
        )
      case 'gemma3':
        llm_metadata.llm_model_type.CopyFrom(
            llm_model_type_pb2.LlmModelType(gemma3=llm_model_type_pb2.Gemma3())
        )
      case 'qwen3':
        llm_metadata.llm_model_type.CopyFrom(
            llm_model_type_pb2.LlmModelType(qwen3=llm_model_type_pb2.Qwen3())
        )
      case 'qwen2p5':
        llm_metadata.llm_model_type.CopyFrom(
            llm_model_type_pb2.LlmModelType(
                qwen2p5=llm_model_type_pb2.Qwen2p5()
            )
        )
      case _:
        raise ValueError(f'Unsupported LLM model type: {llm_model_type}')

  if jinja_prompt_template is not None:
    llm_metadata.jinja_prompt_template = jinja_prompt_template

  llm_metadata_path = os.path.join(workdir, 'llm_metadata_final.pb')
  with open(llm_metadata_path, 'wb') as f:
    f.write(llm_metadata.SerializeToString())

  builder = litertlm_builder.LitertLmFileBuilder()
  builder.add_system_metadata(
      litertlm_builder.Metadata(
          key='Authors',
          value='',
          dtype=litertlm_builder.DType.STRING,
      )
  )
  builder.add_llm_metadata(llm_metadata_path)
  if tokenizer_model_path:
    builder.add_sentencepiece_tokenizer(tokenizer_model_path)
  elif hf_tokenizer_model_path:
    builder.add_hf_tokenizer(hf_tokenizer_model_path)
  else:
    raise ValueError(
        'Either tokenizer_model_path or hf_tokenizer_model_path must be'
        ' provided.'
    )
  builder.add_tflite_model(
      tflite_model_path, llm_metadata_pb2.LlmMetadata.TF_LITE_PREFILL_DECODE
  )

  file_name = pathlib.Path(tflite_model_path).stem + '.litertlm'
  if os.path.dirname(file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
  with open(os.path.join(output_path, file_name), 'wb') as f:
    builder.build(f)
