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
"""Equivalence test for LiteRT Torch LLM models."""

import os
import shutil
import tempfile
from absl import app
from absl import flags
from litert_torch.generative.export_hf import export as litert_torch_export
import torch
import transformers
import litert_lm

_MODEL_ID = flags.DEFINE_string(
    'model_id',
    'google/gemma-3-270m-it',
    'Hugging Face model ID to validate.',
)

_PROMPT = flags.DEFINE_multi_string(
    'prompt',
    default=["What's the capital of France?", 'How about Germany?'],
    help=(
        'Prompt(s) to run the equivalence test on. Can be specified multiple'
        ' times for multi-turn.'
    ),
)

_PROMPT_FILE = flags.DEFINE_string(
    'prompt_file',
    None,
    'Path to a file containing a single prompt. If specified, overrides'
    ' --prompt.',
)

_MAX_NEW_TOKENS = flags.DEFINE_integer(
    'max_new_tokens',
    20,
    'Maximum number of new tokens to generate.',
)

_WORK_DIR = flags.DEFINE_string(
    'work_dir',
    None,
    'Base directory for exporting the model. If not specified, a temporary '
    'directory under HOME is used.',
)

_MAX_NUM_TOKENS = flags.DEFINE_integer(
    'max_num_tokens',
    2048,
    'Maximum number of tokens for the model (cache_length for export,'
    ' max_num_tokens for litert_lm).',
)

_EXTERNALIZE_EMBEDDER = flags.DEFINE_bool(
    'externalize_embedder',
    False,
    'Externalize the embedder during export.',
)

_SINGLE_TOKEN_EMBEDDER = flags.DEFINE_bool(
    'single_token_embedder',
    False,
    'Use single token embedder during export.',
)

_SPLIT_CACHE = flags.DEFINE_bool(
    'split_cache',
    False,
    'Split KV cache during export.',
)

_BACKEND = flags.DEFINE_enum(
    'backend',
    'cpu',
    ['cpu', 'npu'],
    'Hardware backend to use for LiteRT LM.',
)

_LITERT_LM_MODEL_PATH = flags.DEFINE_string(
    'litert_lm_model_path',
    None,
    'Path to the LiteRT LM model to validate. If not specified, will export '
    'from HuggingFace.',
)

_DIFFBASE_LITERT_LM_MODEL_PATH = flags.DEFINE_string(
    'diffbase_litert_lm_model_path',
    None,
    'Path to the LiteRT LM model to compare against. If not specified, will '
    'compare against Transformers.',
)

_SLIDING_WINDOW_RING_BUFFER_SIZE = flags.DEFINE_integer(
    'sliding_window_ring_buffer_size',
    None,
    'Size of the sliding window ring buffer.',
)


def run_transformers(
    model_id: str, prompts: list[str], max_new_tokens: int
) -> list[str]:
  print('Running transformers...')
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
  assert tokenizer is not None, 'Tokenizer is None'
  model = transformers.AutoModelForCausalLM.from_pretrained(
      model_id, torch_dtype=torch.float32
  )

  has_template = tokenizer.chat_template is not None  # pyrefly: ignore[missing-attribute]
  responses = []
  chat = []
  history_str = ''
  for prompt in prompts:
    if has_template:
      chat.append({'role': 'user', 'content': prompt})
      formatted_prompt = tokenizer.apply_chat_template(
          chat, tokenize=False, add_generation_prompt=True
      )
    else:
      history_str += prompt
      formatted_prompt = history_str

    inputs = tokenizer(  # pyrefly: ignore[not-callable]
        formatted_prompt,
        return_tensors='pt',
        add_special_tokens=not has_template,
    )
    with torch.no_grad():
      outputs = model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          do_sample=False,
      )
    input_length = inputs.input_ids.shape[-1]
    generated_tokens = outputs[0][input_length:]
    output_text = tokenizer.decode(  # pyrefly: ignore[missing-attribute]
        generated_tokens, skip_special_tokens=True
    ).strip()
    responses.append(output_text)

    if has_template:
      chat.append({'role': 'assistant', 'content': output_text})
    else:
      history_str += output_text + '\n'

  return responses


def run_litert_lm(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    max_num_tokens: int,
    backend_str: str = 'cpu',
) -> list[str]:
  print('Running litert_lm...')
  if backend_str == 'npu':
    backend = litert_lm.Backend.NPU(litert_dispatch_lib_dir='')  # pyrefly: ignore[missing-attribute]
  else:
    backend = litert_lm.Backend.CPU()  # pyrefly: ignore[missing-attribute]
  engine = litert_lm.Engine(
      model_path,
      backend,
      max_num_tokens=max_num_tokens,
  )

  # We use conversation because it handles chat template automatically if
  # packaged. We want to use greedy decoding.
  sampler_config = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)

  responses = []
  with engine.create_conversation(
      sampler_config=sampler_config
  ) as conversation:
    for prompt in prompts:
      response = conversation.send_message(
          prompt, max_output_tokens=max_new_tokens
      )
      text_pieces = []
      for item in response.get('content', []):
        if item.get('type') == 'text':
          text_pieces.append(item.get('text', ''))
      output_text = ''.join(text_pieces).strip()
      responses.append(output_text)

  return responses


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_id = _MODEL_ID.value
  prompts = _PROMPT.value
  max_new_tokens = _MAX_NEW_TOKENS.value

  if _PROMPT_FILE.value:
    if not os.path.exists(_PROMPT_FILE.value):
      raise FileNotFoundError(f'Prompt file not found: {_PROMPT_FILE.value}')
    with open(_PROMPT_FILE.value, 'r') as f:
      prompts = [f.read().strip()]

  # Create a temp dir for export
  base_dir = _WORK_DIR.value if _WORK_DIR.value else os.path.expanduser('~')
  export_dir = tempfile.mkdtemp(dir=base_dir, prefix='litert_export_')
  print(f'Exporting model to {export_dir}...')

  try:
    if _LITERT_LM_MODEL_PATH.value:
      exported_model_path = _LITERT_LM_MODEL_PATH.value
    else:
      # Export model
      litert_torch_export.export(
          model=model_id,
          output_dir=export_dir,
          quantization_recipe='',  # Disable quantization
          bundle_litert_lm=True,
          cache_length=_MAX_NUM_TOKENS.value,
          externalize_embedder=_EXTERNALIZE_EMBEDDER.value,
          single_token_embedder=_SINGLE_TOKEN_EMBEDDER.value,
          split_cache=_SPLIT_CACHE.value,
          sliding_window_ring_buffer_size=_SLIDING_WINDOW_RING_BUFFER_SIZE.value,
      )

      exported_model_path = os.path.join(export_dir, 'model.litertlm')
    if not os.path.exists(exported_model_path):
      raise FileNotFoundError(
          f'Exported model not found at {exported_model_path}'
      )

    if _DIFFBASE_LITERT_LM_MODEL_PATH.value:
      tf_outputs = run_litert_lm(
          _DIFFBASE_LITERT_LM_MODEL_PATH.value,
          prompts,
          max_new_tokens,
          _MAX_NUM_TOKENS.value,
          _BACKEND.value,
      )
    else:
      tf_outputs = run_transformers(model_id, prompts, max_new_tokens)

    lite_outputs = run_litert_lm(
        exported_model_path,
        prompts,
        max_new_tokens,
        _MAX_NUM_TOKENS.value,
        _BACKEND.value,
    )

    print('\n=== Results ===')
    equivalent = True
    for i, (prompt, tf_out, lite_out) in enumerate(
        zip(prompts, tf_outputs, lite_outputs)
    ):
      print(f'\nTurn {i+1}:')
      print(f'  Prompt:       {prompt!r}')
      print(f'  Transformers: {tf_out!r}')
      print(f'  LiteRT LM:    {lite_out!r}')
      if tf_out != lite_out:
        equivalent = False
        print('  Status:       DIFFER')
      else:
        print('  Status:       EQUIVALENT')

    if equivalent:
      print('\nSUCCESS: All turns are equivalent!')
    else:
      print('\nFAILURE: Some turns differ!')

  finally:
    if not _WORK_DIR.value:
      print(f'Cleaning up export directory: {export_dir}')
      shutil.rmtree(export_dir)
    else:
      print(f'Keeping export directory: {export_dir}')


if __name__ == '__main__':
  app.run(main)
