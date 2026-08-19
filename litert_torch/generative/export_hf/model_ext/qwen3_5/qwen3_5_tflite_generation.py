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
"""Float TFLite export and end-to-end generation check for Qwen 3.5 across short, long, and multi-turn prompts."""

from collections.abc import Sequence
import json
import os
import shutil
import tempfile
from typing import Any

from absl import app
from absl import flags
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import litert_lm
from litert_torch.generative.export_hf import export as litert_torch_export


_MODEL_ID = flags.DEFINE_string(
    "model_id",
    "Qwen/Qwen3.5-0.8B",
    "HuggingFace checkpoint path / model ID for Qwen 3.5.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "",
    "Directory to export float TFLite models to. If empty, a temporary directory is created and preserved.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    32,
    "Maximum number of new tokens to generate.",
)
_PREFILL_CHUNK_SIZE = flags.DEFINE_integer(
    "prefill_chunk_size",
    128,
    "Static prefill chunk length.",
)
_CACHE_LENGTH = flags.DEFINE_integer(
    "cache_length",
    1280,
    "Maximum static context length (`cache_length`).",
)
_USE_JINJA = flags.DEFINE_bool(
    "use_jinja_template",
    False,
    "Whether to package the exact Jinja chat template into LiteRT LM metadata.",
)
_LLM_METADATA_OVERRIDE_PATH = flags.DEFINE_string(
    "llm_metadata_override_path",
    None,
    "Path to LlmMetadataProto text proto override.",
    readonly=True,
)


def extract_litert_lm_response(conv: Any, prompt: str, max_new_tokens: int) -> str:
  response = conv.send_message(prompt, max_output_tokens=max_new_tokens)
  if isinstance(response, str):
    return response.strip()
  text_pieces = []
  if isinstance(response, dict):
    for item in response.get("content", []):
      if isinstance(item, dict) and item.get("type") == "text":
        text_pieces.append(str(item.get("text", "")))
  return "".join(text_pieces).strip()


def run_transformers_chat(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    chat_messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
  formatted_prompt = tokenizer.apply_chat_template(
      chat_messages, tokenize=False, add_generation_prompt=True
  )
  inputs = tokenizer(formatted_prompt, return_tensors="pt")
  eos_ids = []
  if tokenizer.eos_token_id is not None:
    eos_ids.append(tokenizer.eos_token_id)
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  if im_end_id is not None and im_end_id not in eos_ids:
    eos_ids.append(im_end_id)
  with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_ids if eos_ids else None,
    )
  input_length = inputs["input_ids"].shape[-1]
  generated_tokens = outputs[0][input_length:]
  return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def run_tflite_generation_checks() -> None:
  model_id = _MODEL_ID.value
  output_dir = _OUTPUT_DIR.value
  max_new_tokens = _MAX_NEW_TOKENS.value
  prefill_chunk_size = _PREFILL_CHUNK_SIZE.value
  cache_length = _CACHE_LENGTH.value

  if not output_dir:
    output_dir = tempfile.mkdtemp(prefix="qwen3_5_tflite_float_")
  os.makedirs(output_dir, exist_ok=True)
  print(f"=== Exporting {model_id} to float TFLite in: {output_dir} ===")

  litert_torch_export.export(
      model=model_id,
      output_dir=output_dir,
      quantization_recipe="",  # Float TFLite export
      keep_temporary_files=True,
      bundle_litert_lm=True,
      cache_length=cache_length,
      prefill_lengths=[prefill_chunk_size],
      externalize_embedder=True,
      single_token_embedder=True,
      use_jinja_template=True,
      litert_lm_llm_metadata_override=_LLM_METADATA_OVERRIDE_PATH.value,
  )

  exported_model_path = os.path.join(output_dir, "model.litertlm")
  if not os.path.exists(exported_model_path):
    raise FileNotFoundError(f"Exported bundle not found at {exported_model_path}")

  print(f"\n=== Loading HF Reference Model ({model_id}) ===")
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  assert tokenizer is not None
  hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
  hf_model.eval()

  print(f"\n=== Loading LiteRT LM Engine (CPU Backend) from {exported_model_path} ===")
  backend = litert_lm.Backend.CPU()
  engine = litert_lm.Engine(
      exported_model_path,
      backend,
      max_num_tokens=cache_length,
  )
  sampler_config = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0)

  all_passed = True

  # 1. Short Prompt Check
  print("\n---------------------------------------------------------")
  print("Check 1: Short Prompt Generation")
  print("---------------------------------------------------------")
  short_prompt = "Explain why the sky is blue in two sentences."
  short_chat = [{"role": "user", "content": short_prompt}]
  hf_short_out = run_transformers_chat(hf_model, tokenizer, short_chat, max_new_tokens)

  with engine.create_conversation(sampler_config=sampler_config) as conv:
    lite_short_out = extract_litert_lm_response(conv, short_prompt, max_new_tokens)

  print(f"Prompt:        {short_prompt!r}")
  print(f"Transformers:  {hf_short_out!r}")
  print(f"LiteRT TFLite: {lite_short_out!r}")
  match_short = (hf_short_out == lite_short_out)
  print(f"Exact Match:   {match_short}")
  if not match_short:
    all_passed = False

  # 2. Long Prompt Check
  print("\n---------------------------------------------------------")
  print("Check 2: Long Prompt Generation (~80 tokens)")
  print("---------------------------------------------------------")
  long_prompt = (
      "Please summarize the following explanation into a single concise sentence: "
      "Light scatters off the Earth's atmosphere, which is denser at the bottom where it contains more molecules, "
      "causing shorter wavelengths of blue light to bend away from the sun and scatter in all directions across the sky, "
      "whereas longer wavelengths like red and yellow pass straight through without scattering nearly as much."
  )
  long_chat = [{"role": "user", "content": long_prompt}]
  hf_long_out = run_transformers_chat(hf_model, tokenizer, long_chat, max_new_tokens)

  with engine.create_conversation(sampler_config=sampler_config) as conv:
    lite_long_out = extract_litert_lm_response(conv, long_prompt, max_new_tokens)

  print(f"Prompt:        {long_prompt[:60]}... (len={len(tokenizer(long_prompt)['input_ids'])})")
  print(f"Transformers:  {hf_long_out!r}")
  print(f"LiteRT TFLite: {lite_long_out!r}")
  match_long = (hf_long_out == lite_long_out)
  print(f"Exact Match:   {match_long}")
  if not match_long:
    all_passed = False

  # 3. Multi-Turn Conversation Check
  print("\n---------------------------------------------------------")
  print("Check 3: Multi-Turn Conversation (3 Turns)")
  print("---------------------------------------------------------")
  turns = [
      "What is the capital of France?",
      "What is its approximate population?",
      "Name one famous art museum located in that city.",
  ]
  multi_chat = []
  multi_lite_outs = []
  multi_hf_outs = []

  with engine.create_conversation(sampler_config=sampler_config) as conv:
    for i, turn_prompt in enumerate(turns):
      print(f"\n[Turn {i+1}] User: {turn_prompt!r}")
      multi_chat.append({"role": "user", "content": turn_prompt})
      hf_turn_out = run_transformers_chat(hf_model, tokenizer, multi_chat, max_new_tokens)
      multi_hf_outs.append(hf_turn_out)
      multi_chat.append({"role": "assistant", "content": hf_turn_out})

      lite_turn_out = extract_litert_lm_response(conv, turn_prompt, max_new_tokens)
      multi_lite_outs.append(lite_turn_out)

      print(f"Transformers:  {hf_turn_out!r}")
      print(f"LiteRT TFLite: {lite_turn_out!r}")
      match_turn = (hf_turn_out == lite_turn_out)
      print(f"Turn {i+1} Match: {match_turn}")
      if not match_turn:
        all_passed = False

  print("\n=========================================================")
  print(f"Final Verification Summary (All checks exactly matched: {all_passed})")
  print(f"Generated float TFLite models preserved at: {output_dir}")
  print("=========================================================")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  run_tflite_generation_checks()


if __name__ == "__main__":
  app.run(main)
