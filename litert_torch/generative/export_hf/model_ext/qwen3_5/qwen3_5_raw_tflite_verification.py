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
"""Raw float TFLite models (`model.tflite` & `embedder.tflite`) vs. PyTorch exportables vs. HF reference check across short, long, and multi-turn prompts."""

from collections.abc import Sequence
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

from absl import app
from absl import flags
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from ai_edge_litert import interpreter
from litert_torch.generative.export_hf import export as litert_torch_export
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.export_lib import SourceModelArtifacts
from litert_torch.generative.export_hf.model_ext.qwen3_5.exportable_module import create_qwen3_5_attention_mask
from litert_torch.generative.export_hf.model_ext.qwen3_5.exportable_module import LiteRTExportableModuleForQwen3_5Generate
from litert_torch.generative.export_hf.model_ext.qwen3_5.exportable_module import LiteRTExportableModuleForQwen3_5Prefill


_MODEL_ID = flags.DEFINE_string(
    "model_id",
    "Qwen/Qwen3.5-0.8B",
    "HuggingFace checkpoint path / model ID for Qwen 3.5.",
)
_EXPORT_DIR = flags.DEFINE_string(
    "export_dir",
    "/tmp/qwen3_5_tflite_float_gje_vigr",
    "Directory containing exported float TFLite models (`model.tflite` and `embedder.tflite`). If not present, models will be exported.",
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
_LLM_METADATA_OVERRIDE_PATH = flags.DEFINE_string(
    "llm_metadata_override_path",
    "",
    "Path to LlmMetadataProto text proto override.",
)


def ensure_tflite_models_exist(export_dir: str, model_id: str, cache_length: int, prefill_chunk_size: int) -> Tuple[str, str]:
  model_path = os.path.join(export_dir, "model.tflite")
  embedder_path = os.path.join(export_dir, "embedder.tflite")
  if not os.path.exists(model_path) or not os.path.exists(embedder_path):
    print(f"[{model_path} or {embedder_path}] not found. Running float TFLite export right now...")
    os.makedirs(export_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    raw_jinja = getattr(tokenizer, "chat_template", "")
    clean_jinja = raw_jinja.replace(
        "'\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content",
        "'\\n' + content"
    )
    jinja_str = json.dumps(clean_jinja)
    if _LLM_METADATA_OVERRIDE_PATH.value:
      with open(_LLM_METADATA_OVERRIDE_PATH.value, "r") as f:
        override_pbtext = f.read()
    else:
      override_pbtext = f"""pad_token {{
  token_str: "<|endoftext|>"
}}
stop_tokens {{
  token_str: "<|im_end|>"
}}
stop_tokens {{
  token_ids {{
    ids: 248044
  }}
}}
sampler_params {{
  type: TOP_P
  k: 1
  p: 1.0
  temperature: 0.0
}}
max_num_tokens: {cache_length}
llm_model_type {{
  generic_model {{
  }}
}}
jinja_prompt_template: {jinja_str}
"""
    litert_torch_export.export(
        model=model_id,
        output_dir=export_dir,
        quantization_recipe="",  # float tflite
        keep_temporary_files=True,
        bundle_litert_lm=True,
        cache_length=cache_length,
        prefill_lengths=[prefill_chunk_size],
        externalize_embedder=True,
        single_token_embedder=True,
        use_jinja_template=True,
        litert_lm_llm_metadata_override=override_pbtext,
    )
  return model_path, embedder_path


def run_transformers_reference(
    hf_model: AutoModelForCausalLM,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[str, List[int]]:
  eos_ids = []
  if tokenizer.eos_token_id is not None:
    eos_ids.append(tokenizer.eos_token_id)
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  if im_end_id is not None and im_end_id not in eos_ids:
    eos_ids.append(im_end_id)

  with torch.no_grad():
    outputs = hf_model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos_ids if eos_ids else None,
    )
  input_length = input_ids.shape[-1]
  generated_tokens = outputs[0][input_length:].tolist()
  out_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
  return out_text, generated_tokens


def run_pytorch_exportable_pipeline(
    prefill_mod: LiteRTExportableModuleForQwen3_5Prefill,
    decode_mod: LiteRTExportableModuleForQwen3_5Generate,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    prefill_chunk_size: int,
    cache_length: int,
    device: str = "cpu",
) -> Tuple[str, List[int]]:
  prompt_len = input_ids.shape[-1]
  prefill_prompt_len = prompt_len - 1
  pad_token_id = getattr(tokenizer, "pad_token_id", 248044)
  if pad_token_id is None or pad_token_id < 0:
    pad_token_id = 248044

  tokens = torch.full((1, prefill_chunk_size), pad_token_id, dtype=torch.int64, device=device)
  tokens[:, :prefill_prompt_len] = input_ids[:, :prefill_prompt_len]

  input_pos = torch.arange(prefill_chunk_size, dtype=torch.int64, device=device)
  prefill_mask = create_qwen3_5_attention_mask(
      seq_len=prefill_chunk_size,
      cache_length=cache_length,
      input_pos=input_pos,
      dtype=torch.float32,
      device=device,
  )
  prefill_inputs = prefill_mod.get_sample_inputs(prefill_mod.model.config)[f"prefill_{prefill_chunk_size}"][0]
  sample_kv = prefill_inputs["kv_cache"]

  with torch.no_grad():
    prefill_out = prefill_mod(
        tokens=tokens,
        input_pos=input_pos,
        kv_cache=sample_kv,
        mask=prefill_mask,
    )
  updated_kv = prefill_out["kv_cache"]

  # Decode step 0 takes the last prompt token at prefill_prompt_len
  next_token = input_ids[:, prefill_prompt_len : prefill_prompt_len + 1]
  generated_tokens = []
  eos_ids = [tokenizer.eos_token_id]
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  if im_end_id is not None:
    eos_ids.append(im_end_id)

  cur_pos = prefill_prompt_len
  for step in range(max_new_tokens):
    dec_pos = torch.tensor([cur_pos], dtype=torch.int64, device=device)
    dec_mask = create_qwen3_5_attention_mask(
        seq_len=1,
        cache_length=cache_length,
        input_pos=dec_pos,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
      dec_out = decode_mod(
          tokens=next_token,
          input_pos=dec_pos,
          kv_cache=updated_kv,
          mask=dec_mask,
      )
    logits, updated_kv = dec_out["logits"], dec_out["kv_cache"]
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated_tokens.append(next_token.item())
    cur_pos += 1
    if generated_tokens[-1] in eos_ids:
      break

  out_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
  return out_text, generated_tokens


def run_raw_tflite_pipeline(
    prefill_runner: Any,
    decode_runner: Any,
    embedder_prefill_runner: Any,
    embedder_decode_runner: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    prefill_chunk_size: int,
    cache_length: int,
    initial_kv: Dict[str, np.ndarray],
) -> Tuple[str, List[int]]:
  prompt_len = input_ids.shape[-1]
  prefill_prompt_len = prompt_len - 1
  pad_token_id = getattr(tokenizer, "pad_token_id", 248044)
  if pad_token_id is None or pad_token_id < 0:
    pad_token_id = 248044

  tokens = np.full((1, prefill_chunk_size), pad_token_id, dtype=np.int64)
  tokens[:, :prefill_prompt_len] = input_ids.cpu().numpy()[:, :prefill_prompt_len]

  input_pos = np.arange(prefill_chunk_size, dtype=np.int64)
  prefill_mask = create_qwen3_5_attention_mask(
      seq_len=prefill_chunk_size,
      cache_length=cache_length,
      input_pos=torch.from_numpy(input_pos),
      dtype=torch.float32,
      device="cpu",
  ).numpy()

  prefill_kwargs = dict(initial_kv)
  prefill_kwargs["input_pos"] = input_pos
  prefill_kwargs["mask"] = prefill_mask

  if hasattr(prefill_runner, "_inputs") and "embeddings" in prefill_runner._inputs and embedder_prefill_runner is not None:
    embed_details = embedder_prefill_runner.get_input_details()
    embed_dtype = embed_details["token_ids"]["dtype"] if "token_ids" in embed_details else np.int32
    embed_out = embedder_prefill_runner(token_ids=tokens.astype(embed_dtype))
    prefill_kwargs["embeddings"] = list(embed_out.values())[0]
  else:
    prefill_kwargs["tokens"] = tokens

  prefill_details = prefill_runner.get_input_details()
  for k in list(prefill_kwargs.keys()):
    if k in prefill_details and hasattr(prefill_kwargs[k], "astype"):
      prefill_kwargs[k] = prefill_kwargs[k].astype(prefill_details[k]["dtype"])

  prefill_out = prefill_runner(**prefill_kwargs)

  updated_kv = {}
  for k, v in prefill_out.items():
    updated_kv[k] = v

  next_token_id = int(input_ids.cpu().numpy()[0, prefill_prompt_len])
  generated_tokens = []

  eos_ids = [tokenizer.eos_token_id]
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  if im_end_id is not None:
    eos_ids.append(im_end_id)

  cur_pos = prefill_prompt_len
  for step in range(max_new_tokens):
    dec_pos = np.array([cur_pos], dtype=np.int64)
    dec_mask = create_qwen3_5_attention_mask(
        seq_len=1,
        cache_length=cache_length,
        input_pos=torch.from_numpy(dec_pos),
        dtype=torch.float32,
        device="cpu",
    ).numpy()

    dec_kwargs = dict(updated_kv)
    dec_kwargs["input_pos"] = dec_pos
    dec_kwargs["mask"] = dec_mask

    if hasattr(decode_runner, "_inputs") and "embeddings" in decode_runner._inputs and embedder_decode_runner is not None:
      next_token_arr = np.array([[next_token_id]], dtype=np.int64)
      embed_details = embedder_decode_runner.get_input_details()
      embed_dtype = embed_details["token_ids"]["dtype"] if "token_ids" in embed_details else np.int32
      embed_out = embedder_decode_runner(token_ids=next_token_arr.astype(embed_dtype))
      dec_kwargs["embeddings"] = list(embed_out.values())[0]
    else:
      dec_kwargs["tokens"] = np.array([[next_token_id]], dtype=np.int64)

    decode_details = decode_runner.get_input_details()
    for k in list(dec_kwargs.keys()):
      if k in decode_details and hasattr(dec_kwargs[k], "astype"):
        dec_kwargs[k] = dec_kwargs[k].astype(decode_details[k]["dtype"])

    dec_out = decode_runner(**dec_kwargs)
    logits = None
    updated_kv = {}
    for k, v in dec_out.items():
      if "logits" in k or k == "output_0":
        logits = v
      else:
        updated_kv[k] = v

    assert logits is not None
    next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
    generated_tokens.append(next_token_id)
    cur_pos += 1
    if generated_tokens[-1] in eos_ids:
      break

  out_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
  return out_text, generated_tokens


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  model_id = _MODEL_ID.value
  export_dir = _EXPORT_DIR.value
  max_new_tokens = _MAX_NEW_TOKENS.value
  prefill_chunk_size = _PREFILL_CHUNK_SIZE.value
  cache_length = _CACHE_LENGTH.value

  model_path, embedder_path = ensure_tflite_models_exist(export_dir, model_id, cache_length, prefill_chunk_size)

  print(f"\n=== Inspecting Signatures of {model_path} and {embedder_path} ===")
  interp_model = interpreter.Interpreter(model_path=model_path)
  model_sigs = interp_model.get_signature_list()
  print(f"model.tflite signatures: {model_sigs}")

  details = interp_model.get_tensor_details()
  print(f"\n[Graph Inspection of {model_path}] Total tensors: {len(details)}")
  rank5 = [f"{t['name']} shape={t['shape']} dtype={t['dtype']}" for t in details if len(t['shape']) >= 5]
  print(f"Tensors with rank >= 5 (SELECT_V2/SLICE risk): {len(rank5)}")
  for r in rank5[:15]:
    print("  Rank>=5 tensor:", r)
  int64_t = [f"{t['name']} shape={t['shape']}" for t in details if t['dtype'] == np.int64]
  print(f"INT64 tensors (SUM/reduction risk): {len(int64_t)}")
  for i in int64_t[:15]:
    print("  INT64 tensor:", i)

  interp_embed = interpreter.Interpreter(model_path=embedder_path)
  embed_sigs = interp_embed.get_signature_list()
  print(f"embedder.tflite signatures: {embed_sigs}")

  # Determine exact runner signature names
  prefill_sig_name = f"prefill_{prefill_chunk_size}" if f"prefill_{prefill_chunk_size}" in model_sigs else list(model_sigs.keys())[0]
  decode_sig_name = "decode_1" if "decode_1" in model_sigs else ("decode" if "decode" in model_sigs else list(model_sigs.keys())[1])
  print(f"Selected model.tflite signature runners: prefill='{prefill_sig_name}', decode='{decode_sig_name}'")

  prefill_runner = interp_model.get_signature_runner(prefill_sig_name)
  decode_runner = interp_model.get_signature_runner(decode_sig_name)

  embedder_prefill_runner = None
  embedder_decode_runner = None
  if embed_sigs:
    prefill_embed_sig = f"prefill_embedder_{prefill_chunk_size}" if f"prefill_embedder_{prefill_chunk_size}" in embed_sigs else ("serving_default" if "serving_default" in embed_sigs else list(embed_sigs.keys())[0])
    decode_embed_sig = "decode_embedder_1" if "decode_embedder_1" in embed_sigs else ("serving_default" if "serving_default" in embed_sigs else list(embed_sigs.keys())[-1])
    print(f"Selected embedder.tflite signature runners: prefill='{prefill_embed_sig}', decode='{decode_embed_sig}'")
    embedder_prefill_runner = interp_embed.get_signature_runner(prefill_embed_sig)
    embedder_decode_runner = interp_embed.get_signature_runner(decode_embed_sig)

  print(f"\n=== Loading HF Reference Model & PyTorch Exportables ({model_id}) ===")
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  assert tokenizer is not None
  hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
  hf_model.eval()

  export_config = exportable_module_config.ExportableModuleConfig(
      model=model_id,
      task=exportable_module_config.ExportTask.TEXT_GENERATION,
      prefill_lengths=[prefill_chunk_size],
      cache_length=cache_length,
      externalize_embedder=True,
  )
  source_model_artifacts = SourceModelArtifacts(
      model=hf_model,
      model_config=hf_model.config,
      text_model_config=hf_model.config,
      tokenizer=tokenizer,  # pyrefly: ignore[bad-argument-type]
  )
  prefill_mod = LiteRTExportableModuleForQwen3_5Prefill(hf_model, export_config, source_model_artifacts).eval()
  decode_mod = LiteRTExportableModuleForQwen3_5Generate(hf_model, export_config, source_model_artifacts).eval()

  # Construct initial zero KV cache numpy dictionary matching exact TFLite runner inputs (all 48 KV tensors)
  initial_kv_np = {}
  for name, detail in prefill_runner.get_input_details().items():
    if name.startswith("kv_cache_"):
      initial_kv_np[name] = np.zeros(detail["shape"], dtype=detail["dtype"])

  # 1. Short Prompt Check
  print("\n=========================================================")
  print("Check 1: Short Prompt Generation")
  print("=========================================================")
  short_prompt = "Explain why the sky is blue in two sentences."
  chat_msg = [{"role": "user", "content": short_prompt}]
  formatted_prompt = tokenizer.apply_chat_template(chat_msg, tokenize=False, add_generation_prompt=True)
  input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"]

  hf_text, hf_toks = run_transformers_reference(hf_model, tokenizer, input_ids, max_new_tokens)
  pt_text, pt_toks = run_pytorch_exportable_pipeline(prefill_mod, decode_mod, tokenizer, input_ids, max_new_tokens, prefill_chunk_size, cache_length)
  tfl_text, tfl_toks = run_raw_tflite_pipeline(prefill_runner, decode_runner, embedder_prefill_runner, embedder_decode_runner, tokenizer, input_ids, max_new_tokens, prefill_chunk_size, cache_length, initial_kv_np)

  print(f"HF Reference ({len(hf_toks)} toks):        {hf_text!r}")
  print(f"PyTorch Exportable ({len(pt_toks)} toks):  {pt_text!r}")
  print(f"Raw Float TFLite ({len(tfl_toks)} toks):    {tfl_text!r}")
  print(f"Exact Match (HF vs. PyTorch): {hf_toks == pt_toks}")
  print(f"Exact Match (HF vs. TFLite):  {hf_toks == tfl_toks}")
  print(f"Exact Match (PyTorch vs TFLite): {pt_toks == tfl_toks}")

  # 2. Long Prompt Check (~80 tokens)
  print("\n=========================================================")
  print("Check 2: Long Prompt Generation (~80 tokens)")
  print("=========================================================")
  long_prompt = (
      "Please summarize the following explanation into a single concise sentence: "
      "Light scatters off the Earth's atmosphere, which is denser at the bottom where it contains more molecules, "
      "causing shorter wavelengths of blue light to bend away from the sun and scatter in all directions across the sky, "
      "whereas longer wavelengths like red and yellow pass straight through without scattering nearly as much."
  )
  chat_msg = [{"role": "user", "content": long_prompt}]
  formatted_prompt = tokenizer.apply_chat_template(chat_msg, tokenize=False, add_generation_prompt=True)
  input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"]

  hf_text, hf_toks = run_transformers_reference(hf_model, tokenizer, input_ids, max_new_tokens)
  pt_text, pt_toks = run_pytorch_exportable_pipeline(prefill_mod, decode_mod, tokenizer, input_ids, max_new_tokens, prefill_chunk_size, cache_length)
  tfl_text, tfl_toks = run_raw_tflite_pipeline(prefill_runner, decode_runner, embedder_prefill_runner, embedder_decode_runner, tokenizer, input_ids, max_new_tokens, prefill_chunk_size, cache_length, initial_kv_np)

  print(f"HF Reference ({len(hf_toks)} toks):        {hf_text!r}")
  print(f"PyTorch Exportable ({len(pt_toks)} toks):  {pt_text!r}")
  print(f"Raw Float TFLite ({len(tfl_toks)} toks):    {tfl_text!r}")
  print(f"Exact Match (HF vs. PyTorch): {hf_toks == pt_toks}")
  print(f"Exact Match (HF vs. TFLite):  {hf_toks == tfl_toks}")
  print(f"Exact Match (PyTorch vs TFLite): {pt_toks == tfl_toks}")

  # 3. Multi-Turn Check
  print("\n=========================================================")
  print("Check 3: Multi-Turn Conversation (3 Turns)")
  print("=========================================================")
  turns = [
      "What is the capital of France?",
      "What is its approximate population?",
      "Name one famous art museum located in that city.",
  ]
  multi_chat = []
  for i, turn_prompt in enumerate(turns):
    print(f"\n--- [Turn {i+1}] User: {turn_prompt!r} ---")
    multi_chat.append({"role": "user", "content": turn_prompt})
    formatted_prompt = tokenizer.apply_chat_template(multi_chat, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"]

    hf_text, hf_toks = run_transformers_reference(hf_model, tokenizer, input_ids, max_new_tokens)
    pt_text, pt_toks = run_pytorch_exportable_pipeline(prefill_mod, decode_mod, tokenizer, input_ids, max_new_tokens, prefill_chunk_size, cache_length)
    tfl_text, tfl_toks = run_raw_tflite_pipeline(prefill_runner, decode_runner, embedder_prefill_runner, embedder_decode_runner, tokenizer, input_ids, max_new_tokens, prefill_chunk_size, cache_length, initial_kv_np)

    print(f"HF Reference:       {hf_text!r}")
    print(f"PyTorch Exportable: {pt_text!r}")
    print(f"Raw Float TFLite:   {tfl_text!r}")
    print(f"Turn {i+1} Exact Match (HF vs. PyTorch): {hf_toks == pt_toks}")
    print(f"Turn {i+1} Exact Match (HF vs. TFLite):  {hf_toks == tfl_toks}")
    print(f"Turn {i+1} Exact Match (PyTorch vs TFLite): {pt_toks == tfl_toks}")
    multi_chat.append({"role": "assistant", "content": hf_text})

  print("\n=========================================================")
  print("Raw TFLite vs. PyTorch Exportable vs. HF Reference Verification Completed!")
  print("=========================================================")


if __name__ == "__main__":
  app.run(main)
