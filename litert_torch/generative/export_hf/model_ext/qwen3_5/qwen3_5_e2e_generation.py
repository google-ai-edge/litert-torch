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

"""Simple end-to-end static shape prefill and decode generation script for Qwen3.5."""

from collections.abc import Sequence

from absl import app
from absl import flags
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from litert_torch.generative.export_hf.core.exportable_module_config import ExportableModuleConfig
from litert_torch.generative.export_hf.model_ext.qwen3_5 import exportable_module as qwen3_5_exportable


_MODEL_ID = flags.DEFINE_string(
    "model_id",
    "Qwen/Qwen3.5-0.8B",
    "HuggingFace checkpoint path / model ID for Qwen 3.5.",
)
_PROMPT = flags.DEFINE_string(
    "prompt",
    "Explain why the sky is blue in two sentences.",
    "Input prompt for generation.",
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


def run_e2e_generation(
    prompt: str,
    model_id: str = "Qwen/Qwen3.5-0.8B",
    max_new_tokens: int = 32,
    prefill_chunk_size: int = 128,
    cache_length: int = 1280,
) -> None:
    print(f"Loading tokenizer and model from: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert tokenizer is not None
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    hf_model.eval()

    print("Formatting prompt with chat template...")
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    print(f"Prompt length: {prompt_len} tokens")
    if prompt_len > cache_length - max_new_tokens:
        raise ValueError(
            f"Prompt length ({prompt_len}) + max_new_tokens ({max_new_tokens})"
            f" exceeds cache_length ({cache_length})."
        )

    # 1. Reference HF Generation
    print("\n--- [Reference] Running HF Transformers generate ---")
    with torch.no_grad():
        hf_gen_ids = hf_model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False
        )
    hf_text = tokenizer.decode(
        hf_gen_ids[0, prompt_len:], skip_special_tokens=True
    )
    print(f"HF Output ({hf_gen_ids.shape[1] - prompt_len} tokens):\n{hf_text}")

    # 2. Setup LiteRT Static Prefill & Decode Exportable Modules
    print(
        f"\n--- [LiteRT Exportable] Setting up static prefill"
        f" ({prefill_chunk_size}) & decode (1) up to cache length"
        f" {cache_length} ---"
    )
    export_config = ExportableModuleConfig(
        model="qwen3_5",
        batch_size=1,
        cache_length=cache_length,
        prefill_lengths=[prefill_chunk_size],
        prefill_length_dim=None,  # Pure static shapes
        cache_implementation="LiteRTLMCache",
        k_ts_idx=2,
        v_ts_idx=3,
    )

    prefill_mod = (
        qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Prefill(
            hf_model, export_config
        )
    )
    decode_mod = (
        qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Generate(
            hf_model, export_config
        )
    )
    assert prefill_mod.model.config._attn_implementation == "lrt_transposed_attention"
    assert decode_mod.model.config._attn_implementation == "lrt_transposed_attention"

    # Get sample inputs to initialize static KV cache object
    sample_prefill = prefill_mod.get_sample_inputs(hf_model.config)[
        f"prefill_{prefill_chunk_size}"
    ][0]
    kv_cache = sample_prefill["kv_cache"]

    pad_token_id = getattr(tokenizer, "pad_token_id", 248044) or 248044

    # ==========================================
    # PHASE 1: PREFILLING (Static Chunked Processing)
    # ==========================================
    prefill_prompt_len = prompt_len - 1
    prefill_prompt_ids = input_ids[:, :prefill_prompt_len]
    print(
        f"\n--- [Phase 1: Prefill] Processing {prefill_prompt_len} prompt tokens"
        f" across static chunks of {prefill_chunk_size} ---"
    )
    num_chunks = (
        prefill_prompt_len + prefill_chunk_size - 1
    ) // prefill_chunk_size
    for c in range(num_chunks):
        start_idx = c * prefill_chunk_size
        end_idx = min(start_idx + prefill_chunk_size, prefill_prompt_len)
        chunk_slice = prefill_prompt_ids[:, start_idx:end_idx]
        chunk_len = chunk_slice.shape[1]

        if chunk_len < prefill_chunk_size:
            tokens = torch.full(
                (1, prefill_chunk_size),
                pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            tokens[:, :chunk_len] = chunk_slice
        else:
            tokens = chunk_slice

        input_pos = torch.arange(
            start_idx,
            start_idx + prefill_chunk_size,
            dtype=torch.int64,
            device=input_ids.device,
        )
        prefill_mask = qwen3_5_exportable.create_qwen3_5_attention_mask(
            prefill_chunk_size, cache_length, input_pos, dtype=torch.float32, device=input_ids.device
        )

        with torch.no_grad():
            prefill_out = prefill_mod(
                tokens=tokens,
                input_pos=input_pos,
                kv_cache=kv_cache,
                mask=prefill_mask,
            )
            kv_cache = prefill_out["kv_cache"]

    # ==========================================
    # PHASE 2: DECODING (Autoregressive Static Loop)
    # ==========================================
    print(
        "--- [Phase 2: Decode] Autoregressively generating tokens with static"
        " decode exportable ---"
    )
    eos_token_id = tokenizer.eos_token_id

    curr_token = input_ids[:, -1:]  # Last prompt token starts decode step 0
    generated_token_ids = []

    for step in range(max_new_tokens):
        curr_pos = torch.tensor(
            [prefill_prompt_len + step],
            dtype=torch.int64,
            device=input_ids.device,
        )
        decode_mask = qwen3_5_exportable.create_qwen3_5_attention_mask(
            1, cache_length, curr_pos, dtype=torch.float32, device=input_ids.device
        )

        with torch.no_grad():
            decode_out = decode_mod(
                tokens=curr_token,
                input_pos=curr_pos,
                kv_cache=kv_cache,
                mask=decode_mask,
            )
            kv_cache = decode_out["kv_cache"]
            logits = decode_out["logits"]

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        if next_token_id.item() == eos_token_id:
            break
        generated_token_ids.append(next_token_id.item())
        curr_token = next_token_id

    export_text = tokenizer.decode(
        generated_token_ids, skip_special_tokens=True
    )
    print(
        f"\nLiteRT Exportable Output ({len(generated_token_ids)} tokens):\n{export_text}"
    )
    print(f"\nExact match with HF output: {export_text == hf_text}")


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    run_e2e_generation(
        prompt=_PROMPT.value,
        model_id=_MODEL_ID.value,
        max_new_tokens=_MAX_NEW_TOKENS.value,
        prefill_chunk_size=_PREFILL_CHUNK_SIZE.value,
        cache_length=_CACHE_LENGTH.value,
    )


if __name__ == "__main__":
    app.run(main)
