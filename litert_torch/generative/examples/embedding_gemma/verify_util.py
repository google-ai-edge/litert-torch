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
"""Verification utilities for EmbeddingGemma-300M."""

import os

from litert_torch.generative.examples.embedding_gemma import embedding_gemma
from sentence_transformers import SentenceTransformer
import torch
import transformers


DEFAULT_PROMPTS = [
    "What is the meaning of life?",
    "This is an example sentence.",
]

_MODEL_PATH = "google/embeddinggemma-300m"
_LONG_INPUT_PROMPT_PATH = "long_input_prompt_test.txt"


def verify_embedding_gemma_300m(
    checkpoint_dir: str = None,
    prompts: list[str] | None = None,
    long_input_prompt_path: str = None,
    atol: float = 0.0,
    final_l2_norm: bool = True,
) -> bool:
  """Verifies EmbeddingGemma-300M."""

  model_path = _MODEL_PATH
  print(f"Loading the original model from: {model_path}")
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
  try:
    original_model = SentenceTransformer(model_path)
    original_model.eval()
  except Exception as e:  # pylint: disable=broad-except
    print(f"Failed to load original model: {e}")
    return False

  print(f"Loading reauthored model from: {checkpoint_dir}")
  try:
    reauthored_model = embedding_gemma.build_model(
        checkpoint_dir, final_l2_norm=final_l2_norm
    )
    reauthored_model.eval()
  except Exception as e:  # pylint: disable=broad-except
    print(f"Failed to build or load reauthored model: {e}")
    return False

  prompts_to_run = prompts if prompts is not None else DEFAULT_PROMPTS
  if long_input_prompt_path is not None:
    long_prmpt = long_input_prompt_path
    if not os.path.exists(
        long_prmpt
    ):  # try to find the file in the checkpoint dir if not found in the given
      # long_input_prompt_path.
      long_prmpt = os.path.join(checkpoint_dir, _LONG_INPUT_PROMPT_PATH)

    if os.path.exists(long_prmpt):
      with open(long_prmpt, "r", encoding="utf-8") as f:
        content_string = f.read()
        long_prompt = content_string.strip()
        prompts_to_run.append(long_prompt)
    else:
      print(
          "Not running long input prompt test as didnt find any prompt file in"
          f" {long_prmpt} or {long_input_prompt_path}."
      )

  print(f"Tokenizing prompts: {prompts_to_run}")
  inputs = tokenizer(
      prompts_to_run, return_tensors="pt", padding=True, truncation=True
  )
  tokens, attention_mask = inputs["input_ids"], inputs["attention_mask"]
  print(f"Tokenized inputs (tokens): {tokens}")
  print(f"Attention mask: {attention_mask}")

  # --- Compare Final Embeddings ---
  print("\n--- Comparing Final Embeddings ---")
  with torch.no_grad():
    # Get embeddings from the original SentenceTransformer model.
    final_original_output = original_model(
        inputs
    )  # pytype: disable=wrong-arg-types
    original_embedding = final_original_output["sentence_embedding"]

    # Get embeddings from the reauthored model.
    reauthored_embedding = reauthored_model(
        tokens, attention_mask=attention_mask
    )

  print(f"Original embedding shape: {original_embedding.shape}")
  print(f"Reauthored embedding shape: {reauthored_embedding.shape}")
  print(f"Original embedding norm: {torch.norm(original_embedding)}")
  print(f"Reauthored embedding norm: {torch.norm(reauthored_embedding)}")

  if not torch.allclose(original_embedding, reauthored_embedding, atol=atol):
    print("Verification failed: Final outputs do not match!")
    print(
        "Max difference:"
        f" {torch.max(torch.abs(original_embedding - reauthored_embedding))}"
    )
    return False

  print("Verification successful: Final outputs match.")
  return True
