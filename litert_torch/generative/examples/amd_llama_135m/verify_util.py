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
"""Utils for verifying the AMD-Llama-135M model."""
import logging
import os
import pathlib
from typing import Callable, Dict

from litert_torch.generative.examples.amd_llama_135m import amd_llama_135m
from litert_torch.generative.utilities import loader
from litert_torch.generative.utilities import transformers_verifier
from litert_torch.generative.utilities import verifier
import torch
import transformers


DEFAULT_PROMPTS = ["Tell me a story?\nOnce upon a time"]


def verify_amd_llama_135m(
    checkpoint_dir: str,
    weight_filename: str = "model.safetensors",
    max_new_tokens: int = 30,
    prompts: list[str] | None = None,
    initialize_from_local: bool = True,
    custom_loader: Callable[[str], Dict[str, torch.Tensor]] | None = None,
) -> bool:
  """Verifies the reauthored AMD-Llama-135M model with a custom loader."""
  logging.info("Loading the original model from: %s", checkpoint_dir)
  original_model = transformers.AutoModelForCausalLM.from_pretrained(
      checkpoint_dir
  )

  logging.info("Building the reauthored model from: %s", checkpoint_dir)
  if custom_loader is None and not initialize_from_local:
    custom_loader = loader.get_custom_loader("", "safetensors")

  if initialize_from_local:
    # Locate the cached dir.
    cached_config_file = transformers.utils.cached_file(
        checkpoint_dir, transformers.utils.CONFIG_NAME
    )
    reauthored_checkpoint = pathlib.Path(cached_config_file).parent  # pyrefly: ignore[bad-argument-type]
  else:
    reauthored_checkpoint = os.path.join(checkpoint_dir, weight_filename)

  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = amd_llama_135m.build_model(
      checkpoint_path=reauthored_checkpoint,  # pyrefly: ignore[bad-argument-type]
      custom_loader=custom_loader,
      mask_cache_size=verifier.DEFAULT_KV_CACHE_MAX_LEN,
  )

  logging.info("Loading the tokenizer from: %s", checkpoint_dir)
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
  return verifier.verify_reauthored_model(
      original_model=transformers_verifier.TransformersModelWrapper(
          original_model
      ),
      reauthored_model=verifier.ReauthoredModelWrapper(reauthored_model),
      tokenizer=verifier.TokenizerWrapper(tokenizer),
      generate_prompts=DEFAULT_PROMPTS if prompts is None else prompts,
      max_new_tokens=max_new_tokens,
      atol=1e-04,
  )
