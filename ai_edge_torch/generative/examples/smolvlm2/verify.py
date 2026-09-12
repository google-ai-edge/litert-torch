# Copyright 2024 The AI Edge Torch Authors.
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
"""Verifies the reauthored SmolVLM2 Text only model."""

import logging
from typing import Callable, Dict, List, Optional
from absl import app
from absl import flags

from ai_edge_torch.generative.examples.smolvlm2 import smolvlm2
from ai_edge_torch.generative.layers import kv_cache
from ai_edge_torch.generative.utilities import loader
from ai_edge_torch.generative.utilities import transformers_verifier
from ai_edge_torch.generative.utilities import verifier
import torch
import transformers

_PROMPTS_TEXT_ONLY = flags.DEFINE_multi_string(
    "prompts_text_only",
    "<|im_start|>User: What is the capital of"
    " France?<end_of_utterance>\nAssistant:",
    "The input prompts to generate answers only with text.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum number of generated tokens.",
)

_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "The checkpoint to verify.",
)


class ReauthoredSmolVLM2Wrapper(verifier.ReauthoredModelWrapper):
  """Reauthored SmolVLM2 Gemma model wrapper."""

  def _init_kv_cache(self):
    return kv_cache.KVCache.from_model_config(
        self.kv_cache_max_len,
        self.model.config.decoder_config,
        kv_layout=self.kv_layout,
    )


def verify_smolvlm2(
    checkpoint: str,
    prompts: List[str],
    max_new_tokens: int,
    custom_loader: Optional[Callable[[str], Dict[str, torch.Tensor]]] = None,
) -> bool:
  """Verifies the reauthored SmolVLM2 model.

  Args:
      checkpoint: Path to the SmolVLM2 checkpoint.
      prompts: List of prompts for generation.
      max_new_tokens: Maximum number of new tokens to generate.
      custom_loader: A custom loader to load the weights.

  Returns:
      True if the verification passes, False otherwise.
  """
  logging.info("Loading the original model from: %s", checkpoint)
  original_model = transformers.AutoModelForImageTextToText.from_pretrained(
      checkpoint
  )

  if custom_loader is None:
    custom_loader = loader.get_custom_loader("", "safetensors")

  reauthored_checkpoint = "/google/data/rw/users/ay/ayqzhang/smolvlm2_merged.safetensors"

  logging.info("Building the reauthored model from: %s", reauthored_checkpoint)
  reauthored_model = smolvlm2.build_model(
      reauthored_checkpoint,
      custom_loader=custom_loader,
      mask_cache_size=verifier.DEFAULT_KV_CACHE_MAX_LEN,
  )
  wrapped_reauthored_model = ReauthoredSmolVLM2Wrapper(reauthored_model)

  logging.info("Loading the tokenizer from: %s", checkpoint)
  tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
  return verifier.verify_reauthored_model(
      original_model=transformers_verifier.TransformersModelWrapper(
          original_model
      ),
      reauthored_model=wrapped_reauthored_model,
      tokenizer=verifier.TokenizerWrapper(tokenizer),
      generate_prompts=prompts,
      max_new_tokens=max_new_tokens,
      continue_on_failure=False,
      verify_inputs=False,
      atol=0.5e-02,
  )


def main(_):
  verify_smolvlm2(
      checkpoint=_CHECKPOINT.value,
      prompts=_PROMPTS_TEXT_ONLY.value,
      max_new_tokens=_MAX_NEW_TOKENS.value,
  )


if __name__ == "__main__":
  app.run(main)
