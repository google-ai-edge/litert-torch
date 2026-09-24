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
"""Verifies the reauthored EmbeddingGemma-300M model."""

from absl import app
from absl import flags

from litert_torch.generative.examples.embedding_gemma import verify_util

_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    None,
    "The input prompts to generate embeddings for.",
)
_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    None,
    "The directory containing the model checkpoint.",
    required=True,
)
_LONG_INPUT_PROMPT_PATH = flags.DEFINE_string(
    "long_input_prompt_path",
    None,
    "Whether to enable the long input test.",
)
_FINAL_L2_NORM = flags.DEFINE_bool(
    "final_l2_norm",
    True,
    "Whether to apply final L2 normalization to the embeddings.",
)


def main(_):
  if not verify_util.verify_embedding_gemma_300m(
      checkpoint_dir=_CHECKPOINT.value,
      prompts=_PROMPTS.value,
      long_input_prompt_path=_LONG_INPUT_PROMPT_PATH.value,
      final_l2_norm=_FINAL_L2_NORM.value,
  ):
    exit(1)


if __name__ == "__main__":
  app.run(main)
