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

"""Verifies the reauthored SmolLM model."""

from absl import app
from absl import flags
from litert_torch.generative.examples.smollm import verify_util


_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "What is the meaning of life?",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)
_MODEL_VERSION = flags.DEFINE_enum(
    "model_version",
    "v1",
    ["v1", "v2", "v3"],
    "The version of SmolLm to verify.",
)
_CHECKPOINT = {
    "v1": "HuggingFaceTB/SmolLM-135M-Instruct",
    "v2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "v3": "HuggingFaceTB/SmolLM3-3B",
}


def main(_):
  verify_util.verify_smollm_135m(
      model_version=_MODEL_VERSION.value,
      checkpoint_dir=_CHECKPOINT[_MODEL_VERSION.value],
      max_new_tokens=_MAX_NEW_TOKENS.value,
      prompts=_PROMPTS.value,
  )


if __name__ == "__main__":
  app.run(main)
