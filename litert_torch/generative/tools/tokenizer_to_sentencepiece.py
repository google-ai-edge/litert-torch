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

"""Builds a SentencePieceModel protobuf from a HuggingFace tokenizer.

If a SentencePieceModel protobuf file is already available, it copies the
SentencePieceModel protobuf file instead of building a new one.

If not, it tries to build a SentencePieceModel protobuf file from the tokenizer
config files.

Please note that the SentencePirceModel protobuf would not output the same token
IDs as the tokenizer for all input strings because the conversion relies on
heuristics. For example, SentencePiece model built from Llama3.2 tokenizer with
"decode" normalization has around 1% mismatch ratio. It's user's responsibility
to verify the quality of the built SentencePiece model.
"""

import logging

from absl import app
from absl import flags
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as lib
import transformers

import sentencepiece as spm

_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    None,
    "The path to the checkpoint where the tokenizer config files are.",
)

_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    None,
    "The path of the output SentencePieceModel protobuf file.",
)

_STRINGS_TO_VERIFY = flags.DEFINE_list(
    "strings_to_verify",
    [
        "Hello, world! How are you?",
        "Instruct: write a python program to add 2 and 3.",
    ],
    "The strings to verify the SentencePieceModel protobuf file.",
)

_NORMALIZE_TOKENS = flags.DEFINE_enum(
    "normalize_tokens",
    "decode",
    ["none", "gpt2", "decode"],
    "Normalize tokens of the original tokenizer to be compatible with "
    "SentencePiece model.\n"
    "  none:   do not normalize the tokens\n"
    "  gpt2:   apply gpt-2 unicode_to_byte conversion\n"
    "  decode: call tokenizer.decode([token id]) for each token",
)

_NUM_PAIRS_TO_VERIFY = flags.DEFINE_integer(
    "num_pairs_to_verify",
    1000,
    "The number of pairs to verify the SentencePieceModel protobuf file.",
)


def main(_):
  tokenizer = transformers.AutoTokenizer.from_pretrained(_CHECKPOINT.value)
  spm_serialized = lib.convert(tokenizer, _NORMALIZE_TOKENS.value)

  spm_tokenizer = spm.SentencePieceProcessor()
  spm_tokenizer.LoadFromSerializedProto(spm_serialized)
  lib.verify_spm_tokenizer(
      tokenizer,
      spm_tokenizer,
      _STRINGS_TO_VERIFY.value,
      _NUM_PAIRS_TO_VERIFY.value,
  )

  logging.info(
      "Writing the SentencePieceModel protobuf file to: %s", _OUTPUT_PATH.value
  )
  with open(_OUTPUT_PATH.value, "wb") as f:
    f.write(spm_serialized)


if __name__ == "__main__":
  app.run(main)
