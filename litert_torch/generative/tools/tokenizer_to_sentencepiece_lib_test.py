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
"""Tests for tokenizer_to_sentencepiece_lib.

The tests build a small GPT-2 style byte-level BPE tokenizer with the
`tokenizers` library (no network access), convert it, and check that the
SentencePiece model encodes exactly like the original tokenizer — in
particular for the byte-level failure modes: standalone non-ASCII characters
must not collapse to a byte token's id, characters without a piece must go
through byte fallback instead of UNK, and special tokens must match in text
as a single piece.
"""

import tokenizers
from tokenizers import models as tokenizers_models
from tokenizers import pre_tokenizers
from tokenizers import decoders
from tokenizers import trainers
import transformers

from absl.testing import absltest
from absl.testing import parameterized

from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib

from sentencepiece import sentencepiece_model_pb2 as spm_model
import sentencepiece as spm

_PIECE_TYPE = spm_model.ModelProto.SentencePiece

# Standalone characters U+00A1..U+017F (Latin-1 Supplement + Latin Extended-A).
# The GPT-2 byte<->unicode table spells the 256 byte tokens with exactly these
# code points (plus ASCII), which is where a byte token's id can shadow the
# real character.
_LATIN_SWEEP = [chr(c) for c in range(0xA1, 0x180)]

_CORPUS = (
    ["é", "és", "é là"] * 40
    + ["café résumé naïve entrée Zürich señor über garçon"] * 30
    + ["The quick brown fox jumps over the lazy dog."] * 30
    + ["def f(x):\n    return x**2 + 1"] * 20
    + ["2025 12345 3.14159 1,000,000"] * 20
)


def _build_tokenizer(
    with_unk_token: bool = False,
    with_literal_unk: bool = False,
    dropped_alphabet_chars: tuple[str, ...] = (),
) -> transformers.PreTrainedTokenizerFast:
  """Trains a small byte-level BPE tokenizer in memory."""
  tokenizer = tokenizers.Tokenizer(tokenizers_models.BPE())
  tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
  tokenizer.decoder = decoders.ByteLevel()
  special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
  if with_unk_token or with_literal_unk:
    special_tokens.append("<unk>")
  alphabet = [
      c
      for c in pre_tokenizers.ByteLevel.alphabet()
      if c not in dropped_alphabet_chars
  ]
  trainer = trainers.BpeTrainer(
      vocab_size=384,
      special_tokens=special_tokens,
      initial_alphabet=alphabet,
      show_progress=False,
  )
  tokenizer.train_from_iterator(_CORPUS, trainer)
  return transformers.PreTrainedTokenizerFast(
      tokenizer_object=tokenizer,
      eos_token="<|im_end|>",
      pad_token="<|endoftext|>",
      unk_token="<unk>" if with_unk_token else None,
      additional_special_tokens=["<|im_start|>"],
  )


def _convert(
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> tuple[spm_model.ModelProto, spm.SentencePieceProcessor]:
  """Converts the tokenizer and loads the result."""
  serialized = tokenizer_to_sentencepiece_lib.convert(tokenizer)
  proto = spm_model.ModelProto.FromString(serialized)
  processor = spm.SentencePieceProcessor()
  processor.LoadFromSerializedProto(serialized)
  return proto, processor


class TokenizerToSentencepieceLibTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tokenizer = _build_tokenizer()
    cls.proto, cls.processor = _convert(cls.tokenizer)

  def _hf_ids(self, text: str) -> list[int]:
    return self.tokenizer.encode(text, add_special_tokens=False)

  def _spm_ids(self, text: str) -> list[int]:
    return list(self.processor.Encode(text))

  def test_byte_tokens_become_byte_pieces_with_byte_fallback(self):
    self.assertTrue(self.proto.trainer_spec.byte_fallback)
    byte_pieces = [
        p.piece for p in self.proto.pieces if p.type == _PIECE_TYPE.BYTE
    ]
    self.assertLen(byte_pieces, 256)
    self.assertCountEqual(byte_pieces, ["<0x%02X>" % b for b in range(256)])

  def test_standalone_characters_encode_like_the_tokenizer(self):
    mismatches = {
        c: (self._hf_ids(c), self._spm_ids(c))
        for c in _LATIN_SWEEP
        if self._hf_ids(c) != self._spm_ids(c)
    }
    self.assertEmpty(mismatches)

  # Multi-word strings with non-ASCII words can hit the merge-order
  # approximation the module docstring describes (a BPE merge whose
  # intermediate token crosses an UTF-8 character boundary has no piece
  # surface), so the probes here stick to ASCII words, byte-fallback
  # characters, and special tokens, whose encodings the conversion does
  # guarantee.
  @parameterized.named_parameters(
      ("symbols", "20°C × 3 ½ — · Ω"),
      ("emoji", "hello 😀🎉 ok"),
      ("code", "def f(x): return x**2 + 1"),
      ("digits", "2025 20250830 12345 3.14159 1,000,000 v2.1.0"),
      ("chat", "<|im_start|>user\nWhat is the capital of France?<|im_end|>"),
  )
  def test_strings_encode_like_the_tokenizer(self, text):
    self.assertEqual(self._hf_ids(text), self._spm_ids(text))

  def test_unencodable_character_uses_byte_fallback_not_unk(self):
    # No whole-character piece exists for this emoji, so without byte
    # fallback it can only become the UNK piece.
    ids = self._spm_ids("😀")
    self.assertEqual(self._hf_ids("😀"), ids)
    self.assertNotIn(self.proto.trainer_spec.unk_id, ids)

  def test_special_tokens_match_as_a_single_piece(self):
    for token in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
      token_id = self.tokenizer.convert_tokens_to_ids(token)
      self.assertEqual([token_id], self._spm_ids(token))
      self.assertEqual(
          _PIECE_TYPE.USER_DEFINED, self.proto.pieces[token_id].type
      )

  def test_no_unk_token_appends_dedicated_unk_past_the_vocab(self):
    # The tokenizer has pad and eos but no unk_token: neither may be typed
    # UNKNOWN, and the appended <unk> must live past the original vocab.
    vocab_size = len(self.tokenizer.get_vocab())
    unknown_ids = [
        i
        for i, p in enumerate(self.proto.pieces)
        if p.type == _PIECE_TYPE.UNKNOWN
    ]
    self.assertLen(unknown_ids, 1)
    self.assertGreaterEqual(unknown_ids[0], vocab_size)
    self.assertEqual("<unk>", self.proto.pieces[unknown_ids[0]].piece)
    self.assertEqual(unknown_ids[0], self.proto.trainer_spec.unk_id)

  def test_unk_token_stays_the_unknown_piece(self):
    tokenizer = _build_tokenizer(with_unk_token=True)
    proto, _ = _convert(tokenizer)
    unk_id = tokenizer.convert_tokens_to_ids("<unk>")
    self.assertEqual(_PIECE_TYPE.UNKNOWN, proto.pieces[unk_id].type)
    self.assertEqual(unk_id, proto.trainer_spec.unk_id)
    unknown_ids = [
        i for i, p in enumerate(proto.pieces) if p.type == _PIECE_TYPE.UNKNOWN
    ]
    self.assertEqual([unk_id], unknown_ids)

  def test_literal_unk_in_vocab_does_not_collide_with_appended_unk(self):
    # Qwen2.5-style vocab: no unk_token, but a literal "<unk>" token exists
    # in the vocab. SentencePiece rejects duplicate piece surfaces, so the
    # appended UNKNOWN piece must take an unused name.
    tokenizer = _build_tokenizer(with_literal_unk=True)
    self.assertIsNone(tokenizer.unk_token)
    proto, processor = _convert(tokenizer)  # raises on a duplicate surface
    unknown_ids = [
        i for i, p in enumerate(proto.pieces) if p.type == _PIECE_TYPE.UNKNOWN
    ]
    self.assertLen(unknown_ids, 1)
    self.assertEqual("<unk_1>", proto.pieces[unknown_ids[0]].piece)
    self.assertEqual(unknown_ids[0], proto.trainer_spec.unk_id)
    # The literal "<unk>" token keeps its own id and still matches in text.
    literal_id = tokenizer.convert_tokens_to_ids("<unk>")
    self.assertEqual([literal_id], list(processor.Encode("<unk>")))

  def test_missing_byte_tokens_are_appended_past_the_vocab(self):
    # Some vocabs never learned a standalone token for a few bytes (invalid
    # UTF-8 lead bytes like 0xC0/0xC1 never occur in training text), but
    # SentencePiece requires all 256 BYTE pieces when byte_fallback is on.
    byte_to_unicode = {
        b: c
        for b, c in tokenizer_to_sentencepiece_lib._bytes_to_unicode().items()
    }
    dropped = (byte_to_unicode[0xC0], byte_to_unicode[0xC1])
    tokenizer = _build_tokenizer(dropped_alphabet_chars=dropped)
    vocab = tokenizer.get_vocab()
    self.assertNotIn(dropped[0], vocab)
    proto, processor = _convert(tokenizer)
    byte_pieces = [p.piece for p in proto.pieces if p.type == _PIECE_TYPE.BYTE]
    self.assertLen(byte_pieces, 256)
    appended = [
        p.piece for p in list(proto.pieces)[len(vocab) :]
        if p.type == _PIECE_TYPE.BYTE
    ]
    self.assertCountEqual(["<0xC0>", "<0xC1>"], appended)
    # The model still loads and encodes normal text identically.
    text = "The quick brown fox jumps over the lazy dog."
    self.assertEqual(
        tokenizer.encode(text, add_special_tokens=False),
        list(processor.Encode(text)),
    )

  def test_merged_accented_token_keeps_its_surface(self):
    # The corpus trains a standalone token for "é", whose surface must not be
    # dropped as a duplicate of the 0xE9 byte token's GPT-2 spelling. A
    # standalone "é" then encodes to that token's id, exactly like the
    # original tokenizer — not to the byte token's id and not to two byte
    # fallback pieces.
    hf_ids = self._hf_ids("é")
    self.assertLen(hf_ids, 1)
    self.assertEqual(hf_ids, self._spm_ids("é"))


if __name__ == "__main__":
  absltest.main()
