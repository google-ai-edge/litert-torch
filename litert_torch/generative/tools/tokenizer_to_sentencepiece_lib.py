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
import random
from typing import List

import transformers

from sentencepiece import sentencepiece_model_pb2 as spm_model
import sentencepiece as spm


def _bytes_to_unicode():
  """Returns list of utf-8 byte and a corresponding list of unicode strings.

  It's a copy of https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9.
  """
  bs = (
      list(range(ord("!"), ord("~") + 1))
      + list(range(ord("¡"), ord("¬") + 1))
      + list(range(ord("®"), ord("ÿ") + 1))
  )
  cs = bs[:]
  n = 0
  for b in range(2**8):
    if b not in bs:
      bs.append(b)
      cs.append(2**8 + n)
      n += 1
  cs = [chr(n) for n in cs]
  return dict(zip(bs, cs))


# An inverse map of _bytes_to_unicode() to decode unicode tokens in a HF
# transformers tokenizer into utf-8 tokens in the SentencePiece model.
_BYTE_DECODE_MAP = {v: k for k, v in _bytes_to_unicode().items()}


def _normalize_gpt2(token: str) -> str:
  """Normalizes a unicode character to a utf-8 character.

  It's a semantic copy of
  https://github.com/openai/gpt-2/blob/master/src/encoder.py#L105.

  Args:
    token: The unicode character to normalize.

  Returns:
    The utf-8 character.
  """
  return bytearray(
      [_BYTE_DECODE_MAP[c] if c in _BYTE_DECODE_MAP else ord(c) for c in token]
  ).decode("utf-8", "replace")


_NORMALIZE_FUNCS = {
    "none": lambda x, id, _: x,
    "gpt2": lambda x, id, _: _normalize_gpt2(x),
    "decode": lambda _, id, tokenizer: tokenizer.decode([id]),
}


def _is_byte_level(tokenizer: transformers.PreTrainedTokenizer) -> bool:
  """Returns True when the vocab is GPT-2 style byte-level.

  In such a vocab the 256 byte tokens are single characters of the GPT-2
  byte<->unicode table, e.g. the space byte 0x20 is spelled "Ġ".
  """
  vocab = tokenizer.get_vocab()
  return "Ġ" in vocab and all(c in vocab for c in ("Ċ", "Ã", "Â"))


def _add_token(
    token: str,
    id_: int,
    tokenizer: transformers.PreTrainedTokenizer,
    sp_model: spm_model.ModelProto,
    tokens_seen: set[str],
    counts: dict[spm_model.ModelProto.SentencePiece.Type, int],
    normalize_tokens: str = "decode",
):
  """Adds a token to the SentencePieceModel protobuf with a derived type."""
  # Only the tokenizer's own unk_token may become the UNKNOWN piece. Falling
  # back to pad/eos would type that token UNKNOWN, and SentencePiece then both
  # emits its id for unencodable input and never matches its text in prompts.
  unk_token = tokenizer.unk_token
  if unk_token is not None and token == unk_token:
    type_ = spm_model.ModelProto.SentencePiece.UNKNOWN
  elif (
      token in tokenizer.all_special_tokens
      or token in tokenizer.get_added_vocab()
  ):
    # Special tokens must be USER_DEFINED: SentencePiece never matches CONTROL
    # pieces in input text.
    type_ = spm_model.ModelProto.SentencePiece.USER_DEFINED
    sp_model.trainer_spec.user_defined_symbols.append(token)
  elif (
      len(token) == 1
      and token in _BYTE_DECODE_MAP
      and _is_byte_level(tokenizer)
  ):
    # A byte-level token stands for a byte, not for the character its GPT-2
    # spelling happens to be. Emit it as a BYTE piece so that e.g. a
    # standalone "é" does not encode to the 0xE9 byte token's id.
    sp_model.pieces.add(
        piece="<0x%02X>" % _BYTE_DECODE_MAP[token],
        score=-id_,
        type=spm_model.ModelProto.SentencePiece.BYTE,
    )
    counts[spm_model.ModelProto.SentencePiece.BYTE] = (
        counts.get(spm_model.ModelProto.SentencePiece.BYTE, 0) + 1
    )
    return
  else:
    type_ = spm_model.ModelProto.SentencePiece.NORMAL

  count_type = type_
  normalized = _NORMALIZE_FUNCS[normalize_tokens](token, id_, tokenizer)
  if normalized == token:
    pass
  elif normalized in tokens_seen:
    logging.debug(
        'DUPLICATE: token "%s"(id=%d) normalized to "%s"',
        token,
        id_,
        normalized,
    )
    normalized = token
    # Change only the type of counts for logging. When UNUSED is set for SPM
    # model, it seems to have some negative impact, i.e. the ratio of mismatched
    # ID pairs is slightly higher.
    count_type = spm_model.ModelProto.SentencePiece.Type.UNUSED
  else:
    tokens_seen.add(normalized)
  sp_model.pieces.add(piece=normalized, score=-id_, type=type_)
  counts[count_type] = counts.get(count_type, 0) + 1

  # Fill special meta token info. One token can be used for multiple purposes.
  if token == tokenizer.unk_token:
    sp_model.trainer_spec.unk_id = id_
    sp_model.trainer_spec.unk_piece = normalized
    logging.info("Found unk_id: %d, unk_piece: %s", id_, normalized)
  if token == tokenizer.bos_token:
    sp_model.trainer_spec.bos_id = id_
    sp_model.trainer_spec.bos_piece = normalized
    logging.info("Found bos_id: %d, bos_piece: %s", id_, normalized)
  if token == tokenizer.eos_token:
    sp_model.trainer_spec.eos_id = id_
    sp_model.trainer_spec.eos_piece = normalized
    logging.info("Found eos_id: %d, eos_piece: %s", id_, normalized)
  if token == tokenizer.pad_token:
    sp_model.trainer_spec.pad_id = id_
    sp_model.trainer_spec.pad_piece = normalized
    logging.info("Found pad_id: %d, pad_piece: %s", id_, normalized)


def _build_spm_model_from_tokenizer(
    tokenizer: transformers.PreTrainedTokenizer,
    normalize_tokens: str = "decode",
) -> spm_model.ModelProto:
  """Builds a SentencePieceModel protobuf from a tokenizer."""
  sp_model = spm_model.ModelProto()
  sp_model.trainer_spec.model_type = spm_model.TrainerSpec.BPE
  sp_model.trainer_spec.vocab_size = len(tokenizer.vocab)
  sp_model.normalizer_spec.add_dummy_prefix = False
  sp_model.normalizer_spec.remove_extra_whitespaces = False
  sp_model.normalizer_spec.escape_whitespaces = False
  sp_model.denormalizer_spec.CopyFrom(sp_model.normalizer_spec)

  id_to_token = {id: tk for tk, id in tokenizer.vocab.items()}
  tokens_seen = set(tokenizer.vocab.keys())
  if _is_byte_level(tokenizer):
    # The byte tokens become <0xXX> pieces, so their raw spellings (e.g. "é"
    # for byte 0xE9) no longer occupy the surface space; a merged token that
    # decodes to "é" keeps that surface instead of being dropped as a
    # duplicate.
    tokens_seen -= {
        t for t in tokens_seen if len(t) == 1 and t in _BYTE_DECODE_MAP
    }
  counts = {}
  for id_ in range(len(tokenizer.vocab)):
    _add_token(
        id_to_token[id_],
        id_,
        tokenizer,
        sp_model,
        tokens_seen,
        counts,
        normalize_tokens,
    )

  if _is_byte_level(tokenizer):
    sp_model.trainer_spec.byte_fallback = True
    present = {
        p.piece
        for p in sp_model.pieces
        if p.type == spm_model.ModelProto.SentencePiece.BYTE
    }
    missing = [b for b in range(256) if "<0x%02X>" % b not in present]
    if missing:
      # Some vocabs never learned a standalone token for a few bytes, but
      # SentencePiece requires all 256 BYTE pieces when byte_fallback is on.
      # Appended past the vocab, these pieces can only be emitted for bytes
      # the original tokenizer cannot encode either.
      logging.warning(
          "%d byte tokens absent from the vocab;"
          " appending BYTE pieces past the vocab: %s",
          len(missing),
          ["0x%02X" % b for b in missing],
      )
      for b in missing:
        sp_model.pieces.add(
            piece="<0x%02X>" % b,
            score=0.0,
            type=spm_model.ModelProto.SentencePiece.BYTE,
        )
  if not any(
      p.type == spm_model.ModelProto.SentencePiece.UNKNOWN
      for p in sp_model.pieces
  ):
    # SentencePiece requires an UNKNOWN piece. When the tokenizer has no
    # unk_token, append a dedicated <unk> past the vocab instead of typing
    # pad/eos as UNKNOWN. Some vocabs (e.g. Qwen2.5) contain a literal
    # "<unk>" token without it being the unk_token, and SentencePiece
    # rejects duplicate piece surfaces, so pick an unused name.
    existing_pieces = {p.piece for p in sp_model.pieces}
    unk_piece = "<unk>"
    suffix = 0
    while unk_piece in existing_pieces:
      suffix += 1
      unk_piece = "<unk_%d>" % suffix
    sp_model.pieces.add(
        piece=unk_piece,
        score=0.0,
        type=spm_model.ModelProto.SentencePiece.UNKNOWN,
    )
    sp_model.trainer_spec.unk_id = len(sp_model.pieces) - 1
    sp_model.trainer_spec.unk_piece = unk_piece
    logging.info(
        "No unk_token: appended %s at id %d",
        unk_piece,
        sp_model.trainer_spec.unk_id,
    )

  logging.info("number of tokens: %d", len(sp_model.pieces))
  for type_ in counts:
    logging.info(
        "number of %s: %d",
        spm_model.ModelProto.SentencePiece.Type.Name(type_),
        counts[type_],
    )

  return sp_model


def _is_same_ids(ids_by_tokenizer: List[int], ids: List[int]) -> bool:
  """Checks if the IDs are the same to ones by transformer tokenizer."""
  # Transformer tokenizer may insert BOS token at the beginning.
  return ids_by_tokenizer == ids or ids_by_tokenizer[1:] == ids


def _log_not_matched(
    num_not_matched_strict: int, num_not_matched_loose: int, total: int
):
  """Logs the number of not matched pairs."""
  logging.info(
      "Not matched strictly %d/%d pairs: %.2f%%, loosely %d/%d pairs: %.2f%%",
      num_not_matched_strict,
      total,
      100 * num_not_matched_strict / total,
      num_not_matched_loose,
      total,
      100 * num_not_matched_loose / total,
  )


def _encode_by_spm(
    spm_tokenizer: spm.SentencePieceProcessor, string: str
) -> List[int]:
  """Encodes a string by the SentencePiece tokenizer."""
  ids = spm_tokenizer.Encode(string)
  if isinstance(ids, list):
    return ids
  # SentencePieceText
  return [p.id for p in ids.pieces]


def verify_spm_tokenizer(
    tokenizer: transformers.PreTrainedTokenizer,
    spm_tokenizer: spm.SentencePieceProcessor,
    strings_to_verify: List[str] | None = None,
    num_pairs_to_verify: int = 1000,
):
  """Verifies the SentencePiece tokenizer."""
  # First, check if the token IDs encoded by the original tokenizer are the same
  # as the token IDs encoded by the SentencePiece tokenizer.
  strings_to_verify = strings_to_verify or []
  for string in strings_to_verify:
    ids_by_tokenizer = tokenizer.encode(string)
    ids_by_spm = _encode_by_spm(spm_tokenizer, string)
    logging.info("String to verify: %s", string)
    logging.info("Token IDs by the oringal tokenizer: %s", ids_by_tokenizer)
    logging.info("Token IDs by the SentencePiece tokenizer: %s", ids_by_spm)
    if _is_same_ids(ids_by_tokenizer, ids_by_spm):
      logging.info("PASS")
    else:
      logging.warning("FAIL")

  # Second, check if how many strings decoded from the pairs of tokens by the
  # original tokenizer are encoded to the same token IDs by the SentencePiece
  # tokenizer.
  total = num_pairs_to_verify
  num_not_matched_strict = 0
  num_not_matched_loose = 0
  for i in range(total):
    id_pair = random.sample(list(range(len(tokenizer.vocab))), 2)
    string = tokenizer.decode(id_pair)
    ids_by_tokenizer = tokenizer.encode(string)
    ids_by_spm = _encode_by_spm(spm_tokenizer, string)
    if not _is_same_ids(ids_by_tokenizer, ids_by_spm):
      num_not_matched_strict += 1
      if _is_same_ids(ids_by_tokenizer, id_pair):
        num_not_matched_loose += 1
        logging.debug(
            'NOT MATCHED: "%s", ids=%s, tok=%s, spm=%s',
            string,
            id_pair,
            ids_by_tokenizer,
            ids_by_spm,
        )
    if (i + 1) % 100 == 0:
      _log_not_matched(num_not_matched_strict, num_not_matched_loose, i + 1)
  _log_not_matched(num_not_matched_strict, num_not_matched_loose, total)


def convert(tokenizer, normalize_tokens: str = "decode"):
  """Converts a tokenizer to a SentencePieceModel protobuf."""
  if hasattr(tokenizer, "vocab_file") and tokenizer.vocab_file:
    logging.info("vocab_file exists: %s", tokenizer.vocab_file)
    with open(tokenizer.vocab_file, "rb") as f:
      sp_model = spm_model.ModelProto.FromString(f.read())
  else:
    logging.info("vocab_file does not exist. Try to build a new one.")
    sp_model = _build_spm_model_from_tokenizer(tokenizer, normalize_tokens)

  spm_serialized = sp_model.SerializeToString()
  return spm_serialized
