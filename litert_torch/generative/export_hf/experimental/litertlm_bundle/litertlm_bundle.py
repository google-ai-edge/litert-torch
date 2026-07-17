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
"""Object-oriented bundle manager and dynamic packing utilities for .litertlm files."""

import io
import json
import os
import pathlib

try:
  import tomllib
except ImportError:
  import tomli as tomllib
from typing import Any, BinaryIO, cast
import zlib

from google.protobuf import message
from google.protobuf import text_format

from ai_edge_litert.litert_lm import litertlm_builder
from ai_edge_litert.litert_lm import litertlm_peek
from ai_edge_litert.litert_lm.proto import llm_metadata_pb2

gfile = None


def _open(path: str | pathlib.Path, mode: str = 'r'):
  if gfile:
    return gfile.Open(str(path), mode)
  return open(path, mode)


class LitertLmBundle:
  """Manages dynamic unpacking, section inspection, and repacking of .litertlm bundles."""

  def __init__(self, unpack_dir: str, toml_data: dict[str, Any]):
    self.unpack_dir = unpack_dir
    self.toml_data = toml_data
    self.sections: list[dict[str, Any]] = []
    self._component_paths: dict[str, Any] = {}
    self._parse_sections()

  def _parse_sections(self) -> None:
    """Parses model.toml entries to index component paths and extract metadata.

    Iterates through all sections registered in the unpacked model.toml table.
    Populates `_component_paths` with universal section/model type aliases
    (`tf_lite_prefill_decode`,
    `prefill_decode`, `HF_Tokenizer`, etc.) pointing to exact disk paths.
    Triggers decompression
    for `HF_Tokenizer` (`.zlib`) and protobuf parsing for `LlmMetadataProto`.
    """
    for section in self.toml_data.get('section', []):
      section_type = section.get('section_type')
      model_type = section.get('model_type')
      data_path = section.get('data_path')
      if not data_path:
        continue

      full_path = os.path.join(self.unpack_dir, data_path)
      self.sections.append({
          'section_type': section_type,
          'model_type': model_type,
          'data_path': data_path,
          'full_path': full_path,
      })

      if model_type:
        self._component_paths[f'{section_type}_{model_type}'] = full_path
        self._component_paths[model_type] = full_path
        if not model_type.startswith('tf_lite_'):
          self._component_paths[f'tf_lite_{model_type}'] = full_path
        else:
          self._component_paths[model_type[len('tf_lite_') :]] = full_path
      self._component_paths[section_type] = full_path

      if section_type == 'HF_Tokenizer':
        self._extract_hf_tokenizer(full_path, data_path)
      elif section_type in ('LlmMetadataProto', 'LlmMetadata'):
        meta_info = self._parse_metadata_proto(full_path)
        if meta_info.get('chat_tmpl'):
          self._component_paths['chat_template'] = meta_info['chat_tmpl']
        if meta_info.get('stop_ids'):
          self._component_paths['stop_token_ids'] = meta_info['stop_ids']

  def _extract_hf_tokenizer(self, full_path: str, data_path: str) -> None:
    """Decompresses HF_Tokenizer_Zlib section into tokenizer.json for transformers.PreTrainedTokenizerFast.

    When LitertLmFileBuilder.add_hf_tokenizer packs a .json tokenizer during
    export, it writes an
    8-byte little-endian uncompressed_size integer immediately before the zlib
    compressed payload
    (see litertlm_builder.py:681). When add_hf_tokenizer packs a pre-compressed
    .zlib file directly,
    it writes the raw zlib payload without the 8-byte header. We therefore
    attempt decompressing
    with `compressed_data[8:]` first (size-prefixed), falling back to
    `compressed_data` (raw zlib)
    if the size prefix is absent.

    Args:
      full_path: Absolute path to the dumped Section3_HF_Tokenizer_Zlib.zlib
        file.
      data_path: Relative path as registered inside model.toml.
    """
    if not data_path.endswith('.zlib'):
      self._component_paths['tokenizer_json_path'] = full_path
      self._component_paths['transformers_model_path'] = self.unpack_dir
      return

    with _open(full_path, 'rb') as f:
      compressed_data = f.read()
    try:
      decompressed = zlib.decompress(compressed_data[8:])
    except zlib.error:
      decompressed = zlib.decompress(compressed_data)
    json_path = os.path.join(self.unpack_dir, 'tokenizer.json')
    with _open(json_path, 'wb') as f:
      f.write(decompressed)
    self._component_paths['tokenizer_json_path'] = json_path
    self._component_paths['transformers_model_path'] = self.unpack_dir

  def _parse_metadata_proto(self, metadata_path: str) -> dict[str, Any]:
    """Parses LlmMetadataProto section using official llm_metadata_pb2 definition.

    Reads Section4_LlmMetadataProto (.pbtext string dump or binary proto) and
    extracts
    the Jinja chat template (`jinja_prompt_template`), start token, and stop
    token IDs
    without relying on ad-hoc string splitting.

    Args:
      metadata_path: Absolute path to the dumped LlmMetadataProto file.

    Returns:
      A dictionary containing `chat_tmpl`, `start_token_str`, `start_token_id`,
      and `stop_ids`.
    """
    meta_info: dict[str, Any] = {'stop_ids': set()}
    if not os.path.exists(metadata_path):
      return meta_info
    meta = llm_metadata_pb2.LlmMetadata()
    try:
      if metadata_path.endswith('.pbtext'):
        with _open(metadata_path, 'r') as f:
          text_format.Parse(f.read(), meta)
      else:
        with _open(metadata_path, 'rb') as f:
          meta.ParseFromString(f.read())
    except (text_format.ParseError, message.DecodeError):
      return meta_info

    if meta.jinja_prompt_template:
      meta_info['chat_tmpl'] = meta.jinja_prompt_template

    if meta.start_token.token_str:
      meta_info['start_token_str'] = meta.start_token.token_str
    if meta.start_token.token_ids.ids:
      meta_info['start_token_id'] = meta.start_token.token_ids.ids[0]

    for st in meta.stop_tokens:
      if st.token_ids.ids:
        meta_info['stop_ids'].update(st.token_ids.ids)
    return meta_info

  @classmethod
  def unpack(cls, litertlm_path: str, unpack_dir: str) -> 'LitertLmBundle':
    """Unpacks a .litertlm file into unpack_dir and returns a LitertLmBundle instance."""
    os.makedirs(unpack_dir, exist_ok=True)
    output_stream = io.StringIO()
    litertlm_peek.peek_litertlm_file(
        litertlm_path, dump_files_dir=unpack_dir, output_stream=output_stream
    )

    toml_path = os.path.join(unpack_dir, 'model.toml')
    if not os.path.exists(toml_path):
      raise RuntimeError(
          f'Failed to unpack {litertlm_path}: model.toml not found in'
          f' {unpack_dir}'
      )

    with _open(toml_path, 'r') as f:
      toml_data = tomllib.loads(f.read())

    return cls(unpack_dir=unpack_dir, toml_data=toml_data)

  @staticmethod
  def peek(litertlm_path: str) -> str:
    """Returns a string inspection summary of a .litertlm bundle file."""
    return peek_litertlm(litertlm_path)

  @staticmethod
  def pack(
      output_litertlm: str,
      model_path: str,
      embedder_model_path: str | None = None,
      auxiliary_model_path: str | None = None,
      ple_model_path: str | None = None,
      spm_path: str | None = None,
      transformers_model_path: str | None = None,
      metadata: dict[str, Any] | None = None,
      llm_metadata_path: str | None = None,
  ) -> None:
    """Builds and packages model components into a .litertlm bundle file."""
    pack_litertlm(
        output_litertlm=output_litertlm,
        model_path=model_path,
        embedder_model_path=embedder_model_path,
        auxiliary_model_path=auxiliary_model_path,
        ple_model_path=ple_model_path,
        spm_path=spm_path,
        transformers_model_path=transformers_model_path,
        metadata=metadata,
        llm_metadata_path=llm_metadata_path,
    )

  @property
  def component_paths(self) -> dict[str, Any]:
    return self._component_paths

  def get(self, key: str, default: Any = None) -> Any:
    return self._component_paths.get(key, default)

  def __getitem__(self, key: str) -> Any:
    return self._component_paths[key]

  def __contains__(self, key: str) -> bool:
    return key in self._component_paths

  def __iter__(self):
    return iter(self._component_paths)

  def __len__(self) -> int:
    return len(self._component_paths)

  def get_section_path(
      self, section_type: str, model_type: str | None = None
  ) -> str | None:
    for sec in self.sections:
      if sec['section_type'] == section_type and (
          model_type is None or sec['model_type'] == model_type
      ):
        return sec['full_path']
    return None

  def update_section_path(
      self, section_type: str, model_type: str | None, new_path: str
  ) -> None:
    for sec in self.sections:
      if sec['section_type'] == section_type and (
          model_type is None or sec['model_type'] == model_type
      ):
        sec['full_path'] = new_path
        return
    self.sections.append({
        'section_type': section_type,
        'model_type': model_type,
        'data_path': os.path.basename(new_path),
        'full_path': new_path,
    })


def peek_litertlm(litertlm_path: str) -> str:
  """Returns a string inspection summary of a .litertlm bundle file."""
  output_stream = io.StringIO()
  litertlm_peek.peek_litertlm_file(
      litertlm_path, dump_files_dir=None, output_stream=output_stream
  )
  return output_stream.getvalue()


def unpack_litertlm(litertlm_path: str, unpack_dir: str) -> dict[str, Any]:
  """Unpacks a .litertlm file into unpack_dir and resolves model component paths."""
  bundle = LitertLmBundle.unpack(
      litertlm_path=litertlm_path, unpack_dir=unpack_dir
  )
  return bundle.component_paths


def pack_litertlm(
    output_litertlm: str,
    model_path: str,
    embedder_model_path: str | None = None,
    auxiliary_model_path: str | None = None,
    ple_model_path: str | None = None,
    spm_path: str | None = None,
    transformers_model_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    llm_metadata_path: str | None = None,
) -> None:
  """Builds and packages model components into a .litertlm bundle file."""
  builder = litertlm_builder.LitertLmFileBuilder()

  if metadata:
    for key, value in metadata.items():
      builder.add_system_metadata(
          litertlm_builder.Metadata(
              key=key,
              value=str(value),
              dtype=litertlm_builder.DType.STRING,
          )
      )

  builder.add_tflite_model(
      tflite_model_path=model_path,
      model_type=litertlm_builder.TfLiteModelType.PREFILL_DECODE,
  )

  if embedder_model_path:
    builder.add_tflite_model(
        tflite_model_path=embedder_model_path,
        model_type=litertlm_builder.TfLiteModelType.EMBEDDER,
    )

  if auxiliary_model_path:
    builder.add_tflite_model(
        tflite_model_path=auxiliary_model_path,
        model_type=litertlm_builder.TfLiteModelType.AUX,
    )

  if ple_model_path:
    builder.add_tflite_model(
        tflite_model_path=ple_model_path,
        model_type=litertlm_builder.TfLiteModelType.PER_LAYER_EMBEDDER,
    )

  if spm_path:
    builder.add_sentencepiece_tokenizer(sp_tokenizer_path=spm_path)

  if transformers_model_path:
    hf_json = transformers_model_path
    if os.path.isdir(hf_json):
      hf_json = os.path.join(hf_json, 'tokenizer.json')
    if os.path.exists(hf_json):
      builder.add_hf_tokenizer(hf_tokenizer_path=hf_json)

  if llm_metadata_path and os.path.exists(llm_metadata_path):
    builder.add_llm_metadata(llm_metadata_path=llm_metadata_path)

  out_dir = os.path.dirname(output_litertlm)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  with _open(output_litertlm, 'wb') as f:
    builder.build(cast(BinaryIO, f))
