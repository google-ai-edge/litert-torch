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
"""Tests for export_lib."""

import json
import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

from litert_torch.generative.export_hf.core import export_lib


class ExportLibTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_maybe_patch_tokenizer_patches_incorrect_config(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # BPE + Metaspace + BPE chars in vocab
    incorrect_config = {
        "model": {"type": "BPE", "vocab": {"ĠI": 1, "Ċ": 2, "hello": 3}},
        "pre_tokenizer": {"type": "Metaspace", "replacement": " "},
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {
                    "type": "Replace",
                    "pattern": {"String": " "},
                    "content": " ",
                }
            ],
        },
    }

    with open(tokenizer_path, "w") as f:
      json.dump(incorrect_config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      patched_config = json.load(f)

    self.assertEqual(patched_config["pre_tokenizer"]["type"], "ByteLevel")
    self.assertEqual(patched_config["decoder"]["type"], "ByteLevel")
    self.assertTrue(patched_config["pre_tokenizer"]["use_regex"])
    self.assertTrue(patched_config["decoder"]["use_regex"])

  def test_maybe_patch_tokenizer_skips_correct_config(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # BPE + ByteLevel
    correct_config = {
        "model": {"type": "BPE", "vocab": {"ĠI": 1, "Ċ": 2}},
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
    }

    with open(tokenizer_path, "w") as f:
      json.dump(correct_config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      config = json.load(f)

    self.assertEqual(config["pre_tokenizer"]["type"], "ByteLevel")
    self.assertEqual(config["decoder"]["type"], "ByteLevel")
    self.assertNotIn("use_regex", config["pre_tokenizer"])

  def test_maybe_patch_tokenizer_skips_non_bpe(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # WordPiece + Metaspace (dummy)
    config = {
        "model": {"type": "WordPiece", "vocab": {"hello": 1}},
        "pre_tokenizer": {"type": "Metaspace"},
    }

    with open(tokenizer_path, "w") as f:
      json.dump(config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      result_config = json.load(f)

    self.assertEqual(result_config["pre_tokenizer"]["type"], "Metaspace")

  def test_maybe_patch_tokenizer_skips_no_bpe_chars(self):
    tokenizer_path = os.path.join(self.test_dir, "tokenizer.json")

    # BPE + Metaspace but NO BPE chars in vocab
    config = {
        "model": {"type": "BPE", "vocab": {"hello": 1, "world": 2}},
        "pre_tokenizer": {"type": "Metaspace"},
    }

    with open(tokenizer_path, "w") as f:
      json.dump(config, f)

    export_lib._maybe_patch_tokenizer(tokenizer_path)

    with open(tokenizer_path, "r") as f:
      result_config = json.load(f)

    self.assertEqual(result_config["pre_tokenizer"]["type"], "Metaspace")

  def test_asr_export_config_rules(self):
    config = export_lib.exportable_module_config.ExportableModuleConfig(
        model="dummy_asr_model",
        task=export_lib.exportable_module_config.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
        split_cache=True,
        export_vision_encoder=True,
    )
    self.assertEqual(
        config.task,
        export_lib.exportable_module_config.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
    )
    self.assertTrue(config.export_audio_encoder)
    self.assertFalse(config.export_vision_encoder)
    self.assertFalse(config.split_cache)
    self.assertFalse(config.externalize_embedder)
    self.assertFalse(config.externalize_rope)
    self.assertEqual(config.input_sec, 1.0)
    self.assertEqual(config.stateful_after, -1)


if __name__ == "__main__":
  absltest.main()
