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
"""Tests for `ParakeetTDT._load_nemo_model`."""

import os
from unittest import mock

from absl.testing import absltest
from litert_torch.generative.export_hf.model_ext.parakeet import parakeet_tdt


class LoadNemoModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.nemo_asr = mock.MagicMock()
    self.asr_model_cls = self.nemo_asr.models.asr_model.ASRModel
    self.tmp_dir = self.create_tempdir().full_path

  def _touch(self, name: str) -> str:
    path = os.path.join(self.tmp_dir, name)
    with open(path, "wb"):
      pass
    return path

  def test_local_nemo_file_uses_restore_from(self):
    path = self._touch("model.nemo")

    result = parakeet_tdt.ParakeetTDT._load_nemo_model(self.nemo_asr, path)

    self.asr_model_cls.restore_from.assert_called_once_with(path)
    self.asr_model_cls.from_pretrained.assert_not_called()
    self.assertIs(result, self.asr_model_cls.restore_from.return_value)

  def test_local_dir_with_single_nemo_file_uses_restore_from(self):
    path = self._touch("parakeet-tdt-0.6b-v3.nemo")
    self._touch("config.json")

    result = parakeet_tdt.ParakeetTDT._load_nemo_model(
        self.nemo_asr, self.tmp_dir
    )

    self.asr_model_cls.restore_from.assert_called_once_with(path)
    self.asr_model_cls.from_pretrained.assert_not_called()
    self.assertIs(result, self.asr_model_cls.restore_from.return_value)

  def test_local_dir_without_nemo_file_raises(self):
    self._touch("config.json")

    with self.assertRaisesRegex(FileNotFoundError, "No .nemo file found"):
      parakeet_tdt.ParakeetTDT._load_nemo_model(self.nemo_asr, self.tmp_dir)

    self.asr_model_cls.restore_from.assert_not_called()
    self.asr_model_cls.from_pretrained.assert_not_called()

  def test_local_dir_with_multiple_nemo_files_raises(self):
    self._touch("a.nemo")
    self._touch("b.nemo")

    with self.assertRaisesRegex(ValueError, "Multiple .nemo files found"):
      parakeet_tdt.ParakeetTDT._load_nemo_model(self.nemo_asr, self.tmp_dir)

    self.asr_model_cls.restore_from.assert_not_called()
    self.asr_model_cls.from_pretrained.assert_not_called()

  def test_hub_model_id_uses_from_pretrained(self):
    model_id = "nvidia/parakeet-tdt-0.6b-v3"

    result = parakeet_tdt.ParakeetTDT._load_nemo_model(self.nemo_asr, model_id)

    self.asr_model_cls.from_pretrained.assert_called_once_with(model_id)
    self.asr_model_cls.restore_from.assert_not_called()
    self.assertIs(result, self.asr_model_cls.from_pretrained.return_value)


if __name__ == "__main__":
  absltest.main()
