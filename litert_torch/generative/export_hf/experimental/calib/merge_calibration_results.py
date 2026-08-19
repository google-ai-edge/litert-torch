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
"""Script to merge calibration results from multiple tasks."""

from typing import Sequence

from absl import app
from absl import flags
from litert_torch.generative.export_hf.experimental.calib import quant_utils

FLAGS = flags.FLAGS

_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Directory containing calibration results from multiple tasks.',
    required=True,
)

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Directory to save the merged calibration results.',
    required=True,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  merger = quant_utils.CalibrationResultsMerger(
      _INPUT_DIR.value, _OUTPUT_DIR.value
  )
  merger.load_all()
  print(f'--- Tasks loaded: {merger.get_loaded_tasks()}')
  merger.merge()
  merger.save()


if __name__ == '__main__':
  app.run(main)
