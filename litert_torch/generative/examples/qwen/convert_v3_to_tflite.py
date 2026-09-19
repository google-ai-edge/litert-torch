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

"""Example of converting Qwen 3.0 models to multi-signature tflite model."""

from absl import app
from litert_torch.generative.examples.qwen import qwen3
from litert_torch.generative.utilities import converter

flags = converter.define_conversion_flags(
    'qwen', default_transpose_kv_cache=True
)

_MODEL_SIZE = flags.DEFINE_enum(
    'model_size',
    '1.7b',
    ['0.6b', '1.7b', '4b'],
    'The size of the model to convert.',
)

_BUILDER = {
    '0.6b': qwen3.build_0_6b_model,
    '1.7b': qwen3.build_1_7b_model,
    '4b': qwen3.build_4b_model,
}


def main(_):
  converter.build_and_convert_to_tflite_from_flags(_BUILDER[_MODEL_SIZE.value])


if __name__ == '__main__':
  app.run(main)
