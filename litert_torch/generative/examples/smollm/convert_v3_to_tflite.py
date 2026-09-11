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

"""Example of converting SmolLM3 model to multi-signature tflite model."""

from absl import app
from litert_torch.generative.examples.smollm import smollm
from litert_torch.generative.utilities import converter

flags = converter.define_conversion_flags('smollm3')


def main(_):
  converter.build_and_convert_to_tflite_from_flags(smollm.build_model_v3)


if __name__ == '__main__':
  app.run(main)
