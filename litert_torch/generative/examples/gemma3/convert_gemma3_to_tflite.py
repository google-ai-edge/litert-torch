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

"""Example of converting a Gemma3 model to multi-signature tflite model."""

from absl import app
from litert_torch.generative.examples.gemma3 import gemma3
from litert_torch.generative.utilities import converter

flags = converter.define_conversion_flags(
    'gemma3-1b', default_mask_as_input=True, default_transpose_kv_cache=True
)

_MODEL_SIZE = flags.DEFINE_string(
    'model_size',
    '1b',
    'The size of the model to convert.',
)


def main(_):
  model_size = _MODEL_SIZE.value
  # Auto-detect model size if it's the default '1b' or if we want to be proactive.
  # We only override if detection is successful.
  detected_size = gemma3.detect_model_size(flags.FLAGS.checkpoint_path)
  if detected_size and _MODEL_SIZE.present:
    if detected_size != _MODEL_SIZE.value:
      print(f"Note: User specified model_size={_MODEL_SIZE.value}, "
            f"but detected {detected_size}. Using user specification.")
  elif detected_size:
    model_size = detected_size
    print(f"Auto-detected model size: {model_size}")

  if model_size == '1b':
    model_builder = gemma3.build_model_1b
  elif model_size == '270m':
    model_builder = gemma3.build_model_270m
  elif model_size == '4b':
    model_builder = gemma3.build_model_4b
  else:
    raise ValueError(f'Unsupported model size: {model_size}')

  converter.build_and_convert_to_tflite_from_flags(model_builder)


if __name__ == '__main__':
  app.run(main)
