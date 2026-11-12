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
r"""Exporting HF LLM model to LiteRT-LM file."""

import fire
from litert_torch.generative.export_hf import export as lib


# Thing to do:
# * Support re-using re-authored model in torch generative if possible.
# * Support runtime BMM / attention.
# * Support model specific mask generation.
# * Support dynamic rope and long rope.
# * Support quantized safetensor.
# * Support advanced PTQ techniques.
# * Models other than CausalLM.


def main(_=None):
  fire.Fire(lib.export)


if __name__ == '__main__':
  main()
