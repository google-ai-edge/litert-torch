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
"""Transpiles a jinja2 template file to a minijinja compatible template."""

import fire
from litert_torch.generative.export_hf.experimental.minijinja_transpile import transpile as transpile_lib


def transpile_file(file_path: str):
  """Reads a jinja2 template file, transpiles it, and prints the results."""
  with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
  result = transpile_lib.transpile_jinja2(content)
  print(result)


def main(_):
  fire.Fire(transpile_file)


if __name__ == "__main__":
  main()
