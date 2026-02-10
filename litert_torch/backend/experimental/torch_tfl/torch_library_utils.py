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
"""Utility functions for defining custom ops in torch library."""
from typing import Callable, Iterable
import torch


def custom_op_with_fake(
    name: str,
    *,
    mutates_args: str | Iterable[str] = (),
    schema: str | None = None,
):
  """Defines a custom op with a FakeTensor implementation using the same function."""

  def register(fn: Callable[..., object]):
    op = torch.library.custom_op(
        name,
        mutates_args=mutates_args,
        schema=schema,
    )(fn)
    torch.library.register_fake(name)(fn)
    return op

  return register
