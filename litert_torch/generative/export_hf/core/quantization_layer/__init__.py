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
"""Quantization layer wrappers and checkpoint helpers for HuggingFace export."""

from litert_torch.generative.export_hf.core.quantization_layer import gemma_quant
import torch

_QUANT_PARAM_SUFFIXES: dict[str, tuple[str, ...]] = {
    "gemma": gemma_quant.QUANT_PARAM_SUFFIXES,
}


def get_quant_param_suffixes(
    quant_method: str | None = None,
) -> tuple[str, ...]:
  """Returns parameter name suffixes holding quantization scales/zero-points."""
  if quant_method is None:
    return ()
  return _QUANT_PARAM_SUFFIXES.get(quant_method, ())


def load_quant_scales_(
    model: torch.nn.Module,
    weights_loader: gemma_quant.WeightsLoader,
    quant_method: str | None = "gemma",
) -> torch.nn.Module:
  """Loads quantization scale/metadata parameters for `quant_method` onto CPU."""
  if quant_method == "gemma":
    return gemma_quant.load_gemma_quant_scales_(model, weights_loader)
  return model
