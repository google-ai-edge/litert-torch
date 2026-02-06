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

"""Common utils for testing."""

import logging

from litert_torch import backend
from litert_torch import model
from litert_torch.generative.layers import kv_cache as kv_utils
import numpy as np
import torch
from torch.utils import _pytree as pytree


def compare_tflite_torch(
    edge_model: model.Model,
    torch_model: torch.nn.Module,
    tokens: torch.Tensor,
    input_pos: torch.Tensor,
    kv_cache: kv_utils.KVCache,
    signature_name: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    **kwargs,
) -> bool:
  """Compares torch models and TFLite models."""
  values, spec = pytree.tree_flatten({"kv_cache": kv_cache})
  flat_names = backend.export_utils.flat_dict_names(
      spec.children_specs, spec.context
  )
  torch_output = torch_model(tokens, input_pos, kv_cache, **kwargs)

  if "pixel_values" in kwargs:
    kwargs["pixel_values"] = kwargs["pixel_values"].numpy()
  kwargs.update({k: v.numpy() for k, v in zip(flat_names, values)})
  edge_output = edge_model(
      signature_name=signature_name,
      tokens=tokens.numpy(),
      input_pos=input_pos.numpy(),
      **kwargs,
  )

  return compare_logits(
      edge_output["logits"], torch_output["logits"].detach().numpy(), atol, rtol
  )


def compare_logits(
    edge_logits: np.ndarray,
    torch_logits: dict[str, torch.Tensor],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
  """Compares logits from edge model and torch model."""
  if np.allclose(edge_logits, torch_logits, rtol, atol, equal_nan=True):
    return True

  logging.info("edge_logits: %s", edge_logits)
  logging.info("torch_logits: %s", torch_logits)

  orig_atol = atol
  while rtol < 1:
    atol = orig_atol
    while atol < 1:
      if np.allclose(edge_logits, torch_logits, rtol, atol, equal_nan=True):
        logging.info("Got allclose true with atol=%s, rtol=%s", atol, rtol)
        return False
      atol *= 10
    rtol *= 10
  logging.info("allclose failed with reasonable atol and rtol.")
  return False
