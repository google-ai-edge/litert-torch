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

"""Base class of ASR models to be converted to tflite."""

import abc
import dataclasses
from typing import Optional

from litert_torch.backend.composite import StableHLOCompositeBuilder
import numpy as np
import torch
from torch.nn import functional as F
import transformers


class AsrGroupNorm(torch.nn.Module):
  """Custom GroupNorm without GATHER_ND ops for GPU inference."""

  def __init__(
      self,
      num_groups: int,
      eps: float,
      weight: torch.Tensor,
      bias: torch.Tensor,
  ):
    super().__init__()
    self._num_groups = num_groups
    self._eps = eps
    self._weight = weight.reshape(1, -1, 1)
    if bias is None:
      self._bias = torch.zeros_like(self._weight)
    else:
      self._bias = bias.reshape(1, -1, 1)
    self._attr = {
        "sub_type": 0,  # GroupNorm
        "num_groups": self._num_groups,
        "epsilon": self._eps,
    }

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, c, t = x.shape
    x = x.reshape(b, self._num_groups, c // self._num_groups, t)
    mean = x.mean(dim=(2, 3), keepdim=True)
    stddev = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * torch.rsqrt(stddev + self._eps)
    x = x.reshape(b, c, t)
    return x * self._weight + self._bias


class AsrLayerNorm(torch.nn.Module):
  """Custom LayerNorm with SHLO composite ops for GPU inference."""

  def __init__(
      self,
      normalized_shape: torch.Size,
      eps: float,
      weight: torch.Tensor,
      bias: torch.Tensor,
  ):
    super().__init__()
    self._normalized_shape = normalized_shape
    self._eps = eps
    self._weight = weight
    self._bias = bias if bias is not None else torch.zeros_like(weight)
    self._attr = {
        "sub_type": 1,  # LayerNorm
        "epsilon": self._eps,
    }

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    composite = StableHLOCompositeBuilder("odml.group_norm", self._attr)
    x, w, b = composite.mark_inputs(x, self._weight, self._bias)
    y = F.layer_norm(x, self._normalized_shape, w, b, self._eps)
    y = composite.mark_outputs(y)
    return y


class AsrRMSNorm(torch.nn.Module):
  """Custom RMSNorm with SHLO composite ops for GPU inference."""

  def __init__(self, eps: float, weight: torch.Tensor):
    super().__init__()
    self._eps = eps
    self._weight = weight
    self._attr = {"epsilon": self._eps}

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    composite = StableHLOCompositeBuilder("odml.rms_norm", self._attr)
    x, w = composite.mark_inputs(x, self._weight)
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self._eps) * w
    y = composite.mark_outputs(y)
    return y


def _sdpa(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
  """Implements SDPA with SHLO composite ops for GPU inference."""
  attrs = {"scale": scaling}
  composite = StableHLOCompositeBuilder(
      "odml.scaled_dot_product_attention", attrs
  )

  q = query.float().transpose(1, 2)
  k = key.float().transpose(1, 2)
  v = value.float().transpose(1, 2)
  if attention_mask is None:
    q, k, v = composite.mark_inputs(q, k, v)
  else:
    q, k, v, attention_mask = composite.mark_inputs(q, k, v, attention_mask)

  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)
  y, _ = transformers.integrations.sdpa_attention.sdpa_attention_forward(
      module, q, k, v, attention_mask, dropout, scaling, is_causal, **kwargs
  )

  y = composite.mark_outputs(y)
  return y, None


def get_causal_mask(num_tokens: int) -> torch.Tensor:
  """Returns a causal mask of float32 in [1, 1, num_tokens, num_tokens]."""
  cond = torch.tril(torch.ones((num_tokens, num_tokens), dtype=torch.bool))
  cond = cond.unsqueeze(0).unsqueeze(0)
  return torch.where(cond, 0.0, -torch.inf).to(torch.float32)


class AsrProcessor:
  """Base class of processors to convert audio and decode tokens."""

  @abc.abstractmethod
  def get_sampling_rate(self) -> int:
    """Returns the sampling rate of the audio input."""
    raise NotImplementedError()

  @abc.abstractmethod
  def process(self, audio: np.ndarray) -> dict[str, torch.Tensor]:
    """Processes the audio to model inputs."""
    raise NotImplementedError()

  @abc.abstractmethod
  def decode(self, tokens: list[int] | torch.Tensor) -> list[str]:
    """Decodes the tokens to text."""
    raise NotImplementedError()


class TransformersProcessor(AsrProcessor):
  """AsrProcessor with transformers.AutoProcessor."""

  def __init__(self, model_id: str):
    self._processor = transformers.AutoProcessor.from_pretrained(model_id)

  def get_sampling_rate(self) -> int:
    return self._processor.feature_extractor.sampling_rate

  def process(self, audio: np.ndarray) -> dict[str, torch.Tensor]:
    return self._processor(audio=audio, return_tensors="pt")

  def decode(self, tokens: list[int] | torch.Tensor) -> list[str]:
    return self._processor.batch_decode(tokens, skip_special_tokens=True)


class AsrModel:
  """Base class of ASR models to be converted to tflite."""

  @dataclasses.dataclass
  class OriginalModelOutput:
    logits: torch.Tensor
    tokens: torch.Tensor

  def __init__(self, override_transformers: bool = False):
    if not override_transformers:
      return
    # Replace sdpa for GPU inference.
    attn_funs = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
    attn_funs["sdpa"] = _sdpa

  @abc.abstractmethod
  def get_encoder(self) -> torch.nn.Module:
    """Returns the encoder to be converted to 'encode' subgraph in tflite."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decoder(self) -> torch.nn.Module:
    """Returns the decoder to be converted to 'decode' subgraph in tflite."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_processor(self) -> AsrProcessor:
    """Returns the processor to convert audio to model inputs."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_encoder_sample_input(
      self, processed_audio: tuple[str, torch.Tensor]
  ) -> tuple[torch.Tensor, ...]:
    """Builds the encoder inputs as sample args for conversion."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decoder_sample_input(
      self, encoder_output: torch.Tensor, num_tokens: int
  ) -> tuple[torch.Tensor, ...]:
    """Builds the decoder inputs as sample args for conversion."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decode_start_token_id(self) -> int:
    """Returns the token ID to indicate the start of decoding."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get_decode_stop_token_id(self) -> int:
    """Returns the token ID to indicate the end of decoding."""
    raise NotImplementedError()

  @abc.abstractmethod
  def run_original_model(
      self, processed_audio: dict[str, torch.Tensor]
  ) -> OriginalModelOutput:
    """Runs the original model and returns the logits and sequences."""
    raise NotImplementedError()

  def _replace_normalizations(self, module: torch.nn.Module):
    """Replaces normalizations with composite ops for GPU inference."""
    for name, child in list(module.named_children()):
      if isinstance(child, torch.nn.GroupNorm):
        assert child.affine
        setattr(module, name, AsrGroupNorm(
            child.num_groups, child.eps, child.weight, child.bias
        ))
      elif isinstance(child, torch.nn.LayerNorm):
        assert child.elementwise_affine
        setattr(module, name, AsrLayerNorm(
            child.normalized_shape, child.eps, child.weight, child.bias
        ))
      elif isinstance(child, torch.nn.RMSNorm):
        setattr(module, name, AsrRMSNorm(child.eps, child.weight))
      else:
        self._replace_normalizations(child)
