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
# Common normalization layers.

from typing import Callable

from litert_torch.hlfb import StableHLOCompositeBuilder
import torch
from torch import nn
import torch.nn.functional as F


# Implementation for RMSNorm from: https://arxiv.org/abs/1910.07467
class RMSNorm(torch.nn.Module):

  def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
      zero_centered_gamma=False,
      with_scale: bool = False,
      scale_shift: float = 1.0,
      enable_hlfb: bool = False,
      init_fn: Callable[..., torch.Tensor] = lambda *args, **kwargs: None,
  ):
    """Initialize the RMSNorm layer.

    Args:
      dim (int): dimension of the input tensor.
      eps (float): A small float value to ensure numerical stability (default:
        1e-6).
      zero_centered_gamma (bool): Whether or not gamma has an offset.
      with_scale (bool): Whether or not to use a scale parameter.
      scale_shift (float): The shift to apply to the scale parameter.
      enable_hlfb (bool): use HLFB in the op.
      init_fn: The initialization function to use for the parameters. This is
        used to initialize the scale parameter.
    """
    super().__init__()
    self.dim = dim
    self.enable_hlfb = enable_hlfb
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.ones(dim), requires_grad=False)
    init_fn(self.weight)
    self.zero_centered_gamma = zero_centered_gamma
    self.with_scale = with_scale
    if with_scale:
      self.scale = torch.nn.Parameter(
          torch.zeros((dim,), dtype=torch.float32), requires_grad=False
      )
    self.scale_shift = scale_shift

  def _norm(self, x):
    """Apply RMSNorm normalization.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: The normalized output tensor.
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    """Running the forward pass of RMSNorm layer.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: output tensor after applying RMSNorm.
    """
    if self.zero_centered_gamma:
      w = 1 + self.weight
    else:
      w = self.weight

    final_scale = (
        self.scale + self.scale_shift
        if self.with_scale
        else torch.ones((self.dim,), dtype=torch.float32, device=x.device)
    )
    if self.enable_hlfb:
      return rms_norm_with_hlfb(
          x,
          w,
          self.eps,
          final_scale,
      )
    else:
      output = self._norm(x.float()).type_as(x) * final_scale
      return output * w


class GroupNorm(torch.nn.Module):

  def __init__(
      self,
      group_num: int,
      dim: int,
      eps: float = 1e-5,
      enable_hlfb: bool = False,
  ):
    """Initialize the GroupNorm layer.

    Args:
      group_num (int): Number of groups to separate the channels into.
      dim (int): Dimension of the input tensor.
      eps (float): A small float value to ensure numerical stability (default:
        1e-5).
      enable_hlfb (bool): Whether to convert this normalization into a single
        op.
    """
    super().__init__()
    self.enable_hlfb = enable_hlfb
    self.group_num = group_num
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.empty(dim), requires_grad=False)
    self.bias = torch.nn.Parameter(torch.empty(dim), requires_grad=False)

  def forward(self, x):
    """Running the forward pass of GroupNorm layer.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: output tensor after applying GroupNorm.
    """
    return F.group_norm(x, self.group_num, self.weight, self.bias, self.eps)


class LayerNorm(torch.nn.Module):

  def __init__(
      self,
      dim: int,
      eps: float = 1e-5,
      use_bias: bool = True,
      enable_hlfb: bool = False,
  ):
    """Initialize the LayerNorm layer.

    Args:
      dim (int): dimension of the input tensor.
      eps (float): A small float value to ensure numerical stability (default:
        1e-5).
      use_bias (bool): Whether to use bias in LayerNorm.
      enable_hlfb (bool): Whether to convert this normalization into a single
        op.
    """
    super().__init__()
    self.enable_hlfb = enable_hlfb
    self.normalized_shape = (dim,)
    self.eps = eps
    self.weight = torch.nn.Parameter(torch.empty(dim), requires_grad=False)
    self.bias = (
        torch.nn.Parameter(torch.empty(dim), requires_grad=False)
        if use_bias
        else None
    )

  def forward(self, x):
    """Running the forward pass of LayerNorm layer.

    Args:
      x (torch.Tensor): input tensor.

    Returns:
      torch.Tensor: output tensor after applying LayerNorm.
    """
    if self.enable_hlfb and self.bias is not None:
      return layer_norm_with_hlfb(
          x, self.normalized_shape, self.weight, self.bias, self.eps
      )
    return F.layer_norm(
        x, self.normalized_shape, self.weight, self.bias, self.eps
    )


def rms_norm_with_hlfb(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    final_scale: torch.Tensor,
):
  """RMS Normalization with high-level function boundary enabled.

  Args:
    x (torch.Tensor): Input tensor for RMS Normalization, with BCHW shape.
    w (torch.Tensor): The learned parameter tensor for normalization.
    eps (float): A small float value to ensure numerical stability.
    final_scale (torch.Tensor): The final scale to apply to the normalization.

  Returns:
    The output tensor of RMS Normalization.
  """
  builder = StableHLOCompositeBuilder(
      name="odml.rms_norm", attr={"epsilon": eps}
  )

  x, w = builder.mark_inputs(x, w)

  def _norm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

  output = _norm(x.float()).type_as(x) * final_scale
  out = output * w

  out = builder.mark_outputs(out)
  return out


def layer_norm_with_hlfb(
    x: torch.Tensor,
    normalized_shape: list[int],
    w: torch.Tensor,
    b: torch.Tensor,
    eps: float,
):
  """Layer Normalization with high-level function boundary enabled.

  Args:
    x (torch.Tensor): Input tensor for Layer Normalization, with BCHW shape.
    normalized_shape (list[int]): Input shape from an expected input of size,
      same as https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html.
    w (torch.Tensor): The weight tensor for the normalization.
    b (torch.Tensor): The bias tensor for the normalization.
    eps (float): A small float value to ensure numerical stability.

  Returns:
    The output tensor of Layer Normalization.
  """
  builder = StableHLOCompositeBuilder(
      name="odml.group_norm",
      attr={"num_groups": 1, "epsilon": eps, "channel_axis": 1},
  )
  x, w, b = builder.mark_inputs(x, w, b)
  y = F.layer_norm(x, normalized_shape, w, b, eps=eps)
  y = builder.mark_outputs(y)
  return y
