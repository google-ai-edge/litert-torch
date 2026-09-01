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
"""Tests that converted models propagate NaN/Inf like PyTorch does.

Each case feeds inputs containing NaN and +/-Inf through a small module and
checks that the converted model produces NaN in exactly the same positions as
PyTorch (and matching finite values elsewhere).

Known pre-existing mismatches are asserted as such (`known_mismatch=True`) so
that a fix (or a further regression elsewhere) is noticed:
- TFLite `reduce_max` / `reduce_min` drop NaN (v1 and v2).
- The v1 path loses +/-Inf baked into conv / GroupNorm bias constants.
"""

from typing import Callable

from absl.testing import parameterized
import litert_torch
from litert_torch import testing
from litert_torch.generative.examples.stable_diffusion import decoder as sd_decoder
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from absl.testing import absltest as googletest


def _with_non_finite(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
  """Returns a random tensor with a few NaN and +/-Inf entries."""
  gen = torch.Generator().manual_seed(seed)
  x = torch.randn(shape, generator=gen)
  flat = x.view(-1)
  n = flat.numel()
  flat[0] = float("nan")
  flat[n // 3] = float("inf")
  flat[(2 * n) // 3] = -float("inf")
  return x


class _Lambda(nn.Module):

  def __init__(self, fn: Callable[..., torch.Tensor]):
    super().__init__()
    self.fn = fn

  def forward(self, *args):
    return self.fn(*args)


class _GroupNorm(nn.Module):

  def __init__(self, channels: int, groups: int):
    super().__init__()
    self.norm = nn.GroupNorm(groups, channels)
    with torch.no_grad():
      self.norm.weight.uniform_(0.5, 1.5)
      self.norm.bias.uniform_(-0.5, 0.5)

  def forward(self, x):
    return self.norm(x)


class _Attention(nn.Module):
  """Single-head attention as used by the SD decoder mid block."""

  def forward(self, q, k, v):
    scores = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    return torch.matmul(F.softmax(scores, dim=-1), v)


_CASES = (
    ("silu", _Lambda(F.silu), ((1, 4, 8, 8),)),
    ("gelu", _Lambda(F.gelu), ((1, 4, 8, 8),)),
    ("sigmoid", _Lambda(torch.sigmoid), ((1, 4, 8, 8),)),
    ("tanh", _Lambda(torch.tanh), ((1, 4, 8, 8),)),
    ("group_norm", _GroupNorm(4, 2), ((1, 4, 8, 8),)),
    (
        "conv2d",
        nn.Conv2d(4, 4, kernel_size=3, padding=1),
        ((1, 4, 8, 8),),
    ),
    ("softmax", _Lambda(lambda x: F.softmax(x, dim=-1)), ((2, 3, 16),)),
    ("attention", _Attention(), ((1, 16, 4), (1, 16, 4), (1, 16, 4))),
    (
        "upsample_nearest",
        _Lambda(lambda x: F.interpolate(x, scale_factor=2.0, mode="nearest")),
        ((1, 4, 8, 8),),
    ),
    ("mean", _Lambda(lambda x: x.mean(dim=(2, 3))), ((1, 4, 8, 8),)),
    ("sum", _Lambda(lambda x: x.sum(dim=-1)), ((1, 4, 8, 8),)),
    # TFLite reduce_max/reduce_min drop NaN.
    ("amax", _Lambda(lambda x: x.amax(dim=-1)), ((1, 4, 8, 8),), True),
    ("amin", _Lambda(lambda x: x.amin(dim=-1)), ((1, 4, 8, 8),), True),
    ("maximum", _Lambda(torch.maximum), ((1, 4, 8, 8), (1, 4, 8, 8))),
    ("minimum", _Lambda(torch.minimum), ((1, 4, 8, 8), (1, 4, 8, 8))),
    (
        "rsqrt_var",
        _Lambda(lambda x: torch.rsqrt(x.var(dim=-1) + 1e-5)),
        ((1, 4, 8, 8),),
    ),
)


@testing.parameterized_class(testing.V1_V2_PARAMETERS)
class TestNanPropagation(testing.V1V2TestCase, parameterized.TestCase):
  """Converted models must propagate NaN/Inf the way PyTorch does."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def _assert_matches_torch(
      self, module: nn.Module, args, known_mismatch: bool = False
  ) -> None:
    """Asserts edge and torch outputs match, or mismatch if `known_mismatch`."""
    if known_mismatch:
      with self.assertRaises(AssertionError):
        self._assert_matches_torch(module, args)
      return
    module = module.eval()
    torch_out = module(*args).detach().numpy()
    edge_model = litert_torch.convert(module, args)
    edge_out = edge_model(*args)
    np.testing.assert_array_equal(
        np.isnan(edge_out), np.isnan(torch_out), err_msg="NaN positions differ"
    )
    np.testing.assert_allclose(
        edge_out, torch_out, rtol=1e-4, atol=1e-4, equal_nan=True
    )

  @parameterized.named_parameters(*_CASES)
  def test_op(self, module, shapes, known_mismatch=False):
    args = tuple(
        _with_non_finite(shape, seed=i) for i, shape in enumerate(shapes)
    )
    self._assert_matches_torch(module, args, known_mismatch=known_mismatch)

  def test_group_norm_nan_weights(self):
    """NaN/Inf baked into constant weights must propagate too."""
    module = _GroupNorm(4, 2)
    with torch.no_grad():
      module.norm.weight[1] = float("nan")
      module.norm.bias[2] = float("inf")
    # The v1 path loses the Inf baked into the bias constant.
    self._assert_matches_torch(
        module, (torch.randn(1, 4, 8, 8),), known_mismatch=not self.use_v2
    )

  def test_conv2d_nan_weights(self):
    module = nn.Conv2d(4, 4, kernel_size=3, padding=1)
    with torch.no_grad():
      module.weight[1, 0, 0, 0] = float("nan")
      module.bias[3] = -float("inf")
    # The v1 path loses the -Inf baked into the bias constant.
    self._assert_matches_torch(
        module, (torch.randn(1, 4, 8, 8),), known_mismatch=not self.use_v2
    )

  def test_stable_diffusion_decoder_nan_weights(self):
    """SD decoder whose GroupNorm params contain NaN (cf. torch.empty)."""
    model = sd_decoder.Decoder(sd_decoder.get_fake_model_config())
    with torch.no_grad():
      for p in model.parameters():
        p.uniform_(-0.5, 0.5)
      for m in model.modules():
        if hasattr(m, "group_num") and hasattr(m, "weight"):
          m.weight[0] = float("nan")
    self._assert_matches_torch(model, (torch.randn(1, 4, 16, 16) * 0.1,))

  def test_stable_diffusion_decoder_nan_input(self):
    """Full SD decoder with initialized weights and a NaN-containing input."""
    model = sd_decoder.Decoder(sd_decoder.get_fake_model_config())
    with torch.no_grad():
      for p in model.parameters():
        p.uniform_(-0.5, 0.5)
    latents = _with_non_finite((1, 4, 16, 16))
    self._assert_matches_torch(model, (latents,))


if __name__ == "__main__":
  googletest.main()
