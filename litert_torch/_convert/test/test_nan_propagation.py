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

Known pre-existing mismatches are asserted as exactly that mismatch, so that a
fix (or any further regression) is noticed:
- `_DROPS_NAN`: TFLite `reduce_max` / `reduce_min`, and `maximum` / `minimum`
  with NaN in only one operand, drop NaN (v1 and v2). Only NaN positions may
  differ; all other outputs must still match.
- `_LOSES_INF`: the v1 path loses +/-Inf baked into conv / GroupNorm bias
  constants. Only +/-Inf positions may differ; NaN positions and finite
  outputs must still match.
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


# Kinds of known mismatches, see the module docstring.
_DROPS_NAN = "drops_nan"
_LOSES_INF = "loses_inf"


def _with_non_finite(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
  """Returns a random tensor with a NaN, a +Inf and a -Inf entry.

  The positions depend on `seed`, so operands created with different seeds
  have their non-finite values at different positions (e.g. NaN in only one
  operand of a binary op).

  Args:
    shape: The shape of the tensor.
    seed: Seed for the values and the non-finite positions.
  """
  gen = torch.Generator().manual_seed(seed)
  x = torch.randn(shape, generator=gen)
  flat = x.view(-1)
  nan_pos, pos_inf_pos, neg_inf_pos = torch.randperm(
      flat.numel(), generator=gen
  )[:3].tolist()
  flat[nan_pos] = float("nan")
  flat[pos_inf_pos] = float("inf")
  flat[neg_inf_pos] = -float("inf")
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
  """Single-head attention as used by the SD decoder mid block.

  Keys and values are finite constants: a non-finite key or value reaches every
  query and would make the whole output NaN. A non-finite query only affects
  its own output row.
  """

  def __init__(self, seq_len: int, dim: int):
    super().__init__()
    gen = torch.Generator().manual_seed(1)
    self.register_buffer("k", torch.randn(1, seq_len, dim, generator=gen))
    self.register_buffer("v", torch.randn(1, seq_len, dim, generator=gen))

  def forward(self, q):
    scores = torch.matmul(q, self.k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    return torch.matmul(F.softmax(scores, dim=-1), self.v)


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
    ("attention", _Attention(16, 4), ((1, 16, 4),)),
    (
        "upsample_nearest",
        _Lambda(lambda x: F.interpolate(x, scale_factor=2.0, mode="nearest")),
        ((1, 4, 8, 8),),
    ),
    ("mean", _Lambda(lambda x: x.mean(dim=(2, 3))), ((1, 4, 8, 8),)),
    ("sum", _Lambda(lambda x: x.sum(dim=-1)), ((1, 4, 8, 8),)),
    # TFLite reduce_max/reduce_min drop NaN.
    ("amax", _Lambda(lambda x: x.amax(dim=-1)), ((1, 4, 8, 8),), _DROPS_NAN),
    ("amin", _Lambda(lambda x: x.amin(dim=-1)), ((1, 4, 8, 8),), _DROPS_NAN),
    # TFLite maximum/minimum drop NaN when only one operand is NaN.
    (
        "maximum",
        _Lambda(torch.maximum),
        ((1, 4, 8, 8), (1, 4, 8, 8)),
        _DROPS_NAN,
    ),
    (
        "minimum",
        _Lambda(torch.minimum),
        ((1, 4, 8, 8), (1, 4, 8, 8)),
        _DROPS_NAN,
    ),
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
      self,
      module: nn.Module,
      args,
      known_mismatch: str | None = None,
      allow_all_nan: bool = False,
      edge_model=None,
  ):
    """Asserts edge and torch outputs match, up to `known_mismatch`.

    Args:
      module: The module to convert.
      args: The sample inputs.
      known_mismatch: None, `_DROPS_NAN` or `_LOSES_INF`. If set, asserts that
        exactly that mismatch occurs and everything else matches.
      allow_all_nan: Whether an all-NaN torch output is acceptable. Otherwise
        the test would pass for any converter that also outputs all NaN.
      edge_model: An already converted model to run instead of converting
        `module`.

    Returns:
      The converted model.
    """
    module = module.eval()
    torch_out = module(*args).detach().numpy()
    if not allow_all_nan:
      self.assertFalse(
          np.isnan(torch_out).all(), "torch output is all NaN; test is vacuous"
      )
    if edge_model is None:
      edge_model = litert_torch.convert(module, args)
    edge_out = edge_model(*args)

    torch_nan = np.isnan(torch_out)
    torch_inf = np.isinf(torch_out)
    # Positions where the converted model must match exactly (up to tolerance).
    must_match = np.ones_like(torch_nan)
    if known_mismatch == _DROPS_NAN:
      must_match = ~torch_nan
      self.assertTrue(
          np.any(torch_nan & ~np.isnan(edge_out)),
          "expected NaN to be dropped; if fixed, remove the known mismatch",
      )
    elif known_mismatch == _LOSES_INF:
      must_match = ~torch_inf
      self.assertTrue(
          np.any(torch_inf & (edge_out != torch_out)),
          "expected +/-Inf to be lost; if fixed, remove the known mismatch",
      )
    else:
      self.assertIsNone(known_mismatch)

    np.testing.assert_array_equal(
        np.isnan(edge_out)[must_match],
        torch_nan[must_match],
        err_msg="NaN positions differ",
    )
    np.testing.assert_allclose(
        edge_out[must_match],
        torch_out[must_match],
        rtol=1e-4,
        atol=1e-4,
        equal_nan=True,
    )
    return edge_model

  @parameterized.named_parameters(*_CASES)
  def test_op(self, module, shapes, known_mismatch=None):
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
        module,
        (torch.randn(1, 4, 8, 8),),
        known_mismatch=None if self.use_v2 else _LOSES_INF,
    )

  def test_conv2d_nan_weights(self):
    module = nn.Conv2d(4, 4, kernel_size=3, padding=1)
    with torch.no_grad():
      module.weight[1, 0, 0, 0] = float("nan")
      module.bias[3] = -float("inf")
    # The v1 path loses the -Inf baked into the bias constant.
    self._assert_matches_torch(
        module,
        (torch.randn(1, 4, 8, 8),),
        known_mismatch=None if self.use_v2 else _LOSES_INF,
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
    # The NaN reaches every output through the convolutions, so this only checks
    # that NaN isn't dropped. test_stable_diffusion_decoder_nan_input checks
    # that the converted decoder doesn't output NaN everywhere.
    self._assert_matches_torch(
        model, (torch.randn(1, 4, 16, 16) * 0.1,), allow_all_nan=True
    )

  def test_stable_diffusion_decoder_nan_input(self):
    """Full SD decoder with initialized weights and a NaN-containing input."""
    model = sd_decoder.Decoder(sd_decoder.get_fake_model_config())
    with torch.no_grad():
      for p in model.parameters():
        p.uniform_(-0.5, 0.5)
    # With finite latents the output must be finite and match, which rules out
    # a converted model that outputs NaN everywhere.
    edge_model = self._assert_matches_torch(
        model, (torch.randn(1, 4, 16, 16) * 0.1,)
    )
    # A NaN anywhere in the latents reaches every output through the
    # convolutions and attention, so the expected output is all NaN.
    latents = _with_non_finite((1, 4, 16, 16))
    self._assert_matches_torch(
        model, (latents,), allow_all_nan=True, edge_model=edge_model
    )


if __name__ == "__main__":
  googletest.main()
