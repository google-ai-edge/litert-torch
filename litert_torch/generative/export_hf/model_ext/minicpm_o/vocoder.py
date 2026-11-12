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
"""CosyVoice2 Token2Wav (`FlowEncode` + `FlowDit` + `HiFT`) 24 kHz vocoder."""

from __future__ import annotations

import math
from typing import Any
import wave

from litert_torch.generative.export_hf.model_ext.minicpm_o import weights_utils
import torch
from torch import nn
import torch.nn.functional as F

_make_linear = weights_utils.make_linear
_make_embedding = weights_utils.make_embedding

# 1.0-second (25-token) S3TokenizerV2 discrete speech tokens and L2-normalized
# 192-dim CampPlus speaker embedding extracted from MiniCPM-o 4.5's reference
# voice prompt (`assets/system_ref_audio.wav`).
DEFAULT_VOICE_PROMPT_TOKENS: tuple[int, ...] = (
    2646,
    1977,
    1707,
    1950,
    1734,
    3894,
    4137,
    1730,
    1733,
    1055,
    2,
    2198,
    2189,
    245,
    1882,
    1806,
    2058,
    6456,
    2142,
    4349,
    728,
    3643,
    3566,
    2921,
    5348,
)

DEFAULT_VOICE_SPK_EMB: tuple[float, ...] = (
    -0.057898,
    0.013475,
    0.045158,
    0.031746,
    0.075352,
    -0.043855,
    0.019389,
    -0.050324,
    -0.028931,
    0.195181,
    -0.144251,
    -0.063006,
    -0.120187,
    -0.051614,
    -0.001437,
    -0.037310,
    0.023042,
    0.045119,
    -0.105212,
    0.022210,
    0.086806,
    0.011909,
    -0.038803,
    0.087021,
    0.070519,
    0.015503,
    -0.094655,
    0.047297,
    -0.171457,
    0.040670,
    -0.096043,
    0.084931,
    -0.010608,
    0.053378,
    -0.024209,
    0.039696,
    0.005362,
    -0.040355,
    -0.095934,
    0.033964,
    0.077026,
    0.014520,
    -0.099987,
    0.029128,
    -0.018839,
    -0.046773,
    0.030654,
    0.038926,
    0.109632,
    -0.085741,
    0.095212,
    0.002103,
    0.073544,
    0.025766,
    0.001908,
    0.003703,
    -0.093550,
    0.020399,
    -0.151297,
    0.113184,
    -0.191178,
    -0.011058,
    -0.126341,
    -0.047391,
    -0.031899,
    0.067353,
    0.015840,
    0.081736,
    -0.055954,
    0.070745,
    -0.026865,
    0.036182,
    0.015150,
    0.080650,
    -0.015771,
    -0.129775,
    -0.061221,
    -0.023663,
    0.050741,
    0.062868,
    0.048945,
    0.055351,
    -0.078779,
    -0.012961,
    0.017278,
    0.010187,
    0.054335,
    -0.085812,
    0.082667,
    0.005698,
    -0.108338,
    0.082009,
    -0.034831,
    0.030019,
    -0.004983,
    -0.083819,
    -0.014998,
    0.091801,
    -0.176014,
    -0.035501,
    0.140222,
    0.052626,
    -0.106594,
    0.000283,
    -0.052583,
    0.049050,
    0.129498,
    0.069213,
    0.119614,
    0.018544,
    -0.105897,
    -0.048828,
    -0.115597,
    -0.138362,
    -0.092607,
    -0.038027,
    -0.071577,
    -0.044897,
    0.036598,
    0.107662,
    0.000824,
    0.014892,
    -0.012284,
    -0.141932,
    -0.021740,
    -0.066643,
    -0.062298,
    0.094464,
    0.027568,
    0.040159,
    0.017821,
    0.048327,
    -0.022848,
    0.007671,
    -0.000742,
    0.096806,
    0.135016,
    0.015251,
    -0.089117,
    -0.094675,
    0.148687,
    -0.069019,
    -0.105856,
    -0.085959,
    0.101601,
    0.114614,
    0.030098,
    -0.024857,
    0.060976,
    -0.065242,
    -0.078122,
    0.033767,
    0.071654,
    -0.027941,
    -0.060755,
    -0.018369,
    -0.057368,
    0.045321,
    0.087106,
    -0.049172,
    -0.047771,
    -0.039744,
    0.113224,
    -0.083747,
    0.150257,
    -0.032573,
    0.040148,
    -0.086425,
    -0.111649,
    -0.000703,
    0.036045,
    0.038436,
    0.056946,
    -0.020167,
    0.039110,
    0.031839,
    -0.039030,
    0.006115,
    -0.065704,
    0.057719,
    -0.035222,
    -0.023754,
    0.088477,
    0.011716,
    -0.043469,
    0.089464,
    0.063047,
    0.004747,
    -0.005369,
    0.058441,
    0.002205,
    -0.110330,
)

_DEFAULT_VOICE_PROMPT_TOKENS = DEFAULT_VOICE_PROMPT_TOKENS
_DEFAULT_VOICE_SPK_EMB = DEFAULT_VOICE_SPK_EMB

_SLANEY_F_SP = 200.0 / 3.0
_SLANEY_MIN_LOG_HZ = 1000.0
_SLANEY_MIN_LOG_MEL = _SLANEY_MIN_LOG_HZ / _SLANEY_F_SP
_SLANEY_LOGSTEP = math.log(6.4) / 27.0


def _hz_to_mel(hz: float) -> float:
  """Converts frequency in Hz to Slaney mel scale."""
  if hz >= _SLANEY_MIN_LOG_HZ:
    return (
        _SLANEY_MIN_LOG_MEL
        + math.log(max(hz, 1e-10) / _SLANEY_MIN_LOG_HZ) / _SLANEY_LOGSTEP
    )
  return hz / _SLANEY_F_SP


def _mel_to_hz(mel: float) -> float:
  """Converts Slaney mel value to frequency in Hz."""
  if mel >= _SLANEY_MIN_LOG_MEL:
    return _SLANEY_MIN_LOG_HZ * math.exp(
        _SLANEY_LOGSTEP * (mel - _SLANEY_MIN_LOG_MEL)
    )
  return _SLANEY_F_SP * mel


def _build_slaney_mel_basis(
    sr: int = 24000,
    n_fft: int = 1920,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> torch.Tensor:
  """Builds a Slaney-normalized mel filterbank `[n_mels, 1 + n_fft // 2]`."""
  mel_min = _hz_to_mel(fmin)
  mel_max = _hz_to_mel(fmax)
  mel_pts = [
      _mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1))
      for i in range(n_mels + 2)
  ]
  mel_f = torch.tensor(mel_pts, dtype=torch.float64)
  fftfreqs = torch.linspace(0.0, sr / 2.0, 1 + n_fft // 2, dtype=torch.float64)
  fdiff = mel_f[1:] - mel_f[:-1]
  ramps = mel_f.unsqueeze(1) - fftfreqs.unsqueeze(0)
  weights = torch.zeros((n_mels, 1 + n_fft // 2), dtype=torch.float64)
  for i in range(n_mels):
    lower = -ramps[i] / fdiff[i]
    upper = ramps[i + 2] / fdiff[i + 1]
    weights[i] = torch.clamp(torch.minimum(lower, upper), min=0.0)
  enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
  weights *= enorm.unsqueeze(1)
  return weights.to(dtype=torch.float32)


def _extract_ref_mel_24k(
    wav_path: str, num_frames: int = 50, n_mels: int = 80
) -> torch.Tensor:
  """Extracts 24 kHz CosyVoice2 log-mel prompt features from `wav_path`."""
  with wave.open(wav_path, "rb") as wf:
    sr = wf.getframerate()
    n_ch = wf.getnchannels()
    need_sec = (num_frames * 480) / 24000.0 + 0.5
    raw_bytes = wf.readframes(int(sr * need_sec))
  pcm_i16 = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int16).to(
      dtype=torch.float32
  )
  if n_ch > 1:
    pcm_i16 = pcm_i16.reshape(-1, n_ch)[:, 0]
  wav = (pcm_i16 / 32768.0).unsqueeze(0)
  if sr != 24000:
    n_in = wav.shape[1]
    n_out = int(round(n_in * 24000.0 / float(sr)))
    wav = torch.fft.irfft(torch.fft.rfft(wav, n=n_in), n=n_out) * (
        float(n_out) / float(n_in)
    )
  n_fft = 1920
  hop = 480
  pad = (n_fft - hop) // 2
  wav_pad = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
  spec = torch.stft(
      wav_pad,
      n_fft=n_fft,
      hop_length=hop,
      win_length=n_fft,
      window=torch.hann_window(n_fft, dtype=torch.float32),
      center=False,
      return_complex=True,
  )
  mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
  mel_basis = _build_slaney_mel_basis(sr=24000, n_fft=n_fft, n_mels=n_mels)
  mel = torch.log(torch.clamp(torch.matmul(mel_basis, mag), min=1e-5))
  if mel.shape[2] < num_frames:
    mel = F.pad(mel, (0, num_frames - mel.shape[2]))
  return mel[:, :, :num_frames].contiguous()


def _rel_shift(x: torch.Tensor) -> torch.Tensor:
  """Performs relative positional attention shift matching ESPnet."""
  b, h, qlen, pos_len = x.shape
  zeros = torch.zeros((b, h, qlen, 1), dtype=x.dtype, device=x.device)
  x_padded = torch.cat([zeros, x], dim=-1)
  x_padded = x_padded.reshape(b, h, pos_len + 1, qlen)
  return x_padded[:, :, 1:, :].reshape_as(x)[:, :, :, : pos_len // 2 + 1]


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
  """Applies AdaLN-Zero scale and shift modulation."""
  return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _EspnetRelPositionalEncoding(nn.Module):
  """Relative sinusoidal positional encoding matching CosyVoice2 / ESPnet."""

  pe: torch.Tensor

  def __init__(self, d_model: int, max_len: int = 512):
    super().__init__()
    self.d_model = d_model
    self.xscale = math.sqrt(float(d_model))
    pe_positive = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe_negative = torch.zeros(max_len, d_model, dtype=torch.float32)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(math.log(10000.0) / float(d_model))
    )
    pe_positive[:, 0::2] = torch.sin(position * div_term)
    pe_positive[:, 1::2] = torch.cos(position * div_term)
    pe_negative[:, 0::2] = torch.sin(-1.0 * position * div_term)
    pe_negative[:, 1::2] = torch.cos(-1.0 * position * div_term)
    pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
    pe_negative = pe_negative[1:].unsqueeze(0)
    pe = torch.cat([pe_positive, pe_negative], dim=1)
    self.register_buffer("pe", pe, persistent=False)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = x.shape[1]
    mid = self.pe.shape[1] // 2
    pos_emb = self.pe[:, mid - seq_len + 1 : mid + seq_len, :].to(
        dtype=x.dtype, device=x.device
    )
    return x * self.xscale, pos_emb


class _FlowConformerBlock(nn.Module):
  """Relative-position Conformer encoder block for CosyVoice2 `FlowEncode`."""

  def __init__(
      self, d_model: int = 512, num_heads: int = 8, ffn_dim: int = 2048
  ):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    self.norm_mha = nn.LayerNorm(d_model, eps=1e-12)
    self.pos_bias_u = nn.Parameter(
        torch.zeros(num_heads, self.head_dim, dtype=torch.float32)
    )
    self.pos_bias_v = nn.Parameter(
        torch.zeros(num_heads, self.head_dim, dtype=torch.float32)
    )
    self.linear_q = _make_linear(d_model, d_model, bias=True)
    self.linear_k = _make_linear(d_model, d_model, bias=True)
    self.linear_v = _make_linear(d_model, d_model, bias=True)
    self.linear_out = _make_linear(d_model, d_model, bias=True)
    self.linear_pos = _make_linear(d_model, d_model, bias=False)
    self.norm_ff = nn.LayerNorm(d_model, eps=1e-12)
    self.ff_w1 = _make_linear(d_model, ffn_dim, bias=True)
    self.ff_w2 = _make_linear(ffn_dim, d_model, bias=True)

  def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
    b, t, d = x.shape
    n_batch_pos = pos_emb.shape[0]
    p_len = pos_emb.shape[1]
    h = self.norm_mha(x)
    q = self.linear_q(h).reshape(b, t, self.num_heads, self.head_dim)
    k = (
        self.linear_k(h)
        .reshape(b, t, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    v = (
        self.linear_v(h)
        .reshape(b, t, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    p = (
        self.linear_pos(pos_emb)
        .reshape(n_batch_pos, p_len, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    q_with_bias_u = (
        q + self.pos_bias_u.reshape(1, 1, self.num_heads, self.head_dim)
    ).permute(0, 2, 1, 3)
    q_with_bias_v = (
        q + self.pos_bias_v.reshape(1, 1, self.num_heads, self.head_dim)
    ).permute(0, 2, 1, 3)
    matrix_ac = torch.matmul(q_with_bias_u, k.permute(0, 1, 3, 2))
    matrix_bd = _rel_shift(torch.matmul(q_with_bias_v, p.permute(0, 1, 3, 2)))
    scores = (matrix_ac + matrix_bd) * (1.0 / math.sqrt(float(self.head_dim)))
    attn = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, t, d)
    x = x + self.linear_out(attn_out)
    x = x + self.ff_w2(F.silu(self.ff_w1(self.norm_ff(x))))
    return x


class _FlowEncode(nn.Module):
  """CosyVoice2 UpsampleConformerEncoder (`input_embedding` + `encoder`)."""

  def __init__(
      self,
      vocab_size: int = 6561,
      flow_dim: int = 512,
      flow_heads: int = 8,
      flow_ffn_dim: int = 2048,
      num_encoders: int = 6,
      num_up_encoders: int = 4,
      mel_bins: int = 80,
      pre_lookahead_len: int = 3,
  ):
    super().__init__()
    self.pre_lookahead_len = pre_lookahead_len
    self.input_embedding = _make_embedding(vocab_size, flow_dim)
    self.embed_linear = _make_linear(flow_dim, flow_dim, bias=True)
    self.embed_norm = nn.LayerNorm(flow_dim, eps=1e-5)
    self.embed_pos = _EspnetRelPositionalEncoding(flow_dim)
    self.pre_lookahead_conv1 = nn.Conv1d(
        flow_dim,
        flow_dim,
        kernel_size=pre_lookahead_len + 1,
        stride=1,
        padding=0,
    )
    self.pre_lookahead_conv2 = nn.Conv1d(
        flow_dim, flow_dim, kernel_size=3, stride=1, padding=0
    )
    self.encoders = nn.ModuleList([
        _FlowConformerBlock(flow_dim, flow_heads, flow_ffn_dim)
        for _ in range(num_encoders)
    ])
    self.up_conv = nn.Conv1d(
        flow_dim, flow_dim, kernel_size=5, stride=1, padding=0
    )
    self.up_embed_linear = _make_linear(flow_dim, flow_dim, bias=True)
    self.up_embed_norm = nn.LayerNorm(flow_dim, eps=1e-5)
    self.up_embed_pos = _EspnetRelPositionalEncoding(flow_dim)
    self.up_encoders = nn.ModuleList([
        _FlowConformerBlock(flow_dim, flow_heads, flow_ffn_dim)
        for _ in range(num_up_encoders)
    ])
    self.after_norm = nn.LayerNorm(flow_dim, eps=1e-12)
    self.encoder_proj = _make_linear(flow_dim, mel_bins, bias=True)

  def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
    b, s = audio_codes.shape
    out_tok_len = s - self.pre_lookahead_len
    xs = self.input_embedding(audio_codes)
    xs = self.embed_norm(self.embed_linear(xs))
    xs, pos_emb = self.embed_pos(xs)

    outputs = xs.permute(0, 2, 1)
    outputs = F.leaky_relu(self.pre_lookahead_conv1(outputs))
    outputs = F.pad(outputs, (2, 0), mode="constant", value=0.0)
    outputs = self.pre_lookahead_conv2(outputs).permute(0, 2, 1)
    xs = outputs + xs[:, :out_tok_len, :]
    pos_emb = pos_emb[
        :, self.pre_lookahead_len : 2 * s - 1 - self.pre_lookahead_len, :
    ]

    for enc in self.encoders:
      xs = enc(xs, pos_emb)

    xs_up = (
        xs.permute(0, 2, 1)
        .unsqueeze(-1)
        .expand(b, xs.shape[2], out_tok_len, 2)
        .reshape(b, xs.shape[2], out_tok_len * 2)
    )
    xs_up = F.pad(xs_up, (4, 0), mode="constant", value=0.0)
    xs = self.up_conv(xs_up).permute(0, 2, 1)

    xs = self.up_embed_norm(self.up_embed_linear(xs))
    xs, pos_emb = self.up_embed_pos(xs)
    for enc in self.up_encoders:
      xs = enc(xs, pos_emb)

    xs = self.after_norm(xs)
    mu = self.encoder_proj(xs).permute(0, 2, 1)
    return mu


class _FlowDitBlock(nn.Module):
  """DiT block with AdaLN-Zero, QK-norm attention, causal conv, and MLP."""

  def __init__(self, dim: int = 512, num_heads: int = 8, ffn_dim: int = 2048):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.adaLN_linear = _make_linear(  # pylint: disable=invalid-name
        dim, 9 * dim, bias=True
    )
    self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    self.attn_to_q = _make_linear(dim, dim, bias=True)
    self.attn_to_k = _make_linear(dim, dim, bias=True)
    self.attn_to_v = _make_linear(dim, dim, bias=True)
    self.attn_q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
    self.attn_k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
    self.attn_proj = _make_linear(dim, dim, bias=True)
    self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    self.conv_1 = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=0)
    self.conv_norm = nn.LayerNorm(dim, eps=1e-5)
    self.conv_6 = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=0)
    self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
    self.mlp_fc1 = _make_linear(dim, ffn_dim, bias=True)
    self.mlp_fc2 = _make_linear(ffn_dim, dim, bias=True)

  def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    b, n, c_dim = x.shape
    (
        shift_msa,
        scale_msa,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        shift_conv,
        scale_conv,
        gate_conv,
    ) = self.adaLN_linear(F.silu(c)).chunk(9, dim=-1)

    h1 = _modulate(self.norm1(x), shift_msa, scale_msa)
    q = (
        self.attn_to_q(h1)
        .reshape(b, n, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    k = (
        self.attn_to_k(h1)
        .reshape(b, n, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    v = (
        self.attn_to_v(h1)
        .reshape(b, n, self.num_heads, self.head_dim)
        .permute(0, 2, 1, 3)
    )
    q = self.attn_q_norm(q)
    k = self.attn_k_norm(k)
    scores = torch.matmul(q, k.permute(0, 1, 3, 2)) * (
        1.0 / math.sqrt(float(self.head_dim))
    )
    probs = F.softmax(scores, dim=-1)
    attn_out = self.attn_proj(
        torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(b, n, c_dim)
    )
    x = x + gate_msa.unsqueeze(1) * attn_out

    h3 = _modulate(self.norm3(x), shift_conv, scale_conv)
    hc = F.pad(h3.permute(0, 2, 1), (2, 0), mode="constant", value=0.0)
    hc = self.conv_1(hc).permute(0, 2, 1)
    hc = F.mish(self.conv_norm(hc)).permute(0, 2, 1)
    hc = F.pad(hc, (2, 0), mode="constant", value=0.0)
    hc = self.conv_6(hc).permute(0, 2, 1)
    x = x + gate_conv.unsqueeze(1) * hc

    h2 = _modulate(self.norm2(x), shift_mlp, scale_mlp)
    mlp_out = self.mlp_fc2(F.gelu(self.mlp_fc1(h2), approximate="tanh"))
    x = x + gate_mlp.unsqueeze(1) * mlp_out
    return x


class _FlowDit(nn.Module):
  """CosyVoice2 Conditional CFM Euler ODE solver with DiT estimator and CFG."""

  default_spk_emb: torch.Tensor
  prompt_mel: torch.Tensor
  flow_noise: torch.Tensor
  t_freqs: torch.Tensor

  def __init__(
      self,
      flow_dim: int = 512,
      flow_heads: int = 8,
      flow_ffn_dim: int = 2048,
      num_dit_blocks: int = 16,
      mel_bins: int = 80,
      spk_dim: int = 192,
      mel_frames: int = 58,
      prompt_mel_frames: int = 0,
      n_timesteps: int = 6,
      cfg_rate: float = 0.7,
  ):
    super().__init__()
    self.flow_dim = flow_dim
    self.mel_bins = mel_bins
    self.prompt_mel_frames = max(0, prompt_mel_frames)
    self.n_timesteps = n_timesteps
    self.cfg_rate = cfg_rate
    self.spk_embed_affine_layer = _make_linear(spk_dim, mel_bins, bias=True)
    self.t_mlp_0 = _make_linear(256, flow_dim, bias=True)
    self.t_mlp_2 = _make_linear(flow_dim, flow_dim, bias=True)
    self.in_proj = _make_linear(4 * mel_bins, flow_dim, bias=True)
    self.blocks = nn.ModuleList([
        _FlowDitBlock(flow_dim, flow_heads, flow_ffn_dim)
        for _ in range(num_dit_blocks)
    ])
    self.final_norm = nn.LayerNorm(flow_dim, elementwise_affine=False, eps=1e-6)
    self.final_adaLN_linear = _make_linear(  # pylint: disable=invalid-name
        flow_dim, 2 * flow_dim, bias=True
    )
    self.final_linear = _make_linear(flow_dim, mel_bins, bias=True)

    gen = torch.Generator()
    gen.manual_seed(0)
    if spk_dim == len(DEFAULT_VOICE_SPK_EMB):
      raw_spk = torch.tensor([DEFAULT_VOICE_SPK_EMB], dtype=torch.float32)
    else:
      raw_spk = torch.randn(1, spk_dim, generator=gen, dtype=torch.float32)
    raw_spk = raw_spk / torch.clamp(
        torch.norm(raw_spk, dim=-1, keepdim=True), min=1e-12
    )
    self.register_buffer("default_spk_emb", raw_spk, persistent=False)
    self.register_buffer(
        "prompt_mel",
        torch.zeros(
            1, mel_bins, max(1, self.prompt_mel_frames), dtype=torch.float32
        ),
        persistent=False,
    )
    total_frames = self.prompt_mel_frames + mel_frames
    noise = torch.randn(
        1, mel_bins, total_frames, generator=gen, dtype=torch.float32
    )
    self.register_buffer("flow_noise", noise, persistent=False)
    t_freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, 128, dtype=torch.float32) / 128.0
    ).reshape(1, 128)
    self.register_buffer("t_freqs", t_freqs, persistent=False)

  def _embed_timestep(
      self, t_val: float, dtype: torch.dtype, device: torch.device
  ) -> torch.Tensor:
    args = float(t_val) * 1000.0 * self.t_freqs.to(dtype=dtype, device=device)
    t_emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return self.t_mlp_2(F.silu(self.t_mlp_0(t_emb)))

  def _estimate(
      self,
      x: torch.Tensor,
      mu: torch.Tensor,
      t_val: float,
      spks: torch.Tensor,
      cond: torch.Tensor,
  ) -> torch.Tensor:
    t_emb = self._embed_timestep(t_val, x.dtype, x.device)
    h = torch.cat([x, mu, spks, cond], dim=1).permute(0, 2, 1)
    h = self.in_proj(h)
    for blk in self.blocks:
      h = blk(h, t_emb)
    shift, scale = self.final_adaLN_linear(F.silu(t_emb)).chunk(2, dim=-1)
    h = self.final_norm(h) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return self.final_linear(h).permute(0, 2, 1)

  def forward(self, mu: torch.Tensor) -> torch.Tensor:
    b, m, frames = mu.shape
    spk_n = self.default_spk_emb.to(dtype=mu.dtype, device=mu.device)
    spk_n = spk_n / torch.clamp(
        torch.norm(spk_n, dim=-1, keepdim=True), min=1e-12
    )
    spks = self.spk_embed_affine_layer(spk_n).unsqueeze(-1).expand(b, m, frames)
    cond = torch.zeros_like(mu)
    if self.prompt_mel_frames > 0:
      p_frames = min(self.prompt_mel_frames, frames)
      p_mel = (
          self.prompt_mel[:, :, :p_frames]
          .to(dtype=mu.dtype, device=mu.device)
          .expand(b, m, p_frames)
      )
      cond = torch.cat([p_mel, cond[:, :, p_frames:]], dim=2)
    x = self.flow_noise[:, :, :frames].to(dtype=mu.dtype, device=mu.device)
    zero_mu = torch.zeros_like(mu)
    zero_spks = torch.zeros_like(spks)
    zero_cond = torch.zeros_like(cond)
    steps = max(1, self.n_timesteps)
    for step in range(steps):
      u0 = float(step) / float(steps)
      u1 = float(step + 1) / float(steps)
      t0 = 1.0 - math.cos(u0 * 0.5 * math.pi)
      t1 = 1.0 - math.cos(u1 * 0.5 * math.pi)
      dt = t1 - t0
      dphi_cond = self._estimate(x, mu, t0, spks, cond)
      if self.cfg_rate > 0.0:
        dphi_uncond = self._estimate(x, zero_mu, t0, zero_spks, zero_cond)
        dphi_dt = (
            1.0 + self.cfg_rate
        ) * dphi_cond - self.cfg_rate * dphi_uncond
      else:
        dphi_dt = dphi_cond
      x = x + dt * dphi_dt
    if self.prompt_mel_frames > 0 and frames > self.prompt_mel_frames:
      return x[:, :, self.prompt_mel_frames :]
    return x


class _SnakeActivation(nn.Module):
  """Periodic Snake activation `x + (1 / (alpha + 1e-9)) * sin(alpha * x)^2`."""

  def __init__(self, channels: int):
    super().__init__()
    self.alpha = nn.Parameter(torch.ones(channels, dtype=torch.float32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    alpha = self.alpha.reshape(1, -1, 1)
    return x + (1.0 / (alpha + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class _HiFTResBlock(nn.Module):
  """Residual block with dilated 1D convolutions and Snake activations."""

  def __init__(
      self,
      channels: int,
      kernel_size: int = 3,
      dilations: tuple[int, ...] = (1, 3, 5),
  ):
    super().__init__()
    self.convs1 = nn.ModuleList()
    self.convs2 = nn.ModuleList()
    self.activations1 = nn.ModuleList()
    self.activations2 = nn.ModuleList()
    for dilation in dilations:
      pad1 = (kernel_size * dilation - dilation) // 2
      pad2 = (kernel_size - 1) // 2
      self.convs1.append(
          nn.Conv1d(
              channels,
              channels,
              kernel_size=kernel_size,
              stride=1,
              dilation=dilation,
              padding=pad1,
          )
      )
      self.convs2.append(
          nn.Conv1d(
              channels,
              channels,
              kernel_size=kernel_size,
              stride=1,
              dilation=1,
              padding=pad2,
          )
      )
      self.activations1.append(_SnakeActivation(channels))
      self.activations2.append(_SnakeActivation(channels))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for act1, conv1, act2, conv2 in zip(
        self.activations1,
        self.convs1,
        self.activations2,
        self.convs2,
        strict=True,
    ):
      xt = conv1(act1(x))
      xt = conv2(act2(xt))
      x = xt + x
    return x


class _HiFTGenerator(nn.Module):
  """CosyVoice2 neural source-filter HiFT 24 kHz mel-to-waveform generator."""

  harmonic_mult: torch.Tensor
  sine_rand_ini: torch.Tensor
  lerp_w: torch.Tensor
  sine_noise: torch.Tensor
  stft_basis: torch.Tensor
  istft_basis: torch.Tensor
  inv_env: torch.Tensor

  def __init__(
      self,
      mel_bins: int = 80,
      base_channels: int = 512,
      f0_conv_layers: int = 5,
      mel_frames: int = 58,
  ):
    super().__init__()
    self.mel_bins = mel_bins
    self.base_channels = base_channels
    self.f0_convs = nn.ModuleList([
        nn.Conv1d(
            mel_bins if i == 0 else base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )
        for i in range(f0_conv_layers)
    ])
    self.f0_classifier = _make_linear(base_channels, 1, bias=True)
    self.m_source_linear = _make_linear(9, 1, bias=True)

    c1 = base_channels // 2
    c2 = base_channels // 4
    c3 = base_channels // 8
    self.conv_pre = nn.Conv1d(mel_bins, base_channels, kernel_size=7, padding=3)
    self.ups = nn.ModuleList([
        nn.ConvTranspose1d(
            base_channels, c1, kernel_size=16, stride=8, padding=4
        ),
        nn.ConvTranspose1d(c1, c2, kernel_size=11, stride=5, padding=3),
        nn.ConvTranspose1d(c2, c3, kernel_size=7, stride=3, padding=2),
    ])
    self.source_downs = nn.ModuleList([
        nn.Conv1d(18, c1, kernel_size=30, stride=15, padding=7),
        nn.Conv1d(18, c2, kernel_size=6, stride=3, padding=1),
        nn.Conv1d(18, c3, kernel_size=1, stride=1, padding=0),
    ])
    self.source_resblocks = nn.ModuleList([
        _HiFTResBlock(c1, kernel_size=7, dilations=(1, 3, 5)),
        _HiFTResBlock(c2, kernel_size=7, dilations=(1, 3, 5)),
        _HiFTResBlock(c3, kernel_size=11, dilations=(1, 3, 5)),
    ])
    self.resblocks = nn.ModuleList()
    for ch in (c1, c2, c3):
      for k in (3, 7, 11):
        self.resblocks.append(
            _HiFTResBlock(ch, kernel_size=k, dilations=(1, 3, 5))
        )
    self.conv_post = nn.Conv1d(c3, 18, kernel_size=7, padding=3)

    self.register_buffer(
        "harmonic_mult",
        torch.arange(1, 10, dtype=torch.float32).reshape(1, 9, 1),
        persistent=False,
    )
    gen = torch.Generator()
    gen.manual_seed(0)
    rand_ini = torch.rand(1, 9, generator=gen, dtype=torch.float32)
    rand_ini[:, 0] = 0.0
    self.register_buffer(
        "sine_rand_ini", rand_ini.unsqueeze(-1), persistent=False
    )
    lerp_w = (torch.arange(480, dtype=torch.float32) + 0.5) / 480.0
    self.register_buffer(
        "lerp_w", lerp_w.reshape(1, 1, 1, 480), persistent=False
    )
    audio_samples = mel_frames * 480
    sine_noise = torch.randn(
        1, 9, audio_samples, generator=gen, dtype=torch.float32
    )
    self.register_buffer("sine_noise", sine_noise, persistent=False)

    n_fft = 16
    hop = 4
    n_freqs = n_fft // 2 + 1
    n_idx = torch.arange(n_fft, dtype=torch.float32)
    window: torch.Tensor = 0.5 - 0.5 * torch.cos(
        2.0 * math.pi * n_idx / float(n_fft)
    )
    k_idx = torch.arange(n_freqs, dtype=torch.float32).unsqueeze(1)
    angles = 2.0 * math.pi * k_idx * n_idx.unsqueeze(0) / float(n_fft)
    cos_b = torch.cos(angles)
    sin_b = -torch.sin(angles)
    stft_basis = torch.cat([cos_b * window, sin_b * window], dim=0).unsqueeze(1)
    self.register_buffer("stft_basis", stft_basis, persistent=False)

    scale = torch.full((n_freqs, 1), 2.0 / float(n_fft), dtype=torch.float32)
    scale[0, 0] = 1.0 / float(n_fft)
    scale[-1, 0] = 1.0 / float(n_fft)
    istft_basis = torch.cat(
        [cos_b * scale * window, sin_b * scale * window], dim=0
    ).unsqueeze(1)
    self.register_buffer("istft_basis", istft_basis, persistent=False)

    n_stft_frames = mel_frames * 120 + 1
    win_sq = torch.pow(window, 2).reshape(1, 1, n_fft)
    ones = torch.ones(1, 1, n_stft_frames, dtype=torch.float32)
    env = F.conv_transpose1d(ones, win_sq, stride=hop)[:, :, 8:-8]
    inv_env = 1.0 / torch.clamp(env, min=1e-8)
    self.register_buffer("inv_env", inv_env, persistent=False)

  def forward(self, mel: torch.Tensor) -> torch.Tensor:
    b, _, t_mel = mel.shape
    t_audio = t_mel * 480
    h_f0 = mel
    for conv in self.f0_convs:
      h_f0 = F.elu(conv(h_f0))
    f0 = torch.abs(self.f0_classifier(h_f0.permute(0, 2, 1)).squeeze(-1))

    f0_up = f0.unsqueeze(-1).expand(b, t_mel, 480).reshape(b, 1, t_audio)
    harm = self.harmonic_mult.reshape(1, 1, 9).to(
        dtype=mel.dtype, device=mel.device
    )
    rad_raw = f0.unsqueeze(-1) * (harm / 24000.0)
    rad = rad_raw - torch.floor(rad_raw)
    phase_f = torch.cumsum(rad, dim=1).permute(0, 2, 1) * (
        2.0 * math.pi * 480.0
    ) + self.sine_rand_ini.to(dtype=mel.dtype, device=mel.device) * (
        2.0 * math.pi
    )
    lerp_w = self.lerp_w.to(dtype=mel.dtype, device=mel.device)
    head = (
        phase_f[:, :, :1].unsqueeze(-1).expand(b, 9, 1, 240).reshape(b, 9, 240)
    )
    mid = (
        phase_f[:, :, :-1].unsqueeze(-1) * (1.0 - lerp_w)
        + phase_f[:, :, 1:].unsqueeze(-1) * lerp_w
    ).reshape(b, 9, (t_mel - 1) * 480)
    tail = (
        phase_f[:, :, -1:].unsqueeze(-1).expand(b, 9, 1, 240).reshape(b, 9, 240)
    )
    phase = torch.cat([head, mid, tail], dim=2)
    sines = torch.sin(phase) * 0.1
    uv = (f0_up > 10.0).to(mel.dtype)
    noise_amp = uv * 0.003 + (1.0 - uv) * (0.1 / 3.0)
    noise = noise_amp * self.sine_noise[:, :, :t_audio].to(
        dtype=mel.dtype, device=mel.device
    )
    sine_wavs = sines * uv + noise
    s = torch.tanh(self.m_source_linear(sine_wavs.permute(0, 2, 1))).permute(
        0, 2, 1
    )

    s_pad = torch.cat(
        [
            torch.flip(s[:, :, 1:9], dims=[2]),
            s,
            torch.flip(s[:, :, -9:-1], dims=[2]),
        ],
        dim=2,
    )
    s_stft = F.conv1d(
        s_pad,
        self.stft_basis.to(dtype=mel.dtype, device=mel.device),
        stride=4,
    )

    x = self.conv_pre(mel)
    for i, (up, src_down, src_rb) in enumerate(
        zip(self.ups, self.source_downs, self.source_resblocks, strict=True)
    ):
      x = F.leaky_relu(x, 0.1)
      x = up(x)
      if i == 2:
        x = torch.cat([x[:, :, 1:2], x], dim=2)
      si = src_rb(src_down(s_stft))
      x = x + si
      rb0 = self.resblocks[i * 3](x)
      rb1 = self.resblocks[i * 3 + 1](x)
      rb2 = self.resblocks[i * 3 + 2](x)
      x = (rb0 + rb1 + rb2) * (1.0 / 3.0)

    x = self.conv_post(F.leaky_relu(x))
    mag = torch.clamp(torch.exp(x[:, :9, :]), max=100.0)
    sin_p = torch.sin(x[:, 9:, :])
    real = mag * torch.cos(sin_p)
    imag = mag * torch.sin(sin_p)
    spec = torch.cat([real, imag], dim=1)
    y = F.conv_transpose1d(
        spec,
        self.istft_basis.to(dtype=mel.dtype, device=mel.device),
        stride=4,
    )[:, :, 8:-8]
    y = y * self.inv_env[:, :, :t_audio].to(dtype=mel.dtype, device=mel.device)
    return torch.clamp(y.squeeze(1), min=-0.99, max=0.99)


class MiniCpmOAudioVocoderModule(nn.Module):
  """CosyVoice2 Token2Wav (`FlowEncode` + `FlowDit` + `HiFT`) 24 kHz vocoder.

  Input:
    audio_codes: [1, input_tokens] int32 discrete speech tokens in `[0, 6560]`
      (default `32` tokens = `4` history + `25` chunk + `3` lookahead).
  Output:
    dict with `"waveform"`: [1, (input_tokens - pre_lookahead_len) * 960]
    float32 PCM waveform in `[-0.99, 0.99]` at 24 kHz (`27840` samples for 32
    input tokens, from which the runtime slices the middle `25 * 960 = 24000`
    samples).
  """

  prompt_tokens: torch.Tensor

  def __init__(
      self,
      vocab_size: int = 6561,
      input_tokens: int = 32,
      pre_lookahead_len: int = 3,
      flow_dim: int = 512,
      flow_heads: int = 8,
      flow_ffn_dim: int = 2048,
      flow_num_encoders: int = 6,
      flow_num_up_encoders: int = 4,
      flow_num_dit_blocks: int = 16,
      mel_bins: int = 80,
      spk_dim: int = 192,
      n_timesteps: int = 6,
      hift_base_channels: int = 512,
      hift_f0_conv_layers: int = 5,
      prompt_tokens_len: int | None = None,
      cfg_rate: float | None = None,
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.input_tokens = input_tokens
    self.pre_lookahead_len = pre_lookahead_len
    if prompt_tokens_len is None:
      prompt_tokens_len = (
          0
          if (input_tokens < 32 or flow_num_dit_blocks < 4)
          else len(DEFAULT_VOICE_PROMPT_TOKENS)
      )
    self.prompt_tokens_len = max(0, prompt_tokens_len)
    if cfg_rate is None:
      cfg_rate = 0.7 if self.prompt_tokens_len > 0 else 0.0
    mel_frames = max(2, (input_tokens - pre_lookahead_len) * 2)
    prompt_mel_frames = self.prompt_tokens_len * 2
    p_tok_list = list(DEFAULT_VOICE_PROMPT_TOKENS)
    while len(p_tok_list) < max(1, self.prompt_tokens_len):
      p_tok_list.extend(DEFAULT_VOICE_PROMPT_TOKENS)
    p_tok = torch.tensor(
        [p_tok_list[: max(1, self.prompt_tokens_len)]], dtype=torch.int32
    )
    p_tok = torch.clamp(p_tok, min=0, max=max(0, vocab_size - 1))
    self.register_buffer("prompt_tokens", p_tok, persistent=False)

    self.flow_encode = _FlowEncode(
        vocab_size=vocab_size,
        flow_dim=flow_dim,
        flow_heads=flow_heads,
        flow_ffn_dim=flow_ffn_dim,
        num_encoders=flow_num_encoders,
        num_up_encoders=flow_num_up_encoders,
        mel_bins=mel_bins,
        pre_lookahead_len=pre_lookahead_len,
    )
    self.flow_dit = _FlowDit(
        flow_dim=flow_dim,
        flow_heads=flow_heads,
        flow_ffn_dim=flow_ffn_dim,
        num_dit_blocks=flow_num_dit_blocks,
        mel_bins=mel_bins,
        spk_dim=spk_dim,
        mel_frames=mel_frames,
        prompt_mel_frames=prompt_mel_frames,
        n_timesteps=n_timesteps,
        cfg_rate=cfg_rate,
    )
    self.hift = _HiFTGenerator(
        mel_bins=mel_bins,
        base_channels=hift_base_channels,
        f0_conv_layers=hift_f0_conv_layers,
        mel_frames=mel_frames,
    )

  def forward(self, audio_codes: torch.Tensor) -> dict[str, torch.Tensor]:
    safe_codes = torch.clamp(audio_codes, min=0, max=self.vocab_size - 1)
    if self.prompt_tokens_len > 0:
      p_tok = torch.clamp(
          self.prompt_tokens[:, : self.prompt_tokens_len].to(
              dtype=safe_codes.dtype, device=safe_codes.device
          ),
          min=0,
          max=self.vocab_size - 1,
      ).expand(safe_codes.shape[0], -1)
      full_codes = torch.cat([p_tok, safe_codes], dim=1)
    else:
      full_codes = safe_codes
    mu = self.flow_encode(full_codes)
    mel = self.flow_dit(mu)
    wav = self.hift(mel)
    return {"waveform": wav}


_CONFORMER_BLOCK_PARAM_MAP = {
    "pos_bias_u": "self_attn.pos_bias_u",
    "pos_bias_v": "self_attn.pos_bias_v",
    "linear_q.weight": "self_attn.linear_q.weight",
    "linear_q.bias": "self_attn.linear_q.bias",
    "linear_k.weight": "self_attn.linear_k.weight",
    "linear_k.bias": "self_attn.linear_k.bias",
    "linear_v.weight": "self_attn.linear_v.weight",
    "linear_v.bias": "self_attn.linear_v.bias",
    "linear_out.weight": "self_attn.linear_out.weight",
    "linear_out.bias": "self_attn.linear_out.bias",
    "linear_pos.weight": "self_attn.linear_pos.weight",
    "ff_w1.weight": "feed_forward.w_1.weight",
    "ff_w1.bias": "feed_forward.w_1.bias",
    "ff_w2.weight": "feed_forward.w_2.weight",
    "ff_w2.bias": "feed_forward.w_2.bias",
}

_DIT_BLOCK_PARAM_MAP = {
    "adaLN_linear.weight": "adaLN_modulation.1.weight",
    "adaLN_linear.bias": "adaLN_modulation.1.bias",
    "attn_to_q.weight": "attn.to_q.weight",
    "attn_to_q.bias": "attn.to_q.bias",
    "attn_to_k.weight": "attn.to_k.weight",
    "attn_to_k.bias": "attn.to_k.bias",
    "attn_to_v.weight": "attn.to_v.weight",
    "attn_to_v.bias": "attn.to_v.bias",
    "attn_q_norm.weight": "attn.q_norm.weight",
    "attn_q_norm.bias": "attn.q_norm.bias",
    "attn_k_norm.weight": "attn.k_norm.weight",
    "attn_k_norm.bias": "attn.k_norm.bias",
    "attn_proj.weight": "attn.proj.weight",
    "attn_proj.bias": "attn.proj.bias",
    "conv_1.weight": "conv.block.1.weight",
    "conv_1.bias": "conv.block.1.bias",
    "conv_norm.weight": "conv.block.3.weight",
    "conv_norm.bias": "conv.block.3.bias",
    "conv_6.weight": "conv.block.6.weight",
    "conv_6.bias": "conv.block.6.bias",
    "mlp_fc1.weight": "mlp.fc1.weight",
    "mlp_fc1.bias": "mlp.fc1.bias",
    "mlp_fc2.weight": "mlp.fc2.weight",
    "mlp_fc2.bias": "mlp.fc2.bias",
}


def _copy_hift_resblock(
    rb: Any,
    hift_sd: dict[str, torch.Tensor],
    prefix: str,
    strict: bool,
    populated: set[int],
) -> None:
  """Copies weights for a single `_HiFTResBlock` from `hift_sd`."""
  for j, (act1, conv1, act2, conv2) in enumerate(
      zip(
          rb.activations1,
          rb.convs1,
          rb.activations2,
          rb.convs2,
          strict=True,
      )
  ):
    weights_utils.copy_from_dict(
        act1.alpha,
        hift_sd,
        f"{prefix}.activations1.{j}.alpha",
        strict,
        populated,
    )
    weights_utils.copy_wn_from_dict(
        conv1, hift_sd, f"{prefix}.convs1.{j}", strict, populated
    )
    weights_utils.copy_from_dict(
        act2.alpha,
        hift_sd,
        f"{prefix}.activations2.{j}.alpha",
        strict,
        populated,
    )
    weights_utils.copy_wn_from_dict(
        conv2, hift_sd, f"{prefix}.convs2.{j}", strict, populated
    )


def load_token2wav_weights(
    mod: MiniCpmOAudioVocoderModule,
    flow_sd: dict[str, torch.Tensor],
    hift_sd: dict[str, torch.Tensor],
    strict: bool,
    populated: set[int],
) -> None:
  """Copies CosyVoice2 `flow.pt` and `hift.pt` weights into `mod`.

  Args:
    mod: Target `MiniCpmOAudioVocoderModule`.
    flow_sd: State dict from `assets/token2wav/flow.pt` (or `vocoder.flow.*`).
    hift_sd: State dict from `assets/token2wav/hift.pt` (or `vocoder.hift.*`).
    strict: Whether missing or shape-mismatched tensors raise errors.
    populated: Set of `data_ptr()` integers tracking populated parameters.
  """
  fe = mod.flow_encode
  weights_utils.copy_from_dict(
      fe.input_embedding.weight,
      flow_sd,
      "input_embedding.weight",
      strict,
      populated,
  )
  weights_utils.copy_linear_from_dict(
      fe.embed_linear, flow_sd, "encoder.embed.out.0", strict, populated
  )
  weights_utils.copy_linear_from_dict(
      fe.embed_norm, flow_sd, "encoder.embed.out.1", strict, populated
  )
  weights_utils.copy_linear_from_dict(
      fe.pre_lookahead_conv1,
      flow_sd,
      "encoder.pre_lookahead_layer.conv1",
      strict,
      populated,
  )
  weights_utils.copy_linear_from_dict(
      fe.pre_lookahead_conv2,
      flow_sd,
      "encoder.pre_lookahead_layer.conv2",
      strict,
      populated,
  )

  for idx, enc in enumerate(fe.encoders):
    weights_utils.copy_module_parameters(
        enc,
        flow_sd,
        {},
        f"encoder.encoders.{idx}",
        param_map=_CONFORMER_BLOCK_PARAM_MAP,
        strict=strict,
        populated=populated,
    )
  weights_utils.copy_linear_from_dict(
      fe.up_conv, flow_sd, "encoder.up_layer.conv", strict, populated
  )
  weights_utils.copy_linear_from_dict(
      fe.up_embed_linear, flow_sd, "encoder.up_embed.out.0", strict, populated
  )
  weights_utils.copy_linear_from_dict(
      fe.up_embed_norm, flow_sd, "encoder.up_embed.out.1", strict, populated
  )
  for idx, enc in enumerate(fe.up_encoders):
    weights_utils.copy_module_parameters(
        enc,
        flow_sd,
        {},
        f"encoder.up_encoders.{idx}",
        param_map=_CONFORMER_BLOCK_PARAM_MAP,
        strict=strict,
        populated=populated,
    )
  weights_utils.copy_linear_from_dict(
      fe.after_norm, flow_sd, "encoder.after_norm", strict, populated
  )
  weights_utils.copy_linear_from_dict(
      fe.encoder_proj, flow_sd, "encoder_proj", strict, populated
  )

  fd = mod.flow_dit
  weights_utils.copy_linear_from_dict(
      fd.spk_embed_affine_layer,
      flow_sd,
      "spk_embed_affine_layer",
      strict,
      populated,
  )
  weights_utils.copy_linear_from_dict(
      fd.t_mlp_0,
      flow_sd,
      "decoder.estimator.t_embedder.mlp.0",
      strict,
      populated,
  )
  weights_utils.copy_linear_from_dict(
      fd.t_mlp_2,
      flow_sd,
      "decoder.estimator.t_embedder.mlp.2",
      strict,
      populated,
  )
  weights_utils.copy_linear_from_dict(
      fd.in_proj, flow_sd, "decoder.estimator.in_proj", strict, populated
  )
  for idx, blk in enumerate(fd.blocks):
    weights_utils.copy_module_parameters(
        blk,
        flow_sd,
        {},
        f"decoder.estimator.blocks.{idx}",
        param_map=_DIT_BLOCK_PARAM_MAP,
        strict=strict,
        populated=populated,
    )
  weights_utils.copy_linear_from_dict(
      fd.final_adaLN_linear,
      flow_sd,
      "decoder.estimator.final_layer.adaLN_modulation.1",
      strict,
      populated,
  )
  weights_utils.copy_linear_from_dict(
      fd.final_linear,
      flow_sd,
      "decoder.estimator.final_layer.linear",
      strict,
      populated,
  )

  hf = mod.hift
  weights_utils.copy_wn_from_dict(
      hf.conv_pre, hift_sd, "conv_pre", strict, populated
  )
  for idx, up in enumerate(hf.ups):
    weights_utils.copy_wn_from_dict(
        up, hift_sd, f"ups.{idx}", strict, populated
    )
  for idx, sd_conv in enumerate(hf.source_downs):
    weights_utils.copy_wn_from_dict(
        sd_conv, hift_sd, f"source_downs.{idx}", strict, populated
    )

  for idx, srb in enumerate(hf.source_resblocks):
    _copy_hift_resblock(
        srb, hift_sd, f"source_resblocks.{idx}", strict, populated
    )
  for idx, rb in enumerate(hf.resblocks):
    _copy_hift_resblock(rb, hift_sd, f"resblocks.{idx}", strict, populated)
  weights_utils.copy_wn_from_dict(
      hf.conv_post, hift_sd, "conv_post", strict, populated
  )
  for idx, f0_conv in enumerate(hf.f0_convs):
    weights_utils.copy_wn_from_dict(
        f0_conv,
        hift_sd,
        f"f0_predictor.condnet.{idx * 2}",
        strict,
        populated,
    )
  weights_utils.copy_linear_from_dict(
      hf.f0_classifier,
      hift_sd,
      "f0_predictor.classifier",
      strict,
      populated,
  )
  weights_utils.copy_linear_from_dict(
      hf.m_source_linear, hift_sd, "m_source.l_linear", strict, populated
  )
