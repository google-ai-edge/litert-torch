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
"""Token2Wav (DiT flow-matching ODE + BigVGAN) wrapper for Qwen2.5-Omni."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as hf_qwen25


class QwenOmniVocoderWrapper(nn.Module):
  """End-to-end Token2Wav (DiT flow-matching ODE + BigVGAN) wrapper.

  Upstream HF `Qwen2_5OmniToken2WavModel.forward()` cannot be exported directly:
    - `dit.sample()` draws dynamic `torch.randn` noise (unsupported in TFLite)
      and runs `RungeKutta4ODESolver.integrate()` with a data-dependent `while`
      loop over tensor time points and in-place scatter into `torch.empty`.
    - `dit.forward()` passes `int64` `block_diff` into `DiTDecoderLayer`, where
      `block_diff >= -float(...)` promotes to `float64` (rejected by TFLite's
      `tfl.broadcast_to`), and redundantly recomputes step-invariant embeddings
      (including `ECAPA_TimeDelayNet` over the constant speaker `reference_mel`)
      on every RK4 velocity evaluation (`4 * (ode_steps - 1)` times).

  This wrapper reuses upstream HF `dit` submodules (`text_embed`, `time_embed`,
  `input_embed.proj`, `rotary_embed`, `_create_block_diff`,
  `transformer_blocks`, `norm_out`, `proj_out`) and calls upstream
  `code2wav_bigvgan_model` directly as-is (no padding shims or BigVGAN
  re-implementation).
  """

  initial_state: torch.Tensor
  conditioning: torch.Tensor
  spk_emb_cond: torch.Tensor
  spk_emb_uncond: torch.Tensor

  def __init__(
      self,
      token2wav: hf_qwen25.Qwen2_5OmniToken2WavModel,
      chunk_frames: int = 48,
      output_frames: int | None = None,
      ode_steps: int = 10,
      cfg_strength: float = 0.5,
      sway_coef: float = -1.0,
      ref_mel_frames: int | None = None,
      speaker_entry: dict[str, Any] | None = None,
  ):
    super().__init__()
    self.dit = token2wav.code2wav_dit_model
    self.bigvgan = token2wav.code2wav_bigvgan_model
    # `DiTAttention` does not support `"eager"` (expects boolean mask + SDPA).
    # pylint: disable=protected-access
    self.dit.config._attn_implementation = "sdpa"
    # pylint: enable=protected-access

    self.chunk_frames = chunk_frames
    self.output_frames = output_frames
    self.ode_steps = ode_steps
    self.cfg_strength = cfg_strength
    self.sway_coef = sway_coef
    self.repeats = int(self.dit.repeats)
    self.mel_dim = int(self.dit.mel_dim)
    self.upsample_factor = self.repeats * math.prod(
        int(r) for r in self.bigvgan.config.upsample_rates
    )

    # Seeded initial Gaussian noise matching HF `torch.manual_seed(0)` sampling.
    gen = torch.Generator().manual_seed(0)
    init_state = torch.randn(
        (1, chunk_frames * self.repeats, self.mel_dim),
        generator=gen,
        dtype=torch.float32,
    )
    self.register_buffer("initial_state", init_state, persistent=False)

    enc_emb_dim = int(self.dit.config.enc_emb_dim)
    cond = torch.zeros((1, enc_emb_dim), dtype=torch.float32)
    ref_len = ref_mel_frames or 400
    ref_mel = torch.zeros((1, ref_len, self.mel_dim), dtype=torch.float32)
    self.speaker_text_bos_token_id: int | None = None
    if speaker_entry is not None:
      if speaker_entry.get("cond") is not None:
        cond = speaker_entry["cond"].detach().to(torch.float32).clone()
      if speaker_entry.get("ref_mel") is not None:
        ref_mel = speaker_entry["ref_mel"].detach().to(torch.float32).clone()
      if speaker_entry.get("bos_token") is not None:
        self.speaker_text_bos_token_id = int(speaker_entry["bos_token"])
    self.register_buffer("conditioning", cond, persistent=False)

    # Fold `ECAPA_TimeDelayNet` (`spk_encoder`) once at export time: `ref_mel`
    # is a fixed speaker constant independent of `audio_codes`, and
    # `spk_encoder` is the only Token2Wav submodule using reflect-padded
    # `Conv1d`. Precomputing conditional + CFG-null speaker embeddings avoids
    # tracing reflect convs and drops ECAPA-TDNN weights/compute from `.tflite`.
    spk_encoder = self.dit.input_embed.spk_encoder
    spk_param = next(spk_encoder.parameters())
    ref_in = ref_mel.to(device=spk_param.device, dtype=spk_param.dtype)
    with torch.no_grad():
      spk_cond = spk_encoder(ref_in).detach().to(torch.float32)
      spk_uncond = (
          spk_encoder(torch.zeros_like(ref_in)).detach().to(torch.float32)
      )
    self.register_buffer("spk_emb_cond", spk_cond, persistent=False)
    self.register_buffer("spk_emb_uncond", spk_uncond, persistent=False)

  def _dit_sample(self, codes: torch.Tensor) -> torch.Tensor:
    """Runs static unrolled RK4 flow-matching ODE matching HF `dit.sample()`."""
    bsz, code_len = codes.shape
    mel_len = code_len * self.repeats
    y = self.initial_state.expand(bsz, -1, -1)
    apply_cfg = self.cfg_strength >= 1e-5
    cfg_bsz = bsz * 2 if apply_cfg else bsz

    # Hoist step-invariant embeddings outside the unrolled RK4 loop.
    code_emb = self.dit.text_embed(codes, drop_code=False)
    spk = self.spk_emb_cond.expand(bsz, -1)
    cond = self.conditioning.expand(bsz, -1)
    if apply_cfg:
      code_emb = torch.cat(
          [code_emb, self.dit.text_embed(codes, drop_code=True)], dim=0
      )
      spk = torch.cat([spk, self.spk_emb_uncond.expand(bsz, -1)], dim=0)
      cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
    static_cond = torch.cat(
        [
            spk.unsqueeze(1).repeat(1, mel_len, 1),
            code_emb,
            cond.unsqueeze(1).repeat(1, mel_len, 1),
        ],
        dim=-1,
    )

    pos_ids = (
        torch.arange(mel_len, device=codes.device)
        .unsqueeze(0)
        .repeat(cfg_bsz, 1)
    )
    pos_emb = self.dit.rotary_embed(static_cond, pos_ids)
    # Cast to float32 so `DiTDecoderLayer`'s `block_diff >= -float(...)` does
    # not promote int64 to float64 (unsupported by TFLite `tfl.broadcast_to`).
    # pylint: disable=protected-access
    block_diff = self.dit._create_block_diff(static_cond).to(torch.float32)
    # pylint: enable=protected-access

    def velocity(t_val: float, x: torch.Tensor) -> torch.Tensor:
      t = torch.full((cfg_bsz,), t_val, dtype=x.dtype, device=x.device)
      t_emb = self.dit.time_embed(t)
      x_in = torch.cat([x, x], dim=0) if apply_cfg else x
      h = self.dit.input_embed.proj(torch.cat([x_in, static_cond], dim=-1))
      for blk in self.dit.transformer_blocks:
        h = blk(h, t_emb, position_embeddings=pos_emb, block_diff=block_diff)
      out = self.dit.proj_out(self.dit.norm_out(h, t_emb))
      if not apply_cfg:
        return out
      guided, null = torch.chunk(out, 2, dim=0)
      return guided + (guided - null) * self.cfg_strength

    denom = float(max(self.ode_steps - 1, 1))
    ts = [
        u + self.sway_coef * (math.cos(math.pi / 2.0 * u) - 1.0 + u)
        for u in (i / denom for i in range(self.ode_steps))
    ]
    for t0, t1 in zip(ts[:-1], ts[1:]):
      dt = t1 - t0
      k1 = velocity(t0, y)
      k2 = velocity(t0 + dt / 3.0, y + k1 * (dt / 3.0))
      k3 = velocity(t0 + dt * (2.0 / 3.0), y + (k2 - k1 / 3.0) * dt)
      k4 = velocity(t1, y + (k1 - k2 + k3) * dt)
      y = y + (k1 + 3.0 * (k2 + k3) + k4) * (dt / 8.0)

    return y.permute(0, 2, 1)  # [B, mel_dim, mel_len]

  def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
    """Runs static DiT sampling followed by upstream HF BigVGAN.

    Custom `forward()` binds export-time speaker conditioning, runs the static
    `_dit_sample` ODE, restores `[B, output_samples]` after upstream BigVGAN's
    trailing `.squeeze().cpu()`, and trims lookahead frames when configured.

    Args:
      audio_codes: Codec tokens, `int32[1, chunk_frames]` (or `[1, 1, F]`).

    Returns:
      PCM waveform, `float32[1, output_samples]`.
    """
    codes = audio_codes[:, 0, :] if audio_codes.dim() == 3 else audio_codes
    long_codes = codes.to(torch.long)
    mel = self._dit_sample(long_codes)
    wav = self.bigvgan(mel).reshape(long_codes.shape[0], -1)
    if (
        self.output_frames is not None
        and self.output_frames < long_codes.shape[1]
    ):
      wav = wav[:, : self.output_frames * self.upsample_factor]
    return wav
