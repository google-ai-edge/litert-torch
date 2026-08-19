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
"""ECAPA-TDNN speaker encoder architecture for Qwen3-TTS voice profile enrollment."""

import torch
from torch import nn
import torch.nn.functional as F


class SpeakerEncoderConfig:
  """Configuration for Qwen3-TTS ECAPA-TDNN speaker encoder."""

  def __init__(
      self,
      mel_dim: int = 128,
      enc_dim: int = 1024,
      enc_channels: list[int] | None = None,
      enc_kernel_sizes: list[int] | None = None,
      enc_dilations: list[int] | None = None,
      enc_res2net_scale: int = 8,
      enc_se_channels: int = 128,
      enc_attention_channels: int = 128,
      sample_rate: int = 24000,
  ):
    self.mel_dim = mel_dim
    self.enc_dim = enc_dim
    self.enc_channels = (
        enc_channels if enc_channels is not None else [512, 512, 512, 512, 1536]
    )
    self.enc_kernel_sizes = (
        enc_kernel_sizes if enc_kernel_sizes is not None else [5, 3, 3, 3, 1]
    )
    self.enc_dilations = (
        enc_dilations if enc_dilations is not None else [1, 2, 3, 4, 1]
    )
    self.enc_res2net_scale = enc_res2net_scale
    self.enc_se_channels = enc_se_channels
    self.enc_attention_channels = enc_attention_channels
    self.sample_rate = sample_rate


class TimeDelayNetBlock(nn.Module):
  """TDNN 1D convolution layer with reflection padding and ReLU."""

  def __init__(
      self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
  ):
    super().__init__()
    self.conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        padding="same",
        padding_mode="reflect",
    )
    self.activation = nn.ReLU()

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.activation(self.conv(hidden_states))


class Res2NetBlock(nn.Module):
  """Multi-scale Res2Net feature extraction block."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      scale: int = 8,
      kernel_size: int = 3,
      dilation: int = 1,
  ):
    super().__init__()
    in_channel = in_channels // scale
    hidden_channel = out_channels // scale
    self.blocks = nn.ModuleList([
        TimeDelayNetBlock(in_channel, hidden_channel, kernel_size, dilation)
        for _ in range(scale - 1)
    ])
    self.scale = scale

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    outputs = []
    output_part: torch.Tensor = hidden_states
    for i, hidden_part in enumerate(
        torch.chunk(hidden_states, self.scale, dim=1)
    ):
      if i == 0:
        output_part = hidden_part
      elif i == 1:
        output_part = self.blocks[i - 1](hidden_part)
      else:
        output_part = self.blocks[i - 1](hidden_part + output_part)
      outputs.append(output_part)
    return torch.cat(outputs, dim=1)


class SqueezeExcitationBlock(nn.Module):
  """Channel attentive Squeeze-and-Excitation block."""

  def __init__(self, in_channels: int, se_channels: int, out_channels: int):
    super().__init__()
    self.conv1 = nn.Conv1d(
        in_channels,
        se_channels,
        kernel_size=1,
        padding="same",
        padding_mode="reflect",
    )
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv1d(
        se_channels,
        out_channels,
        kernel_size=1,
        padding="same",
        padding_mode="reflect",
    )
    self.sigmoid = nn.Sigmoid()

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    mean = hidden_states.mean(dim=2, keepdim=True)
    mean = self.relu(self.conv1(mean))
    mean = self.sigmoid(self.conv2(mean))
    return hidden_states * mean


class SqueezeExcitationRes2NetBlock(nn.Module):
  """ECAPA-TDNN core building block: TDNN-Res2Net-TDNN-SE."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      res2net_scale: int = 8,
      se_channels: int = 128,
      kernel_size: int = 1,
      dilation: int = 1,
  ):
    super().__init__()
    self.out_channels = out_channels
    self.tdnn1 = TimeDelayNetBlock(
        in_channels, out_channels, kernel_size=1, dilation=1
    )
    self.res2net_block = Res2NetBlock(
        out_channels, out_channels, res2net_scale, kernel_size, dilation
    )
    self.tdnn2 = TimeDelayNetBlock(
        out_channels, out_channels, kernel_size=1, dilation=1
    )
    self.se_block = SqueezeExcitationBlock(
        out_channels, se_channels, out_channels
    )

  def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
    residual = hidden_state
    hidden_state = self.tdnn1(hidden_state)
    hidden_state = self.res2net_block(hidden_state)
    hidden_state = self.tdnn2(hidden_state)
    hidden_state = self.se_block(hidden_state)
    return hidden_state + residual


class AttentiveStatisticsPooling(nn.Module):
  """Attentive statistic pooling layer returning concatenated mean and standard deviation."""

  def __init__(self, channels: int, attention_channels: int = 128):
    super().__init__()
    self.eps = 1e-12
    self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
    self.tanh = nn.Tanh()
    self.conv = nn.Conv1d(
        attention_channels,
        channels,
        kernel_size=1,
        padding="same",
        padding_mode="reflect",
    )

  def _compute_statistics(
      self, x: torch.Tensor, m: torch.Tensor, dim: int = 2
  ) -> tuple[torch.Tensor, torch.Tensor]:
    mean = (m * x).sum(dim)
    std = torch.sqrt(
        (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps)
    )
    return mean, std

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    seq_length = hidden_states.shape[-1]
    mask = torch.ones(
        (hidden_states.shape[0], 1, seq_length),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    total = mask.sum(dim=2, keepdim=True)
    mean, std = self._compute_statistics(hidden_states, mask / total)
    mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
    std = std.unsqueeze(2).repeat(1, 1, seq_length)
    attention = torch.cat([hidden_states, mean, std], dim=1)
    attention = self.conv(self.tanh(self.tdnn(attention)))
    attention = F.softmax(attention, dim=2)
    mean, std = self._compute_statistics(hidden_states, attention)
    pooled_stats = torch.cat((mean, std), dim=1).unsqueeze(2)
    return pooled_stats


class Qwen3TTSSpeakerEncoder(nn.Module):
  """ECAPA-TDNN speaker verification network for generating 1024-d speaker embeddings."""

  def __init__(self, config: SpeakerEncoderConfig | None = None):
    super().__init__()
    if config is None:
      config = SpeakerEncoderConfig()
    self.channels = config.enc_channels
    self.blocks = nn.ModuleList()
    self.blocks.append(
        TimeDelayNetBlock(
            config.mel_dim,
            config.enc_channels[0],
            config.enc_kernel_sizes[0],
            config.enc_dilations[0],
        )
    )
    for i in range(1, len(config.enc_channels) - 1):
      self.blocks.append(
          SqueezeExcitationRes2NetBlock(
              config.enc_channels[i - 1],
              config.enc_channels[i],
              res2net_scale=config.enc_res2net_scale,
              se_channels=config.enc_se_channels,
              kernel_size=config.enc_kernel_sizes[i],
              dilation=config.enc_dilations[i],
          )
      )
    self.mfa = TimeDelayNetBlock(
        config.enc_channels[-1],
        config.enc_channels[-1],
        config.enc_kernel_sizes[-1],
        config.enc_dilations[-1],
    )
    self.asp = AttentiveStatisticsPooling(
        config.enc_channels[-1],
        attention_channels=config.enc_attention_channels,
    )
    self.fc = nn.Conv1d(
        in_channels=config.enc_channels[-1] * 2,
        out_channels=config.enc_dim,
        kernel_size=1,
        padding="same",
        padding_mode="reflect",
    )

  def forward(self, mels: torch.Tensor) -> torch.Tensor:
    """Computes 1024-d speaker embedding from input mels [B, T, 128]."""
    x = mels.transpose(1, 2)  # [B, 128, T]
    hidden_states_list = []
    for layer in self.blocks:
      x = layer(x)
      hidden_states_list.append(x)
    x = torch.cat(hidden_states_list[1:], dim=1)
    x = self.mfa(x)
    x = self.asp(x)
    x = self.fc(x)
    return x.squeeze(-1)  # [B, 1024]
