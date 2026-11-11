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
"""PyTorch architecture modules for LiquidAI LFM2.5-Audio-1.5B (`lfm2_audio`).

Implements the genuine `LiquidAI/LFM2.5-Audio-1.5B` components with exact
`state_dict` key parity against `model.safetensors` and
`audio_detokenizer/model.safetensors`:
- FastConformer audio encoder (`conformer.*`) + 2-layer MLP (`audio_adapter.*`)
- LFM2 text token embedder (`lfm.embed_tokens.weight`)
- Audio input/output projector (`depth_linear.*`, `audio_embedding.*`,
  `codebook_offsets`, `audio_loss_weights`)
- 8-codebook RQ-Transformer Depthformer (`depthformer.*`, `depth_embeddings.*`)
- LFM2-based neural audio detokenizer (`emb.emb.weight`, `lfm.*`, `lin.*`,
  `istft.window`)
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

# ==============================================================================
# 1. FastConformer Encoder + Audio Adapter (`conformer.*`, `audio_adapter.*`)
# ==============================================================================


@dataclasses.dataclass
class ConformerEncoderConfig:
  """Configuration for the LFM2.5-Audio FastConformer encoder."""

  feat_in: int = 128
  n_layers: int = 17
  d_model: int = 512
  feat_out: int = -1
  subsampling: str = 'dw_striding'
  subsampling_factor: int = 8
  subsampling_conv_channels: int = 256
  subsampling_conv_chunking_factor: int = 1
  ff_expansion_factor: int = 4
  self_attention_model: str = 'rel_pos'
  n_heads: int = 8
  att_context_size: Sequence[int] = (-1, -1)
  att_context_style: str = 'regular'
  xscaling: bool = False
  untie_biases: bool = True
  pos_emb_max_len: int = 5000
  conv_kernel_size: int = 9
  conv_norm_type: str = 'batch_norm'
  conv_context_size: Any = None
  dropout: float = 0.0
  dropout_pre_encoder: float = 0.0
  dropout_emb: float = 0.0
  dropout_att: float = 0.0

  @classmethod
  def from_dict(cls, cfg: dict[str, Any]) -> 'ConformerEncoderConfig':
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in cfg.items() if k in valid_fields}
    return cls(**filtered)


def _calc_dw_striding_length(
    length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    num_layers: int,
) -> int:
  """Computes output length after `num_layers` strided 2D convolutions."""
  out = length
  for _ in range(num_layers):
    out = (out + 2 * padding - kernel_size) // stride + 1
  return out


class ConvSubsampling(nn.Module):
  """Depthwise-striding 2D convolution subsampling (`conformer.pre_encode`)."""

  def __init__(
      self,
      subsampling: str,
      subsampling_factor: int,
      feat_in: int,
      feat_out: int,
      conv_channels: int,
  ):
    super().__init__()
    if subsampling != 'dw_striding':
      raise ValueError(f'Unsupported subsampling mode: {subsampling}')
    if subsampling_factor % 2 != 0:
      raise ValueError(
          f'subsampling_factor must be a power of 2, got {subsampling_factor}'
      )
    self._sampling_num = int(math.log(subsampling_factor, 2))
    self._stride = 2
    self._kernel_size = 3
    self._padding = (self._kernel_size - 1) // 2

    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
        ),
        nn.ReLU(inplace=True),
    ]
    for _ in range(self._sampling_num - 1):
      layers.extend([
          nn.Conv2d(
              in_channels=conv_channels,
              out_channels=conv_channels,
              kernel_size=self._kernel_size,
              stride=self._stride,
              padding=self._padding,
              groups=conv_channels,
          ),
          nn.Conv2d(
              in_channels=conv_channels,
              out_channels=conv_channels,
              kernel_size=1,
              stride=1,
              padding=0,
              groups=1,
          ),
          nn.ReLU(inplace=True),
      ])

    out_freq = _calc_dw_striding_length(
        feat_in,
        kernel_size=self._kernel_size,
        stride=self._stride,
        padding=self._padding,
        num_layers=self._sampling_num,
    )
    self.conv = nn.Sequential(*layers)
    self.out = nn.Linear(conv_channels * out_freq, feat_out)

  def forward(
      self, x: torch.Tensor, mel_length: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    # x: [B, T_mel, feat_in]
    x = x.unsqueeze(1)  # [B, 1, T_mel, feat_in]
    if mel_length is None:
      x = self.conv(x)  # [B, C, T_sub, F_sub]
      b, c, t, f = x.shape
      return self.out(x.transpose(1, 2).reshape(b, t, c * f)), None

    cur_len = mel_length.to(dtype=torch.int32).view(-1, 1)
    t_idx = torch.arange(x.shape[2], device=x.device, dtype=torch.int32).view(
        1, -1
    )
    mask = (t_idx < cur_len).to(dtype=x.dtype).view(-1, 1, x.shape[2], 1)
    for layer in self.conv:
      x = x * mask
      x = layer(x)
      if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
        pad_t = layer.padding[0]
        k_t = layer.kernel_size[0]
        s_t = layer.stride[0]
        cur_len = (
            torch.div(cur_len + 2 * pad_t - k_t, s_t, rounding_mode='floor') + 1
        )
        t_idx = torch.arange(
            x.shape[2], device=x.device, dtype=torch.int32
        ).view(1, -1)
        mask = (t_idx < cur_len).to(dtype=x.dtype).view(-1, 1, x.shape[2], 1)
    x = x * mask
    b, c, t, f = x.shape
    return self.out(x.transpose(1, 2).reshape(b, t, c * f)), cur_len.view(-1)


class RelPositionalEncoding(nn.Module):
  """Relative positional encoding matching NeMo / FastConformer."""

  def __init__(
      self,
      d_model: int,
      max_len: int = 5000,
      xscale: float | None = None,
  ):
    super().__init__()
    self.d_model = d_model
    self.xscale = xscale
    self.max_len = max_len
    self.register_buffer(
        'pe',
        self._build_pe(max_len, d_model, dtype=torch.float32),
        persistent=False,
    )

  @staticmethod
  def _build_pe(
      max_len: int, d_model: int, dtype: torch.dtype = torch.float32
  ) -> torch.Tensor:
    positions = torch.arange(
        max_len - 1, -max_len, -1, dtype=torch.float32
    ).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    pe = torch.zeros(2 * max_len - 1, d_model, dtype=dtype)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.unsqueeze(0)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if self.xscale is not None:
      x = x * self.xscale
    input_len = x.size(1)
    center_pos = self.pe.size(1) // 2 + 1
    start_pos = center_pos - input_len
    end_pos = center_pos + input_len - 1
    pos_emb = self.pe[:, start_pos:end_pos].to(dtype=x.dtype, device=x.device)
    return x, pos_emb


class RelPositionMultiHeadAttention(nn.Module):
  """Relative-position multi-head self-attention (`conformer.layers.*.self_attn`)."""

  def __init__(self, n_head: int, n_feat: int):
    super().__init__()
    if n_feat % n_head != 0:
      raise ValueError(
          f'n_feat ({n_feat}) must be divisible by n_head ({n_head})'
      )
    self.d_k = n_feat // n_head
    self.s_d_k = math.sqrt(self.d_k)
    self.h = n_head
    self.linear_q = nn.Linear(n_feat, n_feat)
    self.linear_k = nn.Linear(n_feat, n_feat)
    self.linear_v = nn.Linear(n_feat, n_feat)
    self.linear_out = nn.Linear(n_feat, n_feat)
    self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
    self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
    self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))

  def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
    b, h, qlen, pos_len = x.size()
    x = F.pad(x, pad=(1, 0))
    x = x.view(b, h, -1, qlen)
    return x[:, :, 1:].view(b, h, qlen, pos_len)

  def forward(
      self,
      query: torch.Tensor,
      key: torch.Tensor,
      value: torch.Tensor,
      pos_emb: torch.Tensor,
      att_valid_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    n_batch = query.size(0)
    q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
    k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
    v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)

    n_batch_pos = pos_emb.size(0)
    p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)

    q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
    q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

    matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
    matrix_bd = torch.matmul(q_with_bias_v, p.permute(0, 2, 3, 1))
    matrix_bd = self._rel_shift(matrix_bd)[:, :, :, : matrix_ac.size(-1)]

    scores = (matrix_ac + matrix_bd) / self.s_d_k
    if att_valid_mask is not None:
      scores = torch.where(
          att_valid_mask, scores, torch.full_like(scores, -10000.0)
      )
      attn = torch.softmax(scores, dim=-1) * att_valid_mask.to(
          dtype=scores.dtype
      )
    else:
      attn = torch.softmax(scores, dim=-1)
    x = (
        torch.matmul(attn, v)
        .transpose(1, 2)
        .reshape(n_batch, -1, self.h * self.d_k)
    )
    return self.linear_out(x)


class ConformerFeedForward(nn.Module):
  """Macaron feed-forward module (`conformer.layers.*.feed_forward{1,2}`)."""

  def __init__(self, d_model: int, d_ff: int):
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.activation = nn.SiLU()
    self.linear2 = nn.Linear(d_ff, d_model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear2(self.activation(self.linear1(x)))


class ConformerConvolution(nn.Module):
  """Conformer depthwise-separable convolution module (`conformer.layers.*.conv`)."""

  def __init__(self, d_model: int, kernel_size: int):
    super().__init__()
    if (kernel_size - 1) % 2 != 0:
      raise ValueError(f'kernel_size must be odd, got {kernel_size}')
    self.pointwise_conv1 = nn.Conv1d(
        in_channels=d_model,
        out_channels=d_model * 2,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )
    self.depthwise_conv = nn.Conv1d(
        in_channels=d_model,
        out_channels=d_model,
        kernel_size=kernel_size,
        stride=1,
        padding=(kernel_size - 1) // 2,
        groups=d_model,
        bias=True,
    )
    self.batch_norm = nn.BatchNorm1d(d_model)
    self.activation = nn.SiLU()
    self.pointwise_conv2 = nn.Conv1d(
        in_channels=d_model,
        out_channels=d_model,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )

  def forward(
      self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
  ) -> torch.Tensor:
    # x: [B, T, D]
    x = x.transpose(1, 2)  # [B, D, T]
    x = self.pointwise_conv1(x)  # [B, 2*D, T]
    x = F.glu(x, dim=1)  # [B, D, T]
    if valid_mask is not None:
      x = x * valid_mask
    x = self.depthwise_conv(x)  # [B, D, T]
    x = self.batch_norm(x)
    x = self.activation(x)
    x = self.pointwise_conv2(x)
    return x.transpose(1, 2)


class ConformerLayer(nn.Module):
  """Single FastConformer block (`conformer.layers.*`)."""

  def __init__(
      self,
      d_model: int,
      d_ff: int,
      n_heads: int,
      conv_kernel_size: int,
  ):
    super().__init__()
    self.fc_factor = 0.5
    self.norm_feed_forward1 = nn.LayerNorm(d_model)
    self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff)
    self.norm_self_att = nn.LayerNorm(d_model)
    self.self_attn = RelPositionMultiHeadAttention(
        n_head=n_heads, n_feat=d_model
    )
    self.norm_conv = nn.LayerNorm(d_model)
    self.conv = ConformerConvolution(
        d_model=d_model, kernel_size=conv_kernel_size
    )
    self.norm_feed_forward2 = nn.LayerNorm(d_model)
    self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff)
    self.norm_out = nn.LayerNorm(d_model)

  def forward(
      self,
      x: torch.Tensor,
      pos_emb: torch.Tensor,
      valid_mask: torch.Tensor | None = None,
      att_valid_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    residual = (
        x + self.feed_forward1(self.norm_feed_forward1(x)) * self.fc_factor
    )
    x_norm = self.norm_self_att(residual)
    residual = residual + self.self_attn(
        x_norm,
        x_norm,
        x_norm,
        pos_emb=pos_emb,
        att_valid_mask=att_valid_mask,
    )
    residual = residual + self.conv(
        self.norm_conv(residual), valid_mask=valid_mask
    )
    residual = (
        residual
        + self.feed_forward2(self.norm_feed_forward2(residual)) * self.fc_factor
    )
    return self.norm_out(residual)


class ConformerEncoder(nn.Module):
  """FastConformer encoder matching `liquid_audio` (`conformer.*`)."""

  def __init__(self, cfg: ConformerEncoderConfig):
    super().__init__()
    self.cfg = cfg
    d_ff = cfg.d_model * cfg.ff_expansion_factor
    self.pre_encode = ConvSubsampling(
        subsampling=cfg.subsampling,
        subsampling_factor=cfg.subsampling_factor,
        feat_in=cfg.feat_in,
        feat_out=cfg.d_model,
        conv_channels=cfg.subsampling_conv_channels,
    )
    xscale = math.sqrt(cfg.d_model) if cfg.xscaling else None
    self.pos_enc = RelPositionalEncoding(
        d_model=cfg.d_model,
        max_len=cfg.pos_emb_max_len,
        xscale=xscale,
    )
    self.layers = nn.ModuleList([
        ConformerLayer(
            d_model=cfg.d_model,
            d_ff=d_ff,
            n_heads=cfg.n_heads,
            conv_kernel_size=cfg.conv_kernel_size,
        )
        for _ in range(cfg.n_layers)
    ])

  def forward(
      self,
      audio_signal: torch.Tensor,
      mel_length: torch.Tensor | None = None,
  ) -> torch.Tensor:
    # audio_signal: [B, feat_in, T_mel]
    x = audio_signal.transpose(1, 2)  # [B, T_mel, feat_in]
    x, sub_len = self.pre_encode(
        x, mel_length=mel_length
    )  # [B, ceil(T_mel / 8), d_model]
    x, pos_emb = self.pos_enc(x)
    valid_mask: torch.Tensor | None = None
    att_valid_mask: torch.Tensor | None = None
    if sub_len is not None:
      max_len = x.size(1)
      t_idx = torch.arange(max_len, device=x.device, dtype=torch.int32).view(
          1, -1
      )
      valid_bool = t_idx < sub_len.view(-1, 1)  # [B, T_sub]
      valid_mask = valid_bool.to(dtype=x.dtype).unsqueeze(1)  # [B, 1, T_sub]
      att_valid_mask = (
          valid_bool.unsqueeze(1) & valid_bool.unsqueeze(2)
      ).unsqueeze(1)
    for layer in self.layers:
      x = layer(
          x,
          pos_emb=pos_emb,
          valid_mask=valid_mask,
          att_valid_mask=att_valid_mask,
      )
    return x


class AudioAdapterMLP(nn.Module):
  """2-layer MLP adapter (`audio_adapter.*`) mapping Conformer to LFM2 dim."""

  def __init__(
      self,
      in_dim: int = 512,
      out_dim: int = 2048,
      hidden_dims: Sequence[int] = (2048,),
      use_layer_norm: bool = True,
  ):
    super().__init__()
    layers: list[nn.Module] = []
    if use_layer_norm:
      layers.append(nn.LayerNorm(in_dim))
    dims = [in_dim, *hidden_dims]
    for d0, d1 in zip(dims[:-1], dims[1:]):
      layers.append(nn.Linear(d0, d1))
      layers.append(nn.GELU())
    layers.append(nn.Linear(dims[-1], out_dim))
    self.model = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x)


class Lfm2AudioEncoderModule(nn.Module):
  """Combined FastConformer + MLP audio encoder (`conformer.*`, `audio_adapter.*`)."""

  def __init__(
      self,
      encoder_cfg: ConformerEncoderConfig,
      llm_hidden_size: int = 2048,
      adapter_hidden_dims: Sequence[int] = (2048,),
      adapter_use_layer_norm: bool = True,
  ):
    super().__init__()
    self.conformer = ConformerEncoder(encoder_cfg)
    self.audio_adapter = AudioAdapterMLP(
        in_dim=encoder_cfg.d_model,
        out_dim=llm_hidden_size,
        hidden_dims=adapter_hidden_dims,
        use_layer_norm=adapter_use_layer_norm,
    )

  def forward(
      self,
      mel_features: torch.Tensor,
      mel_length: torch.Tensor | None = None,
  ) -> torch.Tensor:
    # mel_features: [B, 128, T_mel] -> [B, ceil(T_mel / 8), llm_hidden_size]
    encoded = self.conformer(mel_features, mel_length=mel_length)
    return self.audio_adapter(encoded)


# ==============================================================================
# 2. Text Embedder (`lfm.embed_tokens.weight`) & Shared Embeddings
# ==============================================================================


class SharedEmbedding(nn.Module):
  """Embedding + RMSNorm + output projection matching `liquid_audio`."""

  def __init__(
      self,
      dim: int,
      vocab_size: int,
      norm_eps: float = 1e-5,
      tie_embedding: bool = False,
  ):
    super().__init__()
    self.dim = dim
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, dim)
    self.embedding_norm = DepthformerRMSNorm(dim, eps=norm_eps)
    self.to_logits = nn.Linear(dim, vocab_size, bias=False)
    self.tie_embedding = tie_embedding
    if self.tie_embedding:
      self.to_logits.weight = self.embedding.weight

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    return self.embedding(tokens)

  def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
    return self.to_logits(self.embedding_norm(embeddings))


class Lfm2AudioTextEmbedderModule(nn.Module):
  """Text token embedder loaded from `lfm.embed_tokens.weight` (`[65536, 2048]`)."""

  def __init__(self, vocab_size: int = 65536, hidden_size: int = 2048):
    super().__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    return self.embed_tokens(token_ids)


# ==============================================================================
# 3. TTS Projector (`depth_linear.*`, `audio_embedding.*`, `codebook_offsets`)
# ==============================================================================


class Lfm2AudioTtsProjectorModule(nn.Module):
  """Projects LFM2 hidden state to Depthformer inputs and embeds audio codes.

  Matches the top-level `LiquidAI/LFM2.5-Audio-1.5B` bridge parameters:
  - `depth_linear.{weight,bias}`: `[codebooks * depthformer_dim,
  llm_hidden_size]`
  - `audio_embedding.{embedding,lm_head}.weight`: `[audio_vocab_size *
  codebooks, llm_hidden_size]`
  - `codebook_offsets`: `[codebooks]`
  - `audio_loss_weights`: `[codebooks]`
  """

  def __init__(
      self,
      llm_hidden_size: int = 2048,
      depthformer_dim: int = 1024,
      codebooks: int = 8,
      audio_vocab_size: int = 2049,
      tie_audio_embeddings: bool = False,
      audio_loss_weights: Sequence[float] | None = None,
  ):
    super().__init__()
    self.llm_hidden_size = llm_hidden_size
    self.depthformer_dim = depthformer_dim
    self.codebooks = codebooks
    self.audio_vocab_size = audio_vocab_size
    self.depth_linear = nn.Linear(
        llm_hidden_size, codebooks * depthformer_dim, bias=True
    )
    self.audio_embedding = SharedEmbedding(
        dim=llm_hidden_size,
        vocab_size=audio_vocab_size * codebooks,
        tie_embedding=tie_audio_embeddings,
    )
    self.register_buffer(
        'codebook_offsets',
        torch.arange(codebooks, dtype=torch.int64) * audio_vocab_size,
    )
    if audio_loss_weights is None:
      loss_weights = (
          torch.linspace(1.0, 0.0, codebooks, dtype=torch.float32)
          * math.log(100.0)
      ).exp()
    else:
      loss_weights = torch.tensor(
          list(audio_loss_weights)[:codebooks], dtype=torch.float32
      )
    self.register_buffer('audio_loss_weights', loss_weights)

  def forward(
      self, llm_hidden: torch.Tensor, audio_codes: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # llm_hidden: [B, 1, D_llm] -> depthformer_in: [B, codebooks, D_df]
    b = llm_hidden.shape[0]
    depthformer_in = self.depth_linear(llm_hidden).reshape(
        b, self.codebooks, self.depthformer_dim
    )
    # audio_codes: [B, codebooks] -> audio_emb: [B, 1, llm_hidden_size]
    offsets = self.codebook_offsets.to(
        device=audio_codes.device, dtype=torch.long
    ).unsqueeze(0)
    offset_codes = audio_codes.to(torch.long) + offsets
    audio_emb = self.audio_embedding.embedding(offset_codes).sum(
        dim=1, keepdim=True
    )
    return depthformer_in, audio_emb


# ==============================================================================
# 4. Depthformer (`depthformer.*`, `depth_embeddings.*`)
# ==============================================================================


class DepthformerRMSNorm(nn.Module):
  """RMSNorm used in Depthformer (`attn_norm`, `ffn_norm`, `out_norm`, QK norms)."""

  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    normed = x.float() * torch.rsqrt(
        x.float().pow(2).mean(-1, keepdim=True) + self.eps
    )
    return normed.type_as(x) * self.weight


def _apply_interleaved_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Applies interleaved complex RoPE using purely real operations.

  Mathematically identical to `torch.view_as_real(torch.view_as_complex(x) *
  freqs_cis).flatten(3)` in `liquid_audio/model/raw_lm_backbone.py`.

  Args:
    q: Query tensor `[B, T, H_q, D]`.
    k: Key tensor `[B, T, H_kv, D]`.
    cos: Cosine frequencies `[1, T, 1, D // 2]`.
    sin: Sine frequencies `[1, T, 1, D // 2]`.

  Returns:
    Rotated `(q_out, k_out)` with the same shapes and dtypes as `(q, k)`.
  """
  q_pairs = q.float().reshape(*q.shape[:-1], -1, 2)
  k_pairs = k.float().reshape(*k.shape[:-1], -1, 2)
  q_even, q_odd = q_pairs[..., 0], q_pairs[..., 1]
  k_even, k_odd = k_pairs[..., 0], k_pairs[..., 1]
  q_out = torch.stack(
      [q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], dim=-1
  ).flatten(3)
  k_out = torch.stack(
      [k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], dim=-1
  ).flatten(3)
  return q_out.type_as(q), k_out.type_as(k)


class DepthformerBoundedAttention(nn.Module):
  """BoundedAttention submodule holding QK RMSNorms (`bounded_attention`)."""

  def __init__(
      self,
      head_dim: int,
      qk_layernorm: bool = True,
      norm_eps: float = 1e-5,
  ):
    super().__init__()
    self.qk_layernorm = qk_layernorm
    if qk_layernorm:
      self.q_layernorm = DepthformerRMSNorm(head_dim, eps=norm_eps)
      self.k_layernorm = DepthformerRMSNorm(head_dim, eps=norm_eps)


class DepthformerMHA(nn.Module):
  """Grouped-query attention for Depthformer (`depthformer.layers.*.operator`)."""

  def __init__(
      self,
      dim: int = 1024,
      num_heads: int = 32,
      gqa_dim: int = 8,
      qk_layernorm: bool = True,
      norm_eps: float = 1e-5,
  ):
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.gqa_dim = gqa_dim
    self.head_dim = dim // num_heads
    self.kv_dim = gqa_dim * self.head_dim
    self.num_kv_groups = num_heads // gqa_dim

    qkv_dim = (num_heads + 2 * gqa_dim) * self.head_dim
    self.qkv_proj = nn.Linear(dim, qkv_dim, bias=False)
    self.out_proj = nn.Linear(dim, dim, bias=False)
    self.bounded_attention = DepthformerBoundedAttention(
        head_dim=self.head_dim,
        qk_layernorm=qk_layernorm,
        norm_eps=norm_eps,
    )

  def forward_step(
      self,
      x: torch.Tensor,
      cos_step: torch.Tensor,
      sin_step: torch.Tensor,
      step_idx: torch.Tensor,
      k_cache: torch.Tensor,
      v_cache: torch.Tensor,
      positions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: [B, 1, dim]
    b, seqlen, _ = x.shape
    qkv = self.qkv_proj(x)
    q, k, v = torch.split(qkv, [self.dim, self.kv_dim, self.kv_dim], dim=-1)
    q = q.view(b, seqlen, self.num_heads, self.head_dim)
    k = k.view(b, seqlen, self.gqa_dim, self.head_dim)
    v = v.view(b, seqlen, self.gqa_dim, self.head_dim)

    if self.bounded_attention.qk_layernorm:
      q = self.bounded_attention.q_layernorm(q)
      k = self.bounded_attention.k_layernorm(k)

    q, k = _apply_interleaved_rope(q, k, cos_step, sin_step)

    # Update k_cache and v_cache of shape [B, max_seq, gqa_dim * head_dim]
    k_flat = k.reshape(b, 1, self.kv_dim)
    v_flat = v.reshape(b, 1, self.kv_dim)
    write_mask = torch.eq(positions, step_idx).view(1, -1, 1)
    new_k_cache = torch.where(write_mask, k_flat, k_cache)
    new_v_cache = torch.where(write_mask, v_flat, v_cache)

    max_seq = new_k_cache.shape[1]
    k_full = new_k_cache.view(b, max_seq, self.gqa_dim, self.head_dim).permute(
        0, 2, 1, 3
    )
    v_full = new_v_cache.view(b, max_seq, self.gqa_dim, self.head_dim).permute(
        0, 2, 1, 3
    )
    if self.num_kv_groups > 1:
      k_full = k_full.repeat_interleave(self.num_kv_groups, dim=1)
      v_full = v_full.repeat_interleave(self.num_kv_groups, dim=1)

    q_t = q.permute(0, 2, 1, 3)  # [B, num_heads, 1, head_dim]
    scores = torch.matmul(q_t, k_full.transpose(-2, -1)) / math.sqrt(
        self.head_dim
    )
    causal_mask = (positions <= step_idx).view(1, 1, 1, max_seq)
    scores = torch.where(causal_mask, scores, torch.full_like(scores, -1e9))
    attn = torch.softmax(scores, dim=-1)
    out = (
        torch.matmul(attn, v_full).transpose(1, 2).reshape(b, seqlen, self.dim)
    )
    return self.out_proj(out), new_k_cache, new_v_cache


class DepthformerGLU(nn.Module):
  """SwiGLU feed-forward block (`depthformer.layers.*.feed_forward`)."""

  def __init__(
      self, dim: int = 1024, ff_dim: int = 4096, multiple_of: int = 256
  ):
    super().__init__()
    adjusted_ff_dim = int(2 * ff_dim / 3)
    adjusted_ff_dim = multiple_of * (
        (adjusted_ff_dim + multiple_of - 1) // multiple_of
    )
    self.w1 = nn.Linear(dim, adjusted_ff_dim, bias=False)
    self.w2 = nn.Linear(adjusted_ff_dim, dim, bias=False)
    self.w3 = nn.Linear(dim, adjusted_ff_dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DepthformerBlock(nn.Module):
  """Single Depthformer transformer layer (`depthformer.layers.*`)."""

  def __init__(
      self,
      dim: int = 1024,
      ff_dim: int = 4096,
      num_heads: int = 32,
      gqa_dim: int = 8,
      qk_layernorm: bool = True,
      norm_eps: float = 1e-5,
      multiple_of: int = 256,
  ):
    super().__init__()
    self.operator = DepthformerMHA(
        dim=dim,
        num_heads=num_heads,
        gqa_dim=gqa_dim,
        qk_layernorm=qk_layernorm,
        norm_eps=norm_eps,
    )
    self.feed_forward = DepthformerGLU(
        dim=dim, ff_dim=ff_dim, multiple_of=multiple_of
    )
    self.operator_norm = DepthformerRMSNorm(dim, eps=norm_eps)
    self.ffn_norm = DepthformerRMSNorm(dim, eps=norm_eps)

  def forward_step(
      self,
      x: torch.Tensor,
      cos_step: torch.Tensor,
      sin_step: torch.Tensor,
      step_idx: torch.Tensor,
      k_cache: torch.Tensor,
      v_cache: torch.Tensor,
      positions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attn_out, new_k, new_v = self.operator.forward_step(
        self.operator_norm(x),
        cos_step,
        sin_step,
        step_idx,
        k_cache,
        v_cache,
        positions,
    )
    h = x + attn_out
    out = h + self.feed_forward(self.ffn_norm(h))
    return out, new_k, new_v


class RawLMBackbone(nn.Module):
  """Depthformer transformer stack (`depthformer.*`)."""

  def __init__(
      self,
      layers: int = 6,
      dim: int = 1024,
      ff_dim: int = 4096,
      num_heads: int = 32,
      gqa_dim: int = 8,
      qk_layernorm: bool = True,
      norm_eps: float = 1e-5,
      multiple_of: int = 256,
      theta: float = 1000000.0,
      max_seq_len: int = 8,
  ):
    super().__init__()
    self.num_layers = layers
    self.dim = dim
    self.num_heads = num_heads
    self.gqa_dim = gqa_dim
    self.head_dim = dim // num_heads
    self.kv_dim = gqa_dim * self.head_dim
    self.max_seq_len = max_seq_len

    self.layers = nn.ModuleList([
        DepthformerBlock(
            dim=dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            gqa_dim=gqa_dim,
            qk_layernorm=qk_layernorm,
            norm_eps=norm_eps,
            multiple_of=multiple_of,
        )
        for _ in range(layers)
    ])

    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32)[
                : (self.head_dim // 2)
            ]
            / self.head_dim
        )
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    self.register_buffer('_rope_cos', torch.cos(angles), persistent=False)
    self.register_buffer('_rope_sin', torch.sin(angles), persistent=False)
    self.register_buffer(
        '_positions',
        torch.arange(max_seq_len, dtype=torch.int32),
        persistent=False,
    )

  def forward_step(
      self,
      x: torch.Tensor,
      step_idx: torch.Tensor,
      kv_cache: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [B, 1, dim]
    # step_idx: [1] int32 in [0, max_seq_len - 1]
    # kv_cache: [B, 2 * layers, max_seq_len, kv_dim]
    step_mask = (
        torch.eq(self._positions, step_idx).to(self._rope_cos.dtype).view(-1, 1)
    )
    cos_step = (self._rope_cos * step_mask).sum(dim=0).view(1, 1, 1, -1)
    sin_step = (self._rope_sin * step_mask).sum(dim=0).view(1, 1, 1, -1)

    new_caches: list[torch.Tensor] = []
    h = x
    for idx, block in enumerate(self.layers):
      k_c = kv_cache[:, 2 * idx]
      v_c = kv_cache[:, 2 * idx + 1]
      h, new_k, new_v = block.forward_step(
          h, cos_step, sin_step, step_idx, k_c, v_c, self._positions
      )
      new_caches.append(new_k)
      new_caches.append(new_v)

    return h, torch.stack(new_caches, dim=1)


class Lfm2DepthformerModule(nn.Module):
  """8-codebook RQ-Transformer Depthformer (`depthformer.*`, `depth_embeddings.*`)."""

  def __init__(
      self,
      layers: int = 6,
      dim: int = 1024,
      ff_dim: int = 4096,
      num_heads: int = 32,
      gqa_dim: int = 8,
      qk_layernorm: bool = True,
      norm_eps: float = 1e-5,
      multiple_of: int = 256,
      theta: float = 1000000.0,
      codebooks: int = 8,
      vocab_size: int = 2049,
      tie_depth_embeddings: bool = True,
  ):
    super().__init__()
    self.layers = layers
    self.dim = dim
    self.num_heads = num_heads
    self.gqa_dim = gqa_dim
    self.head_dim = dim // num_heads
    self.kv_dim = gqa_dim * self.head_dim
    self.codebooks = codebooks
    self.vocab_size = vocab_size

    self.depthformer = RawLMBackbone(
        layers=layers,
        dim=dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        gqa_dim=gqa_dim,
        qk_layernorm=qk_layernorm,
        norm_eps=norm_eps,
        multiple_of=multiple_of,
        theta=theta,
        max_seq_len=codebooks,
    )
    self.depth_embeddings = nn.ModuleList([
        SharedEmbedding(
            dim=dim,
            vocab_size=vocab_size,
            norm_eps=norm_eps,
            tie_embedding=tie_depth_embeddings,
        )
        for _ in range(codebooks)
    ])

  def forward(
      self,
      cond_emb: torch.Tensor,
      prev_audio_code: torch.Tensor,
      step_idx: torch.Tensor,
      kv_cache: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # cond_emb: [B, 1, dim] (slice `step_idx` of depthformer_in)
    # prev_audio_code: [B, 1] int32
    # step_idx: [1] int32 in [0, codebooks - 1]
    # kv_cache: [B, 2 * layers, codebooks, kv_dim]
    prev_token_long = prev_audio_code.to(torch.long)
    prev_emb = torch.zeros_like(cond_emb)
    for k in range(1, self.codebooks):
      mask = torch.eq(step_idx, k).to(cond_emb.dtype).view(1, 1, 1)
      prev_emb = prev_emb + mask * self.depth_embeddings[k - 1].embedding(
          prev_token_long
      )
    x_in = cond_emb + prev_emb

    h, new_kv_cache = self.depthformer.forward_step(x_in, step_idx, kv_cache)

    logits = torch.zeros(
        (h.shape[0], h.shape[1], self.vocab_size),
        dtype=h.dtype,
        device=h.device,
    )
    for k in range(self.codebooks):
      mask = torch.eq(step_idx, k).to(h.dtype).view(1, 1, 1)
      logits = logits + mask * self.depth_embeddings[k].get_logits(h)

    return logits, new_kv_cache


# ==============================================================================
# 5. LFM2 Audio Detokenizer (`emb.emb.weight`, `lfm.*`, `lin.*`, `istft.window`)
# ==============================================================================


class FusedEmbedding(nn.Module):
  """Multi-codebook fused embedding (`emb.emb.weight` in `audio_detokenizer`)."""

  def __init__(
      self, codebooks: int = 8, vocab_size: int = 2048, dim: int = 512
  ):
    super().__init__()
    self.codebooks = codebooks
    self.vocab_size = vocab_size
    self.dim = dim
    self.emb = nn.Embedding(codebooks * vocab_size, dim)
    self.register_buffer(
        '_offsets',
        (torch.arange(codebooks, dtype=torch.int64) * vocab_size).view(
            1, codebooks, 1
        ),
        persistent=False,
    )

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    # tokens: [B, codebooks, T] -> [B, codebooks, T, dim]
    offsets = self._offsets.to(device=tokens.device)
    return self.emb(tokens.to(torch.long) + offsets)


class _DetokRMSNorm(nn.Module):
  """RMSNorm for Detokenizer LFM2 layers."""

  def __init__(self, hidden_size: int, eps: float = 1e-5):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_norm = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * x_norm.to(input_dtype)


class _DetokMLP(nn.Module):
  """SwiGLU MLP for Detokenizer LFM2 layers (`lfm.layers.*.feed_forward`)."""

  def __init__(
      self,
      hidden_size: int = 512,
      intermediate_size: int = 3328,
      block_auto_adjust_ff_dim: bool = True,
      block_ffn_dim_multiplier: float = 1.0,
      block_multiple_of: int = 256,
  ):
    super().__init__()
    ff_dim = intermediate_size
    if block_auto_adjust_ff_dim:
      ff_dim = int(2 * ff_dim / 3)
      if block_ffn_dim_multiplier is not None:
        ff_dim = int(block_ffn_dim_multiplier * ff_dim)
      ff_dim = block_multiple_of * (
          (ff_dim + block_multiple_of - 1) // block_multiple_of
      )
    self.w1 = nn.Linear(hidden_size, ff_dim, bias=False)
    self.w3 = nn.Linear(hidden_size, ff_dim, bias=False)
    self.w2 = nn.Linear(ff_dim, hidden_size, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _DetokShortConv(nn.Module):
  """Causal gated short conv for Detokenizer LFM2 (`lfm.layers.*.conv`)."""

  def __init__(
      self,
      hidden_size: int = 512,
      l_cache: int = 3,
      conv_bias: bool = False,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.l_cache = l_cache
    self.conv = nn.Conv1d(
        in_channels=hidden_size,
        out_channels=hidden_size,
        kernel_size=l_cache,
        groups=hidden_size,
        bias=conv_bias,
    )
    self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=conv_bias)
    self.out_proj = nn.Linear(hidden_size, hidden_size, bias=conv_bias)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # hidden_states: [B, T, D]
    bcx = self.in_proj(hidden_states).transpose(-1, -2)  # [B, 3*D, T]
    b_gate, c_gate, x = bcx.chunk(3, dim=-2)
    bx = b_gate * x  # [B, D, T]
    bx_padded = F.pad(bx, (self.l_cache - 1, 0))
    conv_out = self.conv(bx_padded)
    y = (c_gate * conv_out).transpose(-1, -2).contiguous()
    return self.out_proj(y)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


class _DetokAttention(nn.Module):
  """Sliding-window / full attention for Detokenizer LFM2 (`lfm.layers.*.self_attn`)."""

  def __init__(
      self,
      hidden_size: int = 512,
      num_attention_heads: int = 16,
      num_key_value_heads: int = 8,
      norm_eps: float = 1e-5,
      sliding_window: int | None = 30,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.head_dim = hidden_size // num_attention_heads
    self.num_key_value_groups = num_attention_heads // num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.sliding_window = sliding_window

    self.q_proj = nn.Linear(
        hidden_size, num_attention_heads * self.head_dim, bias=False
    )
    self.k_proj = nn.Linear(
        hidden_size, num_key_value_heads * self.head_dim, bias=False
    )
    self.v_proj = nn.Linear(
        hidden_size, num_key_value_heads * self.head_dim, bias=False
    )
    self.out_proj = nn.Linear(
        num_attention_heads * self.head_dim, hidden_size, bias=False
    )
    self.q_layernorm = _DetokRMSNorm(self.head_dim, eps=norm_eps)
    self.k_layernorm = _DetokRMSNorm(self.head_dim, eps=norm_eps)

  def forward(
      self,
      hidden_states: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      attn_mask: torch.Tensor,
  ) -> torch.Tensor:
    b, seqlen, _ = hidden_states.shape
    q = self.q_layernorm(
        self.q_proj(hidden_states).view(
            b, seqlen, self.num_heads, self.head_dim
        )
    ).transpose(1, 2)
    k = self.k_layernorm(
        self.k_proj(hidden_states).view(
            b, seqlen, self.num_key_value_heads, self.head_dim
        )
    ).transpose(1, 2)
    v = (
        self.v_proj(hidden_states)
        .view(b, seqlen, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)

    if self.num_key_value_groups > 1:
      k = k.repeat_interleave(self.num_key_value_groups, dim=1)
      v = v.repeat_interleave(self.num_key_value_groups, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling + attn_mask
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = (
        torch.matmul(attn, v)
        .transpose(1, 2)
        .reshape(b, seqlen, self.hidden_size)
    )
    return self.out_proj(out)


class _DetokDecoderLayer(nn.Module):
  """Single LFM2 decoder layer in Detokenizer (`lfm.layers.*`)."""

  def __init__(
      self,
      layer_type: str,
      hidden_size: int = 512,
      intermediate_size: int = 3328,
      block_auto_adjust_ff_dim: bool = True,
      block_ffn_dim_multiplier: float = 1.0,
      block_multiple_of: int = 256,
      num_attention_heads: int = 16,
      num_key_value_heads: int = 8,
      conv_l_cache: int = 3,
      conv_bias: bool = False,
      norm_eps: float = 1e-5,
      sliding_window: int | None = 30,
  ):
    super().__init__()
    self.is_attention_layer = layer_type != 'conv'
    if self.is_attention_layer:
      self.self_attn = _DetokAttention(
          hidden_size=hidden_size,
          num_attention_heads=num_attention_heads,
          num_key_value_heads=num_key_value_heads,
          norm_eps=norm_eps,
          sliding_window=(
              sliding_window if layer_type == 'sliding_attention' else None
          ),
      )
    else:
      self.conv = _DetokShortConv(
          hidden_size=hidden_size,
          l_cache=conv_l_cache,
          conv_bias=conv_bias,
      )
    self.feed_forward = _DetokMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        block_auto_adjust_ff_dim=block_auto_adjust_ff_dim,
        block_ffn_dim_multiplier=block_ffn_dim_multiplier,
        block_multiple_of=block_multiple_of,
    )
    self.operator_norm = _DetokRMSNorm(hidden_size, eps=norm_eps)
    self.ffn_norm = _DetokRMSNorm(hidden_size, eps=norm_eps)

  def forward(
      self,
      hidden_states: torch.Tensor,
      cos: torch.Tensor,
      sin: torch.Tensor,
      attn_mask: torch.Tensor,
  ) -> torch.Tensor:
    residual = hidden_states
    normed = self.operator_norm(hidden_states)
    if self.is_attention_layer:
      hidden_states = residual + self.self_attn(normed, cos, sin, attn_mask)
    else:
      hidden_states = residual + self.conv(normed)
    return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class DetokLfm2Model(nn.Module):
  """Export-friendly Lfm2Model matching `audio_detokenizer` `lfm.*` weights."""

  def __init__(self, lfm_cfg: dict[str, Any]):
    super().__init__()
    vocab_size = int(lfm_cfg.get('vocab_size', 65536))
    hidden_size = int(lfm_cfg.get('hidden_size', 512))
    intermediate_size = int(lfm_cfg.get('intermediate_size', 3328))
    block_auto_adjust_ff_dim = bool(
        lfm_cfg.get('block_auto_adjust_ff_dim', True)
    )
    block_ffn_dim_multiplier = float(
        lfm_cfg.get('block_ffn_dim_multiplier', 1.0)
    )
    block_multiple_of = int(lfm_cfg.get('block_multiple_of', 256))
    num_hidden_layers = int(lfm_cfg.get('num_hidden_layers', 8))
    num_attention_heads = int(lfm_cfg.get('num_attention_heads', 16))
    num_key_value_heads = int(lfm_cfg.get('num_key_value_heads', 8))
    conv_l_cache = int(lfm_cfg.get('conv_L_cache', 3))
    conv_bias = bool(lfm_cfg.get('conv_bias', False))
    norm_eps = float(lfm_cfg.get('norm_eps', 1e-5))
    rope_theta = float(
        lfm_cfg.get('rope_theta', None)
        or (lfm_cfg.get('rope_parameters') or {}).get('rope_theta', 1000000.0)
    )
    sliding_window = lfm_cfg.get('sliding_window', 30)
    layer_types = list(
        lfm_cfg.get(
            'layer_types',
            [
                'conv',
                'conv',
                'sliding_attention',
                'conv',
                'sliding_attention',
                'conv',
                'sliding_attention',
                'conv',
            ][:num_hidden_layers],
        )
    )

    self.hidden_size = hidden_size
    self.head_dim = hidden_size // num_attention_heads
    self.sliding_window = sliding_window
    self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
    self.layers = nn.ModuleList([
        _DetokDecoderLayer(
            layer_type=layer_types[i],
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            block_auto_adjust_ff_dim=block_auto_adjust_ff_dim,
            block_ffn_dim_multiplier=block_ffn_dim_multiplier,
            block_multiple_of=block_multiple_of,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            conv_l_cache=conv_l_cache,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            sliding_window=sliding_window,
        )
        for i in range(num_hidden_layers)
    ])
    self.embedding_norm = _DetokRMSNorm(hidden_size, eps=norm_eps)

    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32)
            / self.head_dim
        )
    )
    self.register_buffer('_inv_freq', inv_freq, persistent=False)

  def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
    seqlen = inputs_embeds.shape[1]
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype

    positions = torch.arange(seqlen, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, self._inv_freq.to(device=device))
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).view(1, 1, seqlen, self.head_dim)
    sin = emb.sin().to(dtype).view(1, 1, seqlen, self.head_dim)

    q_idx = torch.arange(seqlen, device=device).unsqueeze(1)
    kv_idx = torch.arange(seqlen, device=device).unsqueeze(0)
    valid = kv_idx <= q_idx
    if self.sliding_window is not None and self.sliding_window > 0:
      valid = valid & ((q_idx - kv_idx) < self.sliding_window)
    attn_mask = torch.where(
        valid,
        torch.zeros((seqlen, seqlen), dtype=dtype, device=device),
        torch.full((seqlen, seqlen), -1e9, dtype=dtype, device=device),
    ).view(1, 1, seqlen, seqlen)

    hidden_states = inputs_embeds
    for layer in self.layers:
      hidden_states = layer(hidden_states, cos, sin, attn_mask)
    return self.embedding_norm(hidden_states)


class ISTFT(nn.Module):
  """Exportable ISTFT module matching `liquid_audio` (`istft.window`).

  Replaces `torch.fft.irfft` + Python overlap-add loop with an exact real-valued
  inverse DFT basis multiplied by `self.window` and overlap-added via
  `F.conv_transpose1d`.
  """

  def __init__(
      self,
      n_fft: int = 1280,
      hop_length: int = 320,
      win_length: int = 1280,
      padding: str = 'same',
  ):
    super().__init__()
    if padding != 'same':
      raise ValueError(f'Unsupported ISTFT padding: {padding}')
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.pad = (win_length - hop_length) // 2

    window = torch.hann_window(win_length, dtype=torch.float32)
    self.window: torch.Tensor
    self.register_buffer('window', window)

    n_bins = n_fft // 2 + 1
    k = torch.arange(n_bins, dtype=torch.float32).unsqueeze(1)  # [F, 1]
    n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)  # [1, N]
    angles = 2.0 * math.pi * k * n / float(n_fft)
    weights = torch.full((n_bins, 1), 2.0 / float(n_fft), dtype=torch.float32)
    weights[0, 0] = 1.0 / float(n_fft)
    if n_fft % 2 == 0:
      weights[-1, 0] = 1.0 / float(n_fft)

    self.register_buffer(
        '_idft_cos',
        (weights * torch.cos(angles)).unsqueeze(1),
        persistent=False,
    )
    self.register_buffer(
        '_idft_sin',
        (-weights * torch.sin(angles)).unsqueeze(1),
        persistent=False,
    )

  def forward(
      self,
      log_mag: torch.Tensor,
      phase: torch.Tensor,
      valid_stft_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    # log_mag, phase: [B, n_fft // 2 + 1, T_frames]
    mag = torch.exp(log_mag)
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    if valid_stft_mask is not None:
      mask = valid_stft_mask.to(dtype=real.dtype, device=real.device)
      real = real * mask
      imag = imag * mask

    win = self.window.view(1, 1, -1)
    kernel_cos = self._idft_cos.to(dtype=real.dtype, device=real.device) * win
    kernel_sin = self._idft_sin.to(dtype=imag.dtype, device=imag.device) * win
    kernel = torch.cat([kernel_cos, kernel_sin], dim=0)
    spec = torch.cat([real, imag], dim=1)

    y = F.conv_transpose1d(
        spec,
        kernel,
        stride=self.hop_length,
        padding=self.pad,
    )

    if valid_stft_mask is not None:
      ones = valid_stft_mask.to(dtype=real.dtype, device=real.device)
    else:
      ones = torch.ones(
          (1, 1, real.shape[-1]), dtype=real.dtype, device=real.device
      )
    win_sq_kernel = (
        self.window.square()
        .view(1, 1, -1)
        .to(dtype=real.dtype, device=real.device)
    )
    envelope = F.conv_transpose1d(
        ones,
        win_sq_kernel,
        stride=self.hop_length,
        padding=self.pad,
    )
    y = y / torch.clamp(envelope, min=1e-11)
    return y.squeeze(1)


class Lfm2AudioDetokenizerModule(nn.Module):
  """LFM2 neural audio detokenizer (`audio_detokenizer/model.safetensors`)."""

  def __init__(self, detok_cfg: dict[str, Any]):
    super().__init__()
    lfm_cfg = detok_cfg.get('lfm', detok_cfg)
    hidden_size = int(lfm_cfg.get('hidden_size', 512))
    self.codebooks = int(detok_cfg.get('codebooks', 8))
    self.vocab_size = int(
        detok_cfg.get('audio_vocab_size', detok_cfg.get('vocab_size', 2048))
    )
    self.upsample_factor = int(detok_cfg.get('upsample_factor', 6))
    self.n_fft = int(detok_cfg.get('n_fft', 1280))
    self.hop_length = int(detok_cfg.get('hop_length', 320))

    self.emb = FusedEmbedding(
        codebooks=self.codebooks,
        vocab_size=self.vocab_size,
        dim=hidden_size,
    )
    self.lfm = DetokLfm2Model(lfm_cfg)
    self.lin = nn.Linear(hidden_size, self.n_fft + 2)
    self.istft = ISTFT(
        n_fft=self.n_fft,
        hop_length=self.hop_length,
        win_length=self.n_fft,
        padding='same',
    )

  def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
    # audio_codes: [B, codebooks=8, T_codes] -> waveform: [B, 1920 * T_codes]
    # Negative values (e.g. -1) mark right-padded frames when T_valid < T_codes.
    valid_codes_mask = (audio_codes[:, :1, :] >= 0).to(dtype=torch.float32)
    b, _, t_codes = valid_codes_mask.shape
    valid_stft_mask = (
        valid_codes_mask.unsqueeze(-1)
        .expand(b, 1, t_codes, self.upsample_factor)
        .reshape(b, 1, t_codes * self.upsample_factor)
    )
    safe_codes = torch.clamp(audio_codes, min=0, max=self.vocab_size - 1)
    x = self.emb(safe_codes).mean(dim=1)  # [B, T_codes, hidden_size]
    _, _, d = x.shape
    x = (
        x.unsqueeze(2)
        .expand(b, t_codes, self.upsample_factor, d)
        .reshape(b, t_codes * self.upsample_factor, d)
    )
    x = self.lfm(x)
    x = self.lin(x).permute(0, 2, 1)  # [B, n_fft + 2, T_frames]
    log_mag, phase = x.chunk(2, dim=1)
    return self.istft(log_mag, phase, valid_stft_mask=valid_stft_mask)
