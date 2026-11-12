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
"""LiteRT exportable wrappers for HuggingFace gemma_quant layers."""

from collections.abc import Callable
import types
from typing import Any

from litert_torch._convert.fx_passes import lower_torchao_pass
import torch

try:
  from transformers.integrations import gemma_quant as hf_gemma_quant  # pylint: disable=g-import-not-at-top

  QuantizedLinear = hf_gemma_quant.QuantizedLinear
  QuantizedEmbedding = hf_gemma_quant.QuantizedEmbedding
except (ImportError, AttributeError):
  hf_gemma_quant = None
  QuantizedLinear = None  # pylint: disable=invalid-name
  QuantizedEmbedding = None  # pylint: disable=invalid-name

WeightsLoader = Callable[[str], torch.Tensor]

_NUM_BITS_TO_DTYPE: dict[int, str] = {
    2: "i2",
    4: "i4",
    8: "i8",
}

QUANT_PARAM_SUFFIXES: tuple[str, ...] = (
    "weight_scale",
    "embedding_scale",
    "input_activation_scale",
    "output_activation_scale",
)


def quant_dtype_str(num_bits: int) -> str:
  """Returns the MLIR quantization dtype string (`'i2'`, `'i4'`, `'i8'`)."""
  if num_bits not in _NUM_BITS_TO_DTYPE:
    raise ValueError(f"Unsupported num_bits: {num_bits}")
  return _NUM_BITS_TO_DTYPE[num_bits]


def unpack_gemma_weight(
    raw_weight: torch.Tensor, num_bits: int, orig_width: int
) -> torch.Tensor:
  """Unpacks packed Gemma integer weights (`int2`/`int4`/`int8`) into `int8`."""
  if num_bits == 8:
    return raw_weight.to(torch.int8)
  if num_bits == 4:
    packed = raw_weight.to(torch.uint8)
    low = (packed & 0x0F).to(torch.int8) - 8
    high = (packed >> 4).to(torch.int8) - 8
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)[
        ..., :orig_width
    ]
  if num_bits == 2:
    packed = raw_weight.to(torch.uint8)
    v0 = (packed & 0x03).to(torch.int8) - 2
    v1 = ((packed >> 2) & 0x03).to(torch.int8) - 2
    v2 = ((packed >> 4) & 0x03).to(torch.int8) - 2
    v3 = (packed >> 6).to(torch.int8) - 2
    return torch.stack([v0, v1, v2, v3], dim=-1).reshape(
        *packed.shape[:-1], -1
    )[..., :orig_width]
  raise ValueError(f"Unsupported num_bits: {num_bits}")


def create_per_channel_dequant_fn(
    scale: torch.Tensor, num_bits: int
) -> Callable[[torch.Tensor], torch.Tensor]:
  """Creates a 1D per-channel (dim 0) dequantize function for `scale`."""
  if scale.is_meta:
    raise ValueError("Scale tensor must not be on the meta device.")
  attrs = {
      "scale": scale.flatten().tolist(),
      "zero_point": [0],
      "dtype": quant_dtype_str(num_bits),
      "quantization_dimension": 0,
      "narrow_range": False,
  }
  return lower_torchao_pass._create_dequantize_function(attrs)  # pylint: disable=protected-access


def create_per_tensor_fake_quant_fn(
    scale: torch.Tensor | None, dtype_str: str = "i8"
) -> Callable[[torch.Tensor], torch.Tensor] | None:
  """Creates a per-tensor SRQ fake_quant function, or None if uncalibrated."""
  if scale is None:
    return None
  if scale.is_meta:
    raise ValueError("Scale tensor must not be on the meta device.")
  if scale.item() == 0.0:
    return None
  attrs = {
      "scale": [float(scale.item())],
      "zero_point": [0],
      "dtype": dtype_str,
      "narrow_range": False,
  }
  return lower_torchao_pass._create_fake_quant_function(attrs)  # pylint: disable=protected-access


def _embed_tokens_on_cpu_method(
    self: Any,
    ids: torch.Tensor,
    weights_loader: WeightsLoader | None = None,
    prefix: str = "embed_tokens",
) -> torch.Tensor:
  """Evaluates a quantized embedding module on `ids` on CPU."""
  fqn = f"{prefix}.embedding_quantized" if prefix else "embedding_quantized"
  return embed_quantized_tokens_on_cpu(self, ids, weights_loader, fqn)


class LiteRTQuantizedLinear(torch.nn.Module):
  """LiteRT exportable quantized linear layer for Gemma QAT."""

  def __init__(self, q_linear: Any) -> None:
    """Initializes `LiteRTQuantizedLinear` from a `QuantizedLinear` module."""
    super().__init__()
    self.num_bits = int(q_linear.num_bits)
    self.in_features = int(q_linear.in_features)
    self.out_features = int(q_linear.out_features)

    scale = q_linear.weight_scale.float()
    self.is_blockwise = scale.dim() == 2 and scale.shape[-1] > 1
    num_blocks = int(scale.shape[-1]) if self.is_blockwise else 1
    if self.in_features % num_blocks != 0:
      raise ValueError(
          f"in_features ({self.in_features}) must be divisible by num_blocks"
          f" ({num_blocks})."
      )
    self.block_size = self.in_features // num_blocks
    self.total_blocks = self.out_features * num_blocks
    self.dequant_fn = create_per_channel_dequant_fn(scale, self.num_bits)
    self.register_buffer("weight", self._unpack_and_reshape(q_linear.weight))

    if hasattr(q_linear, "bias") and q_linear.bias is not None:
      self.bias = torch.nn.Parameter(
          q_linear.bias.float(), requires_grad=bool(q_linear.bias.requires_grad)
      )
    else:
      self.bias = None

    self.in_fq = create_per_tensor_fake_quant_fn(
        getattr(q_linear, "input_activation_scale", None)
    )
    self.out_fq = create_per_tensor_fake_quant_fn(
        getattr(q_linear, "output_activation_scale", None)
    )
    # For non-blockwise layers without an output SRQ fake_quant (either
    # weight-only DRQ where input_activation_scale == 0.0, or partial SRQ where
    # input_activation_scale > 0 and output_activation_scale == 0.0), emit a
    # DRQ fake_quant (empty scale) immediately before F.linear so FuseQDQPass
    # fuses the quantized weight into a hybrid tfl.fully_connected and keeps
    # float32 bias intact during BiasQuantizerPass.
    self.drq_fq: Callable[[torch.Tensor], torch.Tensor] | None = None
    if not self.is_blockwise and self.out_fq is None:
      drq_attrs = lower_torchao_pass._get_quant_composite_attributes(  # pylint: disable=protected-access
          scale=None,
          zero_point=None,
          dtype="i8",
          quant_dimension=None,
          narrow_range=False,
      )
      self.drq_fq = lower_torchao_pass._create_fake_quant_function(drq_attrs)  # pylint: disable=protected-access

  def _unpack_and_reshape(self, raw_weight: torch.Tensor) -> torch.Tensor:
    int_weight = unpack_gemma_weight(
        raw_weight, self.num_bits, self.in_features
    )
    if self.is_blockwise:
      return int_weight.reshape(self.total_blocks, self.block_size)
    return int_weight

  def load_export_weight(
      self,
      weights_loader: WeightsLoader,
      prefix: str,
      attr: str,
  ) -> torch.Tensor | None:
    """Loads and unpacks `weight` from `weights_loader`, or returns `None`."""
    if attr != "weight":
      return None
    key = f"{prefix}.weight" if prefix else "weight"
    return self._unpack_and_reshape(weights_loader(key))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Applies quantized linear projection with optional SRQ/DRQ fake-quant."""
    if self.in_fq is not None:
      x = self.in_fq(x)
    if self.drq_fq is not None:
      x = self.drq_fq(x)
    dequant_w = self.dequant_fn(self.weight)
    if self.is_blockwise:
      dequant_w = dequant_w.reshape(self.out_features, self.in_features)
    out = torch.nn.functional.linear(x, dequant_w, self.bias)
    if self.out_fq is not None:
      out = self.out_fq(out)
    return out


class LiteRTQuantizedEmbedding(torch.nn.Module):
  """LiteRT exportable quantized embedding layer for Gemma QAT."""

  def __init__(self, q_emb: Any) -> None:
    """Initializes `LiteRTQuantizedEmbedding` from a `QuantizedEmbedding`."""
    super().__init__()
    self.num_bits = int(q_emb.num_bits)
    self.scalar_embed_scale = float(getattr(q_emb, "scalar_embed_scale", 1.0))
    self.num_embeddings = int(q_emb.num_embeddings)
    self.embedding_dim = int(q_emb.embedding_dim)

    scale = q_emb.embedding_scale.float()
    if scale.is_meta:
      raise ValueError("embedding_scale must not be on the meta device.")
    self.embedding_scale = scale.detach().cpu()
    self.is_blockwise = scale.dim() == 2 and scale.shape[-1] > 1
    num_blocks = int(scale.shape[-1]) if self.is_blockwise else 1
    if self.embedding_dim % num_blocks != 0:
      raise ValueError(
          f"embedding_dim ({self.embedding_dim}) must be divisible by"
          f" num_blocks ({num_blocks})."
      )
    self.block_size = self.embedding_dim // num_blocks
    self.total_blocks = self.num_embeddings * num_blocks
    self.dequant_fn = create_per_channel_dequant_fn(scale, self.num_bits)
    self.register_buffer(
        "weight", self._unpack_and_reshape(q_emb.embedding_quantized)
    )

  def _unpack_and_reshape(self, raw_weight: torch.Tensor) -> torch.Tensor:
    int_weight = unpack_gemma_weight(
        raw_weight, self.num_bits, self.embedding_dim
    )
    if self.is_blockwise:
      return int_weight.reshape(self.total_blocks, self.block_size)
    return int_weight

  def load_export_weight(
      self,
      weights_loader: WeightsLoader,
      prefix: str,
      attr: str,
  ) -> torch.Tensor | None:
    """Loads and unpacks `weight` from `weights_loader`, or returns `None`."""
    if attr != "weight":
      return None
    key = f"{prefix}.embedding_quantized" if prefix else "embedding_quantized"
    return self._unpack_and_reshape(weights_loader(key))

  embed_tokens_on_cpu = _embed_tokens_on_cpu_method

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Looks up quantized embeddings and applies `scalar_embed_scale`."""
    dequant_table = self.dequant_fn(self.weight)
    if self.is_blockwise and self.block_size % (8 // self.num_bits) == 0:
      num_blocks = self.embedding_dim // self.block_size
      block_ids = input_ids.unsqueeze(-1) * num_blocks + torch.arange(
          num_blocks, device=input_ids.device, dtype=input_ids.dtype
      )
      output = torch.nn.functional.embedding(block_ids, dequant_table)
      output = output.reshape(*input_ids.shape, self.embedding_dim)
    else:
      if self.is_blockwise:
        dequant_table = dequant_table.reshape(
            self.num_embeddings, self.embedding_dim
        )
      output = torch.nn.functional.embedding(input_ids, dequant_table)
    if self.scalar_embed_scale != 1.0:
      output = output * self.scalar_embed_scale
    return output


def replace_gemma_quant_layers(
    module: torch.nn.Module,
    skip_names: tuple[str, ...] = (),
) -> list[tuple[torch.nn.Module, str, torch.nn.Module]]:
  """Replaces `gemma_quant` layers in `module` with LiteRT exportable layers.

  Args:
    module: Root module to traverse in-place.
    skip_names: Child attribute names to skip (e.g. `"embed_tokens_per_layer"`).

  Returns:
    List of `(parent, child_name, original_child)` tuples for restoration.
  """
  replaced: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []
  if QuantizedLinear is None and QuantizedEmbedding is None:
    return replaced

  for child_name, child in list(module.named_children()):
    if child_name in skip_names:
      continue
    if QuantizedLinear is not None and isinstance(child, QuantizedLinear):
      setattr(module, child_name, LiteRTQuantizedLinear(child))
      replaced.append((module, child_name, child))
    elif QuantizedEmbedding is not None and isinstance(
        child, QuantizedEmbedding
    ):
      setattr(module, child_name, LiteRTQuantizedEmbedding(child))
      replaced.append((module, child_name, child))
    else:
      replaced.extend(replace_gemma_quant_layers(child, skip_names=skip_names))
  return replaced


def embed_quantized_tokens_on_cpu(
    emb: Any,
    ids: torch.Tensor,
    weights_loader: WeightsLoader | None,
    fqn: str,
) -> torch.Tensor:
  """Evaluates a quantized Gemma embedding layer on `ids` on CPU."""
  is_lrt = hasattr(emb, "dequant_fn")
  weight_tensor = emb.weight if is_lrt else emb.embedding_quantized
  if not weight_tensor.is_meta:
    with torch.no_grad():
      return emb(ids.to(weight_tensor.device)).float().detach().cpu().clone()
  if weights_loader is None:
    raise RuntimeError(
        "Input embedding weights are on the meta device, but no"
        " `weights_loader` is available."
    )
  raw_rows = weights_loader(fqn)[ids.long()].cpu()
  int_rows = unpack_gemma_weight(
      raw_rows, int(emb.num_bits), int(emb.embedding_dim)
  ).float()
  if emb.embedding_scale.is_meta:
    if fqn.endswith(".embedding_quantized"):
      scale_fqn = f"{fqn.removesuffix('.embedding_quantized')}.embedding_scale"
    elif fqn.endswith(".weight"):
      scale_fqn = f"{fqn.removesuffix('.weight')}.embedding_scale"
    elif "." in fqn:
      scale_fqn = f"{fqn.rpartition('.')[0]}.embedding_scale"
    else:
      scale_fqn = "embedding_scale"
    scale_tensor = weights_loader(scale_fqn)
  else:
    scale_tensor = emb.embedding_scale
  scale_rows = scale_tensor.float().cpu()[ids.long()]
  if scale_tensor.dim() == 2 and scale_tensor.shape[-1] > 1:
    block_size = int(emb.embedding_dim) // scale_tensor.shape[-1]
    scale_rows = scale_rows.repeat_interleave(block_size, dim=-1)
  elif scale_tensor.dim() == 1:
    scale_rows = scale_rows.unsqueeze(-1)
  scalar_scale = float(getattr(emb, "scalar_embed_scale", 1.0))
  return (int_rows * scale_rows * scalar_scale).to(torch.float32)


def extract_gemma_kv_cache_scales(
    model: torch.nn.Module,
    weights_loader: WeightsLoader,
) -> dict[int, tuple[float, float]]:
  """Extracts per-layer `(k_cache_scale, v_cache_scale)` from `weights_loader`."""
  kv_cache_scales: dict[int, tuple[float, float]] = {}
  for prefix, _ in model.named_modules():
    if (
        prefix.endswith(".self_attn")
        and "vision_tower" not in prefix
        and "audio_tower" not in prefix
    ):
      parts = prefix.split(".")
      layer_idx = next(
          (
              int(parts[i + 1])
              for i, p in enumerate(parts[:-1])
              if p == "layers" and parts[i + 1].isdigit()
          ),
          None,
      )
      if layer_idx is not None:
        try:
          k_s = weights_loader(f"{prefix}.k_cache_scale")
          v_s = weights_loader(f"{prefix}.v_cache_scale")
          kv_cache_scales[layer_idx] = (float(k_s.item()), float(v_s.item()))
        except KeyError:
          pass
  return kv_cache_scales


def load_gemma_quant_scales_(
    model: torch.nn.Module,
    weights_loader: WeightsLoader,
) -> torch.nn.Module:
  """Loads Gemma quantization scale parameters onto CPU and extracts KV scales."""
  for prefix, module in model.named_modules():
    if QuantizedEmbedding is not None and isinstance(
        module, QuantizedEmbedding
    ):
      setattr(
          module,
          "embed_tokens_on_cpu",
          types.MethodType(_embed_tokens_on_cpu_method, module),
      )
    for name, param in list(module._parameters.items()):  # pylint: disable=protected-access
      if param is None or not name.endswith(QUANT_PARAM_SUFFIXES):
        continue
      fqn = f"{prefix}.{name}" if prefix else name
      try:
        value = weights_loader(fqn)
      except KeyError:
        if name.endswith("activation_scale"):
          if param.is_meta:
            module._parameters[name] = torch.nn.Parameter(  # pylint: disable=protected-access
                torch.zeros(param.shape, dtype=param.dtype),
                requires_grad=False,
            )
          continue
        raise
      target_dtype = (
          torch.float32 if value.is_floating_point() else value.dtype
      )
      module._parameters[name] = torch.nn.Parameter(  # pylint: disable=protected-access
          value.to(target_dtype), requires_grad=False
      )
  kv_cache_scales = extract_gemma_kv_cache_scales(model, weights_loader)
  cfg: Any = getattr(model, "config", None)
  if kv_cache_scales and cfg is not None:
    cfg.kv_cache_scales = kv_cache_scales
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
      cfg.text_config.kv_cache_scales = kv_cache_scales
  return model
