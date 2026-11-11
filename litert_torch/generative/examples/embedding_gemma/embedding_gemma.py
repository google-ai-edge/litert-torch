# Copyright 2025 The LiteRT Torch Authors.
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
"""EmbeddingGemma-300M model implementation."""

import os

from litert_torch.generative.layers import attention
from litert_torch.generative.layers import attention_utils
from litert_torch.generative.layers import builder
from litert_torch.generative.layers import model_config as cfg
import safetensors.torch as safetensors_torch
import torch
from torch import nn


# Weight loading constants
# File names
_MODEL_SAFETENSORS = "model.safetensors"
_DENSE1_DIR = "2_Dense"
_DENSE2_DIR = "3_Dense"

# HuggingFace weight keys
_HF_EMBED_TOKENS = "embed_tokens.weight"
_HF_NORM = "norm.weight"
_HF_LAYERS_PREFIX = "layers"
_HF_INPUT_LAYERNORM = "input_layernorm.weight"
_HF_POST_ATTENTION_LAYERNORM = "post_attention_layernorm.weight"
_HF_PRE_FF_LAYERNORM = "pre_feedforward_layernorm.weight"
_HF_POST_FF_LAYERNORM = "post_feedforward_layernorm.weight"
_HF_ATTN_Q_PROJ = "self_attn.q_proj.weight"
_HF_ATTN_K_PROJ = "self_attn.k_proj.weight"
_HF_ATTN_V_PROJ = "self_attn.v_proj.weight"
_HF_ATTN_O_PROJ = "self_attn.o_proj.weight"
_HF_ATTN_Q_NORM = "self_attn.q_norm.weight"
_HF_ATTN_K_NORM = "self_attn.k_norm.weight"
_HF_MLP_GATE_PROJ = "mlp.gate_proj.weight"
_HF_MLP_UP_PROJ = "mlp.up_proj.weight"
_HF_MLP_DOWN_PROJ = "mlp.down_proj.weight"
_ST_LINEAR_WEIGHT = "linear.weight"

# LiteRT Torch weight keys
_AIET_EMBEDDER = "embedder.weight"
_AIET_FINAL_NORM = "final_norm.weight"
_AIET_BLOCKS_PREFIX = "transformer_blocks"
_AIET_PRE_ATTEN_NORM = "pre_atten_norm.weight"
_AIET_POST_ATTEN_NORM = "post_atten_norm.weight"
_AIET_FF_PRE_FF_NORM = "ff.pre_ff_norm.weight"
_AIET_FF_POST_FF_NORM = "ff.post_ff_norm.weight"
_AIET_ATTN_QKV_PROJ = "atten_func.qkv_projection.weight"
_AIET_ATTN_OUTPUT_PROJ = "atten_func.output_projection.weight"
_AIET_ATTN_Q_NORM = "atten_func.query_norm.weight"
_AIET_ATTN_K_NORM = "atten_func.key_norm.weight"
_AIET_MLP_W1 = "ff.w1.weight"
_AIET_MLP_W2 = "ff.w2.weight"
_AIET_MLP_W3 = "ff.w3.weight"
_AIET_DENSE1 = "dense1.weight"
_AIET_DENSE2 = "dense2.weight"


class EncoderBlock(attention.TransformerBlock):
  """EmbeddingGemma encoder block with proper 4-norm architecture."""

  def forward(
      self,
      x: torch.Tensor,
      rope: torch.Tensor | None = None,
      mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Forward with proper normalization order."""

    x_norm = self.pre_atten_norm(x)
    attn_out = self.atten_func(x_norm, rope, mask)
    attn_out = self.post_atten_norm(attn_out)
    x = x + attn_out
    final_output = x + self.ff(x)
    return final_output


class EmbeddingGemma(nn.Module):
  """EmbeddingGemma-300M model."""

  def __init__(self, config: cfg.ModelConfig, final_l2_norm: bool = True):
    super().__init__()
    self.config = config
    self.final_l2_norm = final_l2_norm

    # Token embeddings
    self.embedder = nn.Embedding(
        config.vocab_size, config.embedding_dim, padding_idx=0
    )

    # Transformer blocks
    self.transformer_blocks = nn.ModuleList([
        EncoderBlock(block_config, config)
        for block_config in config.block_configs
    ])

    # Final normalization
    self.final_norm = builder.build_norm(
        config.embedding_dim,
        config.final_norm_config,
    )

    # Dense projections (NO activation)
    self.dense1 = nn.Linear(
        config.embedding_dim, config.dense_intermediate_size, bias=False
    )
    self.dense2 = nn.Linear(
        config.dense_intermediate_size, config.embedding_dim, bias=False
    )

    # RoPE caches - DUAL for local (10k) and global (1M)
    self.rope_local = attention_utils.build_rope_cache(
        size=config.max_seq_len,
        dim=config.block_configs[0].attn_config.head_dim,
        base=config.block_configs[0].attn_config.rotary_base,
        dtype=torch.float32,
    )

    self.rope_global = attention_utils.build_rope_cache(
        size=config.max_seq_len,
        dim=config.block_configs[0].attn_config.head_dim,
        base=config.block_configs[5].attn_config.rotary_base,
        dtype=torch.float32,
    )

  def create_sliding_mask(
      self,
      segment_pos: torch.Tensor,  # [B, L]
      sequence_length: int,
      sliding_window_size: int,
  ) -> torch.Tensor:
    """Creates mask for sliding window attention (PyTorch)."""
    # Use torch.arange to create a tensor with a range of integers in a
    # Dynamo-friendly way.
    sequence_indices = torch.arange(sequence_length, dtype=torch.int32)
    sequence_indices = sequence_indices.view(
        1, 1, -1
    )  # [1, 1, sequence_length]
    segment_pos_expanded = segment_pos.clone().unsqueeze(-1)  # [B, seq_len, 1]

    # Create boolean masks for window boundaries.
    left_boundary = (
        sequence_indices >= segment_pos_expanded - sliding_window_size // 2
    )
    right_boundary = (
        sequence_indices <= segment_pos_expanded + sliding_window_size // 2
    )

    # Combine boolean masks (AND).
    sliding_mask_bool = left_boundary & right_boundary

    # Convert boolean mask to float mask with 0 and -inf using masked_fill.
    sliding_mask = torch.full(
        sliding_mask_bool.shape,
        float("-inf"),
        dtype=torch.float,
        device=sliding_mask_bool.device,
    )
    sliding_mask = sliding_mask.masked_fill(sliding_mask_bool, 0.0)

    return sliding_mask

  def mean_pool(self, hidden_states, attention_mask):
    """Mean pooling with attention mask."""
    if attention_mask is not None:
      input_mask_expanded = (
          attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
      )
      sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
      sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
      return sum_embeddings / sum_mask
    else:
      return torch.mean(hidden_states, dim=1)

  def forward(
      self,
      tokens: torch.Tensor,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Forward pass with layer-specific RoPE and attention."""

    batch_size, seq_len = tokens.shape

    if attention_mask is None:
      attention_mask = torch.ones(batch_size, seq_len, device=tokens.device)

    x = self.embedder(tokens) * self.config.embedding_scale

    # Process each layer
    for i, block in enumerate(self.transformer_blocks):
      # Select RoPE based on layer type
      if (
          self.config.block_configs[i].attn_config.attn_type
          == cfg.AttentionType.GLOBAL
      ):
        # Global: use 1M base
        rope_cos, rope_sin = self.rope_global
      else:  # "sliding_attention"
        # Local: use 10k base
        rope_cos, rope_sin = self.rope_local

      positions = torch.arange(
          0, seq_len, device=tokens.device, dtype=torch.int32
      )
      rope = (rope_cos[positions], rope_sin[positions])
      segment_pos = positions.unsqueeze(0).expand(batch_size, -1)

      # Build mask based on layer type
      # attention_mask: [B, seq_len] -> 1 for valid, 0 for padded
      padding_mask = torch.where(
          attention_mask == 0,
          torch.full_like(attention_mask, float("-inf"), dtype=torch.float),
          torch.zeros_like(attention_mask, dtype=torch.float),
      )  # Shape: [B, seq_len]
      # Shape: [B, 1, 1, seq_len]
      padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)

      if (
          self.config.block_configs[i].attn_config.attn_type
          == cfg.AttentionType.GLOBAL
      ):
        # Global: full bidirectional
        combined_mask = padding_mask_expanded
      else:
        # Local: sliding window
        sliding_mask = self.create_sliding_mask(
            segment_pos=segment_pos,
            sequence_length=seq_len,
            sliding_window_size=self.config.block_configs[
                i
            ].attn_config.sliding_window_size,
        )  # Shape: [B, seq_len, seq_len]
        expanded_sliding_mask = sliding_mask.unsqueeze(1)
        # Shape: [B, 1, seq_len, seq_len]

        combined_mask = expanded_sliding_mask + padding_mask_expanded
        # Shape: [B, 1, seq_len, seq_len]

      x = block(x, rope, combined_mask)

    # Final processing
    x = self.final_norm(x)
    pooled_x = self.mean_pool(x, attention_mask)

    # Dense projections (NO activation)
    pooled_x = self.dense1(pooled_x)
    pooled_x = self.dense2(pooled_x)

    # L2 normalization
    if self.final_l2_norm:
      embedding = torch.nn.functional.normalize(pooled_x, p=2, dim=1)
      return embedding

    return pooled_x


def get_model_config() -> cfg.ModelConfig:
  """Return model config for EmbeddingGemma-300M."""

  norm_config = cfg.NormalizationConfig(
      type=cfg.NormalizationType.RMS_NORM,
      epsilon=1e-6,
      zero_centered=True,
  )

  attn_config_local = cfg.AttentionConfig(
      num_heads=3,
      head_dim=256,
      num_query_groups=1,
      rotary_base=10000,
      rotary_percentage=1.0,
      query_norm_config=norm_config,
      key_norm_config=norm_config,
      sliding_window_size=512,
      attn_type=cfg.AttentionType.LOCAL_SLIDING,
      qkv_transpose_before_split=True,
  )

  attn_config_global = cfg.AttentionConfig(
      num_heads=3,
      head_dim=256,
      num_query_groups=1,
      rotary_base=1000000,
      rotary_percentage=1.0,
      query_norm_config=norm_config,
      key_norm_config=norm_config,
      attn_type=cfg.AttentionType.GLOBAL,
      qkv_transpose_before_split=True,
  )

  ff_config = cfg.FeedForwardConfig(
      type=cfg.FeedForwardType.GATED,
      activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
      intermediate_size=1152,
      pre_ff_norm_config=norm_config,
      post_ff_norm_config=norm_config,
  )

  block_config_local = cfg.TransformerBlockConfig(
      attn_config=attn_config_local,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
      parallel_residual=False,
  )

  block_config_global = cfg.TransformerBlockConfig(
      attn_config=attn_config_global,
      ff_config=ff_config,
      pre_attention_norm_config=norm_config,
      post_attention_norm_config=norm_config,
      parallel_residual=False,
  )

  # Create block configs: 5 local + 1 global, repeated
  block_configs = []
  for i in range(24):
    if (i + 1) % 6 == 0:
      block_configs.append(block_config_global)
    else:
      block_configs.append(block_config_local)

  embedding_dim = 768
  num_layers = 24
  config = cfg.ModelConfig(
      vocab_size=262144,
      num_layers=num_layers,
      max_seq_len=2048,
      embedding_dim=embedding_dim,
      embedding_scale=embedding_dim**0.5,
      block_configs=block_configs,
      final_norm_config=norm_config,
      dense_intermediate_size=3072,
  )

  return config


def build_model(checkpoint_path, final_l2_norm: bool = True) -> EmbeddingGemma:
  """Build model and load weights from HuggingFace checkpoint."""

  config = get_model_config()
  model = EmbeddingGemma(config, final_l2_norm)

  print(f"Loading from checkpoint: {checkpoint_path}")

  # Load main model weights
  model_file = os.path.join(checkpoint_path, _MODEL_SAFETENSORS)
  if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}")

  print(f"Loading {model_file}...")
  state_dict = safetensors_torch.load_file(model_file)
  new_state_dict = {}

  # Embeddings
  if _HF_EMBED_TOKENS in state_dict:
    new_state_dict[_AIET_EMBEDDER] = state_dict[_HF_EMBED_TOKENS]
    print("✓ Loaded embeddings")

  # Final norm
  if _HF_NORM in state_dict:
    new_state_dict[_AIET_FINAL_NORM] = state_dict[_HF_NORM]
    print("✓ Loaded final norm")

  # Transformer layers
  for i in range(24):
    layer_prefix = f"{_HF_LAYERS_PREFIX}.{i}"
    block_prefix = f"{_AIET_BLOCKS_PREFIX}.{i}"

    # Norms
    hf_key = f"{layer_prefix}.{_HF_INPUT_LAYERNORM}"
    aiet_key = f"{block_prefix}.{_AIET_PRE_ATTEN_NORM}"
    if hf_key in state_dict:
      new_state_dict[aiet_key] = state_dict[hf_key]

    hf_key = f"{layer_prefix}.{_HF_POST_ATTENTION_LAYERNORM}"
    aiet_key = f"{block_prefix}.{_AIET_POST_ATTEN_NORM}"
    if hf_key in state_dict:
      new_state_dict[aiet_key] = state_dict[hf_key]

    hf_key = f"{layer_prefix}.{_HF_PRE_FF_LAYERNORM}"
    aiet_key = f"{block_prefix}.{_AIET_FF_PRE_FF_NORM}"
    if hf_key in state_dict:
      new_state_dict[aiet_key] = state_dict[hf_key]

    hf_key = f"{layer_prefix}.{_HF_POST_FF_LAYERNORM}"
    aiet_key = f"{block_prefix}.{_AIET_FF_POST_FF_NORM}"
    if hf_key in state_dict:
      new_state_dict[aiet_key] = state_dict[hf_key]

    # Attention - QKV concatenated
    hf_q_key = f"{layer_prefix}.{_HF_ATTN_Q_PROJ}"
    if hf_q_key in state_dict:
      q = state_dict[hf_q_key]
      k = state_dict[f"{layer_prefix}.{_HF_ATTN_K_PROJ}"]
      v = state_dict[f"{layer_prefix}.{_HF_ATTN_V_PROJ}"]
      new_state_dict[f"{block_prefix}.{_AIET_ATTN_QKV_PROJ}"] = torch.cat(
          [q, k, v], dim=0
      )

    hf_key = f"{layer_prefix}.{_HF_ATTN_O_PROJ}"
    aiet_key = f"{block_prefix}.{_AIET_ATTN_OUTPUT_PROJ}"
    if hf_key in state_dict:
      new_state_dict[aiet_key] = state_dict[hf_key]

    # Q/K norms
    hf_key = f"{layer_prefix}.{_HF_ATTN_Q_NORM}"
    aiet_key = f"{block_prefix}.{_AIET_ATTN_Q_NORM}"
    if hf_key in state_dict:
      new_state_dict[aiet_key] = state_dict[hf_key]

    hf_key = f"{layer_prefix}.{_HF_ATTN_K_NORM}"
    aiet_key = f"{block_prefix}.{_AIET_ATTN_K_NORM}"
    if hf_key in state_dict:
      new_state_dict[aiet_key] = state_dict[hf_key]

    # MLP
    hf_key = f"{layer_prefix}.{_HF_MLP_GATE_PROJ}"
    aiet_key = f"{block_prefix}.{_AIET_MLP_W1}"
    if hf_key in state_dict:
      new_state_dict[aiet_key] = state_dict[hf_key]

    hf_key = f"{layer_prefix}.{_HF_MLP_UP_PROJ}"
    aiet_key = f"{block_prefix}.{_AIET_MLP_W3}"
    if hf_key in state_dict:
      new_state_dict[aiet_key] = state_dict[hf_key]

    hf_key = f"{layer_prefix}.{_HF_MLP_DOWN_PROJ}"
    aiet_key = f"{block_prefix}.{_AIET_MLP_W2}"
    if hf_key in state_dict:
      new_state_dict[aiet_key] = state_dict[hf_key]

  # Dense layers from Sentence-Transformers modules
  try:
    dense1_file = os.path.join(checkpoint_path, _DENSE1_DIR, _MODEL_SAFETENSORS)
    if os.path.exists(dense1_file):
      dense1_state = safetensors_torch.load_file(dense1_file)
      if _ST_LINEAR_WEIGHT in dense1_state:
        new_state_dict[_AIET_DENSE1] = dense1_state[_ST_LINEAR_WEIGHT]
        print("✓ Loaded dense1")
  except Exception as e:  # pylint: disable=broad-except
    print(f"Could not load dense1: {e}")

  try:
    dense2_file = os.path.join(checkpoint_path, _DENSE2_DIR, _MODEL_SAFETENSORS)
    if os.path.exists(dense2_file):
      dense2_state = safetensors_torch.load_file(dense2_file)
      if _ST_LINEAR_WEIGHT in dense2_state:
        new_state_dict[_AIET_DENSE2] = dense2_state[_ST_LINEAR_WEIGHT]
        print("✓ Loaded dense2")
  except Exception as e:  # pylint: disable=broad-except
    print(f"Could not load dense2: {e}")

  # Load into model
  print(f"Loading {len(new_state_dict)} parameters into model...")
  incompatible_keys = model.load_state_dict(new_state_dict, strict=False)

  if incompatible_keys.unexpected_keys:
    print(
        "\nWARNING: The following keys existed in new_state_dict but were not"
        " found in the model architecture. They were SKIPPED:"
    )
    for key in incompatible_keys.unexpected_keys:
      print(f"  - {key}")
  else:
    print("\n✓ All keys in new_state_dict matched parameters in the model.")

  if incompatible_keys.missing_keys:
    print(
        "\nWARNING: The following parameters exist in the model but were not"
        " found in new_state_dict. They were NOT loaded:"
    )
    for key in incompatible_keys.missing_keys:
      print(f"  - {key}")

  print("\n✓ Model loading complete!")

  return model
