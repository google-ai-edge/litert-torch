# Copyright 2026 The LiteRT Torch Authors.
# Copyright 2026 Recursive and the HuggingFace Inc. team. All rights reserved.
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
"""Flint model configuration."""

from transformers import configuration_utils


class FlintConfig(configuration_utils.PreTrainedConfig):
  """Configuration class for the RecursiveSI/flint MoE model."""

  model_type = "flint"
  keys_to_ignore_at_inference = ["past_key_values"]
  base_model_tp_plan = {
      "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
      "layers.*.mlp.experts.down_proj": "rowwise",
      "layers.*.mlp.experts": "moe_tp_experts",
      "layers.*.mlp.shared_expert.gate_proj": "colwise",
      "layers.*.mlp.shared_expert.up_proj": "colwise",
      "layers.*.mlp.shared_expert.down_proj": "rowwise",
      "layers.*.mlp.gate_proj": "colwise",
      "layers.*.mlp.up_proj": "colwise",
      "layers.*.mlp.down_proj": "rowwise",
  }
  base_model_pp_plan = {
      "embed_tokens": (["input_ids"], ["inputs_embeds"]),
      "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
      "norm": (["hidden_states"], ["hidden_states"]),
  }
  base_model_ep_plan = {
      "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
      "layers.*.mlp.experts.down_proj": "grouped_gemm",
      "layers.*.mlp.experts": "ep_dispatch_experts",
  }

  def __init__(
      self,
      vocab_size: int = 150144,
      hidden_size: int = 2048,
      intermediate_size: int = 6144,
      num_hidden_layers: int = 49,
      num_attention_heads: int = 32,
      num_key_value_heads: int = 4,
      hidden_act: str = "silu",
      max_position_embeddings: int = 262144,
      initializer_range: float = 0.02,
      rms_norm_eps: float = 1e-6,
      use_cache: bool = True,
      tie_word_embeddings: bool = False,
      rope_theta: float = 10000.0,
      rope_parameters: dict[str, object] | None = None,
      sliding_window: int = 513,
      attention_dropout: float = 0.0,
      decoder_sparse_step: int = 1,
      moe_intermediate_size: int = 512,
      shared_expert_intermediate_size: int = 1024,
      num_experts_per_tok: int = 6,
      num_experts: int = 256,
      norm_topk_prob: bool = True,
      output_router_logits: bool = False,
      router_aux_loss_coef: float = 0.0005,
      mlp_only_layers: list[int] | None = None,
      layer_types: list[str] | None = None,
      pad_token_id: int | None = None,
      bos_token_id: int | None = None,
      eos_token_id: int | list[int] | None = None,
      head_dim: int = 128,
      attention_bias: bool = False,
      **kwargs,
  ):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.hidden_act = hidden_act
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.rms_norm_eps = rms_norm_eps
    self.use_cache = use_cache
    self.rope_theta = rope_theta
    self.rope_parameters = rope_parameters
    self.sliding_window = sliding_window
    self.attention_dropout = attention_dropout
    self.decoder_sparse_step = decoder_sparse_step
    self.moe_intermediate_size = moe_intermediate_size
    self.shared_expert_intermediate_size = shared_expert_intermediate_size
    self.num_experts_per_tok = num_experts_per_tok
    self.num_experts = num_experts
    self.norm_topk_prob = norm_topk_prob
    self.output_router_logits = output_router_logits
    self.router_aux_loss_coef = router_aux_loss_coef
    self.head_dim = head_dim
    self.attention_bias = attention_bias

    if layer_types is None:
      layer_types = [
          "full_attention"
          if i < 2 or i % 4 == 1 or i == num_hidden_layers - 1
          else "sliding_attention"
          for i in range(num_hidden_layers)
      ]
    self.layer_types = layer_types

    if mlp_only_layers is None:
      mlp_only_layers = [0, 1]
    self.mlp_only_layers = mlp_only_layers

    super().__init__(
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tie_word_embeddings=tie_word_embeddings,
        **kwargs,
    )
    if self.rope_parameters is None:
      self.rope_parameters = {
          "rope_type": "default",
          "rope_theta": float(getattr(self, "rope_theta", 10000.0)),
      }


__all__ = ["FlintConfig"]
