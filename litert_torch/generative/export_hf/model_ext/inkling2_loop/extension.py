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
"""`ModelExtension` for Inkling-2-Mini (`inkling2_loop`, `inkling2_loop_text`).

Supports direct Converter V2 streaming from `thinkingmachines/Inkling-2-Mini`
safetensors checkpoints — including on-the-fly NVFP4 (layers 0-19) and MXFP8
(layers 20-23) expert dequantization, `w13_weight` de-interleaving, cross-pass
message query merging, `attn_res` +1 index offsetting, and `lm_head` vocab
unpadding — as well as pre-converted HF state dicts.
"""

from collections.abc import Callable, Mapping
import copy
import os
import re
from typing import Any

from litert_torch.generative.export_hf.model_ext import extension as extension_lib

# Registers the Inkling2LoopLiteRTLMCache implementation on import.
from litert_torch.generative.export_hf.model_ext.inkling2_loop import cache as _
from litert_torch.generative.export_hf.model_ext.inkling2_loop import exportable_module as inkling2_exportable
from litert_torch.generative.export_hf.model_ext.inkling2_loop import metadata_builder as inkling2_metadata
from litert_torch.generative.export_hf.model_ext.inkling2_loop import modeling_inkling2_loop_static
from litert_torch.generative.layers import moe as moe_lib
import torch
import transformers

_WRAPPER_PREFIXES = (
    "static_model.",
    "model.static_model.",
    "module.",
)

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_EXPERTS_PER_CHUNK = 8


def _strip_wrapper_prefixes(param_name: str) -> str:
  stripped = param_name
  while True:
    for prefix in _WRAPPER_PREFIXES:
      if stripped.startswith(prefix):
        stripped = stripped.removeprefix(prefix)
        break
    else:
      break
  return stripped


def dequantize_inkling2_loop_experts(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    amax: torch.Tensor | None,
    denominator: torch.Tensor | None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  """Dequantizes Inkling-2-Mini NVFP4 or MXFP8 expert weights.

  - NVFP4 (`amax` is not None): 2 E2M1 codes/byte (even in low nibble), one
    FP8 E4M3 scale per 16 values, and per-expert `amax / denominator`.
  - MXFP8 (`amax` is None and `scale` is not None): FP8 E4M3 weights with
    UE8M0 scale `2 ** (scale - 127)` per 32 values.

  Args:
    weight: Raw quantized (or already dense) expert tensor.
    scale: Block scale tensor, or None if already dequantized.
    amax: Per-expert amax tensor for NVFP4, or None for MXFP8.
    denominator: Outer scale denominator for NVFP4, or None for MXFP8.
    out_dtype: Target floating-point dtype.

  Returns:
    Dense expert weight tensor `[num_experts, rows, columns]` in `out_dtype`.
  """
  if scale is None:
    return weight.to(out_dtype)

  num_experts, rows = weight.shape[:2]
  columns = weight.shape[2] * 2 if amax is not None else weight.shape[2]
  output = torch.empty(
      (num_experts, rows, columns), dtype=out_dtype, device=weight.device
  )

  byte_to_values: torch.Tensor | None = None
  expert_factor: torch.Tensor | None = None
  if amax is not None:
    assert denominator is not None
    e2m1 = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=weight.device)
    byte_vals = torch.arange(256, device=weight.device)
    byte_to_values = torch.stack(
        (e2m1[byte_vals & 0xF], e2m1[byte_vals >> 4]), dim=-1
    )
    expert_factor = amax.float() / denominator.float()

  for start in range(0, num_experts, _EXPERTS_PER_CHUNK):
    experts = slice(start, start + _EXPERTS_PER_CHUNK)
    block_scale = scale[experts].float()
    if amax is None:
      values = weight[experts].float().view(*block_scale.shape, 32)
      values = values * torch.exp2(block_scale - 127.0)[..., None]
    else:
      assert byte_to_values is not None and expert_factor is not None
      codes = weight[experts].to(torch.uint8).long()
      values = (
          byte_to_values[codes].view(*block_scale.shape, 16)
          * block_scale[..., None]
      )
      values = values * expert_factor[experts, None, None, None]
    output[experts] = values.reshape(-1, rows, columns).to(out_dtype)
  return output


def deinterleave_dim1(tensor: torch.Tensor) -> torch.Tensor:
  """De-interleaves `w13_weight` along dim 1 into `[gate; up]` chunks."""
  shape = list(tensor.shape)
  shape[1:2] = [shape[1] // 2, 2]
  return (
      tensor.reshape(shape).transpose(1, 2).reshape(tensor.shape).contiguous()
  )


def merge_message_queries(
    queries: torch.Tensor, kv_queries: torch.Tensor
) -> torch.Tensor:
  """Merges `[2*L, D]` query/mlp and kv query tensors into `[L, 4, D]`."""
  q = queries.unflatten(0, (-1, 2))
  kv = kv_queries.unflatten(0, (-1, 2))
  return torch.cat([q[:, :1], kv, q[:, 1:]], dim=1).contiguous()


def _to_channels_last_conv_weight(tensor: torch.Tensor) -> torch.Tensor:
  """Converts depthwise conv1d weights (`[C, 1, K]` or `[C, K]`) to `[1, K, C]`."""
  if tensor.dim() == 2:
    return tensor.transpose(0, 1).unsqueeze(0).contiguous()
  if tensor.dim() == 3 and tensor.shape[1] == 1:
    return tensor.squeeze(1).transpose(0, 1).unsqueeze(0).contiguous()
  return tensor.contiguous()


_LAYER_PARAM_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_MSG_QUERY_RE = re.compile(r"^model\.message_input\.queries\.(\d+)$")
_CONV_WEIGHT_RE = re.compile(r"^(.+)\.conv_weight$")

_CONV_TAP_PREFIX_TO_RAW: dict[str, str] = {
    "self_attn.k_proj.sconv": "attn.k_sconv.weight",
    "self_attn.v_proj.sconv": "attn.v_sconv.weight",
    "post_attention_convolution": "attn_sconv.weight",
    "post_mlp_convolution": "mlp_sconv.weight",
}

_STATIC_TAP_PREFIX_TO_RAW: dict[str, str] = {
    "self_attn.k_proj.sconv": "attn.k_sconv.static_w",
    "self_attn.v_proj.sconv": "attn.v_sconv.static_w",
}

# Direct 1:1 suffix mappings from static model layer params -> raw checkpoint.
_LAYER_SUFFIX_TO_RAW: dict[str, str] = {
    "self_attn.q_proj.linear.weight": "attn.wq_du.weight",
    "self_attn.q_proj.norm.scale": "attn.q_l2_clip.scale",
    "self_attn.k_proj.linear.weight": "attn.wk_dv.weight",
    "self_attn.k_proj.sconv.weight_proj.weight": "attn.k_sconv.weight_proj",
    "self_attn.v_proj.linear.weight": "attn.wv_dv.weight",
    "self_attn.v_proj.sconv.weight_proj.weight": "attn.v_sconv.weight_proj",
    "self_attn.o_proj.linear.weight": "attn.wo_ud.weight",
    "self_attn.o_proj.gate_proj.weight": "attn.gate_attn.weight",
    "self_attn.sink_proj.weight": "attn.attn_sink_proj.weight",
    "self_attn.relative_position_bias.linear.weight": "attn.wr_du.weight",
    "self_attn.relative_position_bias.norm.scale": "attn.rel_l2_clip.scale",
    "input_layernorm.weight": "attn_norm.weight",
    "q_input_layernorm.weight": "q_input_norm.weight",
    "k_input_layernorm.weight": "k_input_norm.weight",
    "v_input_layernorm.weight": "v_input_norm.weight",
    "post_attention_layernorm.weight": "mlp_norm.weight",
    "mlp.gate.weight": "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias": "mlp.gate.bias",
    "q_message_gate.down_proj.weight": "controllers.attn.down_weight",
    "q_message_gate.forget_gate_proj.weight": "controllers.attn.forget_weight",
    "q_message_gate.message_gate_proj.weight": "controllers.attn.gain_weight",
    "k_message_gate.down_proj.weight": "controllers.k.down_weight",
    "k_message_gate.forget_gate_proj.weight": "controllers.k.forget_weight",
    "k_message_gate.message_gate_proj.weight": "controllers.k.gain_weight",
    "v_message_gate.down_proj.weight": "controllers.v.down_weight",
    "v_message_gate.forget_gate_proj.weight": "controllers.v.forget_weight",
    "v_message_gate.message_gate_proj.weight": "controllers.v.gain_weight",
    "mlp_message_gate.down_proj.weight": "controllers.mlp.down_weight",
    "mlp_message_gate.forget_gate_proj.weight": "controllers.mlp.forget_weight",
    "mlp_message_gate.message_gate_proj.weight": "controllers.mlp.gain_weight",
}


def _try_load_with_prefixes(
    try_load: Callable[[str], Any], raw_suffix: str
) -> torch.Tensor | None:
  for prefix in ("model.llm.", "model.language_model.", "model.", ""):
    tensor = try_load(f"{prefix}{raw_suffix}")
    if tensor is not None:
      return tensor
  return None


@extension_lib.register_model_extension(override=False)
class Inkling2LoopExtension(extension_lib.ModelExtension):
  """`ModelExtension` for Thinking Machines Inkling-2-Mini looped MoE."""

  model_types = modeling_inkling2_loop_static.SUPPORTED_MODEL_TYPES
  is_causal_text_model = True
  checkpoint_key_prefixes = ("model.llm.", "model.language_model.")

  def get_export_defaults(
      self,
      config_dict: Mapping[str, Any],
      task: Any = None,
      target: str = "cpu_gpu",
  ) -> extension_lib.ExportDefaults:
    del config_dict, task
    is_npu = str(target or "").startswith("npu")
    export_kwargs: dict[str, Any] = {
        "sampler_top_k": 40,
        "sampler_top_p": 0.95,
        "sampler_temperature": 0.6,
        "use_jinja_template": True,
        "cache_length": 4096,
        "enable_gpu_dynamic_cache": False,
        "apply_gpu_composites": True,
        "externalize_moe_weights": False,
    }
    if not is_npu:
      default_recipe = os.path.join(
          os.path.dirname(os.path.abspath(__file__)),
          "recipes",
          "recipe_w4b32_moe.json",
      )
      export_kwargs["quantization_recipe"] = (
          default_recipe
          if os.path.isfile(default_recipe)
          else "dynamic_wi4b32_moe_afp32"
      )
    return extension_lib.ExportDefaults(
        task=extension_lib.ExportTask.TEXT_GENERATION,
        export_kwargs=export_kwargs,
    )

  def load_config(
      self,
      model_path: str,
      export_config: Any = None,
      trust_remote_code: bool = False,
  ) -> transformers.PretrainedConfig | None:
    del export_config
    is_local = os.path.exists(str(model_path))
    config_dict, _ = transformers.PretrainedConfig.get_config_dict(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=is_local,
    )
    if not config_dict:
      return None
    raw = copy.deepcopy(config_dict)
    raw.pop("dtype", None)
    raw.pop("torch_dtype", None)
    orig_model_type = str(raw.get("model_type") or "inkling2_loop")

    if "text_config" in raw and isinstance(raw["text_config"], Mapping):
      text_raw = copy.deepcopy(dict(raw["text_config"]))
      text_raw.pop("dtype", None)
      text_raw.pop("torch_dtype", None)
      if "eos_token_id" in raw and "eos_token_id" not in text_raw:
        text_raw["eos_token_id"] = raw["eos_token_id"]
      text_cfg = modeling_inkling2_loop_static.Inkling2LoopTextConfig(
          **text_raw
      )
      cfg = modeling_inkling2_loop_static.Inkling2LoopConfig(
          text_config=text_cfg,
          vision_config=raw.get("vision_config"),
          image_token_id=int(raw.get("image_token_id", 200054)),
          eos_token_id=raw.get("eos_token_id", text_cfg.eos_token_id),
      )
      setattr(cfg, "model_type", orig_model_type)
      return cfg

    cfg = modeling_inkling2_loop_static.Inkling2LoopTextConfig(**raw)
    setattr(cfg, "model_type", orig_model_type)
    return cfg

  def update_export_config(
      self,
      export_config: Any,
      model_config: transformers.PretrainedConfig,
  ) -> Any:
    text_cfg = modeling_inkling2_loop_static.unwrap_config(model_config)
    export_config.cache_implementation = "Inkling2LoopLiteRTLMCache"
    # Keep `apply_gpu_composites` enabled by default so KV caches use in-place
    # `odml.cache_update`, attention uses `odml.runtime_bmm`, and all static
    # and dynamic short convolutions use in-place `odml.short_conv_step`
    # (activating `AllocationPolicy::kGpuOptimizedInplace` on GPU).
    export_config.use_sdpa_composite = False
    export_config.sdpa_use_composite = False
    export_config.use_rope_composite = False
    export_config.use_qkv_norm_rope_composite = False
    export_config.use_swiglu_composite = False
    export_config.use_bool_mask = False
    export_config.enable_gpu_dynamic_cache = False
    if getattr(export_config, "pad_token_id", None) is None:
      pad_id = getattr(text_cfg, "pad_token_id", None)
      export_config.pad_token_id = 0 if pad_id is None else int(pad_id)
    return export_config

  def get_hf_model_cls(
      self,
      task: Any,
      model_config: transformers.PretrainedConfig,
      export_config: Any,
  ) -> Any | None:
    del task, model_config, export_config
    return modeling_inkling2_loop_static.Inkling2LoopStaticForCausalLM

  def get_prefill_decode_exportables(
      self,
      model_config: transformers.PretrainedConfig,
      export_config: Any,
  ) -> tuple[Any, Any]:
    del model_config
    if getattr(export_config, "externalize_embedder", False):
      return (
          inkling2_exportable.LiteRTExportableModuleForInkling2LoopPrefillExternalEmbedder,
          inkling2_exportable.LiteRTExportableModuleForInkling2LoopGenerateExternalEmbedder,
      )
    return (
        inkling2_exportable.LiteRTExportableModuleForInkling2LoopPrefill,
        inkling2_exportable.LiteRTExportableModuleForInkling2LoopGenerate,
    )

  def build_llm_metadata(
      self,
      source_model_artifacts: Any,
      export_config: Any,
      exported_model_artifacts: Any,
      llm_metadata: Any,
  ) -> Any:
    return inkling2_metadata.build_inkling2_loop_llm_metadata(
        source_model_artifacts,
        export_config,
        exported_model_artifacts,
        llm_metadata,
    )

  def get_executor_metadata_builder(
      self,
      model_config: transformers.PretrainedConfig,
  ) -> Callable[..., Any]:
    del model_config
    return inkling2_metadata.build_inkling2_loop_executor_metadata

  def resolve_checkpoint_tensor(
      self,
      param_name: str,
      try_load: Callable[[str], Any],
      model_config: Any = None,
      export_config: Any = None,
  ) -> Any | None:
    """Resolves raw `thinkingmachines/Inkling-2-Mini` and HF checkpoints."""
    del export_config
    stripped = _strip_wrapper_prefixes(param_name)

    # 1. Exact key match (already HF-converted state dict).
    direct = try_load(stripped)
    if direct is not None:
      return direct

    text_cfg = (
        modeling_inkling2_loop_static.unwrap_config(model_config)
        if model_config is not None
        else None
    )

    # 2. Top-level non-layer parameters.
    if stripped == "lm_head.weight":
      raw = _try_load_with_prefixes(try_load, "unembed.weight")
      if raw is None:
        return None
      unpadded = int(
          getattr(text_cfg, "unpadded_vocab_size", raw.shape[0]) or raw.shape[0]
      )
      aligned = (
          modeling_inkling2_loop_static.aligned_lm_head_vocab_size(text_cfg)
          if text_cfg is not None
          else unpadded
      )
      out = raw[:aligned].clone().contiguous()
      if aligned > unpadded:
        # Keep padded rows near-zero (paired with -1e4 bias) with a strictly
        # positive per-channel scale so XNNPACK CPU accepts INT8 lm_head.
        out[unpadded:].zero_()
        out[unpadded:, 0] = 1e-4
      return out

    if stripped == "lm_head.bias":
      if text_cfg is None:
        return None
      unpadded = int(text_cfg.unpadded_vocab_size)
      aligned = modeling_inkling2_loop_static.aligned_lm_head_vocab_size(
          text_cfg
      )
      if aligned <= unpadded:
        return None
      bias = torch.zeros((aligned,), dtype=torch.float32)
      bias[unpadded:] = -1e4
      return bias

    if stripped == "model.embed_tokens.weight":
      return _try_load_with_prefixes(try_load, "embed.weight")

    if stripped == "model.embed_tokens.embed_norm.weight":
      return _try_load_with_prefixes(try_load, "embed_norm.weight")

    if stripped == "model.norm.weight":
      return _try_load_with_prefixes(try_load, "norm.weight")

    if stripped == "model.final_input.query":
      return _try_load_with_prefixes(try_load, "attn_res.output_query")

    if stripped == "model.message_input.all_queries":
      merged = try_load("model.message_input.query")
      if merged is None:
        q = _try_load_with_prefixes(try_load, "middle_cycle_cross_pass_queries")
        kv_q = _try_load_with_prefixes(
            try_load, "middle_cycle_cross_pass_kv_queries"
        )
        if q is not None and kv_q is not None:
          merged = merge_message_queries(q, kv_q)
      if merged is not None:
        return merged.reshape(-1, merged.shape[-1]).contiguous()
      return None

    msg_m = _MSG_QUERY_RE.match(stripped)
    if msg_m:
      loop_layer_idx = int(msg_m.group(1))
      merged = try_load("model.message_input.query")
      if merged is None:
        q = _try_load_with_prefixes(try_load, "middle_cycle_cross_pass_queries")
        kv_q = _try_load_with_prefixes(
            try_load, "middle_cycle_cross_pass_kv_queries"
        )
        if q is not None and kv_q is not None:
          merged = merge_message_queries(q, kv_q)
      if merged is not None:
        return merged[loop_layer_idx].contiguous()
      return None

    # 3. Per-layer parameters.
    m = _LAYER_PARAM_RE.match(stripped)
    if not m:
      return None
    layer_idx = int(m.group(1))
    suffix = m.group(2)

    if suffix == "attention_input.query":
      if layer_idx <= 0:
        return None
      return _try_load_with_prefixes(
          try_load, f"attn_res.attn_queries.{layer_idx - 1}"
      )

    if suffix == "mlp_input.query":
      return _try_load_with_prefixes(
          try_load, f"attn_res.mlp_queries.{layer_idx}"
      )

    if suffix in (
        "mlp.experts.flattened_gate_weight",
        "mlp.experts.flattened_ff1_weight",
    ):
      is_gate = suffix == "mlp.experts.flattened_gate_weight"
      # Fallback for pre-converted checkpoints with fused `gate_up_proj`.
      fused = _try_load_with_prefixes(
          try_load, f"layers.{layer_idx}.mlp.experts.gate_up_proj"
      )
      if fused is not None:
        gate_w, up_w = fused.chunk(2, dim=1)
        return moe_lib.flatten_expert_weight(
            (gate_w if is_gate else up_w).contiguous()
        )
      base = f"layers.{layer_idx}.mlp.experts.w13_weight"
      w = _try_load_with_prefixes(try_load, base)
      if w is None:
        return None
      scale = _try_load_with_prefixes(try_load, f"{base}_scale")
      amax = _try_load_with_prefixes(try_load, f"{base}_amax")
      denom = _try_load_with_prefixes(
          try_load, f"{base}_outer_scale_denominator"
      )
      row_slice = slice(0, None, 2) if is_gate else slice(1, None, 2)
      w_half = w[:, row_slice, :].contiguous()
      scale_half = (
          scale[:, row_slice, :].contiguous() if scale is not None else None
      )
      deq = dequantize_inkling2_loop_experts(
          w_half, scale_half, amax, denom, out_dtype=torch.bfloat16
      )
      return moe_lib.flatten_expert_weight(deq)

    if suffix == "mlp.experts.flattened_linear_weight":
      direct_down = _try_load_with_prefixes(
          try_load, f"layers.{layer_idx}.mlp.experts.down_proj"
      )
      if direct_down is not None:
        return moe_lib.flatten_expert_weight(direct_down.contiguous())
      base = f"layers.{layer_idx}.mlp.experts.w2_weight"
      w = _try_load_with_prefixes(try_load, base)
      if w is None:
        return None
      scale = _try_load_with_prefixes(try_load, f"{base}_scale")
      amax = _try_load_with_prefixes(try_load, f"{base}_amax")
      denom = _try_load_with_prefixes(
          try_load, f"{base}_outer_scale_denominator"
      )
      deq = dequantize_inkling2_loop_experts(
          w, scale, amax, denom, out_dtype=torch.bfloat16
      )
      return moe_lib.flatten_expert_weight(deq)

    if suffix == "self_attn.q_o_rel_sink_proj.weight":
      parts = []
      for raw_key, hf_key in (
          ("attn.wq_du.weight", "self_attn.q_proj.linear.weight"),
          ("attn.gate_attn.weight", "self_attn.o_proj.gate_proj.weight"),
          (
              "attn.wr_du.weight",
              "self_attn.relative_position_bias.linear.weight",
          ),
          ("attn.attn_sink_proj.weight", "self_attn.sink_proj.weight"),
      ):
        t = _try_load_with_prefixes(try_load, f"layers.{layer_idx}.{raw_key}")
        if t is None:
          t = try_load(f"model.layers.{layer_idx}.{hf_key}")
        if t is None:
          return None
        parts.append(t)
      return torch.cat(parts, dim=0).contiguous()

    if suffix in (
        "self_attn.k_proj.fused_linear.weight",
        "self_attn.v_proj.fused_linear.weight",
    ):
      kv = "k" if suffix.startswith("self_attn.k_proj.") else "v"
      w_lin = _try_load_with_prefixes(
          try_load, f"layers.{layer_idx}.attn.w{kv}_dv.weight"
      )
      if w_lin is None:
        w_lin = try_load(
            f"model.layers.{layer_idx}.self_attn.{kv}_proj.linear.weight"
        )
      w_dyn = _try_load_with_prefixes(
          try_load, f"layers.{layer_idx}.attn.{kv}_sconv.weight_proj"
      )
      if w_dyn is None:
        w_dyn = try_load(
            f"model.layers.{layer_idx}.self_attn.{kv}_proj.sconv.weight_proj.weight"
        )
      if w_lin is not None and w_dyn is not None:
        return torch.cat([w_lin, w_dyn], dim=0).contiguous()
      return None

    gate_ctrl_map = {
        "q_message_gate.gate_proj.weight": ("attn", "q_message_gate"),
        "k_message_gate.gate_proj.weight": ("k", "k_message_gate"),
        "v_message_gate.gate_proj.weight": ("v", "v_message_gate"),
        "mlp_message_gate.gate_proj.weight": ("mlp", "mlp_message_gate"),
    }
    if suffix in gate_ctrl_map:
      ctrl_name, hf_gate = gate_ctrl_map[suffix]
      w_f = _try_load_with_prefixes(
          try_load, f"layers.{layer_idx}.controllers.{ctrl_name}.forget_weight"
      )
      if w_f is None:
        w_f = try_load(
            f"model.layers.{layer_idx}.{hf_gate}.forget_gate_proj.weight"
        )
      w_m = _try_load_with_prefixes(
          try_load, f"layers.{layer_idx}.controllers.{ctrl_name}.gain_weight"
      )
      if w_m is None:
        w_m = try_load(
            f"model.layers.{layer_idx}.{hf_gate}.message_gate_proj.weight"
        )
      if w_f is not None and w_m is not None:
        return torch.cat([w_f, w_m], dim=0).contiguous()
      return None

    if suffix == "self_attn.relative_position_bias.proj_t":
      raw = _try_load_with_prefixes(
          try_load, f"layers.{layer_idx}.attn.rel_logits_proj.proj"
      )
      if raw is None:
        raw = try_load(
            f"model.layers.{layer_idx}.self_attn.relative_position_bias.proj"
        )
      if raw is not None:
        return raw.transpose(0, 1).contiguous()
      return None

    conv_w_m = _CONV_WEIGHT_RE.match(suffix)
    if conv_w_m:
      conv_prefix = conv_w_m.group(1)
      # Dynamic short-conv (full-attention K/V): checkpoint `static_w` is
      # `[K, C]` indexed by lag (newest -> oldest); flip to `[1, K, C]`
      # oldest -> newest for `odml.short_conv_step`.
      raw_static_key = _STATIC_TAP_PREFIX_TO_RAW.get(conv_prefix)
      if raw_static_key is not None:
        tensor = _try_load_with_prefixes(
            try_load, f"layers.{layer_idx}.{raw_static_key}"
        )
        if tensor is None:
          tensor = try_load(
              f"model.layers.{layer_idx}.{conv_prefix}.static_weight"
          )
        if tensor is not None:
          return tensor.flip(0).unsqueeze(0).contiguous()
      # Static short-conv (sliding K/V, post-attn, post-MLP): `[C, 1, K]` ->
      # `[1, K, C]` oldest -> newest.
      raw_conv_key = _CONV_TAP_PREFIX_TO_RAW.get(conv_prefix)
      tensor = None
      if raw_conv_key is not None:
        tensor = _try_load_with_prefixes(
            try_load, f"layers.{layer_idx}.{raw_conv_key}"
        )
      if tensor is None:
        tensor = try_load(
            f"model.layers.{layer_idx}.{conv_prefix}.conv1d.weight"
        )
      if tensor is not None:
        return _to_channels_last_conv_weight(tensor)
      return None

    raw_suffix = _LAYER_SUFFIX_TO_RAW.get(suffix)
    if raw_suffix is not None:
      return _try_load_with_prefixes(
          try_load, f"layers.{layer_idx}.{raw_suffix}"
      )

    return None
