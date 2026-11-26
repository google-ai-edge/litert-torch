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
"""Prefill and decode exportable modules for Inkling-2-Mini."""

from typing import Any, cast

from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
from litert_torch.generative.export_hf.model_ext.inkling2_loop import cache as inkling2_cache
from litert_torch.generative.export_hf.model_ext.inkling2_loop import modeling_inkling2_loop_static
import torch
from torch import nn
from torch.utils import _pytree as pytree
from transformers import modeling_outputs


def create_causal_mask(
    seq_len: int,
    cache_length: int,
    input_pos: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
  """Creates fp16-safe causal attention mask `(1, 1, seq_len, cache_length)`."""
  cache_positions = torch.arange(cache_length, device=device).view(
      1, 1, 1, cache_length
  )
  q_pos = input_pos.view(1, 1, seq_len, 1).to(device=device)
  causal_bool = cache_positions <= q_pos
  if dtype == torch.bool:
    return causal_bool
  if not dtype.is_floating_point:
    dtype = torch.float32
  return torch.where(
      causal_bool,
      torch.zeros((1,), dtype=dtype, device=device),
      torch.full((1,), -1e4, dtype=dtype, device=device),
  )


class Inkling2LoopStaticModelHFWrapper(nn.Module):
  """HF-compatible wrapper bridging `Inkling2LoopStaticForCausalLM` to export."""

  _can_compile_fullgraph = True
  _supports_attention_backend = True

  def __init__(self, hf_model: Any):
    super().__init__()
    self.static_model = hf_model
    self.config = modeling_inkling2_loop_static.unwrap_config(hf_model.config)

  def get_input_embeddings(self) -> Any:
    return self.static_model.get_input_embeddings()

  def set_input_embeddings(self, value: Any) -> None:
    self.static_model.set_input_embeddings(value)

  def get_output_embeddings(self) -> Any:
    return self.static_model.get_output_embeddings()

  def set_output_embeddings(self, new_embeddings: Any) -> None:
    self.static_model.set_output_embeddings(new_embeddings)

  def set_attn_implementation(self, implementation: str) -> None:
    self.static_model.set_attn_implementation(implementation)

  def forward(
      self,
      input_ids: torch.Tensor | None = None,
      inputs_embeds: torch.Tensor | None = None,
      position_ids: torch.Tensor | None = None,
      past_key_values: Any = None,
      cache_position: torch.Tensor | None = None,
      attention_mask: torch.Tensor | dict[str, Any] | None = None,
      valid_mask: torch.Tensor | None = None,
      **kwargs: Any,
  ) -> modeling_outputs.CausalLMOutputWithPast:
    tokens = input_ids if input_ids is not None else inputs_embeds
    if tokens is None:
      raise ValueError("Either input_ids or inputs_embeds must be provided.")
    if tokens.ndim == 1:
      tokens = tokens.unsqueeze(0)
    if tokens.dtype in (torch.int64, torch.int32):
      tokens = tokens.to(torch.int32)

    positions = position_ids if position_ids is not None else cache_position
    if positions is None:
      positions = torch.arange(
          tokens.shape[1], device=tokens.device, dtype=torch.int32
      ).unsqueeze(0)
    positions = positions.to(torch.int32)
    if positions.ndim == 1:
      positions = positions.unsqueeze(0)

    # Always derive valid_mask from positions for padded prefill chunks so
    # short conv states ignore trailing zero-position padding slots even when
    # core exportable_module passes a token-based mask.
    if tokens.shape[1] > 1:
      seq_idx = torch.arange(
          tokens.shape[1], device=positions.device, dtype=positions.dtype
      )
      valid_mask = ((positions > 0) | (seq_idx == 0)).to(torch.float32)
    elif valid_mask is not None and valid_mask.ndim == 1:
      valid_mask = valid_mask.unsqueeze(0)

    if (
        past_key_values is not None
        and hasattr(past_key_values, "set_cache_runtime_args")
        and getattr(past_key_values, "layers", None)
    ):
      existing_args = past_key_values.layers[0].get_cache_runtime_args()
      if not existing_args or "cache_position" not in existing_args:
        pos_1d = positions[0] if positions.ndim == 2 else positions.view(-1)
        past_key_values.set_cache_runtime_args({"cache_position": pos_1d})

    merged_kwargs = dict(kwargs)
    if attention_mask is not None:
      merged_kwargs["attention_mask"] = attention_mask
    if past_key_values is not None and getattr(past_key_values, "layers", None):
      layer0 = past_key_values.layers[0]
      if hasattr(layer0, "get_k_ts_idx"):
        merged_kwargs.setdefault("k_ts_idx", layer0.get_k_ts_idx())
        merged_kwargs.setdefault("v_ts_idx", layer0.get_v_ts_idx())

    if inputs_embeds is not None:
      embed_dtype = self.static_model.get_input_embeddings().weight.dtype
      logits, past_key_values = self.static_model(
          None,
          positions,
          past_key_values=past_key_values,
          valid_mask=valid_mask,
          inputs_embeds=inputs_embeds.to(embed_dtype),
          **merged_kwargs,
      )
    else:
      logits, past_key_values = self.static_model(
          tokens,
          positions,
          past_key_values=past_key_values,
          valid_mask=valid_mask,
          **merged_kwargs,
      )
    return modeling_outputs.CausalLMOutputWithPast(
        logits=logits, past_key_values=past_key_values
    )


class Inkling2LoopExportableMixin:
  """Mixin configuring wrapper, 48-layer looped cache, and fp16-safe masks."""

  model: Any
  export_config: Any
  device: Any

  def __init__(
      self,
      model: nn.Module,
      export_config: Any,
      source_model_artifacts: Any = None,
  ):
    if not isinstance(model, Inkling2LoopStaticModelHFWrapper):
      model = Inkling2LoopStaticModelHFWrapper(model)
    cast(Any, super()).__init__(model, export_config, source_model_artifacts)

  def get_sample_kv_cache(
      self, model_config: Any
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Builds sample KV and conv-state cache inputs and dynamic shapes."""
    export_config = self.export_config
    kv_cache = inkling2_cache.Inkling2LoopLiteRTLMCache.create_from_config(
        model_config,
        export_config,
        batch_size=export_config.batch_size,
        cache_length=export_config.cache_length,
    ).to(self.device)
    inputs = {"kv_cache": kv_cache}
    flat_tensors, _ = pytree.tree_flatten(kv_cache)
    if export_config.cache_length_dim is None:
      return inputs, {"kv_cache": [None] * len(flat_tensors)}
    flat_shapes: list[dict[int, Any] | None] = []
    for layer in kv_cache.layers:
      k_ts = int(getattr(layer, "k_ts_idx", 2))
      v_ts = int(getattr(layer, "v_ts_idx", 3))
      flat_shapes.append({k_ts: export_config.cache_length_dim})
      flat_shapes.append({v_ts: export_config.cache_length_dim})
      flat_shapes.extend([None] * len(getattr(layer, "conv_states", ())))
    return inputs, {"kv_cache": flat_shapes}

  def _update_sample_masks(self, sample_dict: dict[str, Any]) -> dict[str, Any]:
    """Replaces sample causal masks using `create_causal_mask`."""
    for sample_val in sample_dict.values():
      samples = sample_val if isinstance(sample_val, list) else [sample_val]
      for entry in samples:
        sample = entry[0] if isinstance(entry, tuple) else entry
        if not isinstance(sample, dict):
          continue
        if "mask" in sample and "input_pos" in sample:
          if "tokens" in sample:
            seq_len = sample["tokens"].shape[1]
            device = sample["tokens"].device
          elif "embeddings" in sample:
            seq_len = sample["embeddings"].shape[1]
            device = sample["embeddings"].device
          else:
            seq_len = sample["input_pos"].shape[0]
            device = sample["input_pos"].device
          mask_val = sample["mask"]
          if isinstance(mask_val, torch.Tensor):
            cache_len = mask_val.shape[-1]
            sample["mask"] = create_causal_mask(
                seq_len, cache_len, sample["input_pos"], mask_val.dtype, device
            )
    return sample_dict


class LiteRTExportableModuleForInkling2LoopPrefill(
    Inkling2LoopExportableMixin,
    exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill,
):
  """Bundled-embedder prefill exportable module for Inkling-2-Mini."""

  def get_sample_inputs(
      self, model_config: Any, **kwargs: Any
  ) -> dict[str, Any]:
    return self._update_sample_masks(
        super().get_sample_inputs(model_config, **kwargs)
    )


class LiteRTExportableModuleForInkling2LoopGenerate(
    Inkling2LoopExportableMixin,
    exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate,
):
  """Bundled-embedder decode exportable module for Inkling-2-Mini."""

  def get_sample_inputs(
      self, model_config: Any, **kwargs: Any
  ) -> dict[str, Any]:
    return self._update_sample_masks(
        super().get_sample_inputs(model_config, **kwargs)
    )


class LiteRTExportableModuleForInkling2LoopPrefillExternalEmbedder(
    Inkling2LoopExportableMixin,
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder,
):
  """External-embedder prefill exportable module for Inkling-2-Mini."""

  def get_sample_inputs(
      self, model_config: Any, **kwargs: Any
  ) -> dict[str, Any]:
    return self._update_sample_masks(
        super().get_sample_inputs(model_config, **kwargs)
    )


class LiteRTExportableModuleForInkling2LoopGenerateExternalEmbedder(
    Inkling2LoopExportableMixin,
    external_emb_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder,
):
  """External-embedder decode exportable module for Inkling-2-Mini."""

  def get_sample_inputs(
      self, model_config: Any, **kwargs: Any
  ) -> dict[str, Any]:
    return self._update_sample_masks(
        super().get_sample_inputs(model_config, **kwargs)
    )
