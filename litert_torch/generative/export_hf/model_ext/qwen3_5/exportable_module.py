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

"""Qwen 3.5 exportable modules for LiteRT-LM using Full Model Reauthoring."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import attention as _
from litert_torch.generative.export_hf.core.split_cache import attention as _
from litert_torch.generative.export_hf.core.split_cache import exportable_module as split_cache_module
from litert_torch.generative.export_hf.model_ext.qwen3_5.modeling_qwen3_5_static import Qwen3_5StaticForCausalLM


def create_qwen3_5_attention_mask(
    seq_len: int,
    cache_length: int,
    input_pos: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Creates an attention mask of shape (1, 1, seq_len, cache_length) where valid positions are 0.0 (or True) and future/padding positions are -1e38 (or False)."""
    cache_positions = torch.arange(cache_length, device=device).view(1, 1, 1, cache_length)
    q_pos = input_pos.view(1, 1, seq_len, 1).to(device=device)
    causal_bool = cache_positions <= q_pos
    if dtype == torch.bool:
        return causal_bool
    if not dtype.is_floating_point:
        dtype = torch.float32
    return torch.where(
        causal_bool,
        torch.zeros((1,), dtype=dtype, device=device),
        torch.full((1,), -1e38, dtype=dtype, device=device),
    )


class Qwen3_5StaticModelHFWrapper(nn.Module):
    """HF-compatible wrapper around Qwen3_5StaticForCausalLM that bridges dynamic cache objects to static state tensors while preserving Hugging Face model metadata."""
    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def __init__(self, hf_model: Any):
        super().__init__()
        if isinstance(hf_model, Qwen3_5StaticForCausalLM):
            self.static_model = hf_model
            self.original_hf_model = None
            cfg = hf_model.config
        else:
            self.original_hf_model = hf_model
            self.static_model = Qwen3_5StaticForCausalLM.from_hf_model(hf_model)
            cfg = self.static_model.config
        self.config = getattr(cfg, "text_config", cfg)
        if isinstance(self.config, dict) or self.config is None:
            self.config = cfg

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

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "original_hf_model" in self.__dict__ and self.original_hf_model is not None and hasattr(self.original_hf_model, name):
                return getattr(self.original_hf_model, name)
            if "static_model" in self.__dict__ and hasattr(self.static_model, name):
                return getattr(self.static_model, name)
            raise

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        tokens = input_ids if input_ids is not None else inputs_embeds
        if tokens is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)

        positions = cache_position if cache_position is not None else position_ids
        if positions is None:
            positions = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.int64)
        if positions.ndim == 2 and positions.shape[0] == 1:
            positions = positions.squeeze(0)

        if valid_mask is None and tokens.shape[1] > 1 and input_ids is not None:
            pad_token_id = getattr(self.config, "pad_token_id", None)
            if pad_token_id is not None and pad_token_id >= 0:
                valid_mask = (tokens != pad_token_id)
        if valid_mask is not None and valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)

        merged_kwargs = dict(kwargs)
        if attention_mask is not None:
            merged_kwargs["attention_mask"] = attention_mask

        logits, past_key_values = self.static_model(
            tokens, positions, past_key_values=past_key_values, valid_mask=valid_mask, **merged_kwargs
        )
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


class Qwen3_5ExportableMixin:
    """Mixin that wraps the source HF model with Qwen3_5StaticModelHFWrapper and sets up the attention implementation."""
    model: Any

    def __init__(self, model: torch.nn.Module, export_config: Any, source_model_artifacts: Any = None):
        if not isinstance(model, Qwen3_5StaticModelHFWrapper):
            model = Qwen3_5StaticModelHFWrapper(model)
        super().__init__(model, export_config, source_model_artifacts)  # pytype: disable=wrong-arg-count
        if getattr(export_config, "split_cache", False) or getattr(export_config, "cache_implementation", None) == "SplitCache":
            self.model.set_attn_implementation("lrt_split_cache_attention")
        elif getattr(export_config, "cache_implementation", None) == "LiteRTLMCache":
            self.model.set_attn_implementation("lrt_transposed_attention")

    def _update_sample_masks(self, sample_dict: Dict[str, Any]) -> Dict[str, Any]:
        for sample_list in sample_dict.values():
            for sample in sample_list:
                if "mask" in sample and "input_pos" in sample:
                    seq_len = sample["tokens"].shape[1]
                    cache_len = sample["mask"].shape[-1]
                    mask_dtype = sample["mask"].dtype if hasattr(sample["mask"], "dtype") else torch.float32
                    sample["mask"] = create_qwen3_5_attention_mask(
                        seq_len, cache_len, sample["input_pos"], mask_dtype, sample["tokens"].device
                    )
        return sample_dict


class LiteRTExportableModuleForQwen3_5Prefill(Qwen3_5ExportableMixin, exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill):
    def get_sample_inputs(self, model_config: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._update_sample_masks(super().get_sample_inputs(model_config, **kwargs))


class LiteRTExportableModuleForQwen3_5Generate(Qwen3_5ExportableMixin, exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate):
    def get_sample_inputs(self, model_config: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._update_sample_masks(super().get_sample_inputs(model_config, **kwargs))


class LiteRTSplitCacheExportableModuleForQwen3_5Prefill(Qwen3_5ExportableMixin, split_cache_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill):
    def get_sample_inputs(self, model_config: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._update_sample_masks(super().get_sample_inputs(model_config, **kwargs))


class LiteRTSplitCacheExportableModuleForQwen3_5Generate(Qwen3_5ExportableMixin, split_cache_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate):
    def get_sample_inputs(self, model_config: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._update_sample_masks(super().get_sample_inputs(model_config, **kwargs))
