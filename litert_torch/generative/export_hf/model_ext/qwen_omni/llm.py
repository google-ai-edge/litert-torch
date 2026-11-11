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
"""Thinker LLM decode exportable for Qwen2.5-Omni.

Prefill reuses `LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder`
directly on upstream HF `thinker.model` (`Qwen2_5OmniThinkerTextModel`), and the
text embedder reuses `LiteRTExportableModuleForEmbedder` directly on
`thinker.get_input_embeddings()`. Only decode needs a thin wrapper here.
"""

from __future__ import annotations

from typing import Any

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.external_emb import exportable_module as ext_emb
import torch
from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as hf_qwen25


def ensure_mrope_section(cfg: Any) -> None:
  """Populates `cfg.rope_parameters['mrope_section']` when absent.

  Upstream `Qwen2_5OmniAttention.forward()` indexes
  `config.rope_parameters['mrope_section']`, which real checkpoints define but
  default `Qwen2_5OmniConfig()` leaves unset.

  Args:
    cfg: HF Thinker text or Talker config to patch in-place.
  """
  rope_params = getattr(cfg, "rope_parameters", None)
  if isinstance(rope_params, dict) and "mrope_section" not in rope_params:
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    half = head_dim // 2
    third = half // 3
    rope_params["mrope_section"] = [third, third, half - 2 * third]


class QwenOmniThinkerDecodeModule(
    ext_emb.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder
):
  """Decode exportable for Thinker (`Qwen2_5OmniThinkerTextModel` + `lm_head`).

  Why this wrapper and custom `forward()` are necessary instead of passing
  upstream `Qwen2_5OmniThinkerForConditionalGeneration` directly to the core
  generate module:
  1. `Qwen2_5OmniThinkerForConditionalGeneration` bundles vision/audio towers
     and only exposes final hidden states via HF `@capture_outputs` hooks
     across every decoder layer. Calling `thinker.model` + `thinker.lm_head`
     directly keeps vision/audio submodules out of the LLM graph and reads
     `output.last_hidden_state` without hook overhead.
  2. When `generate_audio=True`, C++ `QwenOmniLlmStage` (`SplitDecodeOutput`)
     requires `logits` to pack `[logits, last_hidden_state]`
     (`vocab_size + hidden_size` floats) so Talker receives Thinker's final
     hidden state over LiteRT-LM's standard single-`logits` decode interface.
  """

  def __init__(
      self,
      thinker: hf_qwen25.Qwen2_5OmniThinkerForConditionalGeneration,
      export_config: exportable_module_config.ExportableModuleConfig,
      *,
      include_hidden_states: bool = True,
  ):
    super().__init__(thinker.model, export_config)
    self.lm_head = thinker.lm_head
    self._include_hidden_states = include_hidden_states

  # pylint: disable=arguments-renamed
  def forward(
      self,
      embeddings,
      input_pos,
      kv_cache,
      mask,
      local_mask=None,
      **kwargs,
  ):
    kwargs.update(self.composite_flags())
    inputs = self.adapt_inputs(
        None,
        embeddings,
        input_pos,
        kv_cache,
        mask,
        local_mask=local_mask,
        use_bool_mask=self.export_config.extra_kwargs.get(
            "use_bool_mask", False
        ),
        **kwargs,
    )
    inputs |= self.attention_kwargs()
    output = self.model(**inputs)
    hidden = output.last_hidden_state
    logits = self.lm_head(hidden).to(torch.float32)
    if self._include_hidden_states:
      logits = torch.cat([logits, hidden.to(torch.float32)], dim=-1)
    return {"kv_cache": output.past_key_values, "logits": logits}
