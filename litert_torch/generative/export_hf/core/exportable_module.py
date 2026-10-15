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
"""Exportable modules."""

import abc

from litert_torch.generative.export_hf.core import attention as _
from litert_torch.generative.export_hf.core import cache as cache_lib
from litert_torch.generative.export_hf.core import cache_base as kv_cache_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import lora as lora_lib
from litert_torch.generative.export_hf.core import utils
from litert_torch.generative.export_hf.core.sliding_window import attention_mask as sliding_window_attention_mask
from litert_torch.generative.export_hf.core.split_cache import attention as _
import torch

ExportableModuleConfig = exportable_module_config.ExportableModuleConfig

# Model keys (`config.model_type` or `export_config.model` substrings) whose
# text-generation global attention mask is guaranteed to be purely causal.
# TODO(b/552147487): Gate this via model-specific export configs/extensions
# rather than an explicit model key allowlist.
_CAUSAL_TEXT_MODEL_KEYS = frozenset({
    "gemma2",
    "gemma3_text",
    "gemma4",
    "gemma4_text",
    "gemma4_tiny",
    "lfm2",
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3_text",
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
})


class ExportableModuleBase(torch.nn.Module, abc.ABC):
  """Base class for exportable modules."""

  def __init__(self, export_config: ExportableModuleConfig):
    super().__init__()
    self._export_config = export_config

  @property
  def export_config(self) -> ExportableModuleConfig:
    return self._export_config

  def attention_kwargs(self):
    k_ts_idx = self.export_config.k_ts_idx
    v_ts_idx = self.export_config.v_ts_idx
    return {
        "k_ts_idx": k_ts_idx,
        "v_ts_idx": v_ts_idx,
        "sdpa_is_causal": self._is_text_only_causal_export(),
    }

  def _is_text_only_causal_export(self) -> bool:
    """Whether the global attention mask is guaranteed to be purely causal.

    Only text-only exports of known causal model keys qualify. Multimodal
    exports (e.g. image tokens interleaved with text) or text generation models
    with non-causal attention patterns may attend bidirectionally, so the mask
    cannot be replaced by an implicit causal bound.
    """
    task = self.export_config.task
    try:
      task = exportable_module_config.ExportTask(task)
    except ValueError:
      return False
    if task != exportable_module_config.ExportTask.TEXT_GENERATION:
      return False
    model = getattr(self, "model", None)
    model_cfg = getattr(model, "config", None)
    text_cfg = getattr(model_cfg, "text_config", model_cfg)
    model_type = getattr(
        text_cfg, "model_type", getattr(model_cfg, "model_type", None)
    )
    if isinstance(model_type, str) and model_type:
      return model_type.lower() in _CAUSAL_TEXT_MODEL_KEYS
    model_id = (self.export_config.model or "").lower()
    return any(key in model_id for key in _CAUSAL_TEXT_MODEL_KEYS)

  def composite_flags(self) -> dict[str, bool]:
    """Returns the attention composite kwargs selected by the export config.

    Modules that override `forward` must merge these into the kwargs passed to
    `adapt_inputs`; otherwise flags such as `use_sdpa_composite` are silently
    dropped for that model.
    """
    flags = {}
    if self.export_config.apply_gpu_composites:
      flags["apply_gpu_composites"] = True
    if self.export_config.use_sdpa_composite:
      flags["use_sdpa_composite"] = True
    return flags

  @abc.abstractmethod
  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the model."""
    ...


class LiteRTExportableModuleForDecoderOnlyLM(ExportableModuleBase):
  """Base class for exportable modules for decoder-only LM."""

  def __init__(
      self,
      model: torch.nn.Module,
      export_config: ExportableModuleConfig,
      source_model_artifacts=None,
  ):
    super().__init__(export_config)
    self.model = model
    self.export_verifier = export_config.assistant_model is not None
    if self.export_verifier:
      self.mtp_verifier_step = export_config.mtp_verifier_step
    else:
      self.mtp_verifier_step = 0
    self.source_model_artifacts = source_model_artifacts

  @property
  def device(self) -> torch.device:
    """Device of the wrapped model's parameters (`meta` for V2 export)."""
    return next(self.model.parameters(), torch.empty(0)).device

  def adapt_inputs(
      self,
      tokens,
      embeddings,
      input_pos,
      kv_cache,
      mask,
      use_bool_mask: bool = False,
      **kwargs,
  ):
    local_mask = kwargs.get("local_mask", None)
    if hasattr(self.model.config, "text_config"):
      text_config = self.model.config.text_config
    else:
      text_config = self.model.config
    sliding_window = getattr(text_config, "sliding_window", None)
    # TODO(weiyiw): This is a hack to check if it's Mistral.
    is_mistral = getattr(self.model.config, "model_type", "") == "mistral"

    valid_mask = None
    # Currently we generate the valid mask either from input tokens or position.
    # But we should consider a unified way to do this or leave it to the
    # pipeline to pass in as an argument.
    if tokens is not None:
      pad_token_id = getattr(text_config, "pad_token_id", 0)
      if pad_token_id is None:
        pad_token_id = 0
      valid_mask = tokens != pad_token_id
    elif input_pos is not None:
      valid_mask_0 = torch.ones_like(input_pos[:1], dtype=torch.bool).unsqueeze(
          0
      )
      if input_pos.shape[0] > 1:
        valid_mask_1 = input_pos[1:] > input_pos[:-1]
      else:
        valid_mask_1 = torch.ones_like(input_pos[1:], dtype=torch.bool)
      valid_mask_1 = valid_mask_1.unsqueeze(0)
      valid_mask = torch.cat([valid_mask_0, valid_mask_1], dim=1)

    if sliding_window is not None:
      layer_types = getattr(text_config, "layer_types", None)
      masks = {
          "full_attention": mask,
      }
      need_sliding_mask = (
          layer_types is not None and "sliding_attention" in layer_types
      ) or is_mistral
      if need_sliding_mask:
        if local_mask is not None:
          masks["sliding_attention"] = local_mask
        elif use_bool_mask:
          if self.export_config.sliding_window_ring_buffer_size:
            assert (
                self.export_config.sliding_window_ring_buffer_size
                >= sliding_window
            ), (
                "Sliding window ring buffer size must be greater than or equal"
                " to the sliding window size."
            )
            is_decode = input_pos.shape[0] == 1
            if (
                kwargs.get("apply_gpu_composites", False)
                and is_decode
            ):
              masks["sliding_attention"] = (
                  sliding_window_attention_mask.build_sliding_window_decode_mask(
                      sliding_window,
                      self.export_config.sliding_window_ring_buffer_size,
                      input_pos,
                      use_bool_mask=True,
                  )
              )
            else:
              masks["sliding_attention"] = (
                  sliding_window_attention_mask.build_full_mask_with_valid_mask(
                      valid_mask,
                      sliding_window,
                      self.export_config.sliding_window_ring_buffer_size,
                      input_pos[0],
                      use_bool_mask=True,
                  )
              )
          else:
            masks["sliding_attention"] = utils.create_sliding_mask(
                input_pos.clone().unsqueeze(0),
                kv_cache.get_max_cache_shape(),
                sliding_window,
                use_bool_mask=True,
            )
        else:

          if self.export_config.sliding_window_ring_buffer_size is not None:
            assert (
                self.export_config.sliding_window_ring_buffer_size
                >= sliding_window
            ), (
                "Sliding window ring buffer size must be greater than or equal"
                " to the sliding window size."
            )
            is_decode = input_pos.shape[0] == 1
            if (
                kwargs.get("apply_gpu_composites", False)
                and is_decode
            ):
              masks["sliding_attention"] = (
                  sliding_window_attention_mask.build_sliding_window_decode_mask(
                      sliding_window,
                      self.export_config.sliding_window_ring_buffer_size,
                      input_pos,
                      use_bool_mask=False,
                  )
              )
            else:
              masks["sliding_attention"] = (
                  sliding_window_attention_mask.build_full_mask_with_valid_mask(
                      valid_mask,
                      sliding_window,
                      self.export_config.sliding_window_ring_buffer_size,
                      input_pos[0],
                      use_bool_mask=False,
                  )
              )
          else:
            masks["sliding_attention"] = (
                utils.create_sliding_mask(
                    input_pos.clone().unsqueeze(0),
                    kv_cache.get_max_cache_shape(),
                    sliding_window,
                )
                + mask
            )
      if is_mistral:
        masks = masks["sliding_attention"]
    else:
      masks = mask

    ret = {}
    if embeddings is not None:
      ret["inputs_embeds"] = embeddings.to(self.export_config.get_cache_dtype())
    else:
      ret["input_ids"] = tokens

    valid_mask = None
    if tokens is not None:
      pad_token_id = None
      if (
          getattr(self.source_model_artifacts, "tokenizer", None) is not None
          and getattr(
              self.source_model_artifacts.tokenizer, "pad_token_id", None
          )
          is not None
      ):
        pad_token_id = self.source_model_artifacts.tokenizer.pad_token_id
      if (
          pad_token_id is None
          and getattr(text_config, "pad_token_id", None) is not None
      ):
        pad_token_id = text_config.pad_token_id
      if not isinstance(pad_token_id, int) or pad_token_id < 0:
        pad_token_id = 0
      valid_mask = tokens != pad_token_id
    elif input_pos is not None:
      valid_mask_0 = torch.ones_like(input_pos[:1], dtype=torch.bool).unsqueeze(
          0
      )
      if input_pos.shape[0] > 1:
        valid_mask_1 = input_pos[1:] > input_pos[:-1]
      else:
        valid_mask_1 = torch.ones_like(input_pos[1:], dtype=torch.bool)
      valid_mask_1 = valid_mask_1.unsqueeze(0)
      valid_mask = torch.cat([valid_mask_0, valid_mask_1], dim=1)

    if valid_mask is not None:
      ret["valid_mask"] = valid_mask

    cache_runtime_args = {"cache_position": input_pos}
    if (
        kwargs.get("apply_gpu_composites", False)
        or kwargs.get("use_sdpa_composite", False)
    ):
      param_tensor = kwargs.get("param_tensor", None)
      if (
          self.export_config.use_fused_sdpa_cache_update
          and param_tensor is not None
      ):
        if self.export_config.sliding_window_ring_buffer_size is None:
          raise ValueError(
              "use_fused_sdpa_cache_update requires"
              " sliding_window_ring_buffer_size."
          )
        # The fused op reads start/end from the raw signature input: a param
        # tensor rewritten in-graph is not visible to backends at runtime.
        ret["use_fused_sdpa_cache_update"] = True
        ret["runtime_param_tensor"] = param_tensor
      if param_tensor is not None:
        # Local attn ring buffer uses param_tensor[3] to store update length.
        # TODO(sulemanshahid): We can move the param tensor as in-graph now.
        # TODO(weiyiw): Drop this rewrite so `param_tensor` stays the signature
        # input and `runtime_param_tensor` is no longer needed.
        update_length = param_tensor[..., 1:2] - param_tensor[..., 0:1]
        param_tensor = torch.cat(
            [param_tensor[..., :3], update_length, param_tensor[..., 4:]],
            dim=-1,
        )
      cache_runtime_args["apply_gpu_composites"] = True
      cache_runtime_args["param_tensor"] = param_tensor
      ret["apply_gpu_composites"] = True
      ret["param_tensor"] = param_tensor
    if self.export_config.sliding_window_ring_buffer_size is not None:
      # Ring buffer cache implementation requires valid mask.
      cache_runtime_args["valid_mask"] = valid_mask
      cache_runtime_args["enable_ring_buffer"] = True
      ret["enable_ring_buffer"] = True
    kv_cache.set_cache_runtime_args(cache_runtime_args)
    if kwargs.get("use_sdpa_composite", False):
      ret["use_sdpa_composite"] = kwargs["use_sdpa_composite"]

    ret.update({
        "position_ids": input_pos.clone().unsqueeze(0),
        "past_key_values": kv_cache,
        "cache_position": input_pos,
        "attention_mask": masks,
        # Other common settings
        "use_cache": True,
    })
    lora_lib.set_lora_weights(self.model, kwargs.get("lora"))
    return ret

  def get_sample_kv_cache(self, model_config):
    """Returns the input sample KV cache for the model."""
    export_config = self.export_config
    kv_cache = kv_cache_lib.CACHE_REGISTRY[
        export_config.cache_implementation  # pyrefly: ignore[bad-index]
    ].create_from_config(
        model_config,
        export_config,
        batch_size=export_config.batch_size,
        cache_length=export_config.cache_length,
    )
    # Sample cache must live on the model's device (`meta` for V2 export).
    kv_cache = kv_cache.to(self.device)
    inputs = {"kv_cache": kv_cache}
    if export_config.cache_length_dim is not None:
      flat_shapes = []
      for layer in kv_cache.layers:
        if isinstance(layer, cache_lib.LiteRTLMConvCacheLayer):
          if getattr(layer, "recurrent_states", None) is not None:
            flat_shapes.append(None)
            flat_shapes.append(None)
          else:
            flat_shapes.append(None)
        elif hasattr(layer, "conv_state") or hasattr(layer, "conv_states"):
          flat_shapes.append(None)
        else:
          k_ts_idx = getattr(layer, "k_ts_idx")
          v_ts_idx = getattr(layer, "v_ts_idx")
          flat_shapes.append({k_ts_idx: export_config.cache_length_dim})
          flat_shapes.append({v_ts_idx: export_config.cache_length_dim})
      dynamic_shapes = {"kv_cache": flat_shapes}
      return inputs, dynamic_shapes
    else:
      import torch.utils._pytree as pytree  # pylint: disable=g-import-not-at-top

      flat_tensors, _ = pytree.tree_flatten(kv_cache)
      return inputs, {"kv_cache": [None] * len(flat_tensors)}


class LiteRTExportableModuleForDecoderOnlyLMPrefill(
    LiteRTExportableModuleForDecoderOnlyLM
):
  """Exportable module for prefill."""

  def forward(
      self,
      tokens,
      input_pos,
      kv_cache,
      mask,
      local_mask=None,
      **kwargs,
  ):
    kwargs.update(self.composite_flags())
    inputs = self.adapt_inputs(
        tokens,
        None,
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
    # Prefill logits are not read by the runtime: sampling uses the decode
    # signature. Emitting them costs a vocabulary-sized `lm_head` matmul on
    # every prefill call, so it is off unless the GPU composite path needs a
    # live consumer. `logits_to_keep=1` keeps it to the final position.
    emit_logits = bool(self.export_config.prefill_logits)
    if emit_logits:
      inputs["logits_to_keep"] = 1
    # Tells the attention layers whether the final hidden states are live;
    # see `core/attention.py:transposed_attention`.
    inputs["prefill_logits"] = emit_logits
    with lora_lib.bind_lora_weights(self.model, kwargs.get("lora")):
      output = self.model(**inputs)
    outputs = {"kv_cache": output.past_key_values}
    if emit_logits:
      outputs["logits"] = output.logits.to(torch.float32)
    return outputs

  def _get_input(
      self, batch_size, prefill_length, prefill_length_dim, model_config
  ):
    del model_config  # Unused.
    device = self.device
    tokens = {
        "tokens": torch.ones(
            (batch_size, prefill_length), dtype=torch.int32, device=device
        )
    }
    tokens_dynamic_shape = (
        {"tokens": {1: prefill_length_dim}} if prefill_length_dim else {}
    )
    return tokens, tokens_dynamic_shape

  def get_sample_inputs(self, model_config, **kwargs):
    export_config = self.export_config
    use_bool_mask = export_config.extra_kwargs.get("use_bool_mask", False)
    device = self.device
    kv_cache_inputs, kv_cache_dynamic_shapes = self.get_sample_kv_cache(
        model_config
    )
    batch_size = export_config.batch_size
    cache_length = export_config.cache_length
    text_cfg = getattr(model_config, "text_config", model_config)
    sliding_window = getattr(text_cfg, "sliding_window", None)
    layer_types = getattr(text_cfg, "layer_types", None)
    model_type = getattr(
        text_cfg, "model_type", getattr(model_config, "model_type", "")
    )
    has_sliding = sliding_window is not None and (
        (layer_types is not None and "sliding_attention" in layer_types)
        or model_type == "mistral"
    )
    ring_buffer_size = export_config.sliding_window_ring_buffer_size
    sample_inputs = {}
    for prefill_length in export_config.prefill_lengths:
      tokens, tokens_dynamic_shape = self._get_input(
          batch_size,
          prefill_length,
          export_config.prefill_length_dim,
          model_config,
      )
      inputs = {
          **tokens,
          "input_pos": torch.ones(
              (prefill_length), dtype=torch.int32, device=device
          ),
          "mask": torch.ones(
              (1, 1, prefill_length, cache_length),
              dtype=torch.bool if use_bool_mask else torch.float32,
              device=device,
          ),
      }
      if ring_buffer_size is not None and has_sliding:
        inputs["local_mask"] = torch.ones(
            (1, 1, prefill_length, ring_buffer_size + prefill_length),
            dtype=torch.bool if use_bool_mask else torch.float32,
            device=device,
        )
      if export_config.apply_gpu_composites:
        inputs["param_tensor"] = torch.ones(
            (1, 1, 1, 7), dtype=torch.int32, device=device
        )

      inputs.update(kv_cache_inputs)
      if export_config.prefill_length_dim is not None:
        dynamic_shapes = {
            **tokens_dynamic_shape,
            "mask": {
                2: export_config.prefill_length_dim,
                3: export_config.cache_length_dim,
            },
            "input_pos": {0: export_config.prefill_length_dim},
        }
        if "local_mask" in inputs:
          dynamic_shapes["local_mask"] = {
              2: export_config.prefill_length_dim,
          }
        dynamic_shapes.update(kv_cache_dynamic_shapes)
        sample_inputs["prefill"] = (inputs, dynamic_shapes)
      else:
        sample_inputs[f"prefill_{prefill_length}"] = (inputs, {})
    return sample_inputs


class LiteRTExportableModuleForDecoderOnlyLMGenerate(
    LiteRTExportableModuleForDecoderOnlyLM
):
  """Exportable module for generate / decode."""

  def forward(
      self,
      tokens,
      input_pos,
      kv_cache,
      mask,
      local_mask=None,
      **kwargs,
  ):
    kwargs.update(self.composite_flags())
    inputs = self.adapt_inputs(
        tokens,
        None,
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
    with lora_lib.bind_lora_weights(self.model, kwargs.get("lora")):
      output = self.model(**inputs, output_hidden_states=self.export_verifier)
    if self.export_verifier:
      extra_outputs = {"activations": output["hidden_states"][-1]}
    else:
      extra_outputs = {}

    return {
        "kv_cache": output.past_key_values,
        "logits": output.logits.to(torch.float32),
        **extra_outputs,
    }

  def _get_input(
      self, batch_size, decode_length, decode_length_dim, model_config
  ):
    del model_config  # Unused.
    device = self.device
    tokens = {
        "tokens": torch.ones(
            (batch_size, decode_length), dtype=torch.int32, device=device
        )
    }
    tokens_dynamic_shape = {"tokens": None} if decode_length_dim else {}
    return tokens, tokens_dynamic_shape

  def get_sample_inputs(self, model_config):  # pyrefly: ignore[bad-override]
    export_config = self.export_config
    use_bool_mask = export_config.extra_kwargs.get("use_bool_mask", False)
    device = self.device
    kv_cache_inputs, kv_cache_dynamic_shapes = self.get_sample_kv_cache(
        model_config
    )
    batch_size = export_config.batch_size
    cache_length = export_config.cache_length
    tokens, tokens_dynamic_shape = self._get_input(
        batch_size,
        1,
        export_config.prefill_length_dim,
        model_config,
    )
    inputs = {
        **tokens,
        "input_pos": torch.ones((1), dtype=torch.int32, device=device),
        "mask": torch.ones(
            (1, 1, 1, cache_length),
            dtype=torch.bool if use_bool_mask else torch.float32,
            device=device,
        ),
    }
    text_cfg = getattr(model_config, "text_config", model_config)
    sliding_window = getattr(text_cfg, "sliding_window", None)
    layer_types = getattr(text_cfg, "layer_types", None)
    model_type = getattr(
        text_cfg, "model_type", getattr(model_config, "model_type", "")
    )
    has_sliding = sliding_window is not None and (
        (layer_types is not None and "sliding_attention" in layer_types)
        or model_type == "mistral"
    )
    ring_buffer_size = export_config.sliding_window_ring_buffer_size
    if ring_buffer_size is not None and has_sliding:
      inputs["local_mask"] = torch.ones(
          (1, 1, 1, ring_buffer_size),
          dtype=torch.bool if use_bool_mask else torch.float32,
          device=device,
      )
    if export_config.apply_gpu_composites:
      inputs["param_tensor"] = torch.ones(
          (1, 1, 1, 7), dtype=torch.int32, device=device
      )

    inputs.update(kv_cache_inputs)
    if export_config.cache_length_dim is not None:
      decode_dynamic_shapes = {
          **tokens_dynamic_shape,
          "mask": {3: export_config.cache_length_dim},
          "input_pos": None,
      }
      if "local_mask" in inputs:
        decode_dynamic_shapes["local_mask"] = None
      decode_dynamic_shapes.update(kv_cache_dynamic_shapes)
    else:
      decode_dynamic_shapes = {}
    decode_graph = {"decode": (inputs, decode_dynamic_shapes)}
    if self.export_verifier:
      verify_length = self.mtp_verifier_step
      tokens, tokens_dynamic_shape = self._get_input(
          batch_size,
          verify_length,
          export_config.prefill_length_dim,
          model_config,
      )
      inputs = {
          **tokens,
          "input_pos": torch.ones(
              (verify_length), dtype=torch.int32, device=device
          ),
          "mask": torch.ones(
              (1, 1, verify_length, cache_length),
              dtype=torch.bool,
              device=device,
          ),
      }
      if ring_buffer_size is not None and has_sliding:
        inputs["local_mask"] = torch.ones(
            (1, 1, verify_length, ring_buffer_size + verify_length),
            dtype=torch.bool if use_bool_mask else torch.float32,
            device=device,
        )
      inputs.update(kv_cache_inputs)
      if export_config.cache_length_dim is not None:
        decode_dynamic_shapes = {
            **tokens_dynamic_shape,
            "mask": {3: export_config.cache_length_dim},
            "input_pos": None,
        }
        if "local_mask" in inputs:
          decode_dynamic_shapes["local_mask"] = None
        decode_dynamic_shapes.update(kv_cache_dynamic_shapes)
      else:
        decode_dynamic_shapes = {}
      decode_graph["verify"] = (inputs, decode_dynamic_shapes)

    return decode_graph
