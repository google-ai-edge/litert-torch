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
"""Streaming weights loader for HuggingFace checkpoints."""

import glob
import json
import os
from typing import Any, Optional
from absl import logging
import huggingface_hub
import safetensors
import torch


class HFCheckpointWeightsLoader:
  """Streams weights on-demand from HuggingFace safetensors checkpoints."""

  def __init__(self, model_path: str, config: Optional[Any] = None):
    self.model_path = model_path
    self.config = config
    self.handles: dict[str, Any] = {}
    self.weight_map: dict[str, str] = {}
    self.checkpoint_dir = self._resolve_checkpoint_dir(model_path)
    self._init_index()

  def _resolve_checkpoint_dir(self, model_path: str) -> str:
    """Resolves local directory or Hugging Face Hub snapshot directory."""
    if os.path.isdir(model_path):
      return model_path

    # Try local cache first
    try:
      d = huggingface_hub.snapshot_download(
          model_path, local_files_only=True
      )
      if glob.glob(os.path.join(d, "*.safetensors")):
        return d
    except Exception:
      pass

    # Download / resolve snapshot
    return huggingface_hub.snapshot_download(model_path)

  def _init_index(self):
    """Initializes safetensors index from model.safetensors.index.json or safetensors files."""
    index_path = os.path.join(
        self.checkpoint_dir, "model.safetensors.index.json"
    )
    if os.path.exists(index_path):
      with open(index_path, "r") as f:
        self.weight_map = json.load(f).get("weight_map", {})
    else:
      files = glob.glob(os.path.join(self.checkpoint_dir, "*.safetensors"))
      for fpath in files:
        fname = os.path.basename(fpath)
        handle = safetensors.safe_open(fpath, framework="pt", device="cpu")
        self.handles[fpath] = handle
        for k in handle.keys():
          self.weight_map[k] = fname
    logging.info(
        "HFCheckpointWeightsLoader initialized from %s with %d tensors.",
        self.checkpoint_dir,
        len(self.weight_map),
    )

  def _get_raw_tensor(self, tensor_name: str) -> torch.Tensor:
    if tensor_name not in self.weight_map:
      raise KeyError(f"Tensor '{tensor_name}' not found in checkpoint.")
    shard_file = self.weight_map[tensor_name]
    shard_path = os.path.join(self.checkpoint_dir, shard_file)
    if shard_path not in self.handles:
      self.handles[shard_path] = safetensors.safe_open(shard_path, framework="pt", device="cpu")
    return self.handles[shard_path].get_tensor(tensor_name)

  def _clean_param_name(self, name: str) -> str:
    prefixes = (
        "model.",
        "module.",
        "static_model.",
        "original_hf_model.",
        "language_model.",
    )
    changed = True
    while changed:
      changed = False
      for pfx in prefixes:
        if name.startswith(pfx):
          name = name[len(pfx) :]
          changed = True
    return name

  def _is_qwen3_5_rms_norm_weight(self, cand: str) -> bool:
    model_type = (
        getattr(self.config, "model_type", "") if self.config is not None else ""
    )
    if model_type not in (
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    ):
      return False
    if cand.endswith((
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    )):
      return True
    if cand in (
        "norm.weight",
        "model.norm.weight",
        "model.language_model.norm.weight",
    ):
      return True
    return False

  def __call__(self, param_name: str) -> torch.Tensor:
    cleaned = self._clean_param_name(param_name)
    candidates = [
        param_name,
        cleaned,
        f"model.{cleaned}",
        f"model.language_model.{cleaned}",
        f"language_model.{cleaned}",
        f"transformer.{cleaned}",
        param_name.removeprefix("model."),
        param_name.removeprefix("module."),
        param_name.removeprefix("static_model."),
        param_name.removeprefix("model.").removeprefix("module."),
        param_name.removeprefix("model.").removeprefix("static_model."),
    ]
    # Deduplicate preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
      if c not in seen:
        seen.add(c)
        unique_candidates.append(c)
    candidates = unique_candidates

    for cand in candidates:
      if cand in self.weight_map:
        tensor = self._get_raw_tensor(cand)
        if self._is_qwen3_5_rms_norm_weight(cand):
          return tensor.to(torch.float32) + 1.0
        return tensor

    # 1. Fused QKV projection fallback
    for cand in candidates:
      if "self_attn.qkv_proj.weight" in cand:
        prefix = cand.replace("self_attn.qkv_proj.weight", "self_attn.")
        q = self(prefix + "q_proj.weight")
        k = self(prefix + "k_proj.weight")
        v = self(prefix + "v_proj.weight")
        return torch.cat([q, k, v], dim=0)
      if "self_attn.qkv_proj.bias" in cand:
        prefix = cand.replace("self_attn.qkv_proj.bias", "self_attn.")
        q = self(prefix + "q_proj.bias")
        k = self(prefix + "k_proj.bias")
        v = self(prefix + "v_proj.bias")
        return torch.cat([q, k, v], dim=0)

    # 2. Fused Gate-Up projection fallback
    for cand in candidates:
      if "mlp.gate_up_proj.weight" in cand:
        prefix = cand.replace("mlp.gate_up_proj.weight", "mlp.")
        gate = self(prefix + "gate_proj.weight")
        up = self(prefix + "up_proj.weight")
        return torch.cat([gate, up], dim=0)
      if "mlp.gate_up_proj.bias" in cand:
        prefix = cand.replace("mlp.gate_up_proj.bias", "mlp.")
        gate = self(prefix + "gate_proj.bias")
        up = self(prefix + "up_proj.bias")
        return torch.cat([gate, up], dim=0)

    # 3. Tied word embeddings fallback (lm_head.weight -> embed_tokens.weight)
    if any("lm_head.weight" in cand for cand in candidates):
      for emb_key in [
          "model.embed_tokens.weight",
          "embed_tokens.weight",
          "model.language_model.embed_tokens.weight",
          "transformer.wte.weight",
      ]:
        if emb_key in self.weight_map:
          return self._get_raw_tensor(emb_key)

    # 4. Standalone embedder module fallback (param is model.weight or weight)
    if any(cand in ("model.weight", "weight") for cand in candidates):
      for emb_key in [
          "model.embed_tokens.weight",
          "embed_tokens.weight",
          "model.language_model.embed_tokens.weight",
          "transformer.wte.weight",
      ]:
        if emb_key in self.weight_map:
          return self._get_raw_tensor(emb_key)

    # 5. Rotary embedding inv_freq buffer fallback
    if any(cand.endswith("inv_freq") for cand in candidates):
      if self.config is not None:
        cfg = getattr(self.config, "text_config", None) or self.config
        if (
            cfg is not None
            and hasattr(cfg, "hidden_size")
            and hasattr(cfg, "num_attention_heads")
        ):
          dim = getattr(cfg, "head_dim", None) or (
              cfg.hidden_size // cfg.num_attention_heads
          )
          base = getattr(cfg, "rope_theta", 10000.0)
          return 1.0 / (
              base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
          )

    raise KeyError(
        f"Parameter '{param_name}' not found in safetensors checkpoint. Candidates: {candidates}"
    )

  def close(self):
    """Closes all open safetensors file handles."""
    self.handles.clear()
