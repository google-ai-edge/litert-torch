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

"""Verifies the reauthored Gemma3 model."""

import glob
import logging
import os

from absl import app
from absl import flags
from gemma import config as gemma_config
import kagglehub
from litert_torch.generative.examples.gemma3 import verify_util


_PROMPTS = flags.DEFINE_multi_string(
    "prompts",
    "What is the meaning of life?",
    "The input prompts to generate answers.",
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    "max_new_tokens",
    30,
    "The maximum size of the generated tokens.",
)
_CHECKPOINT = flags.DEFINE_string(
    "checkpoint",
    "",
    "The checkpoint to verify.",
)
_VARIANT = flags.DEFINE_string(
    "variant",
    "1b",
    "The variant of the model to verify.",
)
_WEIGHT_FILENAME = flags.DEFINE_string(
    "weight_filename",
    None,
    "The weightfilename of the model to verify.",
)


def find_first_ckpt(folder):
  """Finds the first .ckpt file in a folder."""
  ckpt_files = sorted(glob.glob(os.path.join(folder, "*.ckpt")))
  return os.path.basename(ckpt_files[0]) if ckpt_files else None


def get_config_for_270m(dtype):
  # Architecture.GEMMA_3
  return gemma_config.GemmaConfig(
      dtype=dtype,
      architecture=gemma_config.Architecture.GEMMA_3,
      num_hidden_layers=18,
      num_attention_heads=4,
      num_key_value_heads=1,
      hidden_size=640,
      intermediate_size=2048,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      head_dim=256,
      attn_types=(
          gemma_config.AttentionType.LOCAL_SLIDING,
          gemma_config.AttentionType.LOCAL_SLIDING,
          gemma_config.AttentionType.LOCAL_SLIDING,
          gemma_config.AttentionType.LOCAL_SLIDING,
          gemma_config.AttentionType.LOCAL_SLIDING,
          gemma_config.AttentionType.GLOBAL,
      ),
      sliding_window_size=512,
      rope_wave_length={
          gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
          gemma_config.AttentionType.GLOBAL: 1_000_000,
      },
      vocab_size=262_144,
      max_position_embeddings=32_768,
      tokenizer="third_party/py/gemma_pytorch/tokenizer/gemma3_cleaned_262144_v2.spiece.model",
      use_qk_norm=True,
      vision_config=None,
  )


def main(_):
  if _CHECKPOINT.value:
    checkpoint = _CHECKPOINT.value
  else:
    checkpoint = kagglehub.model_download(
        "google/gemma-3/pyTorch/gemma-3-1b-it"
    )

  # If the weight filename is not specified, use the first checkpoint.
  if _WEIGHT_FILENAME.value is None:
    weight_filename = find_first_ckpt(checkpoint)
    logging.info(
        "NOTE: using the first weight file `%s` from `%s`",
        weight_filename,
        checkpoint,
    )
  else:
    weight_filename = _WEIGHT_FILENAME.value

  custom_loader = None
  if weight_filename and weight_filename.endswith(".safetensors"):

    def _loader(path):
      # We use get_custom_loader to get the load_file function from safetensors
      load_fn = verify_util.loader.get_custom_loader(
          path, checkpoint_format="safetensors"
      )
      tensors = load_fn(path)
      new_tensors = {}
      torch = verify_util.torch

      # Rename embedding
      if "model.embed_tokens.weight" in tensors:
        new_tensors["embedder.weight"] = tensors.pop(
            "model.embed_tokens.weight"
        )

      # Iterate keys to find layers
      layer_indices = set()
      for k in tensors.keys():
        if k.startswith("model.layers."):
          parts = k.split(".")
          if parts[2].isdigit():
            layer_indices.add(int(parts[2]))

      for i in layer_indices:
        prefix = f"model.layers.{i}"

        # Norms
        if f"{prefix}.self_attn.q_norm.weight" in tensors:
          new_tensors[f"{prefix}.self_attn.query_norm.weight"] = tensors.pop(
              f"{prefix}.self_attn.q_norm.weight"
          )
        if f"{prefix}.self_attn.k_norm.weight" in tensors:
          new_tensors[f"{prefix}.self_attn.key_norm.weight"] = tensors.pop(
              f"{prefix}.self_attn.k_norm.weight"
          )

        # QKV
        q_key = f"{prefix}.self_attn.q_proj.weight"
        k_key = f"{prefix}.self_attn.k_proj.weight"
        v_key = f"{prefix}.self_attn.v_proj.weight"

        if q_key in tensors and k_key in tensors and v_key in tensors:
          q = tensors.pop(q_key)
          k = tensors.pop(k_key)
          v = tensors.pop(v_key)
          qkv = torch.cat([q, k, v], dim=0)
          new_tensors[f"{prefix}.self_attn.qkv_proj.weight"] = qkv

        q_bias = f"{prefix}.self_attn.q_proj.bias"
        k_bias = f"{prefix}.self_attn.k_proj.bias"
        v_bias = f"{prefix}.self_attn.v_proj.bias"

        if q_bias in tensors and k_bias in tensors and v_bias in tensors:
          qb = tensors.pop(q_bias)
          kb = tensors.pop(k_bias)
          vb = tensors.pop(v_bias)
          qkv_b = torch.cat([qb, kb, vb], dim=0)
          new_tensors[f"{prefix}.self_attn.qkv_proj.bias"] = qkv_b

      # Copy remaining tensors
      for k, v in tensors.items():
        new_tensors[k] = v

      return {"model_state_dict": new_tensors}

    custom_loader = _loader

  # Verify the reauthored model by comparing the outputs with the original one.
  if _VARIANT.value == "270m":
    gemma3_model_path = os.path.join(checkpoint, weight_filename)
    reauthored_model = verify_util.UnifiedGemma3Wrapper(
        verify_util.gemma3.build_model_270m(
            gemma3_model_path,
            custom_loader,
            mask_cache_size=verify_util.verifier.DEFAULT_KV_CACHE_MAX_LEN,
        )
    )

    original_get_model_config = gemma_config.get_model_config

    def get_model_config_patched(variant, dtype="bfloat16"):
      if variant == "270m":
        return get_config_for_270m(dtype)
      return original_get_model_config(variant, dtype)

    verify_util.gemma_config.get_model_config = get_model_config_patched

    verify_util.verify_reauthored_gemma_model(
        checkpoint=checkpoint,
        variant=_VARIANT.value,
        reauthored_model=reauthored_model,
        generate_prompts=_PROMPTS.value,
        forward_input_ids=[[2, 651, 9456, 576, 573, 3520, 3858, 603, 235248]],
        max_new_tokens=_MAX_NEW_TOKENS.value,
        weight_filename=weight_filename,
        custom_loader=custom_loader,
        atol=1e-04,
    )
  else:
    verify_util.verify_gemma3(
        checkpoint,
        _PROMPTS.value,
        _MAX_NEW_TOKENS.value,
        _VARIANT.value,
        weight_filename,
        custom_loader=custom_loader,
    )


if __name__ == "__main__":
  app.run(main)
