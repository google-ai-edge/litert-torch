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
"""Bonsai-FLUX.2 / FLUX.2-klein image generation model export for LiteRT."""

import json
import os
from typing import Any

import litert_torch
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.image_gen import image_gen_model
from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import modeling_flux2
import numpy as np
import torch
from torch import nn
import transformers

from litert_lm_builder.runtime.proto import image_gen_metadata_pb2
from litert_lm_builder.runtime.proto import image_gen_model_type_pb2
from ai_edge_quantizer import qtyping
from ai_edge_quantizer import quantizer as quantizer_lib
from ai_edge_quantizer import recipe_manager as recipe_manager_lib
from tensorflow.lite.python import schema_py_generated as schema_fb  # pylint: disable=g-direct-tensorflow-import

_DEFAULT_QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 9728,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "use_cache": False,
    "vocab_size": 151936,
}


class PromptEmbedder(nn.Module):
  """Extracts intermediate Qwen3 hidden states and stacks them along feature dim."""

  def __init__(
      self, qwen_model: nn.Module, layers: tuple[int, ...] = (9, 18, 27)
  ):
    super().__init__()
    self.model = qwen_model
    self.layers = tuple(layers)

  def forward(
      self, input_ids: torch.Tensor, attention_mask: torch.Tensor
  ) -> torch.Tensor:
    out = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hs = torch.stack([out.hidden_states[k] for k in self.layers], dim=1)
    b, c, s, d = hs.shape
    return hs.permute(0, 2, 1, 3).reshape(b, s, c * d)


class DiTWrapper(nn.Module):
  """Wrapper module for exporting Flux2Transformer2DModel forward pass."""

  def __init__(self, dit_model: nn.Module):
    super().__init__()
    self.m = dit_model

  def forward(
      self,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor,
      timestep: torch.Tensor,
      img_ids: torch.Tensor,
      txt_ids: torch.Tensor,
  ) -> torch.Tensor:
    return self.m(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )[0]


class VaeDecoderWrapper(nn.Module):
  """Wrapper module for exporting AutoencoderKLFlux2 decode pass."""

  def __init__(self, vae_model: nn.Module):
    super().__init__()
    self.vae = vae_model

  def forward(self, latents: torch.Tensor) -> torch.Tensor:
    return self.vae.decode(latents, return_dict=False)[0]


def fix_zero_block_scales(tflite_path: str) -> int:
  """Replaces zero float16 scales in BlockwiseQuantization buffers with min nonzero scale.

  Ternary models (such as Bonsai-FLUX.2) can have 32-weight blocks that are all
  zeros, which produces a blockwise quantization scale of 0.0. XNNPACK rejects
  zero scales in blockwise quantized weights, so replacing 0.0 scales with the
  smallest positive scale in the tensor preserves zero dequantized products
  while satisfying XNNPACK validation.

  Args:
    tflite_path: Path to the quantized .tflite file to patch in place.

  Returns:
    Number of tensors whose blockwise scale buffers were patched.
  """
  with open(tflite_path, "rb") as f:
    buf = bytearray(f.read())
  model = schema_fb.Model.GetRootAsModel(buf, 0)
  patched_tensors = 0
  for sg_idx in range(model.SubgraphsLength()):
    sg = model.Subgraphs(sg_idx)
    for t_idx in range(sg.TensorsLength()):
      t = sg.Tensors(t_idx)
      q = t.Quantization()
      if (
          q is None
          or q.DetailsType()
          != schema_fb.QuantizationDetails.BlockwiseQuantization
      ):
        continue
      bq = schema_fb.BlockwiseQuantization()
      bq.Init(q.Details().Bytes, q.Details().Pos)
      scale_tensor = sg.Tensors(bq.Scales())
      b = model.Buffers(scale_tensor.Buffer())
      if not b.DataIsNone():
        o = b._tab.Offset(4)  # pylint: disable=protected-access
        start = b._tab.Vector(o)  # pylint: disable=protected-access
        length = b.DataLength()
      elif b.Offset() > 0 and b.Size() > 0:
        start = b.Offset()
        length = b.Size()
      else:
        continue
      scales = np.frombuffer(
          buf, dtype=np.float16, count=length // 2, offset=start
      )
      zeros = scales == 0
      if np.any(zeros):
        nonzero = scales[~zeros]
        fill_val = (
            np.min(np.abs(nonzero)) if nonzero.size > 0 else np.float16(1e-4)
        )
        scales[zeros] = fill_val
        patched_tensors += 1
  if patched_tensors > 0:
    with open(tflite_path, "wb") as f:
      f.write(buf)
  return patched_tensors


def _quantize_text_encoder(fp32_path: str, out_path: str) -> str:
  """Quantizes Qwen3 text encoder with int4 block128 FC and int8 channelwise embedding."""
  rm = recipe_manager_lib.RecipeManager()
  rm.add_dynamic_config(
      regex=".*",
      operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
      num_bits=4,
      granularity=qtyping.QuantGranularity.BLOCKWISE_128,
  )
  rm.add_dynamic_config(
      regex=".*",
      operation_name=qtyping.TFLOperationName.EMBEDDING_LOOKUP,
      num_bits=8,
      granularity=qtyping.QuantGranularity.CHANNELWISE,
  )
  qt = quantizer_lib.Quantizer(fp32_path, rm.get_quantization_recipe())
  qt.quantize().export_model(out_path, overwrite=True)
  return out_path


def _quantize_dit(fp32_path: str, out_path: str) -> str:
  """Quantizes FLUX.2 DiT with int8 channelwise non-block FC and int4 block32 block FC."""
  rm = recipe_manager_lib.RecipeManager()
  rm.add_dynamic_config(
      regex=".*",
      operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
      num_bits=8,
      granularity=qtyping.QuantGranularity.CHANNELWISE,
  )
  rm.add_dynamic_config(
      regex=".*TransformerBlock_.*",
      operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
      num_bits=4,
      granularity=qtyping.QuantGranularity.BLOCKWISE_32,
      algorithm_key=recipe_manager_lib.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
  )
  qt = quantizer_lib.Quantizer(fp32_path, rm.get_quantization_recipe())
  qt.quantize().export_model(out_path, overwrite=True)
  fix_zero_block_scales(out_path)
  return out_path


class BonsaiFlux2(image_gen_model.ImageGenModel):
  """Bonsai-FLUX.2 / FLUX.2-klein multi-stage text-to-image model."""

  def __init__(
      self,
      model_path: str,
      export_config: (
          exportable_module_config.ExportableModuleConfig | None
      ) = None,
  ):
    super().__init__(model_path, export_config)
    self._dit_config: dict[str, Any] = {}
    self._vae_config: dict[str, Any] = {}
    self._text_encoder_layers: tuple[int, ...] = (9, 18, 27)
    self._latent_bn_scale: list[float] = []
    self._latent_bn_shift: list[float] = []
    self._load_configs()

  def _load_configs(self) -> None:
    """Loads submodel configs and any pre-existing BN stats."""
    dit_cfg_path = os.path.join(self.model_dir, "transformer", "config.json")
    if os.path.exists(dit_cfg_path):
      with open(dit_cfg_path, "r") as f:
        self._dit_config = json.load(f)

    vae_cfg_path = os.path.join(self.model_dir, "vae", "config.json")
    if os.path.exists(vae_cfg_path):
      with open(vae_cfg_path, "r") as f:
        self._vae_config = json.load(f)

    meta_path = os.path.join(self.model_dir, "pipeline_meta.json")
    if os.path.exists(meta_path):
      with open(meta_path, "r") as f:
        meta = json.load(f)
      if "text_encoder_out_layers" in meta:
        self._text_encoder_layers = tuple(meta["text_encoder_out_layers"])
      if "latent_bn_shift" in meta:
        self._latent_bn_shift = [float(x) for x in meta["latent_bn_shift"]]
      elif "latent_bn_mean" in meta:
        self._latent_bn_shift = [float(x) for x in meta["latent_bn_mean"]]
      if "latent_bn_scale" in meta:
        self._latent_bn_scale = [float(x) for x in meta["latent_bn_scale"]]
      elif "latent_bn_var" in meta:
        eps = float(meta.get("latent_bn_eps", 1e-4))
        self._latent_bn_scale = [
            float(np.sqrt(float(v) + eps)) for v in meta["latent_bn_var"]
        ]

  def _get_max_seq_len(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> int:
    if "max_seq_len" in export_config.extra_kwargs:
      return int(export_config.extra_kwargs["max_seq_len"])
    if export_config.prefill_lengths and export_config.prefill_lengths != [128]:
      return int(export_config.prefill_lengths[0])
    return 256

  def _should_quantize(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> bool:
    recipe = export_config.quantization_recipe
    if not recipe or str(recipe).strip().lower() in (
        "none",
        "null",
        "false",
        "",
    ):
      return False
    return True

  def export_text_encoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports the Qwen3 text encoder to TFLite."""
    print("Exporting text encoder...")
    text_enc_dir = os.path.join(self.model_dir, "text_encoder")
    if not os.path.exists(text_enc_dir):
      text_enc_dir = self.model_dir

    cfg_path = os.path.join(text_enc_dir, "config.json")
    if os.path.exists(cfg_path):
      config = transformers.AutoConfig.from_pretrained(
          text_enc_dir, torch_dtype=torch.float32
      )
    else:
      config = transformers.PretrainedConfig.from_dict(_DEFAULT_QWEN3_CONFIG)

    has_weights = any(
        os.path.exists(os.path.join(text_enc_dir, name))
        for name in ("model.safetensors", "model.safetensors.index.json")
    )
    if export_config.use_random_weights or not has_weights:
      qwen_cfg = transformers.Qwen3Config(
          **{k: v for k, v in config.to_dict().items() if k != "model_type"}
      )
      causal_lm = transformers.Qwen3ForCausalLM(qwen_cfg).to(torch.float32)
    else:
      causal_lm = transformers.AutoModelForCausalLM.from_pretrained(
          text_enc_dir, config=config, torch_dtype=torch.float32
      )
    causal_lm.eval()
    qwen_model = getattr(causal_lm, "model", causal_lm)
    assert isinstance(qwen_model, nn.Module)

    num_layers = getattr(qwen_model.config, "num_hidden_layers", 36)
    layers = tuple(k for k in self._text_encoder_layers if k <= num_layers)
    if not layers:
      layers = tuple(range(1, min(num_layers, 3) + 1))
    self._text_encoder_layers = layers

    embedder = PromptEmbedder(qwen_model, layers=layers).eval()
    seq_len = self._get_max_seq_len(export_config)
    sample_inputs = (
        torch.ones(1, seq_len, dtype=torch.int32),
        torch.ones(1, seq_len, dtype=torch.int32),
    )

    fp32_path = os.path.join(output_dir, "text_encoder_fp32.tflite")
    litert_torch.convert(embedder, sample_inputs).export(fp32_path)

    if self._should_quantize(export_config):
      int4_path = os.path.join(output_dir, "text_encoder_int4.tflite")
      _quantize_text_encoder(fp32_path, int4_path)
      if not export_config.keep_temporary_files and os.path.exists(fp32_path):
        os.remove(fp32_path)
      return int4_path
    return fp32_path

  def export_dit(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports the FLUX.2 DiT transformer to TFLite."""
    print("Exporting DiT transformer...")
    subfolder = (
        "transformer"
        if os.path.exists(os.path.join(self.model_dir, "transformer"))
        else None
    )
    dit = modeling_flux2.Flux2Transformer2DModel.from_pretrained(
        self.model_dir, subfolder=subfolder, torch_dtype=torch.float32
    ).eval()
    self._dit_config = vars(dit.config)

    image_size = export_config.t2i_output_image_size
    grid = image_size // 16
    img_tokens = grid * grid
    seq_len = self._get_max_seq_len(export_config)
    in_channels = dit.config.in_channels
    joint_attention_dim = dit.config.joint_attention_dim
    rope_axes = len(dit.config.axes_dims_rope)

    sample_inputs = (
        torch.zeros(1, img_tokens, in_channels, dtype=torch.float32),
        torch.zeros(1, seq_len, joint_attention_dim, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
        torch.zeros(img_tokens, rope_axes, dtype=torch.float32),
        torch.zeros(seq_len, rope_axes, dtype=torch.float32),
    )

    fp32_path = os.path.join(output_dir, "dit_fp32.tflite")
    litert_torch.convert(DiTWrapper(dit).eval(), sample_inputs).export(
        fp32_path
    )

    if self._should_quantize(export_config):
      int4_path = os.path.join(output_dir, "dit_int4b32.tflite")
      _quantize_dit(fp32_path, int4_path)
      if not export_config.keep_temporary_files and os.path.exists(fp32_path):
        os.remove(fp32_path)
      return int4_path
    return fp32_path

  def export_vae_decoder(
      self,
      output_dir: str,
      export_config: exportable_module_config.ExportableModuleConfig,
  ) -> str:
    """Exports the AutoencoderKLFlux2 decoder to TFLite and records BN stats."""
    print("Exporting VAE decoder...")
    subfolder = (
        "vae" if os.path.exists(os.path.join(self.model_dir, "vae")) else None
    )
    vae = modeling_flux2.AutoencoderKLFlux2.from_pretrained(
        self.model_dir, subfolder=subfolder, torch_dtype=torch.float32
    ).eval()
    self._vae_config = vars(vae.config)

    vae_dir = (
        os.path.join(self.model_dir, subfolder) if subfolder else self.model_dir
    )
    has_vae_weights = any(
        os.path.exists(os.path.join(vae_dir, name))
        for name in (
            "diffusion_pytorch_model.safetensors",
            "model.safetensors",
        )
    )
    if has_vae_weights or not self._latent_bn_scale:
      with torch.no_grad():
        running_var = vae.bn.running_var
        running_mean = vae.bn.running_mean
        assert running_var is not None and running_mean is not None
        self._latent_bn_scale = (
            torch.sqrt(running_var.float() + vae.bn.eps).cpu().tolist()
        )
        self._latent_bn_shift = running_mean.float().cpu().tolist()

    image_size = export_config.t2i_output_image_size
    lat_size = image_size // 8
    latent_channels = vae.config.latent_channels
    sample_inputs = (
        torch.zeros(
            1, latent_channels, lat_size, lat_size, dtype=torch.float32
        ),
    )

    out_path = os.path.join(output_dir, "vae_decoder_fp32.tflite")
    litert_torch.convert(VaeDecoderWrapper(vae).eval(), sample_inputs).export(
        out_path
    )
    return out_path

  def export(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> dict[str, str]:
    """Exports all Bonsai-FLUX.2 submodules to LiteRT TFLite files."""
    output_dir = export_config.work_dir or export_config.output_dir
    if not output_dir:
      raise ValueError("Either output_dir or work_dir must be specified.")
    os.makedirs(output_dir, exist_ok=True)
    artifacts: dict[str, str] = {}

    targets = set(export_config.extra_kwargs.get("targets", ["all"]))
    export_all = "all" in targets

    if export_all or "text_encoder" in targets:
      artifacts["text_encoder"] = self.export_text_encoder(
          output_dir, export_config
      )
    if export_all or "dit" in targets:
      artifacts["dit"] = self.export_dit(output_dir, export_config)
    if export_all or "vae_decoder" in targets:
      artifacts["vae_decoder"] = self.export_vae_decoder(
          output_dir, export_config
      )

    return artifacts

  def get_image_gen_metadata(
      self, export_config: exportable_module_config.ExportableModuleConfig
  ) -> image_gen_metadata_pb2.ImageGenMetadata:
    """Builds ImageGenMetadata proto for the exported Bonsai-FLUX.2 model."""
    in_channels = int(self._dit_config.get("in_channels", 128))
    joint_attention_dim = int(self._dit_config.get("joint_attention_dim", 7680))

    bn_scale = self._latent_bn_scale or [1.0] * in_channels
    bn_shift = self._latent_bn_shift or [0.0] * in_channels

    bonsai_proto = image_gen_model_type_pb2.BonsaiFlux2(
        flux2_params=image_gen_model_type_pb2.Flux2Params(
            seq_len=self._get_max_seq_len(export_config),
            img_size=export_config.t2i_output_image_size,
            packed_ch=in_channels,
            prompt_dim=joint_attention_dim,
            default_steps=4,
            latent_bn_scale=bn_scale,
            latent_bn_shift=bn_shift,
        ),
    )
    return image_gen_metadata_pb2.ImageGenMetadata(
        image_gen_model_type=image_gen_model_type_pb2.ImageGenModelType(
            bonsai_flux2=bonsai_proto
        ),
    )
