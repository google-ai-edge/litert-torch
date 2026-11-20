##################################################
# generative/layers/utilities/whisper_loader.py
##################################################

from dataclasses import dataclass
import glob
import os
from typing import Callable, Dict, List, Tuple

import torch

from ai_edge_torch.generative.layers import model_config

from ai_edge_torch.generative.utilities.loader import load_pytorch_statedict, load_safetensors


@dataclass
class EncoderTensorNames:
    conv1D_0: str
    conv1D_1: str
    layer_norm: str
    attn_key_proj: str
    attn_value_proj: str
    attn_query_proj: str
    attn_output_proj: str
    pre_attn_norm: str
    pre_ff_norm: str
    ff_up_proj: str
    ff_down_proj: str
    embed_positions: str

@dataclass
class DecoderTensorNames:
    layer_norm: str
    attn_key_proj: str
    attn_value_proj: str
    attn_query_proj: str
    attn_output_proj: str
    pre_attn_norm: str
    pre_cross_attn_norm: str
    cross_attn_q_proj: str
    cross_attn_out_proj: str
    pre_ff_norm: str
    ff_up_proj: str
    ff_down_proj: str
    embed_positions: str
    embed_tokens: str


# Depending on Encoder or Decoder, we pop or pass the cross attention
# projections from the source state dictionary.
@dataclass
class CrossAttentionTensorNames:
    cross_attn_key_proj: str
    cross_attn_value_proj: str

class WhisperModelLoader:
  """Utility class for loading and converting checkpoints to ODML transformer layer format."""

  def __init__(self,
               file_name: str,
               names: Union[EncoderTensorNames, DecoderTensorNames],
               cross_attention_names: CrossAttentionTensorNames) -> None:
    """ModelLoader constructor.

    Can be used to load multiple models of the same type.

    Args:
        file_name (str): Path to the checkpoint. Can be a directory or an exact
          file.
        names (TensorNames): An instance of `TensorNames` to determine mappings.
    """
    self._file_name = file_name
    self._names = names
    self._cross_attn_names = cross_attention_names
    self._loader = self._get_loader()
    self._MODEL_PREFIX = "model.encoder" if isinstance(self._names, EncoderTensorNames) else "model.decoder"

  def get_state(self) -> Dict[str, torch.Tensor]:
    return self._loader(self._file_name)

  def load(
      self, model: torch.nn.Module, strict: bool = True
  ) -> Tuple[List[str], List[str]]:
    """Load the model from the checkpoint.

    Args:
        model (torch.nn.Module): The pytorch model that needs to be loaded.
        strict (bool, optional): Whether the converted keys are strictly
          matched. Defaults to True.

    Returns:
        missing_keys (List[str]): a list of str containing the missing keys.
        unexpected_keys (List[str]): a list of str containing the unexpected
        keys.

    Raises:
        ValueError: If conversion results in unmapped tensors and strict mode is
          enabled.
    """
    state = self.get_state()
    state = state["model_state_dict"] if "model_state_dict" in state else state
    converted_state = dict()

    keys_to_pop = list()
    if isinstance(self._names, DecoderTensorNames):
      for key in state:
        if 'model.encoder.' in key:
            keys_to_pop.append(key)
            continue
        if 'encoder_attn.k_proj' in key or 'encoder_attn.v_proj' in key:
            keys_to_pop.append(key)
            continue
      if self._names.embed_tokens is not None:
          converted_state['embed_tokens.weight'] = state.pop(
          f"{self._MODEL_PREFIX}.{self._names.embed_tokens}.weight"
      )
      for i in range(model.config.num_layers):
        self._map_cross_attention_decoder_blocks(i, model.config, state, converted_state)

    for key in keys_to_pop:
      state.pop(key)

    keys_to_pop = list()
    if isinstance(self._names, EncoderTensorNames):
        for key in state:
            if 'encoder_attn.k_proj' in key or 'encoder_attn.v_proj' in key:
                continue
            if 'model.decoder.' in key:
                keys_to_pop.append(key)
                continue
        self._map_conv(state, converted_state)
        for i in range(model.config.num_layers):
            self._map_cross_attention_encoder_blocks(i, model.config, state, converted_state)

    for key in keys_to_pop:
      state.pop(key)

    if self._names.layer_norm is not None:
      layer_norm_name = self._names.layer_norm
      converted_state["layer_norm.weight"] = state.pop(f"{self._MODEL_PREFIX}.{layer_norm_name}.weight")
      if f"{self._MODEL_PREFIX}.{layer_norm_name}.bias" in state:
        converted_state["layer_norm.bias"] = state.pop(f"{self._MODEL_PREFIX}.{layer_norm_name}.bias")

    if self._names.embed_positions is not None:
      converted_state["embed_positions.weight"] = state.pop(
          f"{self._MODEL_PREFIX}.{self._names.embed_positions}.weight"
      )

    for i in range(model.config.num_layers):
      self._map_norm(i, model.config, state, converted_state)
      self._map_feedforward(i, model.config, state, converted_state)
      self._map_attention(i, model.config, state, converted_state)

    if strict and state:
      raise ValueError(
          "Failed to map all tensor. Remaining tensor are:"
          f" {list(state.keys())}"
      )
    model.load_state_dict(converted_state, strict=strict)

  def _map_conv(self,
                state: Dict[str, torch.Tensor],
                converted_state: Dict[str, torch.Tensor],) -> None:

    if self._names.conv1D_0 is not None:
      converted_state["conv1.weight"] = state.pop(f"{self._MODEL_PREFIX}.{self._names.conv1D_0}.weight")
      converted_state["conv1.bias"] = state.pop(f"{self._MODEL_PREFIX}.{self._names.conv1D_0}.bias")

    if self._names.conv1D_1 is not None:
      converted_state["conv2.weight"] = state.pop(f"{self._MODEL_PREFIX}.{self._names.conv1D_1}.weight")
      converted_state["conv2.bias"] = state.pop(f"{self._MODEL_PREFIX}.{self._names.conv1D_1}.bias")


  # Lifts the cross attention k,v projections from the decoder in to the encoder
  # as outputs, rather than having encoder output the latents.
  def _map_cross_attention_encoder_blocks(self,
                                   idx: int,
                                   config: Union[AudioEncoderConfig, TextDecoderConfig],
                                   state: Dict[str, torch.Tensor],
                                   converted_state: Dict[str, torch.Tensor]):
    k_cross_proj = f"k_cross_modules.{idx}"
    v_cross_proj = f"v_cross_modules.{idx}"

    cross_attn_key_proj_name = f"model.decoder.{self._cross_attn_names.cross_attn_key_proj}".format(idx)
    cross_attn_value_proj_name = f"model.decoder.{self._cross_attn_names.cross_attn_value_proj}".format(idx)

    converted_state[f"{k_cross_proj}.weight"] = state.pop(
        f"{cross_attn_key_proj_name}.weight"
    )
    converted_state[f"{v_cross_proj}.weight"] = state.pop(
        f"{cross_attn_value_proj_name}.weight"
    )
    converted_state[f"{v_cross_proj}.bias"] = state.pop(
        f"{cross_attn_value_proj_name}.bias"
    )

  def _map_cross_attention_decoder_blocks(self,
                                   idx: int,
                                   config: Union[AudioEncoderConfig, TextDecoderConfig],
                                   state: Dict[str, torch.Tensor],
                                   converted_state: Dict[str, torch.Tensor]):

    norm_name_pt = self._names.pre_cross_attn_norm.format(idx)
    q_proj_pt = self._names.cross_attn_q_proj.format(idx)
    out_proj_pt = self._names.cross_attn_out_proj.format(idx)

    converted_cross_attn_name = f"transformer_blocks.{idx}.pre_cross_atten_norm"
    q_cross_proj_name = f"transformer_blocks.{idx}.cross_atten_func.q_projection"
    out_cross_proj_name = f"transformer_blocks.{idx}.cross_atten_func.output_projection"

    converted_state[f"{converted_cross_attn_name}.weight"] = state.pop(
        f"model.decoder.{norm_name_pt}.weight"
    )
    converted_state[f"{converted_cross_attn_name}.bias"] = state.pop(
        f"model.decoder.{norm_name_pt}.bias"
    )
    converted_state[f"{q_cross_proj_name}.weight"] = state.pop(
        f"model.decoder.{q_proj_pt}.weight"
    )
    converted_state[f"{q_cross_proj_name}.bias"] = state.pop(
        f"model.decoder.{q_proj_pt}.bias"
    )
    converted_state[f"{out_cross_proj_name}.weight"] = state.pop(
        f"model.decoder.{out_proj_pt}.weight"
    )
    converted_state[f"{out_cross_proj_name}.bias"] = state.pop(
        f"model.decoder.{out_proj_pt}.bias"
    )


  def _get_loader(self) -> Callable[[str], Dict[str, torch.Tensor]]:
    """A best effort method for finding appropriate state loader.

    Raises:
        ValueError: If it fails to find an appropriate loader.

    Returns:
        Callable[[str], Dict[str, torch.Tensor]]: State loader to be used.
    """
    if os.path.isdir(self._file_name):
      if glob.glob(os.path.join(self._file_name, "*.safetensors")):
        return load_safetensors
      if glob.glob(os.path.join(self._file_name, "*.bin")) or glob.glob(
          os.path.join(self._file_name, "*pt")
      ):
        return load_pytorch_statedict

    if self._file_name.endswith(".safetensors"):
      return load_safetensors

    if self._file_name.endswith(".bin") or self._file_name.endswith("pt"):
      return load_pytorch_statedict

    raise ValueError("File format not supported.")

  def _map_feedforward(
      self,
      idx: int,
      config: Union[AudioEncoderConfig, TextDecoderConfig],
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
  ):
    prefix = f"transformer_blocks.{idx}"
    ff_config = config.block_config.ff_config
    if ff_config.type == model_config.FeedForwardType.SEQUENTIAL:
      ff_up_proj_name = self._names.ff_up_proj.format(idx)
      ff_down_proj_name = self._names.ff_down_proj.format(idx)
      converted_state[f"{prefix}.ff.w1.weight"] = state.pop(
          f"{self._MODEL_PREFIX}.{ff_up_proj_name}.weight"
      )
      converted_state[f"{prefix}.ff.w2.weight"] = state.pop(
          f"{self._MODEL_PREFIX}.{ff_down_proj_name}.weight"
      )
      if ff_config.use_bias:
        converted_state[f"{prefix}.ff.w1.bias"] = state.pop(
            f"{self._MODEL_PREFIX}.{ff_up_proj_name}.bias"
        )
        converted_state[f"{prefix}.ff.w2.bias"] = state.pop(
            f"{self._MODEL_PREFIX}.{ff_down_proj_name}.bias"
        )
    else:
        raise ValueError("Expected model_config.FeedForwardType.SEQUENTIAL")

  def _map_attention(
      self,
      idx: int,
      config: Union[AudioEncoderConfig, TextDecoderConfig],
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
  ):
    prefix = f"transformer_blocks.{idx}"
    attn_config = config.block_config.attn_config
    if hasattr(self._names, "attn_fused_qkv_proj"):
        if self._names.attn_fused_qkv_proj:
            fused_qkv_name = self._names.attn_fused_qkv_proj.format(idx)
            converted_state[f"{prefix}.atten_func.qkv_projection.weight"] = state.pop(
                f"{self._MODEL_PREFIX}.{fused_qkv_name}.weight"
            )
    else:
      q_name = self._names.attn_query_proj.format(idx)
      k_name = self._names.attn_key_proj.format(idx)
      v_name = self._names.attn_value_proj.format(idx)
      converted_state[f"{prefix}.atten_func.qkv_projection.weight"] = (
          self._fuse_qkv(
              attn_config,
              state.pop(f"{self._MODEL_PREFIX}.{q_name}.weight"),
              state.pop(f"{self._MODEL_PREFIX}.{k_name}.weight"),
              state.pop(f"{self._MODEL_PREFIX}.{v_name}.weight"),
          )
      )
    if attn_config.qkv_use_bias:
      if hasattr(self._names, "attn_fused_qkv_proj"):
          if self._names.attn_fused_qkv_proj:
              converted_state[f"{prefix}.atten_func.qkv_projection.bias"] = state.pop(
              f"{self._MODEL_PREFIX}.{fused_qkv_name}.bias"
              )
      else:

        k_bias = torch.zeros(state[f"{self._MODEL_PREFIX}.{q_name}.bias"].shape)
        converted_state[f"{prefix}.atten_func.qkv_projection.bias"] = (
            self._fuse_qkv(
                attn_config,
                state.pop(f"{self._MODEL_PREFIX}.{q_name}.bias"),
                k_bias,
                state.pop(f"{self._MODEL_PREFIX}.{v_name}.bias"),
            )
        )

    o_name = self._names.attn_output_proj.format(idx)
    converted_state[f"{prefix}.atten_func.output_projection.weight"] = (
        state.pop(f"{self._MODEL_PREFIX}.{o_name}.weight")
    )
    if attn_config.output_proj_use_bias:
      converted_state[f"{prefix}.atten_func.output_projection.bias"] = (
          state.pop(f"{self._MODEL_PREFIX}.{o_name}.bias")
      )

  def _map_norm(
      self,
      idx: int,
      config: Union[AudioEncoderConfig, TextDecoderConfig],
      state: Dict[str, torch.Tensor],
      converted_state: Dict[str, torch.Tensor],
  ):
    prefix = f"transformer_blocks.{idx}"
    if self._names.pre_attn_norm is not None:
      pre_attn_norm_name = self._names.pre_attn_norm.format(idx)
      converted_state[f"{prefix}.pre_atten_norm.weight"] = state.pop(
          f"{self._MODEL_PREFIX}.{pre_attn_norm_name}.weight"
      )
      if f"{self._MODEL_PREFIX}.{pre_attn_norm_name}.bias" in state:
        converted_state[f"{prefix}.pre_atten_norm.bias"] = state.pop(
            f"{self._MODEL_PREFIX}.{pre_attn_norm_name}.bias"
        )

    if self._names.pre_ff_norm is not None:
      pre_ff_norm_name = self._names.pre_ff_norm.format(idx)
      converted_state[f"{prefix}.ff.pre_ff_norm.weight"] = state.pop(
          f"{self._MODEL_PREFIX}.{pre_ff_norm_name}.weight"
      )
      if f"{self._MODEL_PREFIX}.{pre_ff_norm_name}.bias" in state:
        converted_state[f"{prefix}.ff.pre_ff_norm.bias"] = state.pop(
            f"{self._MODEL_PREFIX}.{pre_ff_norm_name}.bias"
        )

  def _fuse_qkv(
      self,
      attn_config: model_config.AttentionConfig,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
  ) -> torch.Tensor:
    if attn_config.qkv_fused_interleaved:
      q_per_kv = attn_config.num_heads // attn_config.num_query_groups
      qs = torch.split(q, attn_config.head_dim * q_per_kv)
      ks = torch.split(k, attn_config.head_dim)
      vs = torch.split(v, attn_config.head_dim)
      cycled = [t for group in zip(qs, ks, vs) for t in group]
      return torch.cat(cycled)
    else:
      return torch.cat([q, k, v], dim=0)
