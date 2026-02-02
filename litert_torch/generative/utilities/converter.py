# Copyright 2024 The LiteRT Torch Authors.
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

"""Common utility functions for model conversion."""

import enum
import os
import pathlib
import tempfile
from typing import Callable, Dict, Optional, Union
from absl import flags
from litert_torch._convert import interface as converter_utils
from litert_torch.generative.layers import kv_cache as kv_utils
from litert_torch.generative.layers import lora as lora_utils
import litert_torch.generative.layers.model_config as cfg
from litert_torch.generative.quantize import quant_attrs
from litert_torch.generative.quantize import quant_recipes
from litert_torch.generative.utilities import export_config as export_config_lib
from litert_torch.generative.utilities import litertlm_builder
from litert_torch.generative.utilities import loader
from litert_torch.quantize import quant_config as qcfg
import torch

ExportConfig = export_config_lib.ExportConfig


class ExportableModule(torch.nn.Module):

  def __init__(self, module, **extra_kwargs):
    super().__init__()
    self.module = module
    self.extra_kwargs = extra_kwargs

  def forward(self, *export_args, **export_kwargs):
    full_kwargs = {**export_kwargs, **self.extra_kwargs}
    return self.module(*export_args, **full_kwargs)


class QuantizationName(str, enum.Enum):
  """Strings for all supported quantization recipes.

  none: No quantization.
  dynamic_int8: Dynamic range quantization with int8 weights.
  weight_only_int8: Weight only quantization with int8 weights.
  fp16: Float16 quantization.
  dynamic_int4_block32: Dynamic range quantization with int4 weights and block
  size of 32, better model quality but slower inference.
  dynamic_int4_block128: Dynamic range quantization with int4 weights and block
  size of 128, faster inference but worse model quality.
  """

  NONE = 'none'
  DYNAMIC_INT8 = 'dynamic_int8'
  WEIGHT_ONLY_INT8 = 'weight_only_int8'
  FP16 = 'fp16'
  DYNAMIC_INT4_BLOCK32 = 'dynamic_int4_block32'
  DYNAMIC_INT4_BLOCK128 = 'dynamic_int4_block128'


def define_conversion_flags(
    model_name: str,
    default_mask_as_input: bool = False,
    default_transpose_kv_cache: bool = False,
):
  """Defines common flags used for model conversion."""

  flags.DEFINE_string(
      'checkpoint_path',
      os.path.join(pathlib.Path.home(), f'Downloads/llm_data/{model_name}'),
      'The path to the model checkpoint, or directory holding the checkpoint.',
  )
  flags.DEFINE_string(
      'output_path',
      '/tmp/',
      'The path to export the tflite model.',
  )
  flags.DEFINE_string(
      'output_name_prefix',
      f'{model_name}',
      'The prefix of the output tflite model name.',
  )
  flags.DEFINE_multi_integer(
      'prefill_seq_lens',
      (8, 64, 128, 256, 512, 1024),
      'List of the maximum sizes of prefill input tensors.',
  )
  flags.DEFINE_integer(
      'decode_batch_size',
      1,
      'The batch size for the decode signature.',
  )
  flags.DEFINE_integer(
      'kv_cache_max_len',
      1280,
      'The maximum size of KV cache buffer, including both prefill and decode.',
  )
  flags.DEFINE_string(
      'quantize',
      'dynamic_int8',
      'How the model should be quantized. Set to "none" to disable '
      'quantization. See `QuantizationName` for supported quantization types.',
  )
  flags.DEFINE_multi_integer(
      'lora_ranks',
      None,
      'If set, the model will be converted with the provided list of LoRA '
      'ranks.',
  )
  flags.DEFINE_bool(
      'mask_as_input',
      default_mask_as_input,
      'If true, the mask will be passed in as input. Otherwise, mask will be '
      'built by the model internally.',
  )
  flags.DEFINE_bool(
      'transpose_kv_cache',
      default_transpose_kv_cache,
      'If true, the model will be converted with transposed KV cache.',
  )
  flags.DEFINE_bool(
      'custom_checkpoint_loader',
      False,
      'If true, the conversion script will use a custom checkpoint loader '
      'which will read a checkpoint from a remote source.',
  )
  flags.DEFINE_bool(
      'gpu_dynamic_shapes',
      False,
      'It is to support dynamic shapes on GPU effectively. If true, the graph '
      'sets the actual kv_cache size and prefill lengths when the graph is '
      'initialized for inference based on the flags, `kv_cache_max_len` and '
      '`prefill_seq_lens` as the maximum of kv_cache size and prefill lengths '
      'in the graph.',
  )
  flags.DEFINE_bool(
      'export_gpu_dynamic_shape_verifications',
      False,
      'If true, the conversion script will export signatures used only for '
      'verification of GPU dynamic shapes.',
  )
  return flags


# Context length for verifying GPU dynamic shapes.
_CONTEXT_LENGTH_TO_VERIFY_MAGIC_NUMBERS = 1280
# Long prefill length for verifying GPU dynamic shapes.
_LONG_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS = 1024
# Short prefill length for verifying GPU dynamic shapes.
_SHORT_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS = 64


def is_magic_number_(num: int) -> bool:
  """Returns true if the number is a magic number, i.e. prime number > 10."""
  if num < 10:
    return False
  if num % 2 == 0:
    return False
  for i in range(3, int(num / 2), 2):
    if num % i == 0:
      return False
  return True


def get_magic_number_for(org_number: int) -> int:
  """Returns the magic number for the given original number."""
  while not is_magic_number_(org_number):
    org_number += 1
  return org_number


def get_mask_cache_size_from_flags() -> int:
  """Returns the mask cache size according to the flags."""
  if flags.FLAGS.mask_as_input:
    return 0
  if flags.FLAGS.gpu_dynamic_shapes:
    return get_magic_number_for(flags.FLAGS.kv_cache_max_len)
  return flags.FLAGS.kv_cache_max_len


def get_quant_recipe_from_flag(
    quantize: str,
    model_config: cfg.ModelConfig,
) -> Optional[qcfg.QuantConfig]:
  """Processes the quantization flag and returns the corresponding recipe.

  Args:
      quantize: The quantization type.

  Returns:
      The quantization recipe, or None if no quantization is needed.

  Raises:
      ValueError: If the quantization type is not supported.
  """
  match quantize:
    case QuantizationName.NONE:
      return None
    case QuantizationName.DYNAMIC_INT8:
      return quant_recipes.full_dynamic_recipe(mcfg=model_config)
    case QuantizationName.WEIGHT_ONLY_INT8:
      return quant_recipes.full_weight_only_recipe(mcfg=model_config)
    case QuantizationName.FP16:
      return quant_recipes.full_fp16_recipe()
    case QuantizationName.DYNAMIC_INT4_BLOCK32:
      return quant_recipes.full_dynamic_recipe(
          mcfg=model_config,
          weight_dtype=quant_attrs.Dtype.INT4,
          granularity=quant_attrs.Granularity.BLOCKWISE_32,
      )
    case QuantizationName.DYNAMIC_INT4_BLOCK128:
      return quant_recipes.full_dynamic_recipe(
          mcfg=model_config,
          weight_dtype=quant_attrs.Dtype.INT4,
          granularity=quant_attrs.Granularity.BLOCKWISE_128,
      )
    case _:
      raise ValueError(f'Unsupported quantization flag: {quantize}')


def create_quantize_suffix(quantize: str) -> str:
  """Creates a suffix for the output file name based on the quantization type.

  Args:
      quantize: The quantization type.

  Returns:
      A string representing the quantization suffix.

  Raises:
      ValueError: If the quantization type is not supported.
  """
  match quantize:
    case QuantizationName.NONE:
      return 'f32'
    case QuantizationName.DYNAMIC_INT8:
      return 'q8'
    case QuantizationName.WEIGHT_ONLY_INT8:
      return 'q8_wo'
    case QuantizationName.FP16:
      return 'fp16'
    case QuantizationName.DYNAMIC_INT4_BLOCK32:
      return 'q4_block32'
    case QuantizationName.DYNAMIC_INT4_BLOCK128:
      return 'q4_block128'
    case _:
      raise ValueError(f'Unsupported quantization flag: {quantize}')


def _build_mask(mask_len, kv_cache_max_len, causal_mask_value) -> torch.Tensor:
  if isinstance(mask_len, list):
    return [
        _build_mask(i, kv_cache_max_len, causal_mask_value) for i in mask_len
    ]

  mask = torch.full(
      (mask_len, kv_cache_max_len), causal_mask_value, dtype=torch.float32
  )
  return torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)


def convert_to_tflite(
    pytorch_model: torch.nn.Module,
    output_path: str,
    output_name_prefix: str,
    prefill_seq_len: Union[int, list[int]],
    kv_cache_max_len: int,
    pixel_values_size: torch.Size = None,
    pixel_seq_len: int = 0,
    quantize: str = 'dynamic_int8',
    config: cfg.ModelConfig = None,
    lora_ranks: Optional[list[int]] = None,
    export_config: ExportConfig = None,
    extra_model: torch.nn.Module = None,
    extra_prefill_seq_lens: list[int] = None,
    extra_kv_cache_max_len: int = 0,
    extra_signature_prefix: str = '',
):
  """Converts a nn.Module model to multi-signature tflite model.

  A PyTorch model will be converted to a tflite model with several signatures:
    * "prefill_[prefill_seq_len]" (or "prefill" if only one prefill_seq_len is
        passed),
    * "prefill_[preill_seq_len]_pixel" (or "prefill_pixel" if only one
        prefill_seq_len is passed) if num_pixel_values > 0, and
    * "decode".

  "prefill_[prefill_seq_len]" (or "prefill" if only one prefill_seq_len is
  passed) signature takes as a sample input:
    * a tensor of shape [1, prefill_seq_len] of token sequence,
    * a tensor of shape [1, prefill_seq_len] of token positions, and
    * an external KV cache.

  If num_pixel_values > 0, "prefill_[prefill_seq_len]_pixel" (or "prefill_pixel"
  if only one prefill_seq_len is passed) signature takes as a sample input:
    * a tensor of shape [1, prefill_seq_len] of token sequence,
    * a tensor of shape [1, prefill_seq_len] of token positions,
    * an external KV cache, and
    * a tensor of shape [1, num_pixel_values] of pixel values.

  "decode" signature takes as a sample input:
    * a tensor of shape [1, 1] of token sequence,
    * a tensor of shape [1, 1] of the token position, and
    * an external KV cache.

  The final tflite model will be exported to tflite_path.

  Args:
      pytorch_model (torch.nn.Module): PyTorch model to convert to tflite.
      output_path (str): The path to export the tflite model.
      output_name_prefix (str): The prefix of the tflite model name.
      prefill_seq_len (Union[int, list[int]]): The prefill sequence length to
        use. If a list, the model will have multiple prefill signatures.
      kv_cache_max_len (int): The maximum size of KV cache buffer, including
        both prefill and decode.
      pixel_values_size (torch.Size, optional): The size of pixel values to pass
        to the model. If None, the model is not expected to take pixel values.
      pixel_seq_len (int, optional): The length of pixel tokens, or pixel
        embeddings generated by the image encoder with pixel values. The actual
        length of prefill_seq_len will be added by pixel_seq_len when pixel
        values are passed.
      quantize (str, optional): The quantization type. Defaults to
        'dynamic_int8'.
      config (cfg.ModelConfig, optional): The model config used to configure KV
        cache. If None, it uses the config of the pytorch_model.
      lora_ranks (list[int], optional): The ranks of the LORA layers. If None,
        no LoRA signatures will be added.
      export_config (ExportConfig, optional): The export configuration. If None,
        it uses the default export configuration.
      extra_model (torch.nn.Module, optional): PyTorch model to export in
        addition to the pytorch_model. This model can have different
        prefill_seq_lens and kv_cache_max_len.
      extra_prefill_seq_lens (list[int], optional): The prefill sequence
        lengths for extra_model. Meaningful only when extra_model is not None.
      extra_kv_cache_max_len (int, optional): The maximum size of KV cache
        buffer for extra_model. Meaningful only when extra_model is not None.
      extra_signature_prefix (str, optional): The prefix of the extra model
        signatures. Meaningful only when extra_model is not None.
  """
  # pylint: disable=protected-access
  torch._dynamo.config.cache_size_limit = 64

  config = config if config else pytorch_model.config
  prefill_seq_lens = (
      [prefill_seq_len] if isinstance(prefill_seq_len, int) else prefill_seq_len
  )
  loras = [None]
  if lora_ranks is not None:
    for rank in lora_ranks:
      lora = lora_utils.LoRA.zeros(rank, config)
      loras.append(lora)

  quant_suffix = create_quantize_suffix(quantize)
  kv_size = kv_cache_max_len
  lora_suffix = (
      '' if not lora_ranks else f'_lora{",".join(map(str, lora_ranks))}'
  )

  if pixel_values_size is not None:
    assert pixel_seq_len > 0, 'pixel_seq_len must be greater than 0'
    max_prefill_seq_len = max(prefill_seq_lens)
    assert kv_size > max_prefill_seq_len + pixel_seq_len, (
        f'The KV cache size ({kv_size}) must be greater than the maximum '
        f'prefill sequence length ({max_prefill_seq_len}) + pixel sequence '
        f'length ({pixel_seq_len})'
    )

  if export_config is not None:
    if export_config.decode_batch_size > 1:
      output_name_prefix += f'_dbs{export_config.decode_batch_size}'

  output_filename = (
      f'{output_name_prefix}_{quant_suffix}_ekv{kv_size}{lora_suffix}.tflite'
  )
  output_file = os.path.join(output_path, output_filename)

  converter = converter_utils.Converter()
  _add_signatures(
      converter,
      pytorch_model,
      prefill_seq_lens,
      kv_cache_max_len,
      pixel_values_size,
      pixel_seq_len,
      config,
      loras,
      export_config,
  )

  if extra_model is not None and extra_prefill_seq_lens:
    _add_signatures(
        converter,
        extra_model,
        extra_prefill_seq_lens,
        extra_kv_cache_max_len,
        pixel_values_size,
        pixel_seq_len,
        config,
        loras,
        export_config,
        signature_prefix=extra_signature_prefix,
    )

  edge_model = converter.convert(
      quant_config=get_quant_recipe_from_flag(quantize, config),
  )
  edge_model.export(output_file)
  return output_file


def _add_signatures(
    converter: converter_utils.Converter,
    pytorch_model: torch.nn.Module,
    prefill_seq_lens: list[int],
    kv_cache_max_len: int,
    pixel_values_size: torch.Size,
    pixel_seq_len: int,
    config: cfg.ModelConfig,
    loras: list[None | lora_utils.LoRA],
    export_config: ExportConfig,
    signature_prefix: str = '',
):
  """Helper function to export a model to tflite."""
  prefill_tokens_list = []
  prefill_input_pos_list = []
  for seq_len in prefill_seq_lens:
    prefill_tokens_list.append(torch.full((1, seq_len), 0, dtype=torch.int))
    prefill_input_pos_list.append(torch.arange(0, seq_len, dtype=torch.int))

  prefill_pixel_values = None
  prefill_tokens_list_with_pixel = []
  prefill_input_pos_list_with_pixel = []
  if pixel_values_size is not None:
    prefill_pixel_values = torch.full(pixel_values_size, 0, dtype=torch.float32)
    for seq_len in prefill_seq_lens:
      prefill_tokens_list_with_pixel.append(
          torch.full((1, seq_len + pixel_seq_len), 0, dtype=torch.int)
      )
      prefill_input_pos_list_with_pixel.append(
          torch.arange(0, seq_len + pixel_seq_len, dtype=torch.int)
      )

  prefill_masks = None
  if export_config.mask_as_input:
    prefill_masks = _build_mask(
        prefill_seq_lens, kv_cache_max_len, config.causal_mask_value
    )
    if not isinstance(prefill_masks, list):
      prefill_masks = [prefill_masks]
    assert len(prefill_masks) == len(prefill_seq_lens)

  decode_token = torch.tensor(
      [[0] for _ in range(export_config.decode_batch_size)], dtype=torch.int
  )
  decode_input_pos = torch.tensor([0], dtype=torch.int)
  prefill_kv = kv_utils.KVCache.from_model_config(
      kv_cache_max_len, config, kv_layout=export_config.kvcache_layout
  )
  decode_kv = kv_utils.KVCache.from_model_config(
      kv_cache_max_len,
      config,
      batch_size=export_config.decode_batch_size,
      kv_layout=export_config.kvcache_layout,
  )

  # For export, we create a module that captures any non-exportable,
  # arugments, e.g. the generation config object.
  mod = ExportableModule(pytorch_model, export_config=export_config).eval()

  for lora in loras:
    for i in range(len(prefill_seq_lens)):
      prefill_seq_len = prefill_seq_lens[i]
      prefill_signature_name = f'{signature_prefix}prefill_{prefill_seq_len}'

      sample_kwargs = {
          'tokens': prefill_tokens_list[i],
          'input_pos': prefill_input_pos_list[i],
          'kv_cache': prefill_kv,
      }
      if prefill_masks is not None:
        sample_kwargs['mask'] = prefill_masks[i]

      if lora is not None:
        prefill_signature_name += f'_lora_r{lora.get_rank()}'
        sample_kwargs['lora'] = lora

      converter.add_signature(
          prefill_signature_name,
          mod,
          sample_kwargs=sample_kwargs,
      )

      if prefill_pixel_values is not None:
        sample_pixel_kwargs = {
            'tokens': prefill_tokens_list_with_pixel[i],
            'input_pos': prefill_input_pos_list_with_pixel[i],
            'kv_cache': prefill_kv,
            'pixel_values': prefill_pixel_values,
        }
        # mask should be built internally when pixel values are passed.
        if lora is not None:
          sample_pixel_kwargs['lora'] = lora
        converter.add_signature(
            prefill_signature_name + '_pixel',
            mod,
            sample_kwargs=sample_pixel_kwargs,
        )

    sample_kwargs = {
        'tokens': decode_token,
        'input_pos': decode_input_pos,
        'kv_cache': decode_kv,
    }
    if export_config.mask_as_input:
      # Note that the decode mask is not a correct causal mask, but it is okay
      # for the conversion purpose because only the shape matters in conversion.
      # A correct causal mask of decode for a given token position of decode, it
      # should be built like:
      #
      #  torch.triu(mask, diagonal=decode_position).unsqueeze(0).unsqueeze(0)
      #
      sample_kwargs['mask'] = _build_mask(
          1, kv_cache_max_len, config.causal_mask_value
      )
    if lora is not None:
      sample_kwargs['lora'] = lora

    decode_signature_name = f'{signature_prefix}decode'
    if lora is not None:
      decode_signature_name += f'_lora_r{lora.get_rank()}'
    converter.add_signature(
        decode_signature_name,
        mod,
        sample_kwargs=sample_kwargs,
    )


def build_and_convert_to_tflite_from_flags(
    model_builder: Callable[
        [str, Callable[[str], Dict[str, torch.Tensor]], int], torch.nn.Module
    ],
    checkpoint_path: str = None,
    output_name_prefix: str = None,
):
  """Builds a nn.Module model and converts it according to the flags."""
  if checkpoint_path is None:
    checkpoint_path = flags.FLAGS.checkpoint_path
  if output_name_prefix is None:
    output_name_prefix = flags.FLAGS.output_name_prefix

  pytorch_model = model_builder(
      checkpoint_path,
      loader.maybe_get_custom_loader(
          checkpoint_path, flags.FLAGS.custom_checkpoint_loader
      ),
      get_mask_cache_size_from_flags(),
  )

  # Extra model for GPU dynamic shape verification if needed.
  extra_model = None
  extra_prefill_seq_lens = None
  extra_kv_cache_max_len = 0
  if flags.FLAGS.gpu_dynamic_shapes:
    prefill_seq_lens = [
        get_magic_number_for(l) for l in flags.FLAGS.prefill_seq_lens
    ]
    kv_cache_max_len = get_magic_number_for(flags.FLAGS.kv_cache_max_len)

    if flags.FLAGS.export_gpu_dynamic_shape_verifications:
      extra_kv_cache_max_len = _CONTEXT_LENGTH_TO_VERIFY_MAGIC_NUMBERS
      if extra_kv_cache_max_len > flags.FLAGS.kv_cache_max_len:
        extra_kv_cache_max_len = flags.FLAGS.kv_cache_max_len
      extra_model = model_builder(
          checkpoint_path,
          loader.maybe_get_custom_loader(
              checkpoint_path, flags.FLAGS.custom_checkpoint_loader
          ),
          extra_kv_cache_max_len,
      )
      extra_prefill_seq_lens = []
      if extra_kv_cache_max_len > _SHORT_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS:
        extra_prefill_seq_lens.append(
            _SHORT_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS
        )
      if extra_kv_cache_max_len > _LONG_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS:
        extra_prefill_seq_lens.append(
            _LONG_PREFILL_LENGTH_TO_VERIFY_MAGIC_NUMBERS
        )
  else:
    prefill_seq_lens = flags.FLAGS.prefill_seq_lens
    kv_cache_max_len = flags.FLAGS.kv_cache_max_len

  convert_to_tflite(
      pytorch_model,
      output_path=flags.FLAGS.output_path,
      output_name_prefix=output_name_prefix,
      prefill_seq_len=prefill_seq_lens,
      kv_cache_max_len=kv_cache_max_len,
      quantize=flags.FLAGS.quantize,
      lora_ranks=flags.FLAGS.lora_ranks,
      export_config=export_config_lib.get_from_flags(),
      extra_model=extra_model,
      extra_prefill_seq_lens=extra_prefill_seq_lens,
      extra_kv_cache_max_len=extra_kv_cache_max_len,
      extra_signature_prefix='test_' if extra_model is not None else '',
  )


def convert_to_litert(
    pytorch_model: torch.nn.Module,
    output_path: str,
    output_name_prefix: str,
    prefill_seq_len: Union[int, list[int]],
    kv_cache_max_len: int,
    pixel_values_size: torch.Size = None,
    pixel_seq_len: int = 0,
    quantize: str = 'dynamic_int8',
    config: cfg.ModelConfig = None,
    lora_ranks: Optional[list[int]] = None,
    export_config: ExportConfig = None,
    output_format: str = 'tflite',
    **kwargs,
):
  """Converts a nn.Module model to multi-signature tflite model and pack it."""
  with tempfile.TemporaryDirectory() as workdir:
    if output_format == 'litertlm':
      tflite_model_output_path = workdir
    else:
      tflite_model_output_path = output_path
    tflite_model_path = convert_to_tflite(
        pytorch_model,
        tflite_model_output_path,
        output_name_prefix,
        prefill_seq_len,
        kv_cache_max_len,
        pixel_values_size,
        pixel_seq_len,
        quantize,
        config,
        lora_ranks,
        export_config,
    )
    if output_format == 'litertlm':
      tokenizer_model_path = kwargs.pop('tokenizer_model_path', None)
      hf_tokenizer_model_path = kwargs.pop('hf_tokenizer_model_path', None)
      litertlm_builder.build_litertlm(
          tflite_model_path=tflite_model_path,
          workdir=workdir,
          output_path=output_path,
          context_length=kv_cache_max_len,
          tokenizer_model_path=tokenizer_model_path,
          hf_tokenizer_model_path=hf_tokenizer_model_path,
          **kwargs,
      )
