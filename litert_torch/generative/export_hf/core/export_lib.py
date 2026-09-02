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
"""Export library for HF integration."""

import contextlib
import dataclasses
import gc
import json
import os
from typing import Any

import huggingface_hub
from litert_torch import fx_infra
from litert_torch import progress
from litert_torch._convert import interface as converter_utils
from litert_torch.backend.experimental import torch_tfl
from litert_torch.generative.export_hf.core import attention as _
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import patches as _
from litert_torch.generative.export_hf.core import utils
from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_module
from litert_torch.generative.export_hf.core.external_rope import exportable_module as external_rope_module
from litert_torch.generative.export_hf.core.external_rope import preprocess_model as external_rope_preprocess_model
from litert_torch.generative.export_hf.core.mu import mu_pass_lib
from litert_torch.generative.export_hf.core.split_cache import attention as _
from litert_torch.generative.export_hf.core.split_cache import exportable_module as split_cache_module
from litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler import litert_lm_npu_compiler
from litert_torch.generative.export_hf.model_ext import exportables as model_ext_exportables
from litert_torch.generative.export_hf.model_ext import extension as model_ext_extension
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as tokenizer_lib
import torch
from torch import nn
import transformers

from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.core import aot_types
from ai_edge_litert.aot.vendors import import_vendor
from ai_edge_quantizer import quantizer as quantizer_lib
from ai_edge_quantizer import recipe as recipe_lib

ExportTask = exportable_module_config.ExportTask


@dataclasses.dataclass
class SourceModelArtifacts:
  """Source model artifacts."""

  model: torch.nn.Module
  model_config: transformers.PretrainedConfig
  text_model_config: transformers.PretrainedConfig
  tokenizer: transformers.PreTrainedTokenizerBase

  image_processor: transformers.AutoImageProcessor | None = None


@dataclasses.dataclass
class ExportedModelArtifacts:
  """Exported model artifacts."""

  prefill_decode_model_path: str | None = None
  embedder_model_path: str | None = None
  vision_encoder_model_path: str | None = None
  vision_adapter_model_path: str | None = None
  eoi_model_path: str | None = None
  auxiliary_model_path: str | None = None
  tokenizer_model_path: str | None = None
  additional_model_paths: dict[str, str] | None = None

  litert_lm_model_path: str | None = None


def verify_model_compatibility(model, model_config, text_model_config):
  """Verifies model compatibility."""
  # Validating compatibility...
  # NOTE: Currently we don't throw errors for model incompatibilities.
  rope_type = getattr(text_model_config, 'rope_type', 'default')
  if 'dynamic' in rope_type or 'longrope' in rope_type:
    print(utils.ERROR_MESSAGE)
    print('Dynamic and longrope are not supported yet.')
    raise NotImplementedError('Dynamic and longrope are not supported yet.')
  can_compile_fullgraph = getattr(model, '_can_compile_fullgraph', None)
  if can_compile_fullgraph is None:
    print(utils.WARNING_MESSAGE)
    print(
        "Model didn't specify _can_compile_fullgraph. It might not be"
        ' exportable.'
    )
  elif not can_compile_fullgraph:
    print(utils.ERROR_MESSAGE)
    print('Model is not fully compilable.')

  supports_attention_backend = getattr(
      model, '_supports_attention_backend', None
  )
  if supports_attention_backend is None:
    print(utils.WARNING_MESSAGE)
    print(
        "Model didn't specify supports_attention_backend. It might not be"
        ' correctly exported.'
    )
  elif not supports_attention_backend:
    print(utils.ERROR_MESSAGE)
    print('Model does not support attention backend.')

  if (
      hasattr(model_config, 'quantization_config')
      and model_config.quantization_config
  ):
    print(utils.ERROR_MESSAGE)
    raise NotImplementedError('Quantized checkpoint is not supported yet.')


@contextlib.contextmanager
def patch_builtin_tuple_for_export():
  """Temporarily injects a .to() method into Python's built-in tuple type.

  Safely removes it upon exiting the scope.
  """
  tuple_dict = gc.get_referents(tuple.__dict__)[0]

  had_original = 'to' in tuple_dict
  original_to = tuple_dict.get('to')

  def tuple_to(self, *args, **kwargs):
    return tuple(
        item.to(*args, **kwargs) if hasattr(item, 'to') else item
        for item in self
    )

  tuple_dict['to'] = tuple_to

  try:
    yield
  finally:
    if had_original:
      tuple_dict['to'] = original_to
    else:
      del tuple_dict['to']


def pre_split_model_experts(model: nn.Module) -> nn.Module:
  """Splits 3D expert weight tensors into separate parameters to bypass LiteRT constant folding."""
  for module in model.modules():
    # Identify native MoE Experts modules
    if (
        hasattr(module, 'gate_up_proj')
        and getattr(module, 'num_experts', None) is not None
    ):

      # 1. Pre-split gate_up_proj [num_experts, out_features, in_features]
      gate_up_splits = torch.unbind(module.gate_up_proj.data, dim=0)
      module.split_gate_up_proj = nn.ParameterList(
          [nn.Parameter(w) for w in gate_up_splits]
      )
      delattr(module, 'gate_up_proj')  # Deletes the original 3D parameter

      # 2. Pre-split down_proj [num_experts, out_features, in_features]
      down_splits = torch.unbind(module.down_proj.data, dim=0)
      module.split_down_proj = nn.ParameterList(
          [nn.Parameter(w) for w in down_splits]
      )
      delattr(module, 'down_proj')

      # 3. Pre-split optional biases if they exist
      if getattr(module, 'gate_up_bias', None) is not None:
        gate_up_bias_splits = torch.unbind(module.gate_up_bias.data, dim=0)
        module.split_gate_up_bias = nn.ParameterList(
            [nn.Parameter(b) for b in gate_up_bias_splits]
        )
        delattr(module, 'gate_up_bias')

      if getattr(module, 'down_bias', None) is not None:
        down_bias_splits = torch.unbind(module.down_bias.data, dim=0)
        module.split_down_bias = nn.ParameterList(
            [nn.Parameter(b) for b in down_bias_splits]
        )
        delattr(module, 'down_bias')

  return model


@progress.task('Load source model')
def load_model(
    model_path: str,
    export_config: exportable_module.ExportableModuleConfig,
    trust_remote_code: bool = False,
    auto_model_override: str | None = None,
    task: ExportTask | str = ExportTask.TEXT_GENERATION,
) -> SourceModelArtifacts:
  """Loads model from checkpoint."""

  try:
    config = transformers.AutoConfig.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=trust_remote_code,
    )
  except (KeyError, ValueError):
    # Fallback to PretrainedConfig if the model architecture is not built into
    # transformers AutoConfig.
    config_dict, _ = transformers.PretrainedConfig.get_config_dict(
        model_path, trust_remote_code=trust_remote_code
    )
    config = transformers.PretrainedConfig.from_dict(config_dict)

  # Opt-in to global access for per-layer config attributes to avoid blocking heterogeneous pipelines
  if hasattr(config, 'allow_global_per_layer_attribute_access'):
    config.allow_global_per_layer_attribute_access = True
  if hasattr(config, 'text_config') and hasattr(
      config.text_config, 'allow_global_per_layer_attribute_access'
  ):
    config.text_config.allow_global_per_layer_attribute_access = True

  if task == ExportTask.AUTOMATIC_SPEECH_RECOGNITION:
    model_cls = model_ext_exportables.get_speech_model_cls(config.model_type)
    model = model_cls(model_path, override_transformers=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    return SourceModelArtifacts(
        model=model,
        model_config=config,
        text_model_config=config,
        tokenizer=tokenizer,  # pyrefly: ignore[bad-argument-type]
    )

  if task == ExportTask.TEXT_TO_SPEECH:
    model_cls = model_ext_exportables.get_tts_model_cls(config.model_type)
    model = model_cls(model_path, export_config=export_config)
    try:
      tokenizer = transformers.AutoTokenizer.from_pretrained(
          model_path, trust_remote_code=trust_remote_code
      )
    except Exception:  # pylint: disable=broad-exception-caught
      tokenizer = None
    return SourceModelArtifacts(
        model=model,
        model_config=config,
        text_model_config=config,
        tokenizer=tokenizer,  # pyrefly: ignore[bad-argument-type]
    )

  config._attn_implementation = 'lrt_transposed_attention'  # pylint: disable=protected-access

  if export_config.moe_exports_implementation:
    config._experts_implementation = export_config.moe_exports_implementation  # pylint: disable=protected-access
    if hasattr(config, 'text_config'):
      config.text_config._experts_implementation = export_config.moe_exports_implementation  # pylint: disable=protected-access

  if task == ExportTask.TEXT_GENERATION:
    auto_model_cls = transformers.AutoModelForCausalLM
  elif task == ExportTask.IMAGE_TEXT_TO_TEXT:
    auto_model_cls = transformers.AutoModelForImageTextToText
  else:
    raise ValueError(f'Unsupported task: {task}')
  if auto_model_override is not None:
    auto_model_cls = transformers.__dict__[auto_model_override]

  with model_ext_patches.get_patch_context(config.model_type):
    if export_config.use_random_weights:
      model = auto_model_cls.from_config(
          config=config,
          torch_dtype=torch.float32,
          trust_remote_code=trust_remote_code,
      )
    else:
      model = auto_model_cls.from_pretrained(
          model_path,
          config=config,
          torch_dtype=torch.float32,
          trust_remote_code=trust_remote_code,
      )

  if task == ExportTask.TEXT_GENERATION:
    model.generation_config.cache_implementation = 'static'
    model.generation_config.do_sample = False

  text_model_config = config
  if hasattr(config, 'text_config'):
    text_model_config = config.text_config

  if task == ExportTask.TEXT_GENERATION:
    verify_model_compatibility(model, config, text_model_config)
  else:
    # TODO(weiyiw): Add support for other tasks.
    pass

  if task == ExportTask.IMAGE_TEXT_TO_TEXT:
    image_processor = transformers.AutoImageProcessor.from_pretrained(
        model_path
    )
  else:
    image_processor = None

  # TODO(weiyiw): Refactor into a separate function.
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
  if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
    try:
      if utils.get_model_path_type(model_path) == 'repo_id':
        template_file = huggingface_hub.hf_hub_download(
            model_path, filename='chat_template.json'
        )
      else:
        template_file = os.path.join(model_path, 'chat_template.json')
      with open(template_file, 'rt') as f:
        chat_template_str = f.read()
      chat_template_dict = json.loads(chat_template_str)
      if 'chat_template' in chat_template_dict:
        tokenizer.chat_template = chat_template_dict['chat_template']  # pyrefly: ignore[missing-attribute]
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Failed to load chat template: {e}')

  if export_config.moe_exports_implementation == 'litert_moe_sequential':
    model = pre_split_model_experts(model)

  return SourceModelArtifacts(
      model=model,
      model_config=config,
      text_model_config=text_model_config,
      tokenizer=tokenizer,  # pyrefly: ignore[bad-argument-type]
      image_processor=image_processor,
  )


def update_export_config(
    export_config: exportable_module.ExportableModuleConfig,
    source_model_artifacts: SourceModelArtifacts,
) -> exportable_module.ExportableModuleConfig:
  """Updates export config."""
  return model_ext_extension.update_export_config(
      export_config, source_model_artifacts.model_config
  )


def get_prefill_decode_exportable_cls(
    model_config: transformers.PretrainedConfig,
    export_config: exportable_module.ExportableModuleConfig,
):
  """Gets exportable module class."""
  model_specific_exportables = (
      model_ext_exportables.get_prefill_decode_exportables(
          model_config, export_config
      )
  )
  if model_specific_exportables:
    return model_specific_exportables
  if export_config.split_cache:
    return (
        split_cache_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMPrefill,
        split_cache_module.LiteRTSplitCacheExportableModuleForDecoderOnlyLMGenerate,
    )
  elif export_config.externalize_embedder:
    return (
        external_emb_module.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder,
        external_emb_module.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder,
    )
  else:
    return (
        exportable_module.LiteRTExportableModuleForDecoderOnlyLMPrefill,
        exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate,
    )


def _decode_export_config(
    export_config: exportable_module.ExportableModuleConfig,
) -> exportable_module.ExportableModuleConfig:
  """Returns the export config to use for the decode signature.

  Substitutes `decode_cache_length` and `decode_sliding_window_ring_buffer_size`
  for their prefill counterparts; see `ExportableModuleConfig` for what each
  means. Returns `export_config` itself when neither is set, so callers can
  `is`-check to skip rebuilding decode-only modules.
  """
  replacements = {}
  if export_config.decode_cache_length is not None:
    replacements['cache_length'] = export_config.decode_cache_length
  if export_config.decode_sliding_window_ring_buffer_size is not None:
    replacements['sliding_window_ring_buffer_size'] = (
        export_config.decode_sliding_window_ring_buffer_size
    )
  if not replacements:
    return export_config
  return dataclasses.replace(export_config, **replacements)


@progress.task('Export text prefill-decode model')
def export_text_prefill_decode_model(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
):
  """Exports text model to tflite."""
  model = source_model_artifacts.model

  # Patch model instance for export.
  with model_ext_patches.patch_model(
      model, source_model_artifacts.model_config.model_type, export_config
  ):
    text_model_config = source_model_artifacts.text_model_config
    quantization_recipe = export_config.quantization_recipe
    work_dir = export_config.work_dir
    has_dynamic_shape = (
        export_config.cache_length_dim is not None
        or export_config.prefill_length_dim is not None
    )
    if export_config.externalize_rope:
      model = external_rope_preprocess_model.inject_rotary_position_embedding(
          model
      )
    if export_config.split_cache:
      assert (
          not has_dynamic_shape
      ), 'Dynamic shape is not supported for split cache.'
      model.set_attn_implementation('lrt_split_cache_attention')
      # In case of the attn_implementation is not set.
      model.config._attn_implementation = 'lrt_split_cache_attention'  # pylint: disable=protected-access  # pyrefly: ignore[missing-attribute]
    else:
      model.set_attn_implementation('lrt_transposed_attention')

    prefill_module_cls, decode_module_cls = get_prefill_decode_exportable_cls(
        source_model_artifacts.model_config, export_config
    )
    prefill_module = prefill_module_cls(
        model, export_config, source_model_artifacts
    )
    decode_module = decode_module_cls(
        model, _decode_export_config(export_config), source_model_artifacts
    )
    converter = converter_utils.Converter()
    sample_prefill_inputs = prefill_module.get_sample_inputs(text_model_config)
    for signature_name, (
        sample_prefill_inputs,
        prefill_dynamic_shapes,
    ) in sample_prefill_inputs.items():
      if has_dynamic_shape:
        prefill_ep = torch.export.export(
            prefill_module,
            args=(),
            kwargs=sample_prefill_inputs,
            dynamic_shapes=prefill_dynamic_shapes,
        )

        prefill_ep = fx_infra.safe_run_decompositions(
            prefill_ep, fx_infra.decomp.pre_lower_decomp()
        )

        prefill_ep = prefill_ep.run_decompositions(torch_tfl.decomps)

        converter.add_signature(
            signature_name,
            prefill_ep.module(),
            sample_kwargs=sample_prefill_inputs,
            dynamic_shapes=prefill_dynamic_shapes,
        )
      else:
        converter.add_signature(
            signature_name,
            prefill_module.eval(),
            sample_kwargs=sample_prefill_inputs,
        )
    for signature_name, (
        sample_decode_inputs,
        decode_dynamic_shapes,
    ) in decode_module.get_sample_inputs(text_model_config).items():
      if has_dynamic_shape:
        decode_ep = torch.export.export(
            decode_module,
            args=(),
            kwargs=sample_decode_inputs,
            dynamic_shapes=decode_dynamic_shapes,
        )

        decode_ep = fx_infra.safe_run_decompositions(
            decode_ep, fx_infra.decomp.pre_lower_decomp()
        )

        decode_ep = decode_ep.run_decompositions(torch_tfl.decomps)

        converter.add_signature(
            signature_name,
            decode_ep.module(),
            sample_kwargs=sample_decode_inputs,
            dynamic_shapes=decode_dynamic_shapes,
        )
      else:
        converter.add_signature(
            signature_name,
            decode_module.eval(),
            sample_kwargs=sample_decode_inputs,
        )

    with patch_builtin_tuple_for_export():
      lrt_model = converter.convert(
          lightweight_conversion=export_config.experimental_lightweight_conversion,
          strict_export=False,
      )

  lrt_model = mu_pass_lib.update_model(lrt_model)  # pyrefly: ignore[bad-argument-type]
  if export_config.experimental_use_mixed_precision:
    print('Applying mixed precision to model...')
    lrt_model = mu_pass_lib.apply_mixed_precision(lrt_model)

  model_path = os.path.join(work_dir, 'model.tflite')  # pyrefly: ignore[no-matching-overload]
  lrt_model.export(model_path)

  del lrt_model
  del converter
  gc.collect()

  # Quantization
  quantization_recipe_list = (
      quantization_recipe.split(',') if quantization_recipe else [None]
  )
  for recipe in quantization_recipe_list:
    model_path = maybe_quantize_model(model_path, recipe)
    gc.collect()

  return dataclasses.replace(
      exported_model_artifacts,
      prefill_decode_model_path=model_path,
  )


def maybe_quantize_model(
    model_path: str,
    quantization_recipe: str | None = None,
):
  """Quantizes model if recipe is provided."""
  if not quantization_recipe or str(quantization_recipe).strip().lower() in (
      'none',
      'null',
      'false',
      '',
  ):
    return model_path
  return quantize_model(model_path, quantization_recipe)


def _dynamic_wi8_emb4_afp32():
  """Local recipe with 4-bit embedding tables and 8-bit fully connected weights."""
  return recipe_lib.dynamic_wi4c_afp32(
      operation_name=recipe_lib.TFLOperationName.EMBEDDING_LOOKUP,
  ) + recipe_lib.dynamic_wi8c_afp32(
      operation_name=recipe_lib.TFLOperationName.FULLY_CONNECTED,
  )


_LOCAL_QUANTIZATION_RECIPES = {
    'dynamic_wi8_emb4_afp32': _dynamic_wi8_emb4_afp32,
}


@progress.task('Quantize model')
def quantize_model(
    model_path: str,
    quantization_recipe: str,
):
  """Quantizes model."""
  quantized_model_path = (
      model_path.removesuffix('.tflite').removesuffix('_quantized')
      + '_quantized.tflite'
  )
  qt = quantizer_lib.Quantizer(model_path)
  try:
    if quantization_recipe.endswith('.json'):
      recipe = quantization_recipe
    elif quantization_recipe in _LOCAL_QUANTIZATION_RECIPES:
      recipe = _LOCAL_QUANTIZATION_RECIPES[quantization_recipe]()
    else:
      recipe = recipe_lib.__dict__[quantization_recipe]()
    qt.load_quantization_recipe(recipe)
  except Exception as e:
    raise ValueError(
        f'Invalid quantization recipe: {quantization_recipe}. Please check'
        ' the recipe name.'
    ) from e
  qt.quantize().export_model(quantized_model_path, overwrite=True)
  return quantized_model_path


@progress.task('Export embedder model')
def export_embedder_model(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
):
  """Exports embedder."""
  model = source_model_artifacts.model
  text_model_config = source_model_artifacts.text_model_config
  quantization_recipe = export_config.quantization_recipe
  work_dir = export_config.work_dir

  # Patch model instance for export.
  with model_ext_patches.patch_model(
      model, source_model_artifacts.model_config.model_type, export_config
  ):
    embedder_module = external_emb_module.LiteRTExportableModuleForEmbedder(
        model.get_input_embeddings()
    )
    converter = converter_utils.Converter()
    sample_inputs = embedder_module.get_sample_inputs(
        text_model_config, export_config
    )
    for signature_name, (sample_inputs, _) in sample_inputs.items():
      converter.add_signature(
          signature_name,
          embedder_module.eval(),
          sample_kwargs=sample_inputs,
      )
    lrt_model = converter.convert(
        lightweight_conversion=export_config.experimental_lightweight_conversion,
        strict_export=False,
    )
  model_path = os.path.join(work_dir, 'embedder.tflite')  # pyrefly: ignore[no-matching-overload]
  lrt_model.export(model_path)
  quantization_recipe_list = (
      quantization_recipe.split(',') if quantization_recipe else [None]
  )
  for recipe in quantization_recipe_list:
    model_path = maybe_quantize_model(model_path, recipe)
    gc.collect()
  return dataclasses.replace(
      exported_model_artifacts,
      embedder_model_path=model_path,
  )


@progress.task('Export vision encoder models')
def export_vision_encoder_models(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
):
  """Exports vision encoder models."""
  model = source_model_artifacts.model
  image_processor = source_model_artifacts.image_processor
  model_config = source_model_artifacts.model_config
  tokenizer = source_model_artifacts.tokenizer
  quantization_recipe = (
      export_config.vision_encoder_quantization_recipe
      or export_config.quantization_recipe
  )
  work_dir = export_config.work_dir

  model.set_attn_implementation('eager')
  encoder_module_cls, adapter_module_cls, eoi_module_cls = (
      model_ext_exportables.get_vision_exportables(model_config)
  )
  encode_module = encoder_module_cls(model, export_config)
  if adapter_module_cls is not None:
    adapter_module = adapter_module_cls(model, export_config, tokenizer)
  else:
    adapter_module = None
  if eoi_module_cls is not None:
    eoi_module = eoi_module_cls(model, export_config, tokenizer)
  else:
    eoi_module = None
  converter = converter_utils.Converter()
  sample_inputs = encode_module.get_sample_inputs(
      model_config,
      image_processor=image_processor,
      **export_config.extra_kwargs,
  )
  for signature_name, (sample_inputs, _) in sample_inputs.items():
    converter.add_signature(
        signature_name,
        encode_module.eval(),
        sample_kwargs=sample_inputs,
    )
  lrt_model = converter.convert(strict_export=False)
  vision_encoder_path = os.path.join(work_dir, 'vision_encoder.tflite')  # pyrefly: ignore[no-matching-overload]
  lrt_model.export(vision_encoder_path)
  quantization_recipe_list = (
      quantization_recipe.split(',') if quantization_recipe else [None]
  )
  for recipe in quantization_recipe_list:
    vision_encoder_path = maybe_quantize_model(vision_encoder_path, recipe)
    gc.collect()

  if adapter_module:
    converter = converter_utils.Converter()
    sample_inputs = adapter_module.get_sample_inputs(
        model_config,
        image_processor=image_processor,
        **export_config.extra_kwargs,
    )
    for signature_name, (sample_inputs, _) in sample_inputs.items():
      converter.add_signature(
          signature_name,
          adapter_module.eval(),
          sample_kwargs=sample_inputs,
      )
    lrt_model = converter.convert(strict_export=False)
    adapter_path = os.path.join(work_dir, 'vision_adapter.tflite')  # pyrefly: ignore[no-matching-overload]
    lrt_model.export(adapter_path)
    quantization_recipe_list = (
        quantization_recipe.split(',') if quantization_recipe else [None]
    )
    for recipe in quantization_recipe_list:
      adapter_path = maybe_quantize_model(adapter_path, recipe)
      gc.collect()
  else:
    adapter_path = None

  if eoi_module:
    converter = converter_utils.Converter()
    sample_inputs = eoi_module.get_sample_inputs(
        model_config,
        **export_config.extra_kwargs,
    )
    for signature_name, (sample_inputs, _) in sample_inputs.items():
      converter.add_signature(
          signature_name,
          eoi_module.eval(),
          sample_kwargs=sample_inputs,
      )
    lrt_model = converter.convert(strict_export=False)
    eoi_path = os.path.join(work_dir, 'eoi.tflite')  # pyrefly: ignore[no-matching-overload]
    lrt_model.export(eoi_path)
  else:
    eoi_path = None

  return dataclasses.replace(
      exported_model_artifacts,
      vision_encoder_model_path=vision_encoder_path,
      vision_adapter_model_path=adapter_path,
      eoi_model_path=eoi_path,
  )


@progress.task('Export ASR models')
def export_asr_models(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
):
  """Exports ASR models."""
  asr_model = source_model_artifacts.model
  model_config = source_model_artifacts.model_config
  quantization_recipe = export_config.quantization_recipe
  work_dir = export_config.work_dir

  exportables = model_ext_exportables.get_speech_exportables(model_config)
  encode_module_cls = exportables[0]
  decode_module_cls: Any = exportables[1] if len(exportables) > 1 else None

  encode_module = encode_module_cls(asr_model, export_config)
  encoder_sample_inputs_dict = encode_module.get_sample_inputs(model_config)
  encoder_inputs, _ = encoder_sample_inputs_dict['encode']

  decode_module = None
  if decode_module_cls is not None:
    with torch.no_grad():
      encoder_output = encode_module(*encoder_inputs)
    decode_module = decode_module_cls(
        asr_model, export_config, encoder_output=encoder_output
    )

  converter = converter_utils.Converter()

  converter.add_signature(
      'encode',
      encode_module.eval(),
      sample_args=encoder_inputs,
  )

  if decode_module is not None:
    decoder_sample_inputs_dict = decode_module.get_sample_inputs(model_config)
    for signature_name, (sample_args, _) in decoder_sample_inputs_dict.items():
      converter.add_signature(
          signature_name,
          decode_module.eval(),
          sample_args=sample_args,
      )

  with patch_builtin_tuple_for_export():
    lrt_model = converter.convert(
        lightweight_conversion=export_config.experimental_lightweight_conversion,
        strict_export=False,
    )

  lrt_model = mu_pass_lib.update_model(lrt_model)  # pyrefly: ignore[bad-argument-type]
  if export_config.experimental_use_mixed_precision:
    print('Applying mixed precision to model...')
    lrt_model = mu_pass_lib.apply_mixed_precision(lrt_model)

  model_path = os.path.join(work_dir, 'asr_model.tflite')  # pyrefly: ignore[no-matching-overload]
  lrt_model.export(model_path)

  del lrt_model
  del converter
  gc.collect()

  quantization_recipe_list = (
      quantization_recipe.split(',') if quantization_recipe else [None]
  )
  for recipe in quantization_recipe_list:
    model_path = maybe_quantize_model(model_path, recipe)
    gc.collect()

  return dataclasses.replace(
      exported_model_artifacts,
      prefill_decode_model_path=model_path,
  )


@progress.task('Export TTS models')
def export_tts_models(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
):
  """Exports TTS models."""
  tts_model = source_model_artifacts.model
  artifacts = tts_model.export(export_config)
  return dataclasses.replace(
      exported_model_artifacts,
      additional_model_paths=artifacts,
  )


@progress.task('Export auxiliary model')
def export_auxiliary_model(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
):
  """Exports auxiliary model."""
  model = source_model_artifacts.model
  text_model_config = source_model_artifacts.text_model_config
  work_dir = export_config.work_dir
  converter = converter_utils.Converter()
  # RoPE
  rope_module = external_rope_module.RoPEEmbedder(model)
  sample_inputs = rope_module.get_sample_inputs(
      text_model_config, export_config
  )
  for signature_name, (sample_input, _) in sample_inputs.items():
    converter.add_signature(
        signature_name,
        rope_module.eval(),
        sample_kwargs=sample_input,
    )
  # Attention Mask
  sliding_window_sizes = [getattr(text_model_config, 'sliding_window', None)]
  # Mask widths are baked into the builder at construction, so a decode-only
  # size needs its own builder for the `decode_mask` signature.
  decode_config = _decode_export_config(export_config)
  prefill_mask_module = split_cache_module.SplitAttentionMaskBuilder(
      export_config,
      sliding_window_sizes=sliding_window_sizes,
  )
  decode_mask_module = (
      prefill_mask_module
      if decode_config is export_config
      else split_cache_module.SplitAttentionMaskBuilder(
          decode_config,
          sliding_window_sizes=sliding_window_sizes,
      )
  )
  sample_inputs = prefill_mask_module.get_sample_inputs(
      text_model_config, export_config
  )
  for signature_name, (sample_input, _) in sample_inputs.items():
    mask_module = (
        decode_mask_module
        if signature_name == 'decode_mask'
        else prefill_mask_module
    )
    converter.add_signature(
        signature_name,
        mask_module.eval(),
        sample_kwargs=sample_input,
    )
  # Cache Update
  # CacheUpdate is stateless; its sizes come from the sample inputs, so the
  # decode sample must be re-derived to match the decode signature.
  cache_update_module = split_cache_module.CacheUpdate()
  sample_inputs = cache_update_module.get_sample_inputs(
      text_model_config, export_config
  )
  if decode_config is not export_config:
    decode_cache_update_inputs = cache_update_module.get_sample_inputs(
        text_model_config, decode_config
    )
    sample_inputs['decode_cache_update'] = decode_cache_update_inputs[
        'decode_cache_update'
    ]
  for signature_name, (sample_input, _) in sample_inputs.items():
    converter.add_signature(
        signature_name,
        cache_update_module.eval(),
        sample_kwargs=sample_input,
    )
  lrt_model = converter.convert(strict_export=False)
  model_path = os.path.join(work_dir, 'auxiliary.tflite')  # pyrefly: ignore[no-matching-overload]
  lrt_model.export(model_path)
  return dataclasses.replace(
      exported_model_artifacts,
      auxiliary_model_path=model_path,
  )


def export_additional_models_impl(
    name: str,
    exportable_module_cls: torch.nn.Module,
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
) -> ExportedModelArtifacts:
  """Exports additional model."""
  model = source_model_artifacts.model
  text_model_config = source_model_artifacts.text_model_config
  quantization_recipe = export_config.quantization_recipe
  work_dir = export_config.work_dir
  embedder_module = exportable_module_cls(model)
  converter = converter_utils.Converter()
  sample_inputs = embedder_module.get_sample_inputs(
      text_model_config, export_config
  )
  for signature_name, (sample_inputs, _) in sample_inputs.items():
    converter.add_signature(
        signature_name,
        embedder_module.eval(),
        sample_kwargs=sample_inputs,
    )
  lrt_model = converter.convert(strict_export=False)
  model_path = os.path.join(work_dir, f'{name}.tflite')  # pyrefly: ignore[no-matching-overload]
  lrt_model.export(model_path)
  quantization_recipe_list = (
      quantization_recipe.split(',') if quantization_recipe else [None]
  )
  for recipe in quantization_recipe_list:
    model_path = maybe_quantize_model(model_path, recipe)
    gc.collect()
  additional_models = exported_model_artifacts.additional_model_paths or {}
  additional_models[name] = model_path
  return dataclasses.replace(
      exported_model_artifacts,
      additional_model_paths=additional_models,
  )


def export_additional_models(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
) -> ExportedModelArtifacts:
  """Exports embedder."""
  exportable_model_cls_dict = model_ext_exportables.get_additional_exportables(
      source_model_artifacts.model_config
  )
  for name, exportable_module_cls in exportable_model_cls_dict.items():
    with progress.task(f'Export {name} model'):
      exported_model_artifacts = export_additional_models_impl(
          name,
          exportable_module_cls,
          source_model_artifacts,
          export_config,
          exported_model_artifacts,
      )
  return exported_model_artifacts


def _maybe_patch_tokenizer(tokenizer_path: str) -> None:
  """Patches BPE tokenizer if it has Metaspace pre-tokenizer but BPE chars in vocab."""
  if not tokenizer_path.endswith('.json'):
    return
  with open(tokenizer_path, 'r') as f:
    try:
      data = json.load(f)
    except json.JSONDecodeError:
      return

  is_bpe = data.get('model', {}).get('type') == 'BPE'
  has_metaspace = data.get('pre_tokenizer', {}).get('type') == 'Metaspace'

  vocab = data.get('model', {}).get('vocab', {})
  has_bpe_chars = any('Ġ' in k or 'Ċ' in k for k in vocab.keys())

  if is_bpe and has_metaspace and has_bpe_chars:
    print(
        'WARNING: Detected BPE tokenizer with Metaspace pre-tokenizer but BPE'
        ' characters in vocab. Patching to ByteLevel.'
    )
    data['pre_tokenizer'] = {
        'type': 'ByteLevel',
        'add_prefix_space': False,
        'trim_offsets': True,
        'use_regex': True,
    }
    data['decoder'] = {
        'type': 'ByteLevel',
        'add_prefix_space': True,
        'trim_offsets': False,
        'use_regex': True,
    }
    with open(tokenizer_path, 'w') as f:
      json.dump(data, f, indent=2)


@progress.task('Export tokenizer')
def export_tokenizer(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
) -> ExportedModelArtifacts:
  """Exports tokenizer."""
  tokenizer = source_model_artifacts.tokenizer
  work_dir = export_config.work_dir
  if hasattr(tokenizer, 'vocab_file') and tokenizer.vocab_file:
    tokenizer_path = tokenizer.vocab_file
    if tokenizer_path.endswith('tokenizer.model'):
      with open(tokenizer_path, 'rb') as f:
        with open(os.path.join(work_dir, 'tokenizer.model'), 'wb') as f_out:  # pyrefly: ignore[no-matching-overload]
          f_out.write(f.read())
      tokenizer_path = os.path.join(work_dir, 'tokenizer.model')  # pyrefly: ignore[no-matching-overload]
      return dataclasses.replace(
          exported_model_artifacts,
          tokenizer_model_path=tokenizer_path,
      )
  try:
    tokenizer_path = tokenizer.save_pretrained(work_dir, legacy_format=False)  # pyrefly: ignore[bad-argument-type]
    # TODO(weiyiw): This is rough... polish it.
    if isinstance(tokenizer_path, tuple):
      tokenizer_path = [
          x for x in tokenizer_path if x.endswith('tokenizer.json')
      ]
      assert len(tokenizer_path) == 1
      tokenizer_path = tokenizer_path[0]
    _maybe_patch_tokenizer(tokenizer_path)
    return dataclasses.replace(
        exported_model_artifacts,
        tokenizer_model_path=tokenizer_path,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    # Fallback to convert tokenizer to sentencepiece.
    print('Failed to export tokenizer. Converting to sentencepiece.')
    spm_serialized = tokenizer_lib.convert(tokenizer)
    tokenizer_path = os.path.join(work_dir, 'tokenizer.spiece')  # pyrefly: ignore[no-matching-overload]
    with open(tokenizer_path, 'wb') as f:
      f.write(spm_serialized)
  return dataclasses.replace(
      exported_model_artifacts,
      tokenizer_model_path=tokenizer_path,
  )


@progress.task('AOT Compilation')
def aot_compile_model(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
):
  """AOT compiles the model.

  DEPRECATED: This legacy compilation workflow (which compiles individual
  sub-models before packaging) is deprecated and will be removed soon. Please
  use the LiteRT-LM package compiler `compile_litertlm` instead.
  """
  del source_model_artifacts  # Unused.
  assert export_config.aot_backend is not None
  assert export_config.aot_soc_model is not None
  config = {
      'backend_id': export_config.aot_backend,
      'soc_model': export_config.aot_soc_model,
  }
  if export_config.aot_compilation_config_dict is not None:
    config['compilation_config'] = export_config.aot_compilation_config_dict  # pyrefly: ignore[bad-assignment]

  backend_class = import_vendor.import_vendor(export_config.aot_backend)
  backend = backend_class.create(config)

  target = backend.target
  source_model = exported_model_artifacts.prefill_decode_model_path
  assert source_model is not None, 'Prefill-decode model is not found.'
  output_dir = export_config.work_dir
  output_path = source_model.removesuffix('.tflite') + '_aot_compiled.tflite'
  config_dict = export_config.aot_compilation_config_dict or {}
  aot_config = aot_types.CompilationConfig(
      target=target,
      **config_dict,
  )
  aot_results = aot_compile.aot_compile(
      source_model,
      output_dir=output_dir,
      config=aot_config,
  )
  assert aot_results.models, 'AOT compilation failed.'
  assert len(aot_results.models) == 1, 'Currently only support one target.'
  aot_results.models[0].save(output_path, export_only=True)
  return dataclasses.replace(
      exported_model_artifacts,
      prefill_decode_model_path=output_path,
  )


@progress.task('NPU Package Compilation')
def compile_litertlm_bundle(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
):
  """Compiles the packaged .litertlm file using the new NPU compiler."""

  model_name = source_model_artifacts.model_config.model_type
  litert_lm_model_path = exported_model_artifacts.litert_lm_model_path
  assert litert_lm_model_path is not None, 'LiteRT-LM model path not found.'

  backend = export_config.aot_backend
  soc_model = export_config.aot_soc_model
  assert backend is not None, 'aot_backend is required for compilation.'
  assert soc_model is not None, 'aot_soc_model is required for compilation.'

  litert_lm_npu_compiler.compile_litertlm(
      input_litertlm=litert_lm_model_path,
      output_litertlm=litert_lm_model_path,
      backend=backend,
      soc_model=soc_model,
      compile_configs=export_config.compile_configs,
      model_name=model_name,
      overwrite=True,
  )
  return exported_model_artifacts
