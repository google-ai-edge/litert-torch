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

import dataclasses
import gc
import json
import os

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

ExportTask = exportable_module_config.ExportTask
from litert_torch.generative.export_hf.core.external_rope import exportable_module as external_rope_module
from litert_torch.generative.export_hf.core.external_rope import preprocess_model as external_rope_preprocess_model
from litert_torch.generative.export_hf.core.mu import mu_pass_lib
from litert_torch.generative.export_hf.core.split_cache import attention as _
from litert_torch.generative.export_hf.core.split_cache import exportable_module as split_cache_module
from litert_torch.generative.export_hf.model_ext import exportables as model_ext_exportables
from litert_torch.generative.export_hf.model_ext import extension as model_ext_extension
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
from litert_torch.generative.tools import tokenizer_to_sentencepiece_lib as tokenizer_lib
import torch
import transformers

from ai_edge_quantizer import quantizer as quantizer_lib
from ai_edge_quantizer import recipe as recipe_lib


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
  auxiliary_model_path: str | None = None
  tokenizer_model_path: str | None = None
  additional_model_paths: dict[str, str] | None = None

  litert_lm_model_path: str | None = None


def verify_model_compatibility(model, model_config, text_model_config):
  """Verifies model compatibility."""
  del model_config  # Unused.

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


@progress.task('Load source model')
def load_model(
    model_path: str,
    trust_remote_code: bool = False,
    auto_model_override: str | None = None,
    task: ExportTask | str = ExportTask.TEXT_GENERATION,
) -> SourceModelArtifacts:
  """Loads model from checkpoint."""

  config = transformers.AutoConfig.from_pretrained(
      model_path,
      dtype=torch.float32,
      trust_remote_code=trust_remote_code,
  )
  config._attn_implementation = 'lrt_transposed_attention'  # pylint: disable=protected-access

  if task == ExportTask.TEXT_GENERATION:
    auto_model_cls = transformers.AutoModelForCausalLM
  elif task == ExportTask.IMAGE_TEXT_TO_TEXT:
    auto_model_cls = transformers.AutoModelForImageTextToText
  else:
    raise ValueError(f'Unsupported task: {task}')
  if auto_model_override is not None:
    auto_model_cls = transformers.__dict__[auto_model_override]

  with model_ext_patches.get_patch_context(config.model_type):
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
        tokenizer.chat_template = chat_template_dict['chat_template']
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Failed to load chat template: {e}')

  return SourceModelArtifacts(
      model=model,
      model_config=config,
      text_model_config=text_model_config,
      tokenizer=tokenizer,
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


@progress.task('Export text prefill-decode model')
def export_text_prefill_decode_model(
    source_model_artifacts: SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: ExportedModelArtifacts,
):
  """Exports text model to tflite."""
  model = source_model_artifacts.model
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
  else:
    model.set_attn_implementation('lrt_transposed_attention')

  prefill_module_cls, decode_module_cls = get_prefill_decode_exportable_cls(
      source_model_artifacts.model_config, export_config
  )
  prefill_module = prefill_module_cls(model, export_config)
  decode_module = decode_module_cls(model, export_config)
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
  sample_decode_inputs, decode_dynamic_shapes = decode_module.get_sample_inputs(
      text_model_config
  )['decode']
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
        'decode',
        decode_ep.module(),
        sample_kwargs=sample_decode_inputs,
        dynamic_shapes=decode_dynamic_shapes,
    )
  else:
    converter.add_signature(
        'decode',
        decode_module.eval(),
        sample_kwargs=sample_decode_inputs,
    )

  lrt_model = converter.convert(
      lightweight_conversion=export_config.experimental_lightweight_conversion,
      strict_export=False,
  )

  lrt_model = mu_pass_lib.update_model(lrt_model)
  if export_config.experimental_use_mixed_precision:
    print('Applying mixed precision to model...')
    lrt_model = mu_pass_lib.apply_mixed_precision(lrt_model)

  model_path = os.path.join(work_dir, 'model.tflite')
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
  if not quantization_recipe:
    return model_path
  return quantize_model(model_path, quantization_recipe)


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
  model_path = os.path.join(work_dir, 'embedder.tflite')
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
  encoder_module_cls, adapter_module_cls = (
      model_ext_exportables.get_vision_exportables(model_config)
  )
  encode_module = encoder_module_cls(model, export_config)
  adapter_module = adapter_module_cls(model, export_config, tokenizer)
  converter = converter_utils.Converter()
  sample_inputs = encode_module.get_sample_inputs(
      model_config, image_processor=image_processor
  )
  for signature_name, (sample_inputs, _) in sample_inputs.items():
    converter.add_signature(
        signature_name,
        encode_module.eval(),
        sample_kwargs=sample_inputs,
    )
  lrt_model = converter.convert(strict_export=False)
  vision_encoder_path = os.path.join(work_dir, 'vision_encoder.tflite')
  lrt_model.export(vision_encoder_path)
  quantization_recipe_list = (
      quantization_recipe.split(',') if quantization_recipe else [None]
  )
  for recipe in quantization_recipe_list:
    vision_encoder_path = maybe_quantize_model(vision_encoder_path, recipe)
    gc.collect()

  converter = converter_utils.Converter()
  sample_inputs = adapter_module.get_sample_inputs(
      model_config, image_processor=image_processor
  )
  for signature_name, (sample_inputs, _) in sample_inputs.items():
    converter.add_signature(
        signature_name,
        adapter_module.eval(),
        sample_kwargs=sample_inputs,
    )
  lrt_model = converter.convert(strict_export=False)
  adapter_path = os.path.join(work_dir, 'vision_adapter.tflite')
  lrt_model.export(adapter_path)
  quantization_recipe_list = (
      quantization_recipe.split(',') if quantization_recipe else [None]
  )
  for recipe in quantization_recipe_list:
    adapter_path = maybe_quantize_model(adapter_path, recipe)
    gc.collect()
  return dataclasses.replace(
      exported_model_artifacts,
      vision_encoder_model_path=vision_encoder_path,
      vision_adapter_model_path=adapter_path,
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
  attention_mask_module = split_cache_module.SplitAttentionMaskBuilder(
      export_config.cache_length,
      sliding_window_sizes=sliding_window_sizes,
  )
  sample_inputs = attention_mask_module.get_sample_inputs(
      text_model_config, export_config
  )
  for signature_name, (sample_input, _) in sample_inputs.items():
    converter.add_signature(
        signature_name,
        attention_mask_module.eval(),
        sample_kwargs=sample_input,
    )
  # Cache Update
  cache_update_module = split_cache_module.CacheUpdate()
  sample_inputs = cache_update_module.get_sample_inputs(
      text_model_config, export_config
  )
  for signature_name, (sample_input, _) in sample_inputs.items():
    converter.add_signature(
        signature_name,
        cache_update_module.eval(),
        sample_kwargs=sample_input,
    )
  lrt_model = converter.convert(strict_export=False)
  model_path = os.path.join(work_dir, 'auxiliary.tflite')
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
  model_path = os.path.join(work_dir, f'{name}.tflite')
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
        with open(os.path.join(work_dir, 'tokenizer.model'), 'wb') as f_out:
          f_out.write(f.read())
      tokenizer_path = os.path.join(work_dir, 'tokenizer.model')
      return dataclasses.replace(
          exported_model_artifacts,
          tokenizer_model_path=tokenizer_path,
      )
  try:
    tokenizer_path = tokenizer.save_pretrained(work_dir, legacy_format=False)
    # TODO(weiyiw): This is rough... polish it.
    if isinstance(tokenizer_path, tuple):
      tokenizer_path = [
          x for x in tokenizer_path if x.endswith('tokenizer.json')
      ]
      assert len(tokenizer_path) == 1
      tokenizer_path = tokenizer_path[0]
    return dataclasses.replace(
        exported_model_artifacts,
        tokenizer_model_path=tokenizer_path,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    # Fallback to convert tokenizer to sentencepiece.
    print('Failed to export tokenizer. Converting to sentencepiece.')
    spm_serialized = tokenizer_lib.convert(tokenizer)
    tokenizer_path = os.path.join(work_dir, 'tokenizer.spiece')
    with open(tokenizer_path, 'wb') as f:
      f.write(spm_serialized)
  return dataclasses.replace(
      exported_model_artifacts,
      tokenizer_model_path=tokenizer_path,
  )
