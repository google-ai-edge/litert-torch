# Copyright 2026 The LiteRT Torch Authors. All Rights Reserved.
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
"""Tool for compiling TFLite models inside .litertlm file for NPU."""

import contextlib
import json
import os
import pathlib
import re
import sys
import tempfile
try:
  import tomllib
except ImportError:
  import tomli as tomllib
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
from litert_torch.generative.export_hf.experimental.litert_lm_npu_compiler import litert_lm_npu_compiler_configs

gfile = None

from ai_edge_litert.aot.core import aot_types
from ai_edge_litert.aot.core.apply_plugin import ApplyPlugin
from ai_edge_litert.tools import flatbuffer_utils
import litert_lm_builder as litertlm_builder
from litert_lm_builder import litertlm_peek


def _open(path: str | pathlib.Path, mode: str = 'r'):
  if gfile:
    return gfile.Open(str(path), mode)
  return open(path, mode)

FLAGS = flags.FLAGS


def _define_flag(define_fn, name, *args, **kwargs):
  try:
    return define_fn(name, *args, **kwargs)
  except flags.DuplicateFlagError:
    return flags.FLAGS[name]


_define_flag(
    flags.DEFINE_string, 'input_litertlm', None, 'Path to input .litertlm file.'
)
_define_flag(
    flags.DEFINE_string,
    'output_litertlm',
    None,
    'Path to output .litertlm file.',
)
_define_flag(
    flags.DEFINE_enum,
    'backend',
    None,
    ['qualcomm', 'mediatek'],
    'Target NPU backend.',
)
_define_flag(
    flags.DEFINE_string,
    'soc_model',
    None,
    'Target SoC model (e.g., SM8850, MT6993).',
)
_define_flag(
    flags.DEFINE_string,
    'compile_configs',
    None,
    'JSON string or path to JSON file containing compilation flags for each'
    ' model type.',
)
_define_flag(
    flags.DEFINE_string,
    'model_name',
    None,
    'Model name to use default configurations (e.g., gemma4_2b).',
)
_define_flag(
    flags.DEFINE_bool,
    'disable_weight_sharing',
    False,
    'Disable weight sharing for Qualcomm backend compiler.',
)
_define_flag(
    flags.DEFINE_bool,
    'disable_aux_compilation',
    False,
    'Disable compiling aux model.',
)


def _get_soc_manufacturer(backend: str) -> str:
  if backend == 'qualcomm':
    return 'Qualcomm'
  elif backend == 'mediatek':
    return 'MediaTek'
  else:
    raise ValueError(f'Unsupported backend: {backend}')


def _resolve_subgraphs_to_compile(
    model_path: pathlib.Path, model_type: str
) -> list[int] | None:
  """Resolve which subgraphs to compile based on model type and signatures."""
  if model_type != 'aux':
    return None

  try:
    model = flatbuffer_utils.read_model(str(model_path))
    if not model.signatureDefs:
      logging.warning('No signatures found in aux model.')
      return None

    subgraphs_to_compile = []
    for sig in model.signatureDefs:
      key = sig.signatureKey
      if isinstance(key, bytes):
        key = key.decode('utf-8')
      if re.search(r'(prefill_mask|decode_mask)', key):
        subgraphs_to_compile.append(sig.subgraphIndex)

    if subgraphs_to_compile:
      logging.info(
          'Resolved subgraphs to compile for aux model: %s',
          subgraphs_to_compile,
      )
      return sorted(list(set(subgraphs_to_compile)))

  except Exception as e:  # pylint: disable=broad-except
    logging.exception('Failed to parse aux model for signatures')

  return None


_DEFAULT_INTERMEDIATE_ARTIFACTS_DIR = 'intermediate_compiler_artifacts'


def compile_litertlm(
    input_litertlm: str | pathlib.Path,
    output_litertlm: str | pathlib.Path,
    backend: str,
    soc_model: str,
    compile_configs: str | dict[str, Any] | None = None,
    model_name: str | None = None,
    disable_weight_sharing: bool = False,
    disable_aux_compilation: bool = False,
    intermediate_dir: str | pathlib.Path | None = None,
    keep_temporary_files: bool = False,
    overwrite: bool = True,
) -> None:
  """Compiles LiteRT-LM models inside a package for the target NPU.

  Args:
    input_litertlm: Path to the input .litertlm file.
    output_litertlm: Path to the output .litertlm file.
    backend: Target backend ('qualcomm' or 'mediatek').
    soc_model: SoC model ID (e.g. 'SM8850', 'MT6993').
    compile_configs: Optional JSON string or dictionary representing
      compilation configs.
    model_name: Optional model name to resolve model-specific default configs.
    disable_weight_sharing: If True, disables weight sharing for Qualcomm.
    disable_aux_compilation: If True, disables compiling auxiliary model.

  Raises:
    ValueError: If compile options are invalid.
    RuntimeError: If compilation fails.
  """
  if overwrite and output_litertlm and hasattr(output_litertlm, 'unlink'):
    if output_litertlm.exists():
      output_litertlm.unlink()
  elif overwrite and output_litertlm and os.path.exists(str(output_litertlm)):
    os.remove(str(output_litertlm))
  configs = {}
  if compile_configs:
    if isinstance(compile_configs, dict):
      configs = compile_configs
    elif compile_configs.startswith('{'):
      configs = json.loads(compile_configs)
    else:
      with _open(compile_configs, 'r') as f:
        configs = json.load(f)
  else:
    # Resolve default configs
    if model_name:
      key = (backend, model_name)
      if key in litert_lm_npu_compiler_configs.MODEL_SPECIFIC_DEFAULTS:
        configs = litert_lm_npu_compiler_configs.MODEL_SPECIFIC_DEFAULTS[key]
        logging.info(
            'Using model-specific defaults for %s on %s',
            model_name,
            backend,
        )
      else:
        configs = litert_lm_npu_compiler_configs.GENERIC_DEFAULT_CONFIGS.get(
            backend, {}
        )
        logging.info(
            'No model-specific defaults for %s, using generic defaults for %s',
            model_name,
            backend,
        )
    else:
      configs = litert_lm_npu_compiler_configs.GENERIC_DEFAULT_CONFIGS.get(
          backend, {}
      )
      logging.info('Using generic defaults for %s', backend)

  soc_manufacturer = _get_soc_manufacturer(backend)

  if intermediate_dir:
    temp_dir_path = pathlib.Path(intermediate_dir)
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    ctx = contextlib.nullcontext(str(temp_dir_path))
  elif keep_temporary_files:
    temp_dir_path = (
        pathlib.Path(str(output_litertlm)).parent
        / _DEFAULT_INTERMEDIATE_ARTIFACTS_DIR
    )
    temp_dir_path.mkdir(parents=True, exist_ok=True)
    ctx = contextlib.nullcontext(str(temp_dir_path))
  else:
    ctx = tempfile.TemporaryDirectory()

  with ctx as temp_dir:
    temp_dir_path = pathlib.Path(temp_dir)
    logging.info('Unpacking %s to %s', input_litertlm, temp_dir)

    litertlm_peek.peek_litertlm_file(input_litertlm, temp_dir, sys.stdout)

    toml_path = temp_dir_path / 'model.toml'
    if not toml_path.exists():
      raise RuntimeError(
          f'Failed to unpack, model.toml not found in {temp_dir}'
      )

    with _open(toml_path, 'r') as f:
      toml_data = tomllib.loads(f.read())

    if 'section' in toml_data:
      for section in toml_data['section']:
        if section.get('section_type') == 'TFLiteModel':
          model_type = section.get('model_type')
          relative_data_path = section.get('data_path')
          model_path = temp_dir_path / relative_data_path

          should_compile = False
          extra_flags = []

          if model_type in configs:
            if isinstance(configs[model_type], list):
              should_compile = True
              extra_flags = configs[model_type]
            elif isinstance(configs[model_type], dict):
              should_compile = configs[model_type].get('compile', True)
              extra_flags = configs[model_type].get('flags', [])

          if (
              should_compile
              and backend == 'qualcomm'
              and model_type == 'prefill_decode'
          ):
            if not disable_weight_sharing:
              if not any(
                  'qualcomm_enable_weight_sharing' in f for f in extra_flags
              ):
                extra_flags = list(extra_flags)
                extra_flags.append('--qualcomm_enable_weight_sharing=true')
            else:
              extra_flags = [
                  f
                  for f in extra_flags
                  if 'qualcomm_enable_weight_sharing' not in f
              ]
              extra_flags = list(extra_flags)
              extra_flags.append('--qualcomm_enable_weight_sharing=false')

          if model_type == 'aux' and disable_aux_compilation:
            should_compile = False

          if (
              should_compile
              and model_type == 'aux'
              and backend != 'qualcomm'
          ):
            raise ValueError(
                'Compiling aux model is only supported for Qualcomm backend.'
            )

          if should_compile:
            logging.info('Compiling model %s (%s)', model_type, model_path)

            subgraphs = _resolve_subgraphs_to_compile(model_path, model_type)

            kwargs = {}
            for flag in extra_flags:
              flag = flag.lstrip('-')
              if '=' in flag:
                k, v = flag.split('=', 1)
                if v.lower() == 'true':
                  v = True
                elif v.lower() == 'false':
                  v = False
                else:
                  try:
                    v = int(v)
                  except ValueError:
                    pass
                kwargs[k] = v
              else:
                kwargs[flag] = True

            compiled_model_path = model_path.with_suffix('.compiled.tflite')

            input_model = aot_types.Model(path=str(model_path))
            output_model = aot_types.Model(path=str(compiled_model_path))

            compiler = ApplyPlugin(
                subgraphs_to_compile=subgraphs, experimental_capture_stderr=True
            )

            # Resolve plugin directory from ai_edge_litert package
            try:
              apply_plugin_module = sys.modules[ApplyPlugin.__module__]
              if apply_plugin_module and hasattr(
                  apply_plugin_module, '__file__'
              ):
                apply_plugin_file = pathlib.Path(apply_plugin_module.__file__)
                package_root = apply_plugin_file.parent.parent.parent
                plugin_dir = package_root / 'vendors' / backend / 'compiler'
                if not plugin_dir.exists():
                  # Fallback for monorepo source tree layout
                  plugin_dir = (
                      package_root.parent / 'vendors' / backend / 'compiler'
                  )

                if plugin_dir.exists():
                  logging.info('Found plugin directory: %s', plugin_dir)
                  kwargs['libs'] = str(plugin_dir)
                else:
                  logging.warning(
                      'Could not find plugin directory for backend %s', backend
                  )
            except Exception as e:  # pylint: disable=broad-except
              logging.warning('Failed to resolve plugin directory: %s', e)

            try:
              compiler(
                  input_model=input_model,
                  output_model=output_model,
                  soc_manufacturer=soc_manufacturer,
                  soc_model=soc_model.upper(),
                  **kwargs,
              )
              os.replace(compiled_model_path, model_path)
              logging.info('Successfully compiled %s', model_type)
            except Exception as e:
              if model_path.exists():
                logging.error(
                    'Model path %s size: %d',
                    model_path,
                    model_path.stat().st_size,
                )
              else:
                logging.error('Model path %s does not exist!', model_path)
              logging.error('Failed to compile %s: %s', model_type, e)
              raise

    logging.info('Repacking to %s', output_litertlm)
    output_dir = os.path.dirname(output_litertlm)
    if output_dir and not os.path.exists(output_dir):
      os.makedirs(output_dir, exist_ok=True)
    builder = litertlm_builder.LitertLmFileBuilder.from_toml_file(
        str(toml_path)
    )
    with _open(output_litertlm, 'wb') as f:
      builder.build(f)
    logging.info('Done')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not (
      FLAGS.input_litertlm
      and FLAGS.output_litertlm
      and FLAGS.backend
      and FLAGS.soc_model
  ):
    raise app.UsageError(
        'Missing required flags: --input_litertlm, --output_litertlm,'
        ' --backend, --soc_model'
    )

  try:
    compile_litertlm(
        input_litertlm=FLAGS.input_litertlm,
        output_litertlm=FLAGS.output_litertlm,
        backend=FLAGS.backend,
        soc_model=FLAGS.soc_model,
        compile_configs=FLAGS.compile_configs,
        model_name=FLAGS.model_name,
        disable_weight_sharing=FLAGS.disable_weight_sharing,
        disable_aux_compilation=FLAGS.disable_aux_compilation,
    )
  except ValueError as e:
    raise app.UsageError(str(e)) from e


if __name__ == '__main__':
  app.run(main)
