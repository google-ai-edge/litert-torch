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

"""Represents an litert_torch model.

PyTorch models can be converted to this representation through
`litert_torch.convert`.
"""

from __future__ import annotations

import abc
import dataclasses
import os
import re
import shutil
from typing import Callable

import numpy as np
import numpy.typing as npt

from ai_edge_litert import interpreter as interpreter_lib  # pylint: disable=g-direct-tensorflow-import

# The CompiledModel runtime is optional: it is only needed by
# `use_compiled_runtime`. Importing it lazily inside that path would hide the
# dependency, but importing it unguarded would make `import litert_torch` fail
# outright against a LiteRT runtime that predates CompiledModel. Bind None
# instead and raise a targeted error at call time.
try:
  from ai_edge_litert.litert_wrapper.compiled_model_wrapper import compiled_model as compiled_model_lib  # pylint: disable=g-import-not-at-top
  from ai_edge_litert.litert_wrapper.compiled_model_wrapper.hardware_accelerator import HardwareAccelerator  # pylint: disable=g-import-not-at-top
except ImportError:
  try:
    from ai_edge_litert import compiled_model as compiled_model_lib  # pylint: disable=g-import-not-at-top
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator  # pylint: disable=g-import-not-at-top
  except ImportError:
    compiled_model_lib = None
    HardwareAccelerator = None

DEFAULT_SIGNATURE_NAME = 'serving_default'


class ModelExporter(abc.ABC):

  @abc.abstractmethod
  def to_file(self, path: str):
    raise NotImplementedError()

  @abc.abstractmethod
  def to_bytes(self) -> bytes:
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class BytesExporter(ModelExporter):
  content: bytes

  def to_file(self, path: str):
    with open(path, 'wb') as f:
      f.write(self.content)

  def to_bytes(self) -> bytes:
    return self.content


@dataclasses.dataclass(frozen=True)
class FileExporter(ModelExporter):
  path: str

  def __post_init__(self):
    abs_path = os.path.abspath(self.path)
    if not os.path.isfile(abs_path):
      raise FileNotFoundError(f'Model file not found: {abs_path}')
    object.__setattr__(self, 'path', abs_path)

  def to_file(self, path: str):
    if os.path.dirname(path):
      os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.abspath(path) != self.path:
      shutil.copyfile(self.path, path)

  def to_bytes(self) -> bytes:
    with open(self.path, 'rb') as f:
      return f.read()


class Model(abc.ABC):
  """A LiteRT model."""

  @abc.abstractmethod
  def __call__(
      self,
      *args: npt.ArrayLike,
      signature_name: str = DEFAULT_SIGNATURE_NAME,
      **kwargs,
  ) -> npt.ArrayLike | tuple[npt.ArrayLike, ...] | dict[str, npt.ArrayLike]:
    raise NotImplementedError()

  @abc.abstractmethod
  def export(self, path: str):
    raise NotImplementedError()

  @classmethod
  def load(cls, path: str) -> LiteRTModel:
    return LiteRTModel.load(path)


class LiteRTModel(Model):
  """The LiteRT model wrapper and inference runner.

  `__call__` dispatches to one of two mutually exclusive execution modes,
  selected by `use_interpreter_runtime` (the default) and
  `use_compiled_runtime`. Each mode caches its own lazily built runtime object,
  which is invalidated whenever the configuration changes.
  """

  def __init__(self, exporter: ModelExporter | bytes):
    if isinstance(exporter, bytes):
      exporter = BytesExporter(exporter)

    self._exporter = exporter
    if isinstance(exporter, FileExporter):
      self._interpreter_builder = lambda: interpreter_lib.Interpreter(
          model_path=exporter.path,
          experimental_default_delegate_latest_features=True,
      )
    else:
      self._interpreter_builder = lambda: interpreter_lib.Interpreter(
          model_content=exporter.to_bytes(),
          experimental_default_delegate_latest_features=True,
      )
    # Cached tf.lite Interpreter, built on first use by `_get_interpreter`.
    self._interpreter = None
    # Selects the CompiledModel path over the Interpreter path in `__call__`.
    self._use_compiled_runtime = False
    # Cached CompiledModel, built on first use by `_get_compiled_model`.
    self._compiled_model = None
    # HardwareAccelerator bitmask to compile for; None lets LiteRT pick its
    # default (CPU).
    self._hardware_accel = None
    # If True, `_get_compiled_model` rejects a model that did not fully
    # delegate, instead of silently falling back to reference CPU kernels.
    self._require_fully_accelerated = False

  def set_interpreter_builder(
      self, builder: Callable[[], interpreter_lib.Interpreter]
  ) -> None:
    """Sets a custom interpreter builder.

    Args:
      builder: A function that returns a LiteRT Interpreter or its subclass.
    """
    self._interpreter_builder = builder
    self._interpreter = None
    self._use_compiled_runtime = False

  def _get_interpreter(self) -> interpreter_lib.Interpreter:
    if self._interpreter is not None:
      return self._interpreter

    interpreter = self._interpreter_builder()
    interpreter.allocate_tensors()
    self._interpreter = interpreter
    return interpreter

  def model_content(self) -> bytes:
    """Returns the raw bytes of the LiteRT model flatbuffer."""
    return self._exporter.to_bytes()

  def use_compiled_runtime(
      self,
      hardware_accel: HardwareAccelerator | None = None,
      require_fully_accelerated: bool = False,
  ) -> LiteRTModel:
    """Configures the model to execute inference via LiteRT CompiledModel in __call__.

    Args:
      hardware_accel: Optional `HardwareAccelerator` bitmask from the LiteRT
        runtime (e.g. `HardwareAccelerator.GPU | HardwareAccelerator.CPU`),
        forwarded verbatim to `CompiledModel.from_buffer`. Defaults to the
        runtime's own default (CPU) when None. Note that GPU or NPU alone can
        fail on models with unsupported ops; OR in CPU for fallback.
      require_fully_accelerated: If True, raises RuntimeError if any ops in the
        model failed to delegate to the requested hardware accelerator.

    Returns:
      self (for method chaining).
    """
    self._use_compiled_runtime = True
    self._hardware_accel = hardware_accel
    self._require_fully_accelerated = require_fully_accelerated
    self._compiled_model = None
    return self

  def use_interpreter_runtime(self) -> LiteRTModel:
    """Configures the model to execute inference via LiteRT Interpreter in __call__.

    Returns:
      self (for method chaining).
    """
    self._use_compiled_runtime = False
    return self

  def _get_compiled_model(self):
    """Returns the cached CompiledModel, building it on first use.

    Returns:
      The `CompiledModel` compiled for `self._hardware_accel`.

    Raises:
      RuntimeError: If the LiteRT runtime does not provide CompiledModel, or if
        `require_fully_accelerated` was requested and the model did not fully
        delegate to the target accelerator.
    """
    if self._compiled_model is not None:
      return self._compiled_model

    if compiled_model_lib is None:
      raise RuntimeError(
          'use_compiled_runtime() requires a LiteRT runtime that provides'
          ' CompiledModel, which could not be imported. Install/upgrade'
          ' ai_edge_litert, or use use_interpreter_runtime() instead.'
      )

    kwargs_cm = {}
    if self._hardware_accel is not None:
      kwargs_cm['hardware_accel'] = self._hardware_accel

    cm = compiled_model_lib.CompiledModel.from_buffer(
        self.model_content(), **kwargs_cm
    )
    if self._require_fully_accelerated and not cm.is_fully_accelerated():
      raise RuntimeError(
          'Model is not fully accelerated on the requested hardware'
          ' accelerator.'
      )
    self._compiled_model = cm
    return cm

  def _run_compiled(
      self,
      *args: npt.ArrayLike,
      signature_name: str = DEFAULT_SIGNATURE_NAME,
      **kwargs,
  ) -> npt.ArrayLike | tuple[npt.ArrayLike, ...] | dict[str, npt.ArrayLike]:
    """Runs inference through the CompiledModel runtime.

    Inputs whose shape differs from the compiled layout trigger a resize, so
    models converted with `dynamic_shapes` can be called at varying shapes.

    Args:
      *args: Positional model inputs, bound to the signature's `args_N` inputs.
      signature_name: The model signature to invoke.
      **kwargs: Keyword model inputs, bound by name.

    Returns:
      The model outputs, matching the shape of the Interpreter path's return
      value.

    Raises:
      ValueError: If `signature_name` is unknown, if the number of supplied
        inputs does not match the signature, if an input's dtype differs from
        the model's, or if an input's shape differs along a static dimension.
    """
    cm = self._get_compiled_model()
    sig_list = cm.get_signature_list()
    if signature_name not in sig_list:
      raise ValueError(
          f'Invalid signature name provided: {signature_name}. Available:'
          f' {list(sig_list.keys())}'
      )

    if len(sig_list[signature_name]['inputs']) != len(args) + len(kwargs):
      raise ValueError(
          'The model requires'
          f' {len(sig_list[signature_name]["inputs"])} arguments but'
          f' {len(args) + len(kwargs)} was provided.'
      )

    inputs = {f'args_{idx}': np.asarray(args[idx]) for idx in range(len(args))}
    for k, v in kwargs.items():
      inputs[k] = np.asarray(v)

    # Validate dtypes and settle any dynamic shapes before allocating buffers:
    # `create_input_buffer_by_name` sizes each buffer from the *current*
    # compiled layout, so every resize has to happen first.
    input_details = cm.get_input_tensor_details(signature_name)
    arrays = {}
    for name in sig_list[signature_name]['inputs']:
      array = np.ascontiguousarray(inputs[name])
      expected = input_details[name]

      # The Interpreter path rejects dtype mismatches inside SignatureRunner.
      # Writing straight to the buffer would not, so check explicitly rather
      # than silently reinterpreting the caller's data.
      expected_dtype = np.dtype(expected['dtype'])
      if array.dtype != expected_dtype:
        raise ValueError(
            f"Input '{name}' has dtype {array.dtype}, but the model expects"
            f' {expected_dtype}.'
        )
      arrays[name] = array

      if tuple(array.shape) != tuple(expected['shape']):
        # strict=True so that resizing a genuinely static dimension fails
        # loudly instead of being silently accepted.
        cm.resize_input_tensor_by_name(
            signature_name, name, array.shape, strict=True
        )

    input_map = {}
    for name, array in arrays.items():
      buf = cm.create_input_buffer_by_name(signature_name, name)
      buf.write(array)
      input_map[name] = buf

    output_map = {}
    for name in sig_list[signature_name]['outputs']:
      output_map[name] = cm.create_output_buffer_by_name(signature_name, name)

    try:
      cm.run_by_name(signature_name, input_map, output_map)
      outputs = {}
      for name, buf in output_map.items():
        details = buf.get_tensor_details()
        shape = details['shape']
        dtype = details['dtype']
        num_elements = int(np.prod(shape))
        outputs[name] = buf.read(num_elements, dtype).reshape(shape)
    finally:
      for buf in input_map.values():
        buf.destroy()
      for buf in output_map.values():
        buf.destroy()

    output_heuristic = lambda key: bool(re.search(r'output_\d+', key))
    if all(output_heuristic(key) for key in outputs.keys()):
      if len(outputs) == 1:
        return outputs['output_0']
      return tuple(outputs[f'output_{idx}'] for idx in range(len(outputs)))

    return outputs

  def __call__(
      self,
      *args: npt.ArrayLike,
      signature_name: str = DEFAULT_SIGNATURE_NAME,
      **kwargs,
  ) -> npt.ArrayLike | tuple[npt.ArrayLike, ...] | dict[str, npt.ArrayLike]:
    """Runs inference on the LiteRT model using the provided arguments.

    Args:
      *args: The arguments to be passed to the model for inference.
      signature_name: The name of the signature to be used for inference. The
        default signature is used if not provided.
      **kwargs: The arguments with specific names to be passed to the model for
        inference.

    Returns:
      The output of the model. If the model has only one output, the output is
      returned directly.
    """
    if self._use_compiled_runtime:
      return self._run_compiled(*args, signature_name=signature_name, **kwargs)

    interpreter = self._get_interpreter()

    signature_list = interpreter.get_signature_list()
    if signature_name not in signature_list:
      raise ValueError(
          'Invalid signature name provided. Available signatures:'
          f' {", ".join(signature_list.keys())}'
      )

    try:
      runner = interpreter.get_signature_runner(signature_name)
    except ValueError as exception:
      if 'Invalid signature_key provided.' in str(exception):
        raise ValueError(
            'Invalid signature key provided. Available signatures:'
            f' {list(signature_list.keys())}'
        )
      else:
        raise exception

    if len(signature_list[signature_name]['inputs']) != len(args) + len(kwargs):
      raise ValueError(
          'The model requires'
          f' {len(signature_list[signature_name]["inputs"])} arguments but'
          f' {len(args)} was provided.'
      )

    # Gather the input dictionary based on the signature.
    inputs = {f'args_{idx}': args[idx] for idx in range(len(args))}
    inputs = {**inputs, **kwargs}
    outputs = runner(**inputs)

    # When attempting to run a model, check if all the output tensors are named
    # output_<number>. If so, assume the pytorch model returned a tuple and not
    # a dictionary.
    output_heuristic = lambda key: bool(re.search(r'output_\d+', key))
    if all(output_heuristic(key) for key in outputs.keys()):
      if len(outputs) == 1:
        return outputs['output_0']
      else:
        return tuple(outputs[f'output_{idx}'] for idx in range(len(outputs)))

    return outputs

  def export(self, path: str) -> None:
    """Serializes the LiteRT model to a file.

    Args:
      path: The path to file to which the model is serialized.
    """
    if os.path.dirname(path):
      os.makedirs(os.path.dirname(path), exist_ok=True)
    self._exporter.to_file(path)

  @classmethod
  def load(cls, path: str) -> LiteRTModel:
    """Loads a LiteRT model from a file.

    Args:
      path: The path to the model.
    """
    with open(path, 'rb') as f:
      model_content = f.read()

    return cls(model_content)
