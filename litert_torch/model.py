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
from typing import Any
from typing import Callable
import numpy as np
import numpy.typing as npt

from ai_edge_litert import interpreter as interpreter_lib  # pylint: disable=g-direct-tensorflow-import

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


class Model(abc.ABC):
  """A LiteRT model."""

  @abc.abstractmethod
  def __call__(
      self,
      *args: npt.ArrayLike,
      signature_name: str = DEFAULT_SIGNATURE_NAME,
      **kwargs,
  ) -> npt.ArrayLike | tuple[npt.ArrayLike, ...]:
    raise NotImplementedError()

  @abc.abstractmethod
  def export(self, path: str):
    raise NotImplementedError()

  @classmethod
  def load(cls, path: str) -> LiteRTModel:
    return LiteRTModel.load(path)


class LiteRTModel(Model):
  """The LiteRT model wrapper and inference runner."""

  def __init__(self, exporter: ModelExporter | bytes):
    if isinstance(exporter, bytes):
      exporter = BytesExporter(exporter)

    self._exporter = exporter
    self._interpreter_builder = lambda: interpreter_lib.Interpreter(
        model_content=exporter.to_bytes(),
        experimental_default_delegate_latest_features=True,
    )
    self._interpreter = None

  def set_interpreter_builder(
      self, builder: Callable[[], interpreter_lib.Interpreter]
  ) -> None:
    """Sets a custom interpreter builder.

    Args:
      builder: A function that returns a LiteRT Interpreter or its subclass.
    """
    self._interpreter_builder = builder
    self._interpreter = None

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

  def __call__(
      self,
      *args: npt.ArrayLike,
      signature_name: str = DEFAULT_SIGNATURE_NAME,
      **kwargs,
  ) -> npt.ArrayLike | tuple[npt.ArrayLike, ...]:
    """Runs inference on the LiteRT model using the provided arguments.

    Args:
      *args: The arguments to be passed to the model for inference.
      **kwargs: The arguments with specific names to be passed to the model for
        inference.
      signature_name: The name of the signature to be used for inference. The
        default signature is used if not provided.

    Returns:
      The output of the model. If the model has only one output, the output is
      returned directly.
    """
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

  def run_compiled(
      self,
      *args: npt.ArrayLike,
      signature_name: str = DEFAULT_SIGNATURE_NAME,
      hardware_accel: Any = None,
      require_fully_accelerated: bool = False,
      **kwargs,
  ) -> npt.ArrayLike | tuple[npt.ArrayLike, ...] | dict[str, npt.ArrayLike]:
    """Runs inference on the LiteRT model using the LiteRT CompiledModel runtime.

    Args:
      *args: Positional arguments passed to the model signature.
      signature_name: Name of the signature to execute.
      hardware_accel: Optional HardwareAccelerator bitmask (e.g.
        HardwareAccelerator.CPU, HardwareAccelerator.GPU). Defaults to CPU when
        None.
      require_fully_accelerated: If True, raises RuntimeError if any ops in the
        model failed to delegate to the requested hardware accelerator.
      **kwargs: Keyword arguments passed to the model signature.

    Returns:
      The model output(s) as numpy array(s).

    Raises:
      RuntimeError: If require_fully_accelerated is True and the model is not
        fully accelerated on the requested hardware accelerator.
      ValueError: If an invalid signature name is provided.
    """
    try:
      from ai_edge_litert.litert_wrapper.compiled_model_wrapper import compiled_model as cm_lib  # pylint: disable=g-import-not-at-top
    except ImportError:
      from ai_edge_litert import compiled_model as cm_lib  # pylint: disable=g-import-not-at-top

    kwargs_cm = {}
    if hardware_accel is not None:
      kwargs_cm['hardware_accel'] = hardware_accel

    cm = cm_lib.CompiledModel.from_buffer(self.model_content(), **kwargs_cm)
    if require_fully_accelerated and not cm.is_fully_accelerated():
      raise RuntimeError(
          'Model is not fully accelerated on the requested hardware'
          ' accelerator.'
      )

    sig_list = cm.get_signature_list()
    if signature_name not in sig_list:
      raise ValueError(
          f'Invalid signature name provided: {signature_name}. Available:'
          f' {list(sig_list.keys())}'
      )

    inputs = {f'args_{idx}': np.asarray(args[idx]) for idx in range(len(args))}
    for k, v in kwargs.items():
      inputs[k] = np.asarray(v)

    input_map = {}
    for name in sig_list[signature_name]['inputs']:
      buf = cm.create_input_buffer_by_name(signature_name, name)
      buf.write(np.ascontiguousarray(inputs[name]))
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
