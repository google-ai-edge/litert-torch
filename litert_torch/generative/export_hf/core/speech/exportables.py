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

"""Exportable modules for speech (ASR) models."""

import math
from typing import Any
from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
from litert_torch.generative.export_hf.core.speech import asr_model as asr_model_lib
import numpy as np
import torch


class LiteRTExportableModuleForAsrEncode(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for ASR encoder."""

  def __init__(self, asr_model: asr_model_lib.AsrModel, export_config):
    super().__init__(export_config)
    self.asr_model = asr_model
    self.encoder = asr_model.get_encoder()

  def forward(self, *args, **kwargs):
    return self.encoder(*args, **kwargs)

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[Any, dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the ASR encoder."""
    processor = self.asr_model.get_processor()
    sr = processor.get_sampling_rate()
    input_sec = getattr(self.export_config, "input_sec", 1.0)
    dummy_audio = np.zeros(int(input_sec * sr), dtype=np.float32)
    processed = processor.process(dummy_audio)
    encoder_inputs = self.asr_model.get_encoder_sample_input(processed)
    return {"encode": (encoder_inputs, {})}


class LiteRTExportableModuleForAsrDecode(
    exportable_module_base.ExportableModuleBase
):
  """Exportable module for ASR decoder."""

  def __init__(
      self,
      asr_model: asr_model_lib.AsrModel,
      export_config,
      encoder_output: tuple[torch.Tensor, ...] | None = None,
  ):
    super().__init__(export_config)
    self.asr_model = asr_model
    self.decoder = asr_model.get_decoder()
    self._encoder_output = encoder_output

  def forward(self, *args, **kwargs):
    return self.decoder(*args, **kwargs)

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[Any, dict[str, torch.export.Dim]]]:
    """Returns the sample inputs for the ASR decoder."""
    if self._encoder_output is None:
      raise ValueError(
          "encoder_output must be provided to AsrDecode exportable."
      )
    input_sec = getattr(self.export_config, "input_sec", 1.0)
    stateful_after = getattr(self.export_config, "stateful_after", -1)

    num_tokens = math.ceil(input_sec * 8 / 32) * 32
    if 0 <= stateful_after < num_tokens:
      num_tokens = stateful_after if stateful_after > 0 else 1

    decoder_inputs = self.asr_model.get_decoder_sample_input(
        self._encoder_output, num_tokens
    )
    signatures = {"decode": (decoder_inputs, {})}
    if stateful_after > 1:
      decoder_inputs_1 = self.asr_model.get_decoder_sample_input(
          self._encoder_output, 1
      )
      signatures["decode_1"] = (decoder_inputs_1, {})
    return signatures
