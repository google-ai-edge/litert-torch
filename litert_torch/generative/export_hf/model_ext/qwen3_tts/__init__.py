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
"""LiteRT Torch model extensions for Qwen3-TTS."""

from litert_torch.generative.export_hf.model_ext.qwen3_tts import exportable_module
from litert_torch.generative.export_hf.model_ext.qwen3_tts import patch
from litert_torch.generative.export_hf.model_ext.qwen3_tts import qwen3_tts
from litert_torch.generative.export_hf.model_ext.qwen3_tts import speaker_encoder

__all__ = ["exportable_module", "patch", "qwen3_tts", "speaker_encoder"]
