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
"""Imports and registers all built-in ModelExtension implementations."""

from litert_torch.generative.export_hf.model_ext.bonsai_flux2 import extension as bonsai_flux2_ext
from litert_torch.generative.export_hf.model_ext.flux2_klein import extension as flux2_klein_ext
from litert_torch.generative.export_hf.model_ext.gemma3 import extension as gemma3_ext
from litert_torch.generative.export_hf.model_ext.gemma3n import extension as gemma3n_ext
from litert_torch.generative.export_hf.model_ext.gemma4 import extension as gemma4_ext
from litert_torch.generative.export_hf.model_ext.gemma4_unified import extension as gemma4_unified_ext
from litert_torch.generative.export_hf.model_ext.kokoro_tts import extension as kokoro_tts_ext
from litert_torch.generative.export_hf.model_ext.lfm2 import extension as lfm2_ext
from litert_torch.generative.export_hf.model_ext.lfm2_vl import extension as lfm2_vl_ext
from litert_torch.generative.export_hf.model_ext.minicpm_o import extension as minicpm_o_ext
from litert_torch.generative.export_hf.model_ext.moonshine import extension as moonshine_ext
from litert_torch.generative.export_hf.model_ext.parakeet import extension as parakeet_ext
from litert_torch.generative.export_hf.model_ext.qwen3 import extension as qwen3_ext
from litert_torch.generative.export_hf.model_ext.qwen3_5 import extension as qwen3_5_ext
from litert_torch.generative.export_hf.model_ext.qwen3_tts import extension as qwen3_tts_ext
from litert_torch.generative.export_hf.model_ext.whisper import extension as whisper_ext

BonsaiFlux2Extension = bonsai_flux2_ext.BonsaiFlux2Extension
Flux2KleinExtension = flux2_klein_ext.Flux2KleinExtension
Gemma3Extension = gemma3_ext.Gemma3Extension
Gemma3nExtension = gemma3n_ext.Gemma3nExtension
Gemma4Extension = gemma4_ext.Gemma4Extension
Gemma4UnifiedExtension = gemma4_unified_ext.Gemma4UnifiedExtension
KokoroTtsExtension = kokoro_tts_ext.KokoroTtsExtension
Lfm2Extension = lfm2_ext.Lfm2Extension
Lfm2VlExtension = lfm2_vl_ext.Lfm2VlExtension
MiniCpmOExtension = minicpm_o_ext.MiniCpmOExtension
MoonshineExtension = moonshine_ext.MoonshineExtension
ParakeetCtcExtension = parakeet_ext.ParakeetCtcExtension
ParakeetTdtExtension = parakeet_ext.ParakeetTdtExtension
Qwen3AsrExtension = qwen3_ext.Qwen3AsrExtension
Qwen3Extension = qwen3_ext.Qwen3Extension
Qwen3TtsExtension = qwen3_tts_ext.Qwen3TtsExtension
Qwen3_5Extension = qwen3_5_ext.Qwen3_5Extension
WhisperExtension = whisper_ext.WhisperExtension
