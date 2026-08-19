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
"""Export HF model extensions."""

from litert_torch.generative.export_hf.model_ext.gemma3 import patch as _
from litert_torch.generative.export_hf.model_ext.gemma3n import patch as _
from litert_torch.generative.export_hf.model_ext.gemma4 import patch as _
from litert_torch.generative.export_hf.model_ext.gemma4_unified import patch as _
from litert_torch.generative.export_hf.model_ext.lfm2 import patch as _
from litert_torch.generative.export_hf.model_ext.lfm2_vl import patch as _
from litert_torch.generative.export_hf.model_ext.moonshine import patch as _
from litert_torch.generative.export_hf.model_ext.parakeet import patch as _
from litert_torch.generative.export_hf.model_ext.qwen3 import patch as _
from litert_torch.generative.export_hf.model_ext.qwen3_tts import patch as _
from litert_torch.generative.export_hf.model_ext.whisper import patch as _
from litert_torch.generative.layers import moe

try:
  from transformers.integrations import moe as transformers_moe

  transformers_moe.ALL_EXPERTS_FUNCTIONS.register(
      "litert_moe", moe.litert_moe_experts_forward
  )
  transformers_moe.ALL_EXPERTS_FUNCTIONS.register(
      "litert_moe_sequential", moe.litert_sequential_experts_forward
  )
except (ImportError, AttributeError):
  pass
