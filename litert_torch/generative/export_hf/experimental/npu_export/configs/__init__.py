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
"""Vendor configuration definitions for NPU compilation."""

from litert_torch.generative.export_hf.experimental.npu_export.configs import vendor_configs

VENDOR_CONFIGS = vendor_configs.VENDOR_CONFIGS
__all__ = [
    "VENDOR_CONFIGS",
]
