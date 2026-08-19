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
"""Default configurations for NPU compiler backend."""

GENERIC_DEFAULT_CONFIGS = {
    'qualcomm': {
        'prefill_decode': {
            'compile': True,
            'flags': [
                '--qualcomm_optimization_level=O3',
                '--qualcomm_log_level=off',
            ],
        },
        'aux': {
            'compile': True,
            'flags': [
                '--qualcomm_optimization_level=O3',
                '--qualcomm_log_level=off',
            ],
        },
    },
    'mediatek': {
        'prefill_decode': {
            'compile': True,
            'flags': [
                '--mediatek_enable_gemma_compiler_optimizations=true',
                '--mediatek_performance_mode_type=turbo_boost',
                '--mediatek_enable_l1_cache_optimizations=true',
                '--mediatek_optimization_hint=low_latency',
            ],
        },
    },
    # Add Google Tensor and other default backend configs here.
}

MODEL_SPECIFIC_DEFAULTS = {
    # Keep empty for future model-specific overrides
}
