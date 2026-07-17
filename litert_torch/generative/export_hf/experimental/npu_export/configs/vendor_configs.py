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
"""Vendor and SoC level default configurations for LiteRT NPU export pipeline."""

from typing import Any

VENDOR_CONFIGS: dict[str, dict[str, Any]] = {
    "qualcomm": {
        "backend": "qualcomm",
        "default_soc": "sm8850",
        "k_ts_idx": 3,
        "v_ts_idx": 2,
        "a16w8": True,
        "allow_float_operations": False,
    },
    "mediatek": {
        "backend": "mediatek",
        "default_soc": "mt6993",
        "k_ts_idx": 2,
        "v_ts_idx": 3,
        "a16w8": True,
        "allow_float_operations": True,
    },
}

# Sparse per-SoC override map for SoCs requiring non-default vendor settings.
SOC_CONFIGS: dict[str, dict[str, Any]] = {}
