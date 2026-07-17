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
"""Utility for inspecting and validating authoritative SoC hardware support."""

import os


def get_supported_socs(vendor: str) -> list[str]:
  """Reads supported SoCs from the symlinked supported_soc.csv for a vendor."""
  clean_vendor = vendor.strip().lower()
  csv_path = os.path.join(
      os.path.dirname(__file__), "vendors", clean_vendor, "supported_soc.csv"
  )

  if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"Authoritative SoC CSV not found for vendor '{vendor}' at: {csv_path}"
    )

  supported = []
  with open(csv_path, "r") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split(",")
      if len(parts) >= 2:
        model = parts[1].strip().lower()
        model_clean = model.split("(")[0].strip()
        if model_clean:
          supported.append(model_clean)

  return sorted(list(set(supported)))


def validate_soc(vendor: str, soc_model: str) -> str:
  """Validates a target SoC against authoritative vendor supported_soc.csv."""
  clean_vendor = vendor.strip().lower()
  clean_soc = soc_model.split("(")[0].strip().lower()
  supported = get_supported_socs(clean_vendor)
  if clean_soc not in supported:
    raise ValueError(
        f"SoC model '{soc_model}' is not in the authoritative list of supported"
        f" SoCs for vendor '{clean_vendor}' (supported_soc.csv). Supported"
        f" SoCs include: {supported}"
    )
  return clean_soc
