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
"""Debug info generation for ODML Torch."""

from . import _build

build_nodename_debuginfo = _build.build_nodename_debuginfo
build_mlir_file_debuginfo = _build.build_mlir_file_debuginfo
build_mlir_debuginfo = _build.build_mlir_debuginfo
