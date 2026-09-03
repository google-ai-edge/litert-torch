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
from litert_torch.testing import export
from litert_torch.testing import model_coverage
from litert_torch.testing import test_utils

export_with_tensor_inputs_only = export.export_with_tensor_inputs_only
V1_V2_PARAMETERS = test_utils.V1_V2_PARAMETERS
V1V2TestCase = test_utils.V1V2TestCase
parameterized_class = test_utils.parameterized_class
