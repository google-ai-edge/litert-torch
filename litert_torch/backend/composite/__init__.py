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
from litert_torch.backend.composite import mark_tensor as _mark_tensor
from litert_torch.backend.composite import stablehlo_composite_builder as _stablehlo_composite_builder

mark_tensor_op = _mark_tensor.mark_tensor_op
serialize_composite_attr = _mark_tensor.serialize_composite_attr
deserialize_composite_attr = _mark_tensor.deserialize_composite_attr
StableHLOCompositeBuilder = (
    _stablehlo_composite_builder.StableHLOCompositeBuilder
)
