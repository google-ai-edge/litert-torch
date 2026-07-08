#!/usr/bin/env bash
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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

IGNORE_LARGE_TESTS="--ignore-glob=*_large.py --ignore-glob=*equivalence_test.py"
if [[ "$RUN_LARGE_TESTS" == "true" ]]; then
  IGNORE_LARGE_TESTS=""
fi

# TODO(b/362799258) Setup CIs to test odml-torch path and remove test ignore
PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH \
    python -m pytest $IGNORE_LARGE_TESTS $SCRIPT_DIR -n auto
