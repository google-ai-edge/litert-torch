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
"""Tests for split cache exportable modules."""

from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.split_cache import exportable_module
import torch

from absl.testing import absltest as googletest


class _MockLfm2Config:
  num_hidden_layers = 2
  layer_types = ['conv', 'full_attention']
  conv_L_cache = 3
  hidden_size = 32
  num_attention_heads = 4
  num_key_value_heads = 2


class CacheUpdateTest(googletest.TestCase):

  def test_cache_update_with_conv_layer(self):
    export_config = exportable_module_config.ExportableModuleConfig(
        model='dummy_model', prefill_lengths=[4], cache_length=16
    )
    for valid, takes_slice in (
        ([[True, True, True, False]], True),
        # A call without real tokens keeps the conv state.
        ([[False, False, False, False]], False),
    ):
      with self.subTest(valid=valid):
        inputs, _ = exportable_module.CacheUpdate.get_sample_inputs(
            _MockLfm2Config(), export_config
        )['prefill_cache_update_4']
        conv_cache = inputs['kv_cache'].layers[0]
        conv_slice, attention_slice = inputs['kv_slice'].layers
        old_state = torch.randn_like(conv_cache.conv_states)
        conv_cache.conv_states = old_state.clone()
        conv_slice.conv_states = torch.randn_like(conv_slice.conv_states)
        attention_slice.keys = torch.randn_like(attention_slice.keys)
        attention_slice.values = torch.randn_like(attention_slice.values)
        inputs['input_pos'] = torch.arange(5, 9, dtype=torch.int32)
        inputs['valid_mask'] = torch.tensor(valid)

        out = exportable_module.CacheUpdate()(**inputs)['kv_cache']

        torch.testing.assert_close(
            out.layers[0].conv_states,
            conv_slice.conv_states if takes_slice else old_state,
            rtol=0,
            atol=0,
        )
        # The attention layer is still written at `input_pos`.
        torch.testing.assert_close(
            out.layers[1].keys[:, :, 5:9],
            attention_slice.keys,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            out.layers[1].values[..., 5:9],
            attention_slice.values,
            rtol=0,
            atol=0,
        )


if __name__ == '__main__':
  googletest.main()
