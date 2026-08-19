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
"""Tests for Qwen 3.5 LiteRT exportable modules (prefill, decode, and split cache)."""

from absl.testing import absltest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5Config, Qwen3_5TextConfig
from transformers import DynamicCache

from litert_torch.generative.export_hf.core import cache as cache_lib
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core.split_cache import cache as split_cache_lib
from litert_torch.generative.export_hf.model_ext.qwen3_5 import exportable_module as qwen3_5_exportable


class Qwen35ExportableTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.manual_seed(42)
        cls.config = Qwen3_5TextConfig(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            head_dim=64,
            pad_token_id=0,
            rope_parameters={"rope_type": "default"},
        )
        cls.hf_model = AutoModelForCausalLM.from_config(cls.config)
        cls.hf_model.eval()

    def test_a_prefill_and_decode_exportable_equivalence(self):
        export_config = exportable_module_config.ExportableModuleConfig(
            model="qwen3_5_dummy",
            task=exportable_module_config.ExportTask.TEXT_GENERATION,
            batch_size=1,
            cache_length=512,
            prefill_lengths=[128],
        )
        prefill_mod = qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Prefill(self.hf_model, export_config)
        decode_mod = qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Generate(self.hf_model, export_config)
        self.assertEqual(prefill_mod.model.config._attn_implementation, "lrt_transposed_attention")
        self.assertEqual(decode_mod.model.config._attn_implementation, "lrt_transposed_attention")

        # 1. Prefill Chunk (128 tokens)
        sample_prefill_inputs = prefill_mod.get_sample_inputs(self.config)["prefill_128"][0]
        tokens = sample_prefill_inputs["tokens"]
        input_pos = torch.arange(128, dtype=torch.int64)
        kv_cache = sample_prefill_inputs["kv_cache"]
        mask = sample_prefill_inputs["mask"]

        with torch.no_grad():
            # HF reference prefill
            hf_cache = DynamicCache(config=self.hf_model.config)
            hf_out = self.hf_model(input_ids=tokens, past_key_values=hf_cache, use_cache=True)

            # Exportable prefill
            prefill_out = prefill_mod(tokens=tokens, input_pos=input_pos, kv_cache=kv_cache, mask=mask)
            export_cache = prefill_out["kv_cache"]

        # Verify cache states match exactly after prefill
        assert self.config.layer_types is not None
        for i in range(self.config.num_hidden_layers):
            layer_type = self.config.layer_types[i]
            if layer_type == "linear_attention":
                conv_L = self.config.linear_conv_kernel_dim - 1
                hf_conv = getattr(hf_cache.layers[i], "conv_states")
                hf_conv = hf_conv[0] if isinstance(hf_conv, dict) else hf_conv
                hf_rec = getattr(hf_cache.layers[i], "recurrent_states")
                hf_rec = hf_rec[0] if isinstance(hf_rec, dict) else hf_rec
                torch.testing.assert_close(export_cache.layers[i].conv_states, hf_conv[:, :, -conv_L:], rtol=1e-4, atol=1e-4)  # pyrefly: ignore[missing-attribute]
                torch.testing.assert_close(export_cache.layers[i].recurrent_states, hf_rec, rtol=1e-4, atol=1e-4)  # pyrefly: ignore[missing-attribute]
            else:
                torch.testing.assert_close(export_cache.layers[i].keys[:, :, :128], hf_cache.layers[i].keys, rtol=1e-4, atol=1e-4)  # pyrefly: ignore[missing-attribute]
                torch.testing.assert_close(export_cache.layers[i].values[:, :, :, :128], hf_cache.layers[i].values.transpose(2, 3), rtol=1e-4, atol=1e-4)  # pyrefly: ignore[missing-attribute]

        # 2. Decode Step (1 token)
        next_token = torch.randint(0, self.config.vocab_size, (1, 1), dtype=torch.int64)
        next_pos = torch.tensor([128], dtype=torch.int64)

        with torch.no_grad():
            hf_decode_out = self.hf_model(input_ids=next_token, position_ids=next_pos.unsqueeze(0), past_key_values=hf_cache, use_cache=True)
            decode_mask = qwen3_5_exportable.create_qwen3_5_attention_mask(1, 512, next_pos)
            decode_out = decode_mod(tokens=next_token, input_pos=next_pos, kv_cache=export_cache, mask=decode_mask)

        torch.testing.assert_close(decode_out["logits"], hf_decode_out.logits, rtol=1e-4, atol=1e-4)

    def test_b_e2e_exportable_generation(self):
        export_config = exportable_module_config.ExportableModuleConfig(
            model="qwen3_5_dummy",
            task=exportable_module_config.ExportTask.TEXT_GENERATION,
            batch_size=1,
            cache_length=512,
            prefill_lengths=[9],
        )
        prefill_mod = qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Prefill(self.hf_model, export_config)
        decode_mod = qwen3_5_exportable.LiteRTExportableModuleForQwen3_5Generate(self.hf_model, export_config)
        self.assertEqual(prefill_mod.model.config._attn_implementation, "lrt_transposed_attention")
        self.assertEqual(decode_mod.model.config._attn_implementation, "lrt_transposed_attention")

        prompt = torch.randint(0, self.config.vocab_size, (1, 10), dtype=torch.int64)

        # HF generation
        with torch.no_grad():
            hf_generated = self.hf_model.generate(prompt, max_new_tokens=8, do_sample=False)

        # Exportable generation loop
        sample_prefill = prefill_mod.get_sample_inputs(self.config)["prefill_9"][0]
        sample_prefill["tokens"] = prompt[:, :-1]
        sample_prefill["input_pos"] = torch.arange(9, dtype=torch.int64)

        with torch.no_grad():
            prefill_out = prefill_mod(**sample_prefill)
            kv_cache = prefill_out["kv_cache"]

            curr_token = prompt[:, -1:]
            generated_tokens = [prompt]
            for step in range(8):
                pos = torch.tensor([9 + step], dtype=torch.int64)
                mask = qwen3_5_exportable.create_qwen3_5_attention_mask(1, 512, pos)
                decode_out = decode_mod(tokens=curr_token, input_pos=pos, kv_cache=kv_cache, mask=mask)
                kv_cache = decode_out["kv_cache"]
                next_token = decode_out["logits"][:, -1:, :].argmax(dim=-1)
                generated_tokens.append(next_token)
                curr_token = next_token

        # Verify all generated tokens match exactly
        export_generated = torch.cat(generated_tokens, dim=-1)
        self.assertTrue(torch.equal(export_generated[:, :hf_generated.shape[1]], hf_generated))


if __name__ == "__main__":
    absltest.main()
