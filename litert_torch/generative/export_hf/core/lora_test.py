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
"""Tests for LoRA support in export_hf."""

import os
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import exportable_module_config
from litert_torch.generative.export_hf.core import lora as lora_lib
from litert_torch.generative.export_hf.core import weights_loader as weights_loader_lib
from litert_torch.generative.export_hf.model_ext import patches as model_ext_patches
import numpy as np
import torch
from torch import nn
import torch.utils._pytree as pytree
import transformers

from ai_edge_litert import interpreter as tfl_interpreter  # pylint: disable=g-direct-tensorflow-import


def _create_tiny_llama_model():
  config = transformers.LlamaConfig(
      vocab_size=64,
      hidden_size=32,
      intermediate_size=64,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=8,
      max_position_embeddings=64,
  )
  config._attn_implementation = "lrt_transposed_attention"  # pylint: disable=protected-access
  model = transformers.LlamaForCausalLM(config).eval()
  return model, config


def _create_tiny_gemma3_model():
  config = transformers.Gemma3TextConfig(
      vocab_size=64,
      hidden_size=32,
      intermediate_size=64,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=1,
      head_dim=8,
      max_position_embeddings=64,
  )
  config._attn_implementation = "lrt_transposed_attention"  # pylint: disable=protected-access
  model = transformers.Gemma3ForCausalLM(config).eval()
  return model, config


def _flatten_inputs_for_tflite(sample_inputs, lora=None):
  """Flattens PyTorch sample inputs and optional LoRA PyTree into numpy dict for TFLite SignatureRunner."""
  flat_dict = {}
  # Flatten sample_inputs (tokens, input_pos, mask, kv_cache)
  for k, v in sample_inputs.items():
    if k == "kv_cache":
      leaves, _ = pytree.tree_flatten(v)
      # kv_cache leaves are k_0, v_0, k_1, v_1, ...
      for layer_idx in range(len(leaves) // 2):
        flat_dict[f"kv_cache_k_{layer_idx}"] = (
            leaves[2 * layer_idx].detach().numpy()
        )
        flat_dict[f"kv_cache_v_{layer_idx}"] = (
            leaves[2 * layer_idx + 1].detach().numpy()
        )
    elif isinstance(v, torch.Tensor):
      flat_dict[k] = v.detach().numpy()

  if lora is not None:
    leaves_with_keys, _ = pytree.tree_flatten_with_path(lora)
    for path, leaf in leaves_with_keys:
      # path[0].key is e.g. 'atten_q_a_prime_weight_0'
      key_str = path[0].key
      flat_dict[f"lora_{key_str}"] = leaf.detach().numpy()
  return flat_dict


class LoraExportHfTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    torch.manual_seed(42)

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_config_lora_ranks_normalization(self):
    cfg1 = exportable_module_config.ExportableModuleConfig(
        model="dummy", lora_ranks=8
    )
    self.assertEqual(cfg1.lora_ranks, [8])

    cfg2 = exportable_module_config.ExportableModuleConfig(
        model="dummy", lora_ranks="8, 16"
    )
    self.assertEqual(cfg2.lora_ranks, [8, 16])

    cfg3 = exportable_module_config.ExportableModuleConfig(
        model="dummy", lora_ranks=[4, 8]
    )
    self.assertEqual(cfg3.lora_ranks, [4, 8])

  def test_lora_linear_patch_and_shapes(self):
    model, _ = _create_tiny_llama_model()
    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy", lora_ranks=[8]
    )

    lora = lora_lib.create_zeros_lora(rank=8, model=model)
    self.assertLen(lora.adapters, 2)
    attn0 = lora.adapters[0].attention
    # hidden_size=32, num_heads=4, num_kv_heads=2, head_dim=8
    self.assertEqual(attn0.query.a_prime.shape, (32, 8))
    self.assertEqual(attn0.query.b_prime.shape, (8, 32))
    self.assertEqual(attn0.key.a_prime.shape, (32, 8))
    self.assertEqual(attn0.key.b_prime.shape, (8, 16))
    self.assertEqual(attn0.value.a_prime.shape, (32, 8))
    self.assertEqual(attn0.value.b_prime.shape, (8, 16))
    self.assertEqual(attn0.output.a_prime.shape, (32, 8))
    self.assertEqual(attn0.output.b_prime.shape, (8, 32))

    # Patching hooks projections in place: module types and parameter FQNs
    # (which checkpoint weights loaders rely on) are unchanged.
    state_dict_keys = list(model.state_dict().keys())
    q_proj = model.model.layers[0].self_attn.q_proj
    x = torch.randn(1, 3, 32)
    base_out = q_proj(x)
    with lora_lib.patch_model_for_lora(model, export_config):
      self.assertIsInstance(q_proj, nn.Linear)
      self.assertEqual(list(model.state_dict().keys()), state_dict_keys)
      random_lora = lora_lib.create_random_lora(rank=8, model=model)
      with lora_lib.bind_lora_weights(model, random_lora):
        lora_out = q_proj(x)
      np.testing.assert_allclose(
          lora_out.detach().numpy(),
          (
              base_out
              + (x @ random_lora.adapters[0].attention.query.a_prime)
              @ random_lora.adapters[0].attention.query.b_prime
          ).detach().numpy(),
          atol=1e-5,
      )
      # Unbound: base output.
      np.testing.assert_allclose(
          q_proj(x).detach().numpy(), base_out.detach().numpy()
      )
    # Hooks removed after exiting context.
    self.assertEmpty(q_proj._forward_hooks)  # pylint: disable=protected-access
    self.assertEqual(list(model.state_dict().keys()), state_dict_keys)

  def test_export_llama_with_lora_signatures_and_numerical_accuracy(self):
    model, config = _create_tiny_llama_model()
    export_config = exportable_module_config.ExportableModuleConfig(
        model="dummy",
        work_dir=self.test_dir,
        output_dir=self.test_dir,
        prefill_lengths=[8],
        cache_length=16,
        lora_ranks=[4, 8],
        quantization_recipe=None,
        cache_implementation="LiteRTLMCache",
    )
    source_artifacts = export_lib.SourceModelArtifacts(
        model=model,
        model_config=config,
        text_model_config=config,
        tokenizer=None,  # pyrefly: ignore[bad-argument-type]
    )
    exported_artifacts = export_lib.ExportedModelArtifacts()

    exported_artifacts = export_lib.export_text_prefill_decode_model(
        source_artifacts, export_config, exported_artifacts
    )
    self.assertIsNotNone(exported_artifacts.prefill_decode_model_path)

    interpreter = tfl_interpreter.Interpreter(
        model_path=exported_artifacts.prefill_decode_model_path
    )
    signatures = interpreter.get_signature_list()
    expected_sigs = {
        "prefill_8",
        "prefill_8_lora_r4",
        "prefill_8_lora_r8",
        "decode",
        "decode_lora_r4",
        "decode_lora_r8",
    }
    self.assertEqual(set(signatures.keys()), expected_sigs)

    # Verify LoRA input tensor names on decode_lora_r4
    decode_r4_inputs = set(signatures["decode_lora_r4"]["inputs"])
    for layer_idx in range(2):
      for proj in ("q", "k", "v", "o"):
        for mat in ("a", "b"):
          self.assertIn(
              f"lora_atten_{proj}_{mat}_prime_weight_{layer_idx}",
              decode_r4_inputs,
          )

    # Numerical comparison between PyTorch and TFLite on decode
    random_lora = lora_lib.create_random_lora(rank=4, model=model)
    with (
        model_ext_patches.patch_model(model, config.model_type, export_config),
        lora_lib.patch_model_for_lora(model, export_config),
    ):
      model.set_attn_implementation("lrt_transposed_attention")
      decode_module = (
          exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate(
              model, export_config, source_artifacts
          ).eval()
      )
      sample_inputs_dict = decode_module.get_sample_inputs(config)
      decode_inputs, _ = sample_inputs_dict["decode"]

      with torch.no_grad():
        pt_base_out = decode_module(**decode_inputs)
        pt_lora_out = decode_module(**decode_inputs, lora=random_lora)

    pt_base_logits = pt_base_out["logits"].detach().numpy()
    pt_lora_logits = pt_lora_out["logits"].detach().numpy()

    # Ensure LoRA actually changes model output
    self.assertGreater(
        np.max(np.abs(pt_lora_logits - pt_base_logits)),
        1e-2,
    )

    # Run TFLite base decode signature
    base_runner = interpreter.get_signature_runner("decode")
    tfl_base_inputs = _flatten_inputs_for_tflite(decode_inputs, lora=None)
    tfl_base_out = base_runner(**tfl_base_inputs)
    np.testing.assert_allclose(
        pt_base_logits, tfl_base_out["logits"], atol=1e-4, rtol=1e-4
    )

    # Run TFLite decode_lora_r4 signature
    lora_runner = interpreter.get_signature_runner("decode_lora_r4")
    tfl_lora_inputs = _flatten_inputs_for_tflite(
        decode_inputs, lora=random_lora
    )
    tfl_lora_out = lora_runner(**tfl_lora_inputs)
    np.testing.assert_allclose(
        pt_lora_logits, tfl_lora_out["logits"], atol=1e-4, rtol=1e-4
    )

  def test_config_rejects_lora_with_fuse_qkv(self):
    with self.assertRaisesRegex(ValueError, "fuse_qkv"):
      exportable_module_config.ExportableModuleConfig(
          model="dummy", lora_ranks=[4], fuse_qkv=True
      )

  @parameterized.named_parameters(
      ("llama", _create_tiny_llama_model),
      ("gemma3", _create_tiny_gemma3_model),
  )
  # TODO(weiyiw): Re-enable once Converter V2 is available in open source.
  @absltest.skip("Converter V2 is not available in open source yet.")
  def test_export_with_lora_v2_converter(self, model_fn):
    """LoRA export with Converter V2 streaming weights from a checkpoint."""
    ref_model, config = model_fn()
    ckpt_dir = os.path.join(self.test_dir, "ckpt")
    ref_model.save_pretrained(ckpt_dir, safe_serialization=True)

    # Mirrors the `use_v2` branch of `export_lib.load_model`: parameters live on
    # meta and are streamed from the checkpoint by the converter.
    with weights_loader_lib.init_params_on_meta():
      model = type(ref_model)(config).eval()
    weights_loader = weights_loader_lib.HFCheckpointWeightsLoader(
        ckpt_dir,
        tie_word_embeddings=(
            model.get_output_embeddings().weight
            is model.get_input_embeddings().weight
        ),
    )
    weights_loader_lib.load_persistent_buffers_(model, weights_loader)

    export_config = exportable_module_config.ExportableModuleConfig(
        model=ckpt_dir,
        work_dir=self.test_dir,
        output_dir=self.test_dir,
        prefill_lengths=[8],
        cache_length=16,
        lora_ranks=[4],
        quantization_recipe=None,
        cache_implementation="LiteRTLMCache",
        use_v2=True,
        delete_in_memory_params=True,
    )
    source_artifacts = export_lib.SourceModelArtifacts(
        model=model,
        model_config=config,
        text_model_config=config,
        tokenizer=None,  # pyrefly: ignore[bad-argument-type]
        weights_loader=weights_loader,
    )
    exported_artifacts = export_lib.export_text_prefill_decode_model(
        source_artifacts, export_config, export_lib.ExportedModelArtifacts()
    )

    interpreter = tfl_interpreter.Interpreter(
        model_path=exported_artifacts.prefill_decode_model_path
    )
    signatures = interpreter.get_signature_list()
    self.assertEqual(
        set(signatures.keys()),
        {"prefill_8", "prefill_8_lora_r4", "decode", "decode_lora_r4"},
    )
    decode_lora_inputs = set(signatures["decode_lora_r4"]["inputs"])
    for layer_idx in range(config.num_hidden_layers):
      for proj in ("q", "k", "v", "o"):
        for mat in ("a", "b"):
          self.assertIn(
              f"lora_atten_{proj}_{mat}_prime_weight_{layer_idx}",
              decode_lora_inputs,
          )

    # Reference: eager PyTorch on the real-weight model.
    random_lora = lora_lib.create_random_lora(rank=4, model=ref_model)
    with (
        model_ext_patches.patch_model(
            ref_model, config.model_type, export_config
        ),
        lora_lib.patch_model_for_lora(ref_model, export_config),
    ):
      ref_model.set_attn_implementation("lrt_transposed_attention")
      decode_module = (
          exportable_module.LiteRTExportableModuleForDecoderOnlyLMGenerate(
              ref_model, export_config, source_artifacts
          ).eval()
      )
      decode_inputs, _ = decode_module.get_sample_inputs(config)["decode"]
      with torch.no_grad():
        pt_base_logits = decode_module(**decode_inputs)["logits"].numpy()
        pt_lora_logits = decode_module(**decode_inputs, lora=random_lora)[
            "logits"
        ].numpy()

    tfl_base_logits = interpreter.get_signature_runner("decode")(
        **_flatten_inputs_for_tflite(decode_inputs)
    )["logits"]
    tfl_lora_logits = interpreter.get_signature_runner("decode_lora_r4")(
        **_flatten_inputs_for_tflite(decode_inputs, lora=random_lora)
    )["logits"]
    np.testing.assert_allclose(
        pt_base_logits, tfl_base_logits, atol=1e-4, rtol=1e-4
    )
    np.testing.assert_allclose(
        pt_lora_logits, tfl_lora_logits, atol=1e-4, rtol=1e-4
    )


if __name__ == "__main__":
  absltest.main()
