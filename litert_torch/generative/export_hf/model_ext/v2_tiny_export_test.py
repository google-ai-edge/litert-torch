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
"""Fast hermetic Converter V2 `.tflite` export tests across model families."""

from typing import Any

from litert_torch._convert import converter_v2
from litert_torch.generative.export_hf.model_ext import builtin_extensions
from litert_torch.generative.export_hf.model_ext import testing as ext_testing
import torch
import transformers
from transformers.models.gemma4 import configuration_gemma4
from transformers.models.gemma4 import modeling_gemma4
from transformers.models.lfm2 import modeling_lfm2
from transformers.models.qwen3_5 import configuration_qwen3_5
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5 import modular_qwen3_5

from absl.testing import absltest as googletest
from absl.testing import parameterized


def _build_tiny_qwen3() -> tuple[Any, torch.nn.Module]:
  ext = builtin_extensions.Qwen3Extension()
  cfg = transformers.Qwen3Config(
      vocab_size=64,
      hidden_size=32,
      intermediate_size=64,
      num_hidden_layers=2,
      num_attention_heads=2,
      num_key_value_heads=2,
      head_dim=16,
      max_position_embeddings=64,
  )
  with ext.patch_context():
    model = transformers.Qwen3ForCausalLM(cfg).eval()
  return ext, model


def _build_tiny_qwen3_5() -> tuple[Any, torch.nn.Module]:
  ext = builtin_extensions.Qwen3_5Extension()
  cfg = configuration_qwen3_5.Qwen3_5TextConfig(
      vocab_size=64,
      hidden_size=64,
      intermediate_size=128,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=16,
      linear_num_value_heads=4,
      linear_num_key_heads=2,
      linear_key_head_dim=16,
      linear_value_head_dim=16,
      linear_conv_kernel_dim=4,
      layer_types=['linear_attention', 'full_attention'],
      rope_parameters={
          'rope_theta': 10000.0,
          'partial_rotary_factor': 0.25,
      },
      rms_norm_eps=1e-6,
  )
  cfg._attn_implementation = 'eager'
  with ext.patch_context():
    model = modeling_qwen3_5.Qwen3_5ForCausalLM(cfg).eval()
  # Populate non-zero RMSNorm weights so `(1.0 + weight)` streaming and partial
  # RoPE channel permutation are exercised against non-trivial values.
  with torch.no_grad():
    for module in model.modules():
      if isinstance(
          module,
          (modeling_qwen3_5.Qwen3_5RMSNorm, modular_qwen3_5.Qwen3_5RMSNorm),
      ):
        module.weight.normal_(mean=0.0, std=0.1)
  return ext, model


def _build_tiny_gemma3() -> tuple[Any, torch.nn.Module]:
  ext = builtin_extensions.Gemma3Extension()
  cfg = transformers.Gemma3TextConfig(
      vocab_size=64,
      hidden_size=32,
      intermediate_size=64,
      num_hidden_layers=2,
      num_attention_heads=2,
      num_key_value_heads=1,
      head_dim=16,
      sliding_window=8,
      layer_types=['sliding_attention', 'full_attention'],
      max_position_embeddings=64,
  )
  with ext.patch_context():
    model = transformers.Gemma3ForCausalLM(cfg).eval()
  return ext, model


def _build_tiny_gemma4() -> tuple[Any, torch.nn.Module]:
  ext = builtin_extensions.Gemma4Extension()
  cfg = configuration_gemma4.Gemma4TextConfig(
      vocab_size=64,
      vocab_size_per_layer_input=64,
      hidden_size=32,
      intermediate_size=64,
      hidden_size_per_layer_input=32,
      num_hidden_layers=3,
      num_kv_shared_layers=1,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=16,
      global_head_dim=16,
      sliding_window=16,
      layer_types=['sliding_attention', 'full_attention', 'sliding_attention'],
      final_logit_softcapping=30.0,
      attention_dropout=0.0,
      hidden_activation='gelu_pytorch_tanh',
  )
  cfg._attn_implementation = 'eager'
  with ext.patch_context():
    model = modeling_gemma4.Gemma4ForCausalLM(cfg).eval()
  return ext, model


def _build_tiny_lfm2() -> tuple[Any, torch.nn.Module]:
  ext = builtin_extensions.Lfm2Extension()
  cfg = modeling_lfm2.Lfm2Config(
      vocab_size=64,
      num_attention_heads=4,
      num_key_value_heads=2,
      hidden_size=64,
      intermediate_size=128,
      block_auto_adjust_ff_dim=False,
      num_hidden_layers=2,
      layer_types=['conv', 'full_attention'],
      full_attn_idxs=[1],
      rope_parameters={'rope_type': 'default', 'rope_theta': 1000000.0},
  )
  cfg._attn_implementation = 'eager'
  with ext.patch_context():
    model = modeling_lfm2.Lfm2ForCausalLM(cfg).eval()
  return ext, model


_MODEL_BUILDERS = {
    'qwen3': _build_tiny_qwen3,
    'qwen3_5': _build_tiny_qwen3_5,
    'gemma3': _build_tiny_gemma3,
    'gemma4': _build_tiny_gemma4,
    'lfm2': _build_tiny_lfm2,
}


class V2TinyExportTest(parameterized.TestCase, googletest.TestCase):

  def setUp(self):
    super().setUp()
    if not converter_v2.is_supported():
      self.skipTest(
          'Converter V2 requires a newer TensorFlow / tf-nightly build.'
      )

  @parameterized.named_parameters(
      ('qwen3', 'qwen3'),
      ('qwen3_5', 'qwen3_5'),
      ('gemma3', 'gemma3'),
      ('gemma4', 'gemma4'),
      ('lfm2', 'lfm2'),
  )
  def test_v2_weights_loader_exact_parity(self, family: str):
    torch.manual_seed(0)
    ext, model = _MODEL_BUILDERS[family]()
    resolved = ext_testing.assert_v2_weights_resolvable(
        ext, model, check_values=True
    )
    self.assertNotEmpty(resolved)

  @parameterized.named_parameters(
      (
          'qwen3_cpu_gpu',
          'qwen3',
          'auto',
          'cpu_gpu',
          {'externalize_embedder': True},
          ('model.tflite', 'embedder.tflite'),
          4,
      ),
      (
          'qwen3_npu_qualcomm',
          'qwen3',
          'auto',
          'npu:qualcomm',
          {},
          ('model.tflite', 'embedder.tflite', 'auxiliary.tflite'),
          None,
      ),
      (
          'qwen3_5_cpu_gpu_bundled',
          'qwen3_5',
          'auto',
          'cpu_gpu',
          {
              'externalize_embedder': False,
              'apply_gpu_composites': False,
          },
          ('model.tflite',),
          4,
      ),
      (
          'qwen3_5_cpu_gpu_composites',
          'qwen3_5',
          'auto',
          'cpu_gpu',
          {'externalize_embedder': True},
          ('model.tflite', 'embedder.tflite'),
          4,
      ),
      (
          'gemma3_cpu_gpu_ring_buffer',
          'gemma3',
          'auto',
          'cpu_gpu',
          {
              'externalize_embedder': True,
              'single_token_embedder': True,
              'sliding_window_ring_buffer_size': 16,
          },
          ('model.tflite', 'embedder.tflite'),
          4,
      ),
      (
          'gemma3_npu_google_tensor',
          'gemma3',
          'auto',
          'npu:google_tensor',
          {},
          ('model.tflite', 'embedder.tflite', 'auxiliary.tflite'),
          None,
      ),
      (
          'gemma4_cpu_gpu_ple',
          'gemma4',
          'text_generation',
          'cpu_gpu',
          {},
          ('model.tflite', 'embedder.tflite', 'per_layer_embedder.tflite'),
          4,
      ),
      (
          'gemma4_npu_qualcomm_ple',
          'gemma4',
          'text_generation',
          'npu:qualcomm',
          {},
          (
              'model.tflite',
              'embedder.tflite',
              'per_layer_embedder.tflite',
              'auxiliary.tflite',
          ),
          None,
      ),
      (
          'lfm2_cpu_gpu_bundled',
          'lfm2',
          'auto',
          'cpu_gpu',
          {
              'externalize_embedder': False,
              'apply_gpu_composites': False,
          },
          ('model.tflite',),
          4,
      ),
      (
          'lfm2_cpu_gpu_composites',
          'lfm2',
          'auto',
          'cpu_gpu',
          {'externalize_embedder': True},
          ('model.tflite', 'embedder.tflite'),
          4,
      ),
  )
  def test_tiny_v2_export(
      self,
      family: str,
      task: str,
      target: str,
      export_kwargs: dict[str, Any],
      expected_tflites: tuple[str, ...],
      max_tensor_rank: int | None,
  ):
    torch.manual_seed(0)
    ext, model = _MODEL_BUILDERS[family]()
    signatures_by_file = ext_testing.assert_tiny_v2_export(
        ext=ext,
        model=model,
        task=task,
        target=target,
        prefill_lengths=(8,),
        cache_length=16,
        export_kwargs=export_kwargs,
        max_tensor_rank=max_tensor_rank,
    )
    for expected_file in expected_tflites:
      self.assertIn(expected_file, signatures_by_file)
    self.assertIn('prefill_8', signatures_by_file['model.tflite'])
    self.assertIn('decode', signatures_by_file['model.tflite'])


if __name__ == '__main__':
  googletest.main()
