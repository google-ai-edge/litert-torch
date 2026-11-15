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
"""Tests for LiteRT gemma_quant layer wrappers."""

import copy

from absl.testing import parameterized
import litert_torch
from litert_torch.generative.export_hf.core import quantization_layer
from litert_torch.generative.export_hf.core import weights_loader as weights_loader_lib
from litert_torch.generative.export_hf.core.quantization_layer import gemma_quant as lrt_gemma_quant
import numpy as np
import torch
from torch import nn
from transformers.integrations import gemma_quant as hf_gemma_quant

from absl.testing import absltest as googletest


class GemmaQuantTest(parameterized.TestCase):

  @parameterized.parameters(2, 4, 8)
  def test_litert_quantized_linear_srq(self, num_bits):
    torch.manual_seed(42)
    orig_linear = hf_gemma_quant.QuantizedLinear(
        in_features=32, out_features=16, num_bits=num_bits
    ).eval()
    if num_bits in (2, 4):
      orig_linear.weight.data = torch.randint(
          0, 256, orig_linear.weight.shape, dtype=torch.uint8
      )
    else:
      orig_linear.weight.data = torch.randint(
          -128, 127, orig_linear.weight.shape, dtype=torch.int8
      )
    orig_linear.weight_scale.data = torch.rand(16, 1, dtype=torch.float32) * 0.1
    orig_linear.input_activation_scale.data = torch.tensor(
        0.1, dtype=torch.float32
    )
    orig_linear.output_activation_scale.data = torch.tensor(
        0.2, dtype=torch.float32
    )

    lrt_linear = lrt_gemma_quant.LiteRTQuantizedLinear(orig_linear)
    x = torch.randn((2, 32), dtype=torch.float32)

    with torch.no_grad():
      expected = orig_linear(x)
      actual = lrt_linear(x)

    self.assertTrue(
        torch.allclose(expected, actual, rtol=1e-4, atol=1e-4),
        f"Mismatch in SRQ QuantizedLinear (num_bits={num_bits}).\nExpected:"
        f" {expected}\nActual: {actual}",
    )

  @parameterized.parameters(2, 4)
  def test_litert_quantized_linear_blockwise(self, num_bits):
    torch.manual_seed(42)
    in_features = 64
    out_features = 16
    num_blocks = 2
    block_size = in_features // num_blocks
    orig_linear = hf_gemma_quant.QuantizedLinear(
        in_features=in_features, out_features=out_features, num_bits=num_bits
    ).eval()
    orig_linear.weight.data = torch.randint(
        0, 256, orig_linear.weight.shape, dtype=torch.uint8
    )
    orig_linear.weight_scale = torch.nn.Parameter(
        torch.rand(out_features, num_blocks, dtype=torch.float32) * 0.1,
        requires_grad=False,
    )

    lrt_linear = lrt_gemma_quant.LiteRTQuantizedLinear(orig_linear)
    self.assertTrue(lrt_linear.is_blockwise)
    self.assertEqual(lrt_linear.block_size, block_size)

    x = torch.randn((2, in_features), dtype=torch.float32)
    with torch.no_grad():
      if num_bits == 2:
        int_w = hf_gemma_quant._unpack_int2(orig_linear.weight, in_features)  # pylint: disable=protected-access
      else:
        int_w = hf_gemma_quant._unpack_int4(orig_linear.weight, in_features)  # pylint: disable=protected-access
      expanded_scale = orig_linear.weight_scale.repeat_interleave(
          block_size, dim=-1
      )
      expected = torch.nn.functional.linear(x, int_w.float() * expanded_scale)
      actual = lrt_linear(x)

    self.assertTrue(
        torch.allclose(expected, actual, rtol=1e-4, atol=1e-4),
        f"Mismatch in blockwise QuantizedLinear (num_bits={num_bits}).",
    )

  @parameterized.parameters((2, 1), (4, 1), (2, 2), (4, 2))
  def test_litert_quantized_embedding(self, num_bits, num_blocks):
    torch.manual_seed(42)
    num_embeddings = 32
    embedding_dim = 64
    orig_emb = hf_gemma_quant.QuantizedEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        output_dtype=torch.float32,
        embed_scale=2.0,
        num_bits=num_bits,
    ).eval()
    orig_emb.embedding_quantized.data = torch.randint(
        0, 256, orig_emb.embedding_quantized.shape, dtype=torch.uint8
    )
    orig_emb.embedding_scale = torch.nn.Parameter(
        torch.rand(num_embeddings, num_blocks, dtype=torch.float32) * 0.1,
        requires_grad=False,
    )

    lrt_emb = lrt_gemma_quant.LiteRTQuantizedEmbedding(orig_emb)
    self.assertEqual(lrt_emb.is_blockwise, num_blocks > 1)

    input_ids = torch.tensor([[1, 5, 10], [20, 25, 30]], dtype=torch.long)
    with torch.no_grad():
      expected = orig_emb(input_ids)
      actual = lrt_emb(input_ids)

    self.assertTrue(
        torch.allclose(expected, actual, rtol=1e-4, atol=1e-4),
        f"Mismatch in LiteRTQuantizedEmbedding (bits={num_bits},"
        f" blocks={num_blocks}).",
    )

  @parameterized.parameters(
      # (num_bits, num_blocks, in_scale, out_scale, use_bias)
      (2, 1, 0.0, 0.0, False),
      (4, 1, 0.1, 0.2, True),
      (4, 1, 0.1, 0.0, True),
      (4, 2, 0.0, 0.0, True),
      (8, 1, 0.0, 0.0, True),
  )
  def test_litert_quantized_linear_convert_v2(
      self, num_bits, num_blocks, in_scale, out_scale, use_bias
  ):
    if not litert_torch.converter_v2.is_supported():
      self.skipTest(
          "Converter V2 is not supported by the installed TensorFlow."
      )
    torch.manual_seed(42)
    in_features = 64
    out_features = 16
    orig_linear = hf_gemma_quant.QuantizedLinear(
        in_features=in_features,
        out_features=out_features,
        bias=use_bias,
        num_bits=num_bits,
    ).eval()
    if num_bits in (2, 4):
      orig_linear.weight.data = torch.randint(
          0, 256, orig_linear.weight.shape, dtype=torch.uint8
      )
    else:
      orig_linear.weight.data = torch.randint(
          -127, 127, orig_linear.weight.shape, dtype=torch.int8
      )
    orig_linear.weight_scale = torch.nn.Parameter(
        (torch.rand(out_features, num_blocks, dtype=torch.float32) * 0.1 + 0.01)
        / (1 << (num_bits - 1)),
        requires_grad=False,
    )
    orig_linear.input_activation_scale.data = torch.tensor(
        in_scale, dtype=torch.float32
    )
    orig_linear.output_activation_scale.data = torch.tensor(
        out_scale, dtype=torch.float32
    )
    if use_bias:
      orig_linear.bias.data = torch.randn(out_features, dtype=torch.float32)

    lrt_linear = lrt_gemma_quant.LiteRTQuantizedLinear(orig_linear).eval()
    x = torch.randn((2, in_features), dtype=torch.float32)
    with torch.no_grad():
      expected = lrt_linear(x).detach().numpy()

    edge_model = litert_torch.convert(lrt_linear, (x,), use_v2=True)
    interp = edge_model._get_interpreter()  # pylint: disable=protected-access
    op_names = {op["op_name"] for op in interp._get_ops_details()}  # pylint: disable=protected-access
    self.assertNotIn("STABLEHLO_CUSTOM_CALL", op_names)
    self.assertIn("FULLY_CONNECTED", op_names)
    tensor_dtypes = {d["dtype"] for d in interp.get_tensor_details()}
    self.assertIn(np.int8, tensor_dtypes)
    actual = edge_model(x)
    np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=5e-2)

  @parameterized.parameters((2, 1), (4, 1), (4, 2))
  def test_litert_quantized_embedding_convert_v2(self, num_bits, num_blocks):
    if not litert_torch.converter_v2.is_supported():
      self.skipTest(
          "Converter V2 is not supported by the installed TensorFlow."
      )
    torch.manual_seed(42)
    num_embeddings = 32
    embedding_dim = 64
    orig_emb = hf_gemma_quant.QuantizedEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        output_dtype=torch.float32,
        embed_scale=2.0,
        num_bits=num_bits,
    ).eval()
    orig_emb.embedding_quantized.data = torch.randint(
        0, 256, orig_emb.embedding_quantized.shape, dtype=torch.uint8
    )
    orig_emb.embedding_scale = torch.nn.Parameter(
        torch.rand(num_embeddings, num_blocks, dtype=torch.float32) * 0.1
        + 0.01,
        requires_grad=False,
    )

    lrt_emb = lrt_gemma_quant.LiteRTQuantizedEmbedding(orig_emb).eval()
    input_ids = torch.tensor([[1, 5, 10], [20, 25, 30]], dtype=torch.int32)
    with torch.no_grad():
      expected = lrt_emb(input_ids).detach().numpy()

    edge_model = litert_torch.convert(lrt_emb, (input_ids,), use_v2=True)
    interp = edge_model._get_interpreter()  # pylint: disable=protected-access
    op_names = {op["op_name"] for op in interp._get_ops_details()}  # pylint: disable=protected-access
    self.assertNotIn("STABLEHLO_CUSTOM_CALL", op_names)
    self.assertIn("EMBEDDING_LOOKUP", op_names)
    self.assertNotIn("DEQUANTIZE", op_names)
    actual = edge_model(input_ids)
    np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=5e-2)

  def test_load_export_weight_and_validation(self):
    orig_linear = hf_gemma_quant.QuantizedLinear(
        in_features=32, out_features=8, num_bits=4
    ).eval()
    orig_linear.weight.data = torch.randint(
        0, 256, orig_linear.weight.shape, dtype=torch.uint8
    )
    orig_linear.weight_scale = nn.Parameter(
        torch.ones(8, 2, dtype=torch.float32), requires_grad=False
    )
    lrt_linear = lrt_gemma_quant.LiteRTQuantizedLinear(orig_linear)
    loader_w = {"proj.weight": orig_linear.weight}.__getitem__
    self.assertIsNone(lrt_linear.load_export_weight(loader_w, "proj", "bias"))
    loaded_w = lrt_linear.load_export_weight(loader_w, "proj", "weight")
    torch.testing.assert_close(loaded_w, lrt_linear.weight)

    orig_emb = hf_gemma_quant.QuantizedEmbedding(
        num_embeddings=16,
        embedding_dim=32,
        output_dtype=torch.float32,
        num_bits=4,
    ).eval()
    orig_emb.embedding_quantized.data = torch.randint(
        0, 256, orig_emb.embedding_quantized.shape, dtype=torch.uint8
    )
    # 1D scale shape `(num_embeddings,)` is per-channel, not blockwise.
    orig_emb.embedding_scale = nn.Parameter(
        torch.ones(16, dtype=torch.float32), requires_grad=False
    )
    lrt_emb = lrt_gemma_quant.LiteRTQuantizedEmbedding(orig_emb)
    self.assertFalse(lrt_emb.is_blockwise)
    loader_emb = {
        "emb.embedding_quantized": orig_emb.embedding_quantized
    }.__getitem__
    self.assertIsNone(lrt_emb.load_export_weight(loader_emb, "emb", "other"))
    loaded_emb = lrt_emb.load_export_weight(loader_emb, "emb", "weight")
    torch.testing.assert_close(loaded_emb, lrt_emb.weight)

    # Meta scales and non-divisible block counts raise ValueError.
    meta_scale = torch.empty((8, 1), dtype=torch.float32, device="meta")
    with self.assertRaises(ValueError):
      lrt_gemma_quant.create_per_channel_dequant_fn(meta_scale, 4)
    with self.assertRaises(ValueError):
      lrt_gemma_quant.create_per_tensor_fake_quant_fn(meta_scale)

    orig_linear.weight_scale = nn.Parameter(
        torch.ones(8, 3, dtype=torch.float32), requires_grad=False
    )
    with self.assertRaises(ValueError):
      lrt_gemma_quant.LiteRTQuantizedLinear(orig_linear)

    orig_emb.embedding_scale = nn.Parameter(
        torch.ones(16, 3, dtype=torch.float32), requires_grad=False
    )
    with self.assertRaises(ValueError):
      lrt_gemma_quant.LiteRTQuantizedEmbedding(orig_emb)

  def test_replace_gemma_quant_layers_and_skip_names(self):
    class TinyBlock(nn.Module):

      def __init__(self):
        super().__init__()
        self.embed_tokens = hf_gemma_quant.QuantizedEmbedding(
            16, 32, output_dtype=torch.float32, num_bits=4
        )
        self.embed_tokens_per_layer = hf_gemma_quant.QuantizedEmbedding(
            16, 32, output_dtype=torch.float32, num_bits=4
        )
        self.proj = hf_gemma_quant.QuantizedLinear(32, 16, num_bits=4)

    block = TinyBlock().eval()
    replaced = lrt_gemma_quant.replace_gemma_quant_layers(
        block, skip_names=("embed_tokens_per_layer",)
    )
    self.assertLen(replaced, 2)
    self.assertIsInstance(
        block.embed_tokens, lrt_gemma_quant.LiteRTQuantizedEmbedding
    )
    self.assertIsInstance(block.proj, lrt_gemma_quant.LiteRTQuantizedLinear)
    self.assertIsInstance(
        block.embed_tokens_per_layer, hf_gemma_quant.QuantizedEmbedding
    )

  @parameterized.parameters((2, 1), (4, 1), (4, 2))
  def test_embed_quantized_tokens_on_cpu_real_and_meta(
      self, num_bits, num_blocks
  ):
    torch.manual_seed(42)
    num_embeddings = 16
    embedding_dim = 32
    orig_emb = hf_gemma_quant.QuantizedEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        output_dtype=torch.float32,
        embed_scale=2.0,
        num_bits=num_bits,
    ).eval()
    orig_emb.embedding_quantized.data = torch.randint(
        0, 256, orig_emb.embedding_quantized.shape, dtype=torch.uint8
    )
    orig_emb.embedding_scale = nn.Parameter(
        torch.rand(num_embeddings, num_blocks, dtype=torch.float32) * 0.1
        + 0.01,
        requires_grad=False,
    )
    ids = torch.tensor([[1, 5, 12]], dtype=torch.int32)
    with torch.no_grad():
      expected = orig_emb(ids).float()

    actual_real = lrt_gemma_quant.embed_quantized_tokens_on_cpu(
        orig_emb,
        ids,
        weights_loader=None,
        fqn="embed_tokens.embedding_quantized",
    )
    torch.testing.assert_close(actual_real, expected, rtol=1e-4, atol=1e-4)

    raw_weight = orig_emb.embedding_quantized.detach().clone()
    scale_weight = orig_emb.embedding_scale.detach().clone()

    class _ModelWithQuantEmb(nn.Module):

      def __init__(self):
        super().__init__()
        self.embed_tokens = hf_gemma_quant.QuantizedEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            output_dtype=torch.float32,
            embed_scale=2.0,
            num_bits=num_bits,
        ).eval()

      def get_input_embeddings(self):
        return self.embed_tokens

    with weights_loader_lib.init_params_on_meta():
      meta_model = _ModelWithQuantEmb()
    loader = {
        "embed_tokens.embedding_quantized": raw_weight,
        "embed_tokens.embedding_scale": scale_weight,
    }.__getitem__
    # Works when `meta_emb.embedding_scale` is still on `meta`.
    actual_unpatched_meta = lrt_gemma_quant.embed_quantized_tokens_on_cpu(
        meta_model.embed_tokens,
        ids,
        weights_loader=loader,
        fqn="embed_tokens.embedding_quantized",
    )
    torch.testing.assert_close(
        actual_unpatched_meta, expected, rtol=1e-4, atol=1e-4
    )

    # `load_quant_scales_` binds `embed_tokens_on_cpu` onto unpatched
    # `QuantizedEmbedding`, so `weights_loader.embed_tokens_on_cpu` works both
    # before and after `replace_gemma_quant_layers`.
    quantization_layer.load_quant_scales_(meta_model, loader)
    actual_via_wl_unpatched = weights_loader_lib.embed_tokens_on_cpu(
        meta_model, [1, 5, 12], weights_loader=loader
    )
    torch.testing.assert_close(
        actual_via_wl_unpatched, expected, rtol=1e-4, atol=1e-4
    )

    lrt_gemma_quant.replace_gemma_quant_layers(meta_model)
    self.assertIsInstance(
        meta_model.embed_tokens, lrt_gemma_quant.LiteRTQuantizedEmbedding
    )
    actual_via_wl_patched = weights_loader_lib.embed_tokens_on_cpu(
        meta_model, [1, 5, 12], weights_loader=loader
    )
    torch.testing.assert_close(
        actual_via_wl_patched, expected, rtol=1e-4, atol=1e-4
    )

  def test_extract_gemma_kv_cache_scales_ignores_vision_and_audio(self):
    class Model(nn.Module):

      def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module(), nn.Module()])
        self.layers[0].self_attn = nn.Module()
        self.layers[1].self_attn = nn.Module()
        self.vision_tower = nn.Module()
        self.vision_tower.layers = nn.ModuleList([nn.Module()])
        self.vision_tower.layers[0].self_attn = nn.Module()
        self.audio_tower = nn.Module()
        self.audio_tower.layers = nn.ModuleList([nn.Module()])
        self.audio_tower.layers[0].self_attn = nn.Module()

    tensors = {
        "layers.0.self_attn.k_cache_scale": torch.tensor(0.125),
        "layers.0.self_attn.v_cache_scale": torch.tensor(0.25),
        "vision_tower.layers.0.self_attn.k_cache_scale": torch.tensor(0.5),
        "vision_tower.layers.0.self_attn.v_cache_scale": torch.tensor(0.5),
        "audio_tower.layers.0.self_attn.k_cache_scale": torch.tensor(0.5),
        "audio_tower.layers.0.self_attn.v_cache_scale": torch.tensor(0.5),
    }
    scales = lrt_gemma_quant.extract_gemma_kv_cache_scales(
        Model(), tensors.__getitem__
    )
    self.assertEqual(scales, {0: (0.125, 0.25)})

  def test_load_quant_scales_and_suffixes(self):
    self.assertEqual(
        quantization_layer.get_quant_param_suffixes("gemma"),
        lrt_gemma_quant.QUANT_PARAM_SUFFIXES,
    )
    self.assertEqual(quantization_layer.get_quant_param_suffixes(None), ())

    with weights_loader_lib.init_params_on_meta():
      proj = nn.Linear(4, 4)
      proj.weight_scale = nn.Parameter(
          torch.ones(4, 1, dtype=torch.float32), requires_grad=False
      )
      proj.input_activation_scale = nn.Parameter(
          torch.tensor(0.0, dtype=torch.float32), requires_grad=False
      )
      proj.per_dim_scale = nn.Parameter(
          torch.zeros(4, dtype=torch.float32), requires_grad=False
      )
    self.assertTrue(proj.weight_scale.is_meta)

    # `quant_method=None` is a no-op.
    quantization_layer.load_quant_scales_(
        proj, {}.__getitem__, quant_method=None
    )
    self.assertTrue(proj.weight_scale.is_meta)

    # Missing required `weight_scale` raises KeyError.
    with self.assertRaises(KeyError):
      quantization_layer.load_quant_scales_(
          proj, {}.__getitem__, quant_method="gemma"
      )

    scales = {"weight_scale": torch.full((4, 1), 0.25, dtype=torch.bfloat16)}
    quantization_layer.load_quant_scales_(
        proj, scales.__getitem__, quant_method="gemma"
    )
    self.assertFalse(proj.weight_scale.is_meta)
    self.assertEqual(proj.weight_scale.dtype, torch.float32)
    torch.testing.assert_close(
        proj.weight_scale, torch.full((4, 1), 0.25, dtype=torch.float32)
    )
    self.assertFalse(proj.input_activation_scale.is_meta)
    self.assertEqual(proj.input_activation_scale.item(), 0.0)
    self.assertTrue(proj.per_dim_scale.is_meta)

    # Calling `load_quant_scales_` again when `input_activation_scale` is
    # already on CPU (`is_meta == False`) and missing from `scales` keeps its
    # existing value without raising KeyError.
    quantization_layer.load_quant_scales_(
        proj, scales.__getitem__, quant_method="gemma"
    )
    self.assertEqual(proj.input_activation_scale.item(), 0.0)

  def test_gemma_quant_linear_meta_convert_v2(self):
    if not litert_torch.converter_v2.is_supported():
      self.skipTest(
          "Converter V2 is not supported by the installed TensorFlow."
      )
    torch.manual_seed(42)
    in_features = 64
    out_features = 16
    orig_linear = hf_gemma_quant.QuantizedLinear(
        in_features=in_features, out_features=out_features, num_bits=4
    ).eval()
    orig_linear.weight.data = torch.randint(
        0, 256, orig_linear.weight.shape, dtype=torch.uint8
    )
    orig_linear.weight_scale = nn.Parameter(
        torch.rand(out_features, 2, dtype=torch.float32) * 0.1 + 0.01,
        requires_grad=False,
    )
    orig_linear.input_activation_scale.data = torch.tensor(
        0.1, dtype=torch.float32
    )
    orig_linear.output_activation_scale.data = torch.tensor(
        0.2, dtype=torch.float32
    )

    ref_linear = lrt_gemma_quant.LiteRTQuantizedLinear(
        copy.deepcopy(orig_linear)
    ).eval()
    x = torch.randn((2, in_features), dtype=torch.float32)
    with torch.no_grad():
      expected = ref_linear(x).detach().numpy()

    ckpt = orig_linear.state_dict()
    with weights_loader_lib.init_params_on_meta():
      meta_orig = hf_gemma_quant.QuantizedLinear(
          in_features=in_features, out_features=out_features, num_bits=4
      ).eval()
      meta_orig.weight_scale = nn.Parameter(
          torch.empty(out_features, 2, dtype=torch.float32),
          requires_grad=False,
      )
    quantization_layer.load_quant_scales_(meta_orig, ckpt.__getitem__)
    meta_linear = lrt_gemma_quant.LiteRTQuantizedLinear(meta_orig).eval()
    self.assertEqual(meta_linear.weight.device.type, "meta")

    with weights_loader_lib.prepare_for_export(
        ckpt.__getitem__, meta_linear
    ) as loader:
      edge_model = litert_torch.convert(
          meta_linear,
          (x.to("meta"),),
          use_v2=True,
          weights_loader=loader,
      )
    actual = edge_model(x)
    np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
  googletest.main()
