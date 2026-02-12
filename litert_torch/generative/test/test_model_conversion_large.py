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

"""Testing model conversion for a few gen-ai models."""

import litert_torch
from litert_torch.generative.examples.amd_llama_135m import amd_llama_135m
from litert_torch.generative.examples.deepseek import deepseek
from litert_torch.generative.examples.gemma import gemma1
from litert_torch.generative.examples.gemma import gemma2
from litert_torch.generative.examples.hammer import hammer
from litert_torch.generative.examples.llama import llama
from litert_torch.generative.examples.openelm import openelm
from litert_torch.generative.examples.paligemma import decoder
from litert_torch.generative.examples.paligemma import decoder2
from litert_torch.generative.examples.paligemma import paligemma
from litert_torch.generative.examples.phi import phi2
from litert_torch.generative.examples.phi import phi3
from litert_torch.generative.examples.phi import phi4
from litert_torch.generative.examples.qwen import qwen
from litert_torch.generative.examples.qwen_vl import qwen_vl
from litert_torch.generative.examples.smollm import smollm
from litert_torch.generative.examples.stable_diffusion import clip as sd_clip
from litert_torch.generative.examples.stable_diffusion import decoder as sd_decoder
from litert_torch.generative.examples.stable_diffusion import diffusion as sd_diffusion
from litert_torch.generative.layers import kv_cache
from litert_torch.generative.test import utils as test_utils
import numpy as np
import torch

from absl.testing import absltest as googletest
from ai_edge_litert import interpreter


class TestModelConversion(googletest.TestCase):
  """Unit tests that check for model conversion and correctness."""

  def setUp(self):
    super().setUp()
    self._interpreter_builder = (
        lambda model_content: lambda: interpreter.Interpreter(
            model_content=model_content,
            experimental_default_delegate_latest_features=True,
        )
    )
    self._kv_cache_max = 128
    # Default cache_size_limit, 8 is hit and aborts often when the tests are
    # running all together. Doubles it to avoid abortion.
    torch._dynamo.config.cache_size_limit = 16
    np.random.seed(1234)  # Make np.random deterministic.

  def _test_model(self, config, model, signature_name, atol, rtol):
    seq_len = 10
    tokens = torch.zeros((1, seq_len), dtype=torch.int, device="cpu")
    input_pos = torch.arange(0, seq_len, dtype=torch.int)
    kv = kv_cache.KVCache.from_model_config(self._kv_cache_max, config)

    edge_model = litert_torch.signature(
        signature_name,
        model,
        sample_kwargs={
            "tokens": tokens,
            "input_pos": input_pos,
            "kv_cache": kv,
        },
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.model_content())
    )

    tokens = torch.arange(1, seq_len + 1, dtype=torch.int).unsqueeze(0)
    self.assertTrue(
        test_utils.compare_tflite_torch(
            edge_model,
            model,
            tokens,
            input_pos,
            kv,
            signature_name=signature_name,
            atol=atol,
            rtol=rtol,
        )
    )

  def test_gemma1(self):
    config = gemma1.get_fake_model_config()
    pytorch_model = gemma1.Gemma1(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-3, rtol=1e-5)

  def test_gemma2(self):
    config = gemma2.get_fake_model_config()
    pytorch_model = gemma2.Gemma2(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-4, rtol=1e-5)

  def test_llama(self):
    config = llama.get_fake_model_config()
    pytorch_model = llama.Llama(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-3, rtol=1e-5)

  def test_phi2(self):
    config = phi2.get_fake_model_config()
    pytorch_model = phi2.Phi2(config, self._kv_cache_max).eval()
    # Phi-2 logits are very big, so we need a larger absolute tolerance.
    self._test_model(config, pytorch_model, "prefill", atol=1e-3, rtol=1e-5)

  def test_phi3(self):
    config = phi3.get_fake_model_config()
    pytorch_model = phi3.Phi3_5Mini(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-5, rtol=1e-5)

  def test_phi4(self):
    config = phi4.get_fake_model_config()
    pytorch_model = phi4.Phi4Mini(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-3, rtol=1e-5)

  def test_smollm(self):
    config = smollm.get_fake_model_config()
    pytorch_model = smollm.SmolLM(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-4, rtol=1e-5)

  def test_smollm2(self):
    config = smollm.get_fake_model_config_v2()
    pytorch_model = smollm.SmolLM2(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-4, rtol=1e-5)

  def test_openelm(self):
    config = openelm.get_fake_model_config()
    pytorch_model = openelm.OpenELM(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-4, rtol=1e-5)

  def test_qwen(self):
    config = qwen.get_fake_model_config()
    pytorch_model = qwen.Qwen(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-3, rtol=1e-5)

  def test_deepseek(self):
    config = deepseek.get_fake_model_config()
    pytorch_model = deepseek.DeepSeekDistillQwen(
        config, self._kv_cache_max
    ).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-5, rtol=1e-5)

  def test_hammer(self):
    config = hammer.get_fake_model_config()
    pytorch_model = hammer.Hammer(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-5, rtol=1e-5)

  def test_amd_llama_135m(self):
    config = amd_llama_135m.get_fake_model_config()
    pytorch_model = amd_llama_135m.AmdLlama(config, self._kv_cache_max).eval()
    self._test_model(config, pytorch_model, "prefill", atol=1e-5, rtol=1e-5)

  def _test_paligemma_model(self, decoder_class, decoder_config, atol, rtol):
    config = paligemma.get_fake_model_config(decoder_config)
    pytorch_model = paligemma.PaliGemma(
        config, decoder_class, mask_cache_size=self._kv_cache_max
    ).eval()

    image_config = config.image_encoder_config.image_embedding
    num_patches = (image_config.image_size // image_config.patch_size) ** 2

    # Make sure the token size is longer than the number of image patches.
    seq_len = num_patches + 10
    tokens = torch.zeros((1, seq_len), dtype=torch.int)
    input_pos = torch.arange(0, seq_len, dtype=torch.int)
    kv = kv_cache.KVCache.from_model_config(
        self._kv_cache_max, config.decoder_config
    )
    pixel_values = torch.zeros((1, 3, 8, 8), dtype=torch.float32)

    edge_model = litert_torch.signature(
        "prefill_pixel",
        pytorch_model,
        sample_kwargs={
            "tokens": tokens,
            "input_pos": input_pos,
            "kv_cache": kv,
            "pixel_values": pixel_values,
        },
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.model_content())
    )

    tokens = torch.arange(1, seq_len + 1, dtype=torch.int).unsqueeze(0)
    self.assertTrue(
        test_utils.compare_tflite_torch(
            edge_model,
            pytorch_model,
            tokens,
            input_pos,
            kv,
            pixel_values=pixel_values,
            signature_name="prefill_pixel",
            atol=atol,
            rtol=rtol,
        )
    )

  def test_paligemma1(self):
    self._test_paligemma_model(
        decoder.Decoder, decoder.get_fake_decoder_config, atol=1e-3, rtol=1e-5
    )

  def test_paligemma2(self):
    self._test_paligemma_model(
        decoder2.Decoder2,
        decoder2.get_fake_decoder2_config,
        atol=1e-3,
        rtol=1e-5,
    )

  def test_qwen_vl_model(self):
    config = qwen_vl.get_fake_model_config()
    pytorch_model = qwen_vl.QwenVL(config, self._kv_cache_max).eval()

    grid_thw = pytorch_model.image_encoder.get_grid_thw()
    pixel_values_size = pytorch_model.image_encoder.get_pixel_values_size(
        grid_thw
    )

    # Make sure the token size is longer than the number of pixel values.
    seq_len = pixel_values_size[0] + 10
    tokens = torch.zeros((1, seq_len), dtype=torch.int)
    input_pos = torch.arange(0, seq_len, dtype=torch.int)
    kv = kv_cache.KVCache.from_model_config(
        self._kv_cache_max, config.decoder_config
    )
    pixel_values = torch.zeros(pixel_values_size, dtype=torch.float32)

    edge_model = litert_torch.signature(
        "prefill_pixel",
        pytorch_model,
        sample_kwargs={
            "tokens": tokens,
            "input_pos": input_pos,
            "kv_cache": kv,
            "pixel_values": pixel_values,
        },
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.model_content())
    )

    tokens = torch.arange(1, seq_len + 1, dtype=torch.int).unsqueeze(0)
    self.assertTrue(
        test_utils.compare_tflite_torch(
            edge_model,
            pytorch_model,
            tokens,
            input_pos,
            kv,
            pixel_values=pixel_values,
            signature_name="prefill_pixel",
            atol=1e-3,
            rtol=1e-5,
        )
    )

  @googletest.skipIf(litert_torch.config.in_oss, reason="flaky")
  def test_stable_diffusion_clip(self):
    config = sd_clip.get_fake_model_config()
    prompt_tokens = torch.from_numpy(
        np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)
    )

    pytorch_model = sd_clip.CLIP(config).eval()
    torch_output = pytorch_model(prompt_tokens)

    edge_model = litert_torch.signature(
        "encode", pytorch_model, (prompt_tokens,)
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.model_content())
    )
    edge_output = edge_model(
        prompt_tokens.numpy(),
        signature_name="encode",
    )
    self.assertTrue(
        test_utils.compare_logits(
            edge_output,
            torch_output.detach().numpy(),
            atol=1e-4,
            rtol=1e-5,
        )
    )

  @googletest.skipIf(litert_torch.config.in_oss, reason="b/413106901")
  def test_stable_diffusion_diffusion(self):
    config = sd_diffusion.get_fake_model_config(2)
    # Reduce stddev(scale) of input values to avoid too big output logits which
    # fails comparisons with reasonable tolerances.
    latents = torch.from_numpy(
        np.random.normal(size=(2, 4, 8, 8), scale=0.1).astype(np.float32)
    )
    context = torch.from_numpy(
        np.random.normal(size=(2, 4, 4), scale=0.1).astype(np.float32)
    )
    time_embedding = torch.from_numpy(
        np.random.normal(size=(2, 2), scale=0.1).astype(np.float32)
    )

    pytorch_model = sd_diffusion.Diffusion(config).eval()
    torch_output = pytorch_model(latents, context, time_embedding)

    edge_model = litert_torch.signature(
        "diffusion", pytorch_model, (latents, context, time_embedding)
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.model_content())
    )
    edge_output = edge_model(
        latents.numpy(),
        context.numpy(),
        time_embedding.numpy(),
        signature_name="diffusion",
    )
    self.assertTrue(
        test_utils.compare_logits(
            edge_output,
            torch_output.detach().numpy(),
            atol=1e-4,
            rtol=1e-5,
        )
    )

  def test_stable_diffusion_decoder(self):
    config = sd_decoder.get_fake_model_config()
    # Reduce stddev(scale) of input values to avoid too big output logits which
    # fails comparisons with reasonable tolerances.
    latents = torch.from_numpy(
        np.random.normal(size=(1, 4, 64, 64), scale=0.1).astype(np.float32)
    )

    pytorch_model = sd_decoder.Decoder(config).eval()
    torch_output = pytorch_model(latents)

    edge_model = litert_torch.signature(
        "decode", pytorch_model, (latents,)
    ).convert()
    edge_model.set_interpreter_builder(
        self._interpreter_builder(edge_model.model_content())
    )
    edge_output = edge_model(
        latents.numpy(),
        signature_name="decode",
    )
    self.assertTrue(
        test_utils.compare_logits(
            edge_output,
            torch_output.detach().numpy(),
            atol=1e-3,
            rtol=1e-5,
        )
    )


if __name__ == "__main__":
  googletest.main()
