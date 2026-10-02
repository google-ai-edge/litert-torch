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
"""Tests for litert_lm_builder."""

import os
import types
from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import litert_lm_builder
import litert_lm_builder as litertlm_builder
from litert_lm_builder.runtime.proto import image_gen_metadata_pb2
from litert_lm_builder.runtime.proto import image_gen_model_type_pb2

_EMPTY_CHAT_TEMPLATES = ((None, None), (None, None), (None, None))


class _FakeEncoding:

  def __init__(self, input_ids):
    self.input_ids = input_ids


class _FakeTokenizer:
  """Minimal tokenizer stub for start_token gating tests."""

  def __init__(
      self,
      bos_token,
      bos_token_id,
      prepends_bos,
      eos_token="</s>",
      template_carries_bos=None,
  ):
    self.bos_token = bos_token
    self.bos_token_id = bos_token_id
    self.eos_token = eos_token
    # None: no chat template. True/False: template present, starting with the
    # BOS or not (like the OLMo-2 `{{ bos_token }}...` vs granite templates).
    self.chat_template = None
    if template_carries_bos is not None:
      prefix = "{{ bos_token }}" if template_carries_bos else ""
      self.chat_template = prefix + "<|user|>{{ content }}"
    self._prepends_bos = prepends_bos

  def __call__(self, text, add_special_tokens=True):
    ids = []
    if add_special_tokens and self._prepends_bos:
      ids.append(self.bos_token_id)
    if isinstance(self.bos_token, str) and text.startswith(self.bos_token):
      ids.append(self.bos_token_id)
    ids.extend([7, 8])
    return _FakeEncoding(ids)

  def apply_chat_template(
      self,
      messages,
      chat_template=None,
      tokenize=False,
      add_generation_prompt=False,
  ):
    template = (
        chat_template if chat_template is not None else self.chat_template
    )
    rendered = template.replace("{{ bos_token }}", str(self.bos_token))
    return rendered.replace("{{ content }}", messages[0]["content"])


class _UnprobeableTokenizer:
  """Tokenizer that cannot be called, e.g. a non-HF tokenizer."""

  def __init__(self, bos_token, eos_token="</s>"):
    self.bos_token = bos_token
    self.eos_token = eos_token
    self.chat_template = None


class _FakeModel:
  generation_config = None


def _build_llm_metadata(tokenizer, chat_templates=_EMPTY_CHAT_TEMPLATES):
  return litert_lm_builder.build_llm_metadata(
      source_model_artifacts=export_lib.SourceModelArtifacts(
          model=_FakeModel(),
          model_config=None,
          text_model_config=None,
          tokenizer=tokenizer,
      ),
      export_config=exportable_module.ExportableModuleConfig(
          model="test-model"
      ),
      chat_templates=chat_templates,
      exported_model_artifacts=export_lib.ExportedModelArtifacts(),
      litert_lm_model_type_override="generic",
  )


class TokenizerPrependsBosTest(parameterized.TestCase):

  def test_true_when_bos_is_prepended(self):
    tokenizer = _FakeTokenizer(
        bos_token="<s>", bos_token_id=1, prepends_bos=True
    )
    self.assertTrue(litert_lm_builder._tokenizer_prepends_bos(tokenizer))

  def test_false_when_declared_bos_is_not_prepended(self):
    # add_bos_token: false checkpoints (e.g. granite-4.1) declare a BOS that
    # tokenization never prepends.
    tokenizer = _FakeTokenizer(
        bos_token="<|end_of_text|>",
        bos_token_id=100257,
        prepends_bos=False,
        eos_token="<|end_of_text|>",
    )
    self.assertFalse(litert_lm_builder._tokenizer_prepends_bos(tokenizer))

  def test_true_when_chat_template_carries_the_bos(self):
    # OLMo-2-style: tokenization never prepends the BOS, but the chat
    # template begins with `{{ bos_token }}`. parse_chat_template strips that
    # prefix on the assumption start_token restores it, so it must be kept.
    tokenizer = _FakeTokenizer(
        bos_token="<|endoftext|>",
        bos_token_id=100257,
        prepends_bos=False,
        eos_token="<|endoftext|>",
        template_carries_bos=True,
    )
    self.assertTrue(litert_lm_builder._tokenizer_prepends_bos(tokenizer))

  def test_false_when_chat_template_does_not_carry_the_bos(self):
    tokenizer = _FakeTokenizer(
        bos_token="<|end_of_text|>",
        bos_token_id=100257,
        prepends_bos=False,
        eos_token="<|end_of_text|>",
        template_carries_bos=False,
    )
    self.assertFalse(litert_lm_builder._tokenizer_prepends_bos(tokenizer))

  def test_true_when_shipped_template_carries_the_bos(self):
    # jinja_chat_template_override case: the shipped template is not the
    # tokenizer's own (which may not exist), so the gate must probe the
    # template that is actually packaged.
    tokenizer = _FakeTokenizer(
        bos_token="<|endoftext|>",
        bos_token_id=100257,
        prepends_bos=False,
        eos_token="<|endoftext|>",
    )
    self.assertTrue(
        litert_lm_builder._tokenizer_prepends_bos(
            tokenizer, "{{ bos_token }}<|user|>{{ content }}"
        )
    )
    self.assertFalse(
        litert_lm_builder._tokenizer_prepends_bos(
            tokenizer, "<|user|>{{ content }}"
        )
    )

  def test_false_when_bos_token_id_is_unresolvable(self):
    tokenizer = _FakeTokenizer(
        bos_token="<s>", bos_token_id=None, prepends_bos=False
    )
    self.assertFalse(litert_lm_builder._tokenizer_prepends_bos(tokenizer))

  def test_int_bos_token_used_as_id_when_id_attribute_is_missing(self):
    class _IntBosTokenizer:
      bos_token = 5
      eos_token = "</s>"

      def __call__(self, text):
        return _FakeEncoding([5, 7, 8])

    self.assertTrue(
        litert_lm_builder._tokenizer_prepends_bos(_IntBosTokenizer())
    )

  def test_true_when_tokenizer_cannot_be_probed(self):
    # Preserves the previous behavior for tokenizers we cannot call.
    self.assertTrue(
        litert_lm_builder._tokenizer_prepends_bos(
            _UnprobeableTokenizer(bos_token="<s>")
        )
    )


class BuildLlmMetadataStartTokenTest(parameterized.TestCase):

  def test_no_start_token_when_bos_is_not_prepended(self):
    # Regression test for https://github.com/google-ai-edge/litert-torch/issues/1194:
    # a bos == eos checkpoint with add_bos_token: false must not get a
    # start_token, otherwise the runtime prepends the EOS to the first turn.
    tokenizer = _FakeTokenizer(
        bos_token="<|end_of_text|>",
        bos_token_id=100257,
        prepends_bos=False,
        eos_token="<|end_of_text|>",
    )
    llm_metadata = _build_llm_metadata(tokenizer)
    self.assertFalse(llm_metadata.HasField("start_token"))

  def test_start_token_written_when_bos_is_prepended(self):
    tokenizer = _FakeTokenizer(
        bos_token="<s>", bos_token_id=1, prepends_bos=True
    )
    llm_metadata = _build_llm_metadata(tokenizer)
    self.assertEqual(llm_metadata.start_token.token_str, "<s>")

  def test_start_token_written_when_chat_template_carries_the_bos(self):
    tokenizer = _FakeTokenizer(
        bos_token="<|endoftext|>",
        bos_token_id=100257,
        prepends_bos=False,
        eos_token="<|endoftext|>",
        template_carries_bos=True,
    )
    llm_metadata = _build_llm_metadata(tokenizer)
    self.assertEqual(llm_metadata.start_token.token_str, "<|endoftext|>")

  def test_start_token_follows_the_shipped_jinja_template(self):
    # A tokenizer with no template of its own, packaged with an override
    # template: the BOS must follow the template actually shipped.
    tokenizer = _FakeTokenizer(
        bos_token="<|endoftext|>",
        bos_token_id=100257,
        prepends_bos=False,
        eos_token="<|endoftext|>",
    )
    with_bos = _build_llm_metadata(
        tokenizer, chat_templates="{{ bos_token }}<|user|>{{ content }}"
    )
    self.assertEqual(with_bos.start_token.token_str, "<|endoftext|>")
    without_bos = _build_llm_metadata(
        tokenizer, chat_templates="<|user|>{{ content }}"
    )
    self.assertFalse(without_bos.HasField("start_token"))

  def test_start_token_written_when_tokenizer_cannot_be_probed(self):
    llm_metadata = _build_llm_metadata(_UnprobeableTokenizer(bos_token="<s>"))
    self.assertEqual(llm_metadata.start_token.token_str, "<s>")

  def test_no_start_token_when_bos_token_is_missing(self):
    tokenizer = _FakeTokenizer(
        bos_token=None, bos_token_id=None, prepends_bos=False
    )
    llm_metadata = _build_llm_metadata(tokenizer)
    self.assertFalse(llm_metadata.HasField("start_token"))

  def test_build_llm_metadata_qwen3_asr_model_type(self):
    class _FakeConfig:
      model_type = "qwen3_asr"
      text_config = types.SimpleNamespace(model_type="qwen3")
    class _FakeAsrModel:
      config = _FakeConfig()
      generation_config = None
    tokenizer = _FakeTokenizer(bos_token=None, bos_token_id=None, prepends_bos=False)
    source_artifacts = export_lib.SourceModelArtifacts(
        model=_FakeAsrModel(),
        model_config=_FakeConfig(),  # pyrefly: ignore[bad-argument-type]
        text_model_config=_FakeConfig.text_config,  # pyrefly: ignore[bad-argument-type]
        tokenizer=tokenizer,
    )
    export_config = exportable_module.ExportableModuleConfig(
        model="dummy",
        task=export_lib.exportable_module_config.ExportTask.AUTOMATIC_SPEECH_RECOGNITION,
    )
    exported_artifacts = export_lib.ExportedModelArtifacts()
    metadata = litert_lm_builder.build_llm_metadata(
        source_artifacts, export_config, "", exported_artifacts
    )
    self.assertTrue(metadata.llm_model_type.HasField("qwen3"))

  def test_build_llm_metadata_gemma3_text_model_type(self):
    class _FakeConfig:
      model_type = "gemma3_text"

    class _FakeModelWithConfig:
      config = _FakeConfig()
      generation_config = None

    tokenizer = _FakeTokenizer(
        bos_token=None, bos_token_id=None, prepends_bos=False
    )
    source_artifacts = export_lib.SourceModelArtifacts(
        model=_FakeModelWithConfig(),
        model_config=_FakeConfig(),  # pyrefly: ignore[bad-argument-type]
        text_model_config=_FakeConfig(),  # pyrefly: ignore[bad-argument-type]
        tokenizer=tokenizer,
    )
    export_config = exportable_module.ExportableModuleConfig(
        model="dummy",
    )
    exported_artifacts = export_lib.ExportedModelArtifacts()
    metadata = litert_lm_builder.build_llm_metadata(
        source_artifacts, export_config, "", exported_artifacts
    )
    self.assertTrue(metadata.llm_model_type.HasField("gemma3"))

  def test_package_image_gen_model_bonsai_flux2(self):
    temp_dir = self.create_tempdir().full_path
    text_enc_path = os.path.join(temp_dir, "text_encoder.tflite")
    dit_path = os.path.join(temp_dir, "dit.tflite")
    vae_path = os.path.join(temp_dir, "vae_decoder.tflite")
    tok_path = os.path.join(temp_dir, "tokenizer.json")
    for p, content in [
        (text_enc_path, b"TFL3_TEXT_ENC"),
        (dit_path, b"TFL3_DIT"),
        (vae_path, b"TFL3_VAE"),
        (tok_path, b'{"version": "1.0"}'),
    ]:
      with open(p, "wb") as f:
        f.write(content)

    class _FakeImageGenModel:

      def get_image_gen_metadata(self, export_config):
        return image_gen_metadata_pb2.ImageGenMetadata(
            image_gen_model_type=image_gen_model_type_pb2.ImageGenModelType(
                bonsai_flux2=image_gen_model_type_pb2.BonsaiFlux2(
                    flux2_params=image_gen_model_type_pb2.Flux2Params(
                        seq_len=256,
                        img_size=export_config.t2i_output_image_size,
                        packed_ch=128,
                        prompt_dim=7680,
                        default_steps=4,
                    ),
                )
            )
        )

    source_artifacts = export_lib.SourceModelArtifacts(
        model=_FakeImageGenModel(),  # pyrefly: ignore[bad-argument-type]
        model_config=None,
        text_model_config=None,
        tokenizer=None,
    )
    export_config = exportable_module.ExportableModuleConfig(
        model="bonsai-flux2",
        output_dir=temp_dir,
        work_dir=temp_dir,
        task=export_lib.exportable_module_config.ExportTask.TEXT_TO_IMAGE,
        t2i_output_image_size=256,
    )
    exported_artifacts = export_lib.ExportedModelArtifacts(
        additional_model_paths={
            "text_encoder": text_enc_path,
            "dit": dit_path,
            "vae_decoder": vae_path,
        },
        tokenizer_model_path=tok_path,
    )
    result = litert_lm_builder.package_model(
        source_artifacts, export_config, exported_artifacts
    )
    self.assertIsNotNone(result.litert_lm_model_path)
    self.assertTrue(os.path.exists(result.litert_lm_model_path))


if __name__ == "__main__":
  absltest.main()
