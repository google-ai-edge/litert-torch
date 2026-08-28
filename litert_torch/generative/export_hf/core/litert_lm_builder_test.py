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

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import litert_lm_builder as litertlm_builder

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
  return litertlm_builder.build_llm_metadata(
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
    self.assertTrue(litertlm_builder._tokenizer_prepends_bos(tokenizer))

  def test_false_when_declared_bos_is_not_prepended(self):
    # add_bos_token: false checkpoints (e.g. granite-4.1) declare a BOS that
    # tokenization never prepends.
    tokenizer = _FakeTokenizer(
        bos_token="<|end_of_text|>",
        bos_token_id=100257,
        prepends_bos=False,
        eos_token="<|end_of_text|>",
    )
    self.assertFalse(litertlm_builder._tokenizer_prepends_bos(tokenizer))

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
    self.assertTrue(litertlm_builder._tokenizer_prepends_bos(tokenizer))

  def test_false_when_chat_template_does_not_carry_the_bos(self):
    tokenizer = _FakeTokenizer(
        bos_token="<|end_of_text|>",
        bos_token_id=100257,
        prepends_bos=False,
        eos_token="<|end_of_text|>",
        template_carries_bos=False,
    )
    self.assertFalse(litertlm_builder._tokenizer_prepends_bos(tokenizer))

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
        litertlm_builder._tokenizer_prepends_bos(
            tokenizer, "{{ bos_token }}<|user|>{{ content }}"
        )
    )
    self.assertFalse(
        litertlm_builder._tokenizer_prepends_bos(
            tokenizer, "<|user|>{{ content }}"
        )
    )

  def test_false_when_bos_token_id_is_unresolvable(self):
    tokenizer = _FakeTokenizer(
        bos_token="<s>", bos_token_id=None, prepends_bos=False
    )
    self.assertFalse(litertlm_builder._tokenizer_prepends_bos(tokenizer))

  def test_int_bos_token_used_as_id_when_id_attribute_is_missing(self):
    class _IntBosTokenizer:
      bos_token = 5
      eos_token = "</s>"

      def __call__(self, text):
        return _FakeEncoding([5, 7, 8])

    self.assertTrue(
        litertlm_builder._tokenizer_prepends_bos(_IntBosTokenizer())
    )

  def test_true_when_tokenizer_cannot_be_probed(self):
    # Preserves the previous behavior for tokenizers we cannot call.
    self.assertTrue(
        litertlm_builder._tokenizer_prepends_bos(
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


if __name__ == "__main__":
  absltest.main()
