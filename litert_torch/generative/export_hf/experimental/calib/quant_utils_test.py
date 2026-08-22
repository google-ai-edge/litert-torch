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
"""Tests for calibration prompt formatting in quant_utils."""

from absl.testing import absltest
from litert_torch.generative.export_hf.experimental.calib import quant_utils


class _FakeTxTokenizer:
  """Stands in for a transformers tokenizer with a Qwen-style template."""

  chat_template = 'non-empty'

  def apply_chat_template(self, messages, tokenize=False,
                          add_generation_prompt=False):
    del tokenize
    body = ''.join(
        f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in messages
    )
    if add_generation_prompt:
      body += '<|im_start|>assistant\n'
    return body


class _FakeTokenizer:

  def __init__(self, tx_tokenizer=None):
    self.tx_tokenizer = tx_tokenizer


def _qwen_tokenizer():
  return _FakeTokenizer(_FakeTxTokenizer())


EXPECTED_QWEN = (
    '<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n'
)


class GetExamplePromptTest(absltest.TestCase):

  def test_bare_string_uses_model_chat_template(self):
    self.assertEqual(
        quant_utils.get_example_prompt('hello', True, _qwen_tokenizer()),
        EXPECTED_QWEN,
    )

  def test_text_dict_uses_model_chat_template(self):
    self.assertEqual(
        quant_utils.get_example_prompt(
            {'text': 'hello'}, True, _qwen_tokenizer()
        ),
        EXPECTED_QWEN,
    )

  def test_prompt_dict_uses_model_chat_template(self):
    self.assertEqual(
        quant_utils.get_example_prompt(
            {'prompt': 'hello'}, True, _qwen_tokenizer()
        ),
        EXPECTED_QWEN,
    )

  def test_inputs_dict_uses_model_chat_template(self):
    self.assertEqual(
        quant_utils.get_example_prompt(
            {'inputs': 'hello'}, True, _qwen_tokenizer()
        ),
        EXPECTED_QWEN,
    )

  def test_messages_dict_matches_plain_text_branch(self):
    """The shipped {'text'} format must calibrate on the same string as the
    {'messages'} format that already used the chat template."""
    from_text = quant_utils.get_example_prompt(
        {'text': 'hello'}, True, _qwen_tokenizer()
    )
    from_messages = quant_utils.get_example_prompt(
        {'messages': [{'role': 'user', 'content': 'hello'}]},
        True,
        _qwen_tokenizer(),
    )
    self.assertEqual(from_text, from_messages)

  def test_no_gemma_markers_leak_into_non_gemma_prompt(self):
    prompt = quant_utils.get_example_prompt('hello', True, _qwen_tokenizer())
    self.assertNotIn('<start_of_turn>', prompt)
    self.assertNotIn('<end_of_turn>', prompt)

  def test_falls_back_to_constants_without_tokenizer(self):
    self.assertEqual(
        quant_utils.get_example_prompt('hello', True, None),
        quant_utils.PROMPT_TEMPLATE_PREFIX
        + 'hello'
        + quant_utils.PROMPT_TEMPLATE_SUFFIX,
    )

  def test_falls_back_when_tokenizer_has_no_chat_template(self):
    tx = _FakeTxTokenizer()
    tx.chat_template = None
    self.assertEqual(
        quant_utils.get_example_prompt('hello', True, _FakeTokenizer(tx)),
        quant_utils.PROMPT_TEMPLATE_PREFIX
        + 'hello'
        + quant_utils.PROMPT_TEMPLATE_SUFFIX,
    )

  def test_formatting_disabled_returns_raw_prompt(self):
    self.assertEqual(
        quant_utils.get_example_prompt('hello', False, _qwen_tokenizer()),
        'hello',
    )

  def test_raising_template_falls_back_instead_of_crashing(self):
    """A template that rejects the message shape must not abort calibration."""

    class _RaisingTx(_FakeTxTokenizer):

      def apply_chat_template(self, messages, tokenize=False,
                              add_generation_prompt=False):
        raise ValueError('No user query found in messages.')

    self.assertEqual(
        quant_utils.get_example_prompt(
            'hello', True, _FakeTokenizer(_RaisingTx())
        ),
        quant_utils.PROMPT_TEMPLATE_PREFIX
        + 'hello'
        + quant_utils.PROMPT_TEMPLATE_SUFFIX,
    )

  def test_template_that_drops_the_prompt_falls_back(self):
    """Templates wanting list-shaped content drop a plain string; calibrating
    on the resulting empty turn is worse than using the default markers."""

    class _DroppingTx(_FakeTxTokenizer):

      def apply_chat_template(self, messages, tokenize=False,
                              add_generation_prompt=False):
        del messages, tokenize, add_generation_prompt
        return '<|im_start|>User: <end_of_utterance>\nAssistant:'

    prompt = quant_utils.get_example_prompt(
        'hello', True, _FakeTokenizer(_DroppingTx())
    )
    self.assertIn('hello', prompt)

  def test_boilerplate_substring_prompt_is_not_dropped(self):
    """A prompt that happens to appear in the template's own boilerplate must
    not defeat the check that the template preserved it."""

    class _DroppingTx(_FakeTxTokenizer):

      def apply_chat_template(self, messages, tokenize=False,
                              add_generation_prompt=False):
        del messages, tokenize, add_generation_prompt
        return 'User: <end_of_utterance>\nAssistant:'

    prompt = quant_utils.get_example_prompt(
        {'text': 'User:'}, True, _FakeTokenizer(_DroppingTx())
    )
    self.assertEqual(
        prompt,
        quant_utils.PROMPT_TEMPLATE_PREFIX
        + 'User:'
        + quant_utils.PROMPT_TEMPLATE_SUFFIX,
    )

  def test_non_string_prompt_raises(self):
    with self.assertRaises(TypeError):
      quant_utils.get_example_prompt({'inputs': ['a', 'b']}, True,
                                     _qwen_tokenizer())


if __name__ == '__main__':
  absltest.main()
