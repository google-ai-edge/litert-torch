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
"""Tests for calibration prompt formatting and chat template integration."""

import json
import os
from unittest import mock

from absl.testing import absltest
from litert_torch.generative.export_hf.experimental.calib import quant_utils
from litert_torch.generative.export_hf.experimental.calib import tokenizer as tokenizer_lib
import transformers


class QuantUtilsChatTemplateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Mock a Qwen / ChatML style tokenizer
    self.chatml_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

    # Mock a Gemma style tokenizer
    self.gemma_template = (
        "{% for message in messages %}"
        "{{'<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<start_of_turn>model\n' }}"
        "{% endif %}"
    )

  def test_format_chat_prompt_qwen_chatml(self):
    mock_tx = mock.MagicMock()
    mock_tx.chat_template = self.chatml_template
    mock_tx.apply_chat_template.side_effect = lambda messages, tokenize=False, add_generation_prompt=True: (
        f"<|im_start|>{messages[0]['role']}\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
    )

    tok = tokenizer_lib.Tokenizer.__new__(tokenizer_lib.Tokenizer)
    tok.spm = None
    tok.tx_tokenizer = mock_tx
    tok._image_preprocessor = None

    prompt = "Hello, introduce yourself."
    formatted = quant_utils.get_example_prompt(
        prompt, enable_formatting=True, tokenizer=tok
    )
    self.assertEqual(
        formatted,
        "<|im_start|>user\nHello, introduce yourself.<|im_end|>\n<|im_start|>assistant\n",
    )

  def test_format_chat_prompt_dict_text_and_prompt(self):
    mock_tx = mock.MagicMock()
    mock_tx.chat_template = self.chatml_template
    mock_tx.apply_chat_template.side_effect = lambda messages, tokenize=False, add_generation_prompt=True: (
        f"<|im_start|>{messages[0]['role']}\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
    )

    tok = tokenizer_lib.Tokenizer.__new__(tokenizer_lib.Tokenizer)
    tok.spm = None
    tok.tx_tokenizer = mock_tx
    tok._image_preprocessor = None

    # Test {"text": ...}
    res1 = quant_utils.get_example_prompt(
        {"text": "Sample text prompt"}, enable_formatting=True, tokenizer=tok
    )
    self.assertIn("<|im_start|>user\nSample text prompt<|im_end|>", res1)

    # Test {"prompt": ...}
    res2 = quant_utils.get_example_prompt(
        {"prompt": "Sample prompt"}, enable_formatting=True, tokenizer=tok
    )
    self.assertIn("<|im_start|>user\nSample prompt<|im_end|>", res2)

    # Test {"inputs": ...}
    res3 = quant_utils.get_example_prompt(
        {"inputs": "Sample inputs"}, enable_formatting=True, tokenizer=tok
    )
    self.assertIn("<|im_start|>user\nSample inputs<|im_end|>", res3)

  def test_format_chat_prompt_dict_messages(self):
    mock_tx = mock.MagicMock()
    mock_tx.chat_template = self.chatml_template
    mock_tx.apply_chat_template.side_effect = lambda messages, tokenize=False, add_generation_prompt=True: (
        f"<|im_start|>{messages[0]['role']}\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
    )

    tok = tokenizer_lib.Tokenizer.__new__(tokenizer_lib.Tokenizer)
    tok.spm = None
    tok.tx_tokenizer = mock_tx
    tok._image_preprocessor = None

    messages_example = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a test function."},
        ]
    }
    res = quant_utils.get_example_prompt(
        messages_example, enable_formatting=True, tokenizer=tok
    )
    self.assertIn("<|im_start|>user\nWrite a test function.<|im_end|>", res)

  def test_format_chat_prompt_enable_formatting_false(self):
    mock_tx = mock.MagicMock()
    mock_tx.chat_template = self.chatml_template

    tok = tokenizer_lib.Tokenizer.__new__(tokenizer_lib.Tokenizer)
    tok.spm = None
    tok.tx_tokenizer = mock_tx
    tok._image_preprocessor = None

    prompt = "Raw prompt without formatting."
    res = quant_utils.get_example_prompt(
        prompt, enable_formatting=False, tokenizer=tok
    )
    self.assertEqual(res, "Raw prompt without formatting.")

    res_dict = quant_utils.get_example_prompt(
        {"text": "Raw dict text."}, enable_formatting=False, tokenizer=tok
    )
    self.assertEqual(res_dict, "Raw dict text.")

  def test_format_chat_prompt_fallback_when_no_tokenizer(self):
    prompt = "Hello fallback"
    res = quant_utils.get_example_prompt(
        prompt, enable_formatting=True, tokenizer=None
    )
    self.assertEqual(
        res,
        "<start_of_turn>user\nHello fallback<end_of_turn>\n<start_of_turn>model\n",
    )

  def test_format_chat_prompt_guard_on_exception(self):
    mock_tx = mock.MagicMock()
    mock_tx.chat_template = "some_template"
    mock_tx.apply_chat_template.side_effect = ValueError(
        "Template error simulation"
    )

    tok = tokenizer_lib.Tokenizer.__new__(tokenizer_lib.Tokenizer)
    tok.spm = None
    tok.tx_tokenizer = mock_tx
    tok._image_preprocessor = None

    prompt = "Hello error"
    res = quant_utils.get_example_prompt(
        prompt, enable_formatting=True, tokenizer=tok
    )
    # Should gracefully fall back to Gemma turn markers rather than crashing
    self.assertEqual(
        res,
        "<start_of_turn>user\nHello error<end_of_turn>\n<start_of_turn>model\n",
    )

  def test_format_chat_prompt_dropping_template_falls_back(self):
    # SmolVLM2-style template: expects list-shaped content, so a plain string
    # is dropped and only the template's own boilerplate is rendered.
    mock_tx = mock.MagicMock()
    mock_tx.chat_template = "some_template"
    mock_tx.apply_chat_template.return_value = (
        "<|im_start|>User: <end_of_utterance>\nAssistant:"
    )

    tok = tokenizer_lib.Tokenizer.__new__(tokenizer_lib.Tokenizer)
    tok.spm = None
    tok.tx_tokenizer = mock_tx
    tok._image_preprocessor = None

    res = quant_utils.get_example_prompt(
        "Hello dropped", enable_formatting=True, tokenizer=tok
    )
    self.assertEqual(
        res,
        "<start_of_turn>user\nHello dropped<end_of_turn>\n<start_of_turn>model\n",
    )

  def test_format_chat_prompt_boilerplate_substring_prompt_falls_back(self):
    # Regression: a prompt that happens to appear in the dropping template's
    # own boilerplate ('User:') must not defeat the drop detection.
    mock_tx = mock.MagicMock()
    mock_tx.chat_template = "some_template"
    mock_tx.apply_chat_template.return_value = (
        "<|im_start|>User: <end_of_utterance>\nAssistant:"
    )

    tok = tokenizer_lib.Tokenizer.__new__(tokenizer_lib.Tokenizer)
    tok.spm = None
    tok.tx_tokenizer = mock_tx
    tok._image_preprocessor = None

    res = quant_utils.get_example_prompt(
        {"text": "User:"}, enable_formatting=True, tokenizer=tok
    )
    self.assertEqual(
        res,
        "<start_of_turn>user\nUser:<end_of_turn>\n<start_of_turn>model\n",
    )

  def test_tokenizer_loads_standalone_chat_template_json(self):
    temp_dir = self.create_tempdir().full_path
    template_data = {
        "chat_template": "{{ messages[0]['content'] }} [CUSTOM_TEMPLATE]"
    }
    with open(os.path.join(temp_dir, "chat_template.json"), "w") as f:
      json.dump(template_data, f)

    with mock.patch.object(
        transformers.AutoTokenizer, "from_pretrained"
    ) as mock_from_pretrained:
      mock_auto_tok = mock.MagicMock()
      mock_auto_tok.chat_template = None
      mock_from_pretrained.return_value = mock_auto_tok

      tok = tokenizer_lib.Tokenizer(transformers_model_path=temp_dir)
      self.assertEqual(
          tok.tx_tokenizer.chat_template,
          "{{ messages[0]['content'] }} [CUSTOM_TEMPLATE]",
      )


if __name__ == "__main__":
  absltest.main()
