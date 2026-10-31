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

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from litert_torch.generative.export_hf.experimental.calib import quant_utils
from litert_torch.generative.export_hf.experimental.calib import quantize as quantize_lib
from litert_torch.generative.export_hf.experimental.calib import tokenizer as tokenizer_lib
import numpy as np
import transformers

from litert_quantizer import qtyping
from litert_quantizer import quantizer


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

  def test_scale_calibration_results_skips_mask_tensors(self):
    calibration_result = {
        "prefill_128_mask_global": {
            "min": np.array([-100.0], dtype=np.float32),
            "max": np.array([0.0], dtype=np.float32),
        },
        "mask_local_0_output": {
            "min": np.array([-50.0], dtype=np.float32),
            "max": np.array([0.0], dtype=np.float32),
        },
        "decoder_layer_0_activation": {
            "min": np.array([-10.0], dtype=np.float32),
            "max": np.array([20.0], dtype=np.float32),
        },
    }
    scaled = quantize_lib._scale_calibration_results(calibration_result, 1.5)
    np.testing.assert_allclose(
        scaled["prefill_128_mask_global"]["min"], [-100.0]
    )
    np.testing.assert_allclose(scaled["mask_local_0_output"]["min"], [-50.0])
    np.testing.assert_allclose(
        scaled["decoder_layer_0_activation"]["min"], [-15.0]
    )
    np.testing.assert_allclose(
        scaled["decoder_layer_0_activation"]["max"], [30.0]
    )

  def test_symmetrize_mask_and_logits_calibration(self):
    main_calib = {
        "prefill_128_mask_global": {
            "min": np.array([-100.0], dtype=np.float32),
            "max": np.array([0.0], dtype=np.float32),
        },
        "decode_logits_output": {
            "min": np.array([-12.0], dtype=np.float32),
            "max": np.array([25.0], dtype=np.float32),
        },
        "hidden_states": {
            "min": np.array([-3.0], dtype=np.float32),
            "max": np.array([7.0], dtype=np.float32),
        },
    }
    aux_calib = {
        "prefill_mask_128_mask_local_output": {
            "min": np.array([-80.0], dtype=np.float32),
            "max": np.array([0.0], dtype=np.float32),
        },
    }
    quantize_lib._symmetrize_mask_and_logits_calibration(main_calib, aux_calib)
    np.testing.assert_allclose(
        main_calib["prefill_128_mask_global"]["min"], [-100.0]
    )
    np.testing.assert_allclose(
        main_calib["prefill_128_mask_global"]["max"], [100.0]
    )
    np.testing.assert_allclose(
        main_calib["decode_logits_output"]["min"], [-25.0]
    )
    np.testing.assert_allclose(
        main_calib["decode_logits_output"]["max"], [25.0]
    )
    np.testing.assert_allclose(main_calib["hidden_states"]["min"], [-3.0])
    np.testing.assert_allclose(main_calib["hidden_states"]["max"], [7.0])
    np.testing.assert_allclose(
        aux_calib["prefill_mask_128_mask_local_output"]["min"], [-80.0]
    )
    np.testing.assert_allclose(
        aux_calib["prefill_mask_128_mask_local_output"]["max"], [80.0]
    )

  def test_add_main_model_quant_recipe_rules(self):
    mock_q = mock.MagicMock()
    quant_utils.add_main_model_quant_recipe(
        mock_q, allow_float_operations=True, a16w8=True
    )
    mock_q.add_static_config.assert_any_call(
        regex=".*(q_norm|k_norm).*",
        operation_name=qtyping.TFLOperationName.STABLEHLO_COMPOSITE,
        activation_num_bits=16,
        weight_num_bits=8,
    )
    mock_q.update_quantization_recipe.assert_any_call(
        regex="^(?!.*(q_norm|k_norm)).*",
        operation_name=qtyping.TFLOperationName.STABLEHLO_COMPOSITE,
        algorithm_key=quantizer.AlgorithmName.NO_QUANTIZE,
    )
    mock_q.update_quantization_recipe.assert_any_call(
        regex=".*(post_qkv|(DecoderLayer|decoder_layer)_[0-9]+;).*",
        operation_name=qtyping.TFLOperationName.ADD,
        algorithm_key=quantizer.AlgorithmName.NO_QUANTIZE,
    )

    mock_q_int = mock.MagicMock()
    quant_utils.add_main_model_quant_recipe(
        mock_q_int, allow_float_operations=False, a16w8=True
    )
    mock_q_int.add_static_config.assert_any_call(
        regex=".*(post_qkv|(DecoderLayer|decoder_layer)_[0-9]+;).*",
        operation_name=qtyping.TFLOperationName.ADD,
        activation_num_bits=16,
        weight_num_bits=8,
    )


class QuantizeOptionsTest(parameterized.TestCase):
  """Checks how quantize() and its CLI pass the recipe options on."""

  def setUp(self):
    super().setUp()
    if not flags.FLAGS.is_parsed():
      flags.FLAGS.mark_as_parsed()

  def _run_quantize(self, **kwargs):
    """Runs quantize() on a main and an aux model with a mocked quantizer."""
    tmp_dir = self.create_tempdir().full_path
    main_q = mock.MagicMock(name="main_quantizer")
    aux_q = mock.MagicMock(name="aux_quantizer")

    def _load_calibration_results(unused_path):
      return {"x": {"min": np.zeros(1), "max": np.ones(1)}}, {}

    with mock.patch.object(
        quantize_lib.algorithm_manager, "register_quantized_op"
    ), mock.patch.object(
        quantize_lib.calibration_utils,
        "load_calibration_results",
        side_effect=_load_calibration_results,
    ), mock.patch.object(
        quantize_lib.quant_utils, "align_kv_cache_params"
    ), mock.patch.object(
        quantize_lib.quantizer, "Quantizer", side_effect=[main_q, aux_q]
    ), mock.patch.object(
        quantize_lib.quant_utils,
        "add_main_model_quant_recipe",
        side_effect=lambda q, **unused_kwargs: q,
    ) as mock_main_recipe:
      quantize_lib.quantize(
          model_path=os.path.join(tmp_dir, "model.tflite"),
          calibration_path=os.path.join(tmp_dir, "model.tflite.json"),
          output_path=os.path.join(tmp_dir, "out", "model.tflite"),
          aux_model_path=os.path.join(tmp_dir, "aux.tflite"),
          aux_calibration_path=os.path.join(tmp_dir, "aux.tflite.json"),
          aux_output_path=os.path.join(tmp_dir, "out", "aux.tflite"),
          **kwargs,
      )
    return main_q, aux_q, mock_main_recipe

  @parameterized.named_parameters(
      ("default", {}, True, False, 16),
      (
          "a16w8_float_normalizer",
          {"a16w8": True, "use_float_input_output_normalizer": True},
          True,
          True,
          16,
      ),
      ("a8w8", {"a16w8": False}, False, False, 8),
  )
  def test_quantize_recipe_options(
      self,
      kwargs,
      expected_a16w8,
      expected_allow_float_operations,
      expected_aux_bits,
  ):
    main_q, aux_q, mock_main_recipe = self._run_quantize(**kwargs)
    mock_main_recipe.assert_called_once_with(
        main_q,
        allow_float_operations=expected_allow_float_operations,
        a16w8=expected_a16w8,
    )
    self.assertCountEqual(
        aux_q.add_static_config.call_args_list,
        [
            mock.call(
                regex=".*",
                operation_name=op,
                activation_num_bits=expected_aux_bits,
                weight_num_bits=8,
            )
            for op in (
                qtyping.TFLOperationName.OUTPUT,
                qtyping.TFLOperationName.DYNAMIC_UPDATE_SLICE,
                qtyping.TFLOperationName.INPUT,
            )
        ],
    )

  @parameterized.named_parameters(
      ("defaults", {}, True, True),
      ("a8w8", {"a16w8": False}, False, True),
      (
          "int_normalizer",
          {"use_float_input_output_normalizer": False},
          True,
          False,
      ),
  )
  def test_main_passes_flags_to_quantize(
      self, flag_values, expected_a16w8, expected_normalizer
  ):
    with flagsaver.flagsaver(
        model_path="model.tflite",
        calibration_path="model.tflite.json",
        **flag_values,
    ), mock.patch.object(quantize_lib, "quantize") as mock_quantize:
      quantize_lib.main(["quantize"])
    _, kwargs = mock_quantize.call_args
    self.assertEqual(kwargs["a16w8"], expected_a16w8)
    self.assertEqual(
        kwargs["use_float_input_output_normalizer"], expected_normalizer
    )


if __name__ == "__main__":
  absltest.main()
