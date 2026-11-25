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
import re
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.experimental.calib import quant_utils
from litert_torch.generative.export_hf.experimental.calib import quantize as quantize_lib
from litert_torch.generative.export_hf.experimental.calib import tokenizer as tokenizer_lib
import numpy as np
import transformers

from ai_edge_litert import schema_py_generated as schema_fb
from ai_edge_litert.tools import flatbuffer_utils
from litert_quantizer import qtyping
from litert_quantizer import quantizer
from litert_quantizer.utils import tfl_flatbuffer_utils


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


_LAYER = "Model/DecoderLayer_0"
_ATTN = f"{_LAYER}/Attention_self_attn"


class _ModelBuilder:
  """Builds a small TFLite model with flatbuffer object API types."""

  def __init__(self, v2_names: bool):
    self._v2_names = v2_names
    self._name_counts = {}
    self.model = schema_fb.ModelT()
    self.model.version = 3
    self.model.buffers = [schema_fb.BufferT()]
    self.model.operatorCodes = []
    self.model.subgraphs = []

  def name(self, v1_name: str, op_name: str) -> str:
    """Returns `v1_name`, or a Converter V2 style name such as `tfl.add1`."""
    if not self._v2_names:
      return v1_name
    count = self._name_counts.get(op_name, 0)
    self._name_counts[op_name] = count + 1
    return f"tfl.{op_name}{count or ''}"

  def subgraph(self) -> tuple[int, schema_fb.SubGraphT]:
    subgraph = schema_fb.SubGraphT()
    subgraph.tensors, subgraph.operators = [], []
    subgraph.inputs, subgraph.outputs = [], []
    self.model.subgraphs.append(subgraph)
    return len(self.model.subgraphs) - 1, subgraph

  def tensor(self, subgraph, name: str, const: bool = False) -> int:
    tensor = schema_fb.TensorT()
    tensor.name = name.encode()
    tensor.shape = [4, 4]
    tensor.type = schema_fb.TensorType.FLOAT32
    tensor.buffer = 0
    if const:
      buffer = schema_fb.BufferT()
      buffer.data = np.frombuffer(np.ones(16, np.float32).tobytes(), np.uint8)
      self.model.buffers.append(buffer)
      tensor.buffer = len(self.model.buffers) - 1
    subgraph.tensors.append(tensor)
    return len(subgraph.tensors) - 1

  def op(self, subgraph, builtin_code, inputs, outputs, options=None):
    codes = [c.builtinCode for c in self.model.operatorCodes]
    if builtin_code not in codes:
      op_code = schema_fb.OperatorCodeT()
      op_code.builtinCode = builtin_code
      op_code.deprecatedBuiltinCode = min(
          builtin_code,
          schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES,
      )
      op_code.version = 1
      self.model.operatorCodes.append(op_code)
      codes.append(builtin_code)
    op = schema_fb.OperatorT()
    op.opcodeIndex = codes.index(builtin_code)
    op.inputs, op.outputs = list(inputs), list(outputs)
    if options is not None:
      op.builtinOptions2Type = (
          schema_fb.BuiltinOptions2.StableHLOCompositeOptions
      )
      op.builtinOptions2 = options
    subgraph.operators.append(op)

  def rms_norm(self, subgraph, x: int, module: str, output_name=None) -> int:
    """Adds an `odml.rms_norm` composite and its decomposition subgraph."""
    weight = self.tensor(subgraph, f"{module}.weight", const=True)
    out = self.tensor(subgraph, output_name or module)
    index, decomposition = self.subgraph()
    impl = f"odml.rms_norm.impl_{index}"
    dx = self.tensor(decomposition, f"{impl}_arg0")
    dw = self.tensor(decomposition, f"{impl}_arg1")
    decomposition.inputs = [dx, dw]
    b = schema_fb.BuiltinOperator
    tensor_ids = iter(range(6))

    def step(builtin_code, op_name, inputs):
      v1_name = f"{module};{next(tensor_ids)}"
      t = self.tensor(decomposition, self.name(v1_name, op_name))
      self.op(decomposition, builtin_code, inputs, [t])
      return t

    squared = step(b.MUL, "mul", [dx, dx])
    mean = step(b.MEAN, "mean", [squared])
    mean_eps = step(b.ADD, "add", [mean])
    rsqrt = step(b.RSQRT, "rsqrt", [mean_eps])
    normed = step(b.MUL, "mul", [dx, rsqrt])
    decomposition.outputs = [step(b.MUL, "mul", [normed, dw])]
    options = schema_fb.StableHLOCompositeOptionsT()
    options.name = b"odml.rms_norm"
    options.decompositionSubgraphIndex = index
    options.version = 1
    self.op(subgraph, b.STABLEHLO_COMPOSITE, [x, weight], [out], options)
    return out


def _build_decoder_layer_model(v2_names: bool) -> bytes:
  """Returns one decoder layer, named like a Converter V1 or V2 export.

  x -> input_layernorm -> FC -> q_norm -> MUL(q, ADD(FC, mask))
    -> post_attention_layernorm -> residual ADD(x, .) -> norm -> output

  The final norm writes the subgraph output, so its output is named after the
  signature output with both converters. The graph is only used to resolve the
  quantization recipe; it is not run.

  Args:
    v2_names: Whether to name the tensors like Converter V2 does.
  """
  m = _ModelBuilder(v2_names)
  _, main = m.subgraph()
  b = schema_fb.BuiltinOperator
  x = m.tensor(main, "decode_embeddings")
  mask = m.tensor(main, "decode_mask")
  main.inputs = [x, mask]
  h = m.rms_norm(main, x, f"{_LAYER}/RMSNorm_input_layernorm")
  fc_weight = m.tensor(main, f"{_ATTN}/q_proj.weight", const=True)
  q = m.tensor(main, m.name(f"{_ATTN};0", "fully_connected"))
  m.op(main, b.FULLY_CONNECTED, [h, fc_weight], [q])
  q_normed = m.rms_norm(main, q, f"{_ATTN}/RMSNorm_q_norm")
  # An ADD in the attention whose input also feeds q_norm: not a residual ADD.
  masked = m.tensor(main, m.name(f"{_ATTN};1", "add"))
  m.op(main, b.ADD, [q, mask], [masked])
  attn = m.tensor(main, m.name(f"{_ATTN};2", "mul"))
  m.op(main, b.MUL, [q_normed, masked], [attn])
  post = m.rms_norm(main, attn, f"{_LAYER}/RMSNorm_post_attention_layernorm")
  residual = m.tensor(main, m.name(_LAYER, "add"))
  m.op(main, b.ADD, [x, post], [residual])
  out = m.rms_norm(main, residual, "Model/RMSNorm_norm", "decode_output")
  main.outputs = [out]
  return bytes(flatbuffer_utils.convert_object_to_bytearray(m.model))


def _resolve_recipe(q: quantizer.Quantizer) -> list[tuple[int, str, str, Any]]:
  """Returns (subgraph, op, algorithm, op config) for each op in graph order."""
  model = q._float_model
  resolved = []
  for subgraph_index, subgraph in enumerate(model.subgraphs):
    ops = list(subgraph.operators)
    ops += tfl_flatbuffer_utils.get_subgraph_input_output_operators(subgraph)
    for op in ops:
      if isinstance(op, qtyping.IOOperator):
        op_name = op.op_key
      else:
        op_name = tfl_flatbuffer_utils.TFL_OP_CODE_TO_NAME[
            model.operatorCodes[op.opcodeIndex].builtinCode
        ]
      scope = tfl_flatbuffer_utils.get_op_scope(op, subgraph.tensors)
      algorithm, config = q._recipe_manager.get_quantization_configs(
          op_name, scope
      )
      resolved.append((subgraph_index, op_name, algorithm, config))
  return resolved


def _num_bits(config) -> int:
  return config.activation_tensor_config.num_bits


class QuantRecipeOpNamesTest(parameterized.TestCase):
  """Tests the main model recipe on Converter V1 and V2 tensor names."""

  def _quantizer(self, v2_names, allow_float_operations, a16w8):
    q = quantizer.Quantizer(_build_decoder_layer_model(v2_names))
    return quant_utils.add_main_model_quant_recipe(
        q, allow_float_operations=allow_float_operations, a16w8=a16w8
    )

  @parameterized.product(
      allow_float_operations=[False, True], a16w8=[False, True]
  )
  def test_v2_names_resolve_like_v1_names(self, allow_float_operations, a16w8):
    v1 = self._quantizer(False, allow_float_operations, a16w8)
    v2 = self._quantizer(True, allow_float_operations, a16w8)
    self.assertEqual(_resolve_recipe(v2), _resolve_recipe(v1))

  @parameterized.parameters(False, True)
  def test_rms_norm_decompositions_kept_in_float(self, v2_names):
    q = self._quantizer(v2_names, allow_float_operations=False, a16w8=False)
    decomposition_ops = [r for r in _resolve_recipe(q) if r[0] > 0]
    # 4 decompositions with 6 ops and the INPUT and OUTPUT ops each.
    self.assertLen(decomposition_ops, 32)
    for _, op_name, algorithm, config in decomposition_ops:
      if op_name in (
          qtyping.TFLOperationName.INPUT,
          qtyping.TFLOperationName.OUTPUT,
      ):
        self.assertEqual(_num_bits(config), 16)
      else:
        self.assertEqual(algorithm, quantizer.AlgorithmName.NO_QUANTIZE)

  @parameterized.parameters(False, True)
  def test_residual_add(self, v2_names):
    # Main subgraph ops: norm, FC, q_norm, attention ADD, MUL, norm,
    # residual ADD, norm.
    attention_add, residual_add = 3, 6
    q = self._quantizer(v2_names, allow_float_operations=False, a16w8=False)
    resolved = _resolve_recipe(q)
    self.assertEqual(_num_bits(resolved[residual_add][3]), 16)
    self.assertEqual(_num_bits(resolved[attention_add][3]), 8)

    q = self._quantizer(v2_names, allow_float_operations=True, a16w8=True)
    resolved = _resolve_recipe(q)
    self.assertEqual(
        resolved[residual_add][2], quantizer.AlgorithmName.NO_QUANTIZE
    )
    self.assertEqual(_num_bits(resolved[attention_add][3]), 16)

  @parameterized.parameters(False, True)
  def test_v1_names_keep_name_regex_rules(self, allow_float_operations):
    q = self._quantizer(False, allow_float_operations, a16w8=True)
    regexes = {rule["regex"] for rule in q.get_quantization_recipe()}
    if allow_float_operations:
      expected = {
          ".*",
          ".*rotated.*",
          ".*(q_norm|k_norm).*",
          "^(?!.*(q_norm|k_norm)).*",
          ".*(post_qkv|(DecoderLayer|decoder_layer)_[0-9]+;).*",
          ".*apply_skip_scale.*",
      }
    else:
      expected = {
          ".*",
          ".*rotated.*",
          ".*(post_qkv|(DecoderLayer|decoder_layer)_[0-9]+;).*",
          ".*apply_skip_scale.*",
          "^(?!.*(rotated|rotate|reshaped)).*[rR][mM][sS]_?[nN][oO][rR][mM].*",
      }
    self.assertEqual(regexes, expected)

  def test_warns_when_rule_matches_no_op(self):
    m = _ModelBuilder(v2_names=True)
    _, main = m.subgraph()
    x = m.tensor(main, "x")
    y = m.tensor(main, "tfl.add")
    m.op(main, schema_fb.BuiltinOperator.ADD, [x, x], [y])
    main.inputs, main.outputs = [x], [y]
    q = quantizer.Quantizer(
        bytes(flatbuffer_utils.convert_object_to_bytearray(m.model))
    )
    with self.assertLogs(logger="absl", level="WARNING") as logs:
      quant_utils.add_main_model_quant_recipe(q, allow_float_operations=True)
    output = "\n".join(logs.output)
    self.assertIn("(post_qkv|(DecoderLayer|decoder_layer)_[0-9]+;)", output)
    self.assertIn(".*(q_norm|k_norm).*", output)

  def test_extend_regex_with_scopes(self):
    self.assertEqual(
        quant_utils._extend_regex_with_scopes(".*norm.*", ["a/norm;"]),
        ".*norm.*",
    )
    regex = quant_utils._extend_regex_with_scopes(
        ".*norm.*", ["a/norm;", "tfl.add1;"]
    )
    self.assertRegex("a/norm;", regex)
    self.assertRegex("tfl.add1;", regex)
    self.assertIsNone(re.search(regex, "tfl.add12;"))
    self.assertIsNone(re.search(regex, "xtfl.add1;"))


if __name__ == "__main__":
  absltest.main()
