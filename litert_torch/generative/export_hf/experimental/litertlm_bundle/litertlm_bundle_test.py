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
"""Tests for litertlm_bundle."""

import os
from absl.testing import absltest
from litert_torch.generative.export_hf.experimental.calib import loader as loader_lib
from litert_torch.generative.export_hf.experimental.calib import sampling_executor as tfl_sampling_executor
from litert_torch.generative.export_hf.experimental.calib import tokenizer as tokenizer_lib
from litert_torch.generative.export_hf.experimental.litertlm_bundle import litertlm_bundle


class LitertLmBundleTest(absltest.TestCase):

  def test_pack_unpack_and_peek_litertlm(self):
    tmpdir = self.create_tempdir()

    model_file = tmpdir.create_file('main.tflite', content=b'TFL3_main_data')
    embedder_file = tmpdir.create_file(
        'embedder.tflite', content=b'TFL3_embedder_data'
    )
    aux_file = tmpdir.create_file('aux.tflite', content=b'TFL3_aux_data')
    spm_file = tmpdir.create_file('tokenizer.model', content=b'SPM_TOKENIZER')

    output_litertlm = os.path.join(tmpdir.full_path, 'bundle.litertlm')

    litertlm_bundle.LitertLmBundle.pack(
        output_litertlm=output_litertlm,
        prefilldecode_model_path=model_file.full_path,
        embedder_model_path=embedder_file.full_path,
        auxiliary_model_path=aux_file.full_path,
        spm_path=spm_file.full_path,
        metadata={'test_key': 'test_value'},
    )

    self.assertTrue(os.path.exists(output_litertlm))

    peek_output = litertlm_bundle.LitertLmBundle.peek(output_litertlm)
    self.assertIn('Key: test_key, Value (String): test_value', peek_output)
    self.assertIn('tf_lite_prefill_decode', peek_output)
    self.assertIn('tf_lite_embedder', peek_output)
    self.assertIn('tf_lite_aux', peek_output)

    unpack_dir = os.path.join(tmpdir.full_path, 'unpacked')
    unpacked = litertlm_bundle.LitertLmBundle.unpack(
        output_litertlm, unpack_dir
    )

    self.assertEqual(unpacked.metadata.get('test_key'), 'test_value')
    self.assertEqual(unpacked.get('test_key'), 'test_value')
    self.assertIn('tf_lite_prefill_decode', unpacked)
    self.assertIn('tf_lite_embedder', unpacked)
    self.assertIn('tf_lite_aux', unpacked)
    self.assertIn('SP_Tokenizer', unpacked)

    with open(unpacked['tf_lite_prefill_decode'], 'rb') as f:
      self.assertEqual(f.read(), b'TFL3_main_data')
    with open(unpacked['tf_lite_embedder'], 'rb') as f:
      self.assertEqual(f.read(), b'TFL3_embedder_data')
    with open(unpacked['tf_lite_aux'], 'rb') as f:
      self.assertEqual(f.read(), b'TFL3_aux_data')
    with open(unpacked['SP_Tokenizer'], 'rb') as f:
      self.assertEqual(f.read(), b'SPM_TOKENIZER')

  def test_gemma3_prompt_tokenization_if_available(self):
    bundle_path = '/tmp/gemma3_270m_exported/model.litertlm'
    if not os.path.exists(bundle_path):
      return

    tmpdir = self.create_tempdir()
    unpacked = litertlm_bundle.LitertLmBundle.unpack(
        bundle_path, tmpdir.full_path
    )
    tok = tokenizer_lib.Tokenizer(
        transformers_model_path=unpacked.get('transformers_model_path')
    )
    prompt = 'What is the highest building in the world?'
    formatted = tok.tx_tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok.tokenize_internal(formatted)
    print('\n================ GEMMA 3 TOKENIZATION DEBUG ================')
    print('Formatted prompt:', repr(formatted))
    print('Token IDs (len', len(ids), '):', ids.tolist())
    for idx, tid in enumerate(ids.tolist()):
      print(f'  [{idx}] ID {tid}: {repr(tok.detokenize_internal([tid]))}')
    print('Detokenized back:', repr(tok.detokenize_internal(ids.tolist())))

    config = loader_lib.ConversationExecutorConfig(
        model_path=bundle_path,
    )
    executor = tfl_sampling_executor.ConversationExecutor(config)
    req = tfl_sampling_executor.Request(
        contents=[tfl_sampling_executor.DataItem(text=prompt)]
    )
    out = executor.sample_text(req, max_sample_step=15)
    print('Raw executor out:', repr(out))
    print('============================================================\n')


if __name__ == '__main__':
  absltest.main()
