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
"""Unit tests for the Qualcomm HTP prefill-mask size check."""

import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from litert_torch.generative.export_hf.experimental.npu_export import config_manager
import transformers


def _hf_config(
    num_heads: int, num_kv_heads: int | None = None, nested: bool = False
) -> types.SimpleNamespace:
  """Returns a stand-in for the HF config the check reads."""
  cfg = types.SimpleNamespace(num_attention_heads=num_heads)
  if num_kv_heads is not None:
    cfg.num_key_value_heads = num_kv_heads
  return types.SimpleNamespace(text_config=cfg) if nested else cfg


def _pipeline_config(
    cache_length: int,
    prefill_lengths: list[int] | None = None,
    backend: str = "qualcomm",
    soc_model: str = "SM8850",
) -> config_manager.NpuPipelineConfig:
  return config_manager.NpuPipelineConfig(
      model_id="org/model",
      backend=backend,
      soc_model=soc_model,
      prefill_lengths=[128] if prefill_lengths is None else prefill_lengths,
      cache_length=cache_length,
  )


class PrefillMaskCheckTest(parameterized.TestCase):

  def _check(
      self,
      cfg: config_manager.NpuPipelineConfig,
      hf_config: types.SimpleNamespace | None = None,
      error: Exception | None = None,
  ) -> mock.Mock:
    """Runs the check with the HF config read stubbed and returns the stub."""
    with mock.patch.object(
        transformers.AutoConfig,
        "from_pretrained",
        return_value=hf_config,
        side_effect=error,
    ) as from_pretrained:
      config_manager.warn_if_prefill_mask_exceeds_htp_limit(cfg)
    return from_pretrained

  def test_warns_above_the_limit_with_the_largest_safe_cache_length(self):
    # 4 query heads on 1 KV head: 2 B x 4 x 128 x (1024 + 128) = 1.125 MiB.
    with self.assertLogs("absl", level="WARNING") as logs:
      self._check(_pipeline_config(1024), _hf_config(4, 1))
    self.assertLen(logs.records, 1)
    message = logs.records[0].getMessage()
    self.assertIn("1.125 MiB", message)
    self.assertIn("896", message)

  @parameterized.named_parameters(
      # 2 B x 4 x 128 x (896 + 128) = 1.000 MiB, on the limit.
      ("on_the_limit", 896, _hf_config(4, 1)),
      # 16 query heads on 8 KV heads: 2 B x 2 x 128 x 1152 = 0.5625 MiB.
      ("two_query_heads_per_kv_head", 1024, _hf_config(16, 8)),
      # The exporter builds no mask concat for one query head per KV head.
      ("one_query_head_per_kv_head", 4096, _hf_config(8, 8)),
      ("no_num_key_value_heads", 4096, _hf_config(12)),
  )
  def test_stays_silent(self, cache_length, hf_config):
    with self.assertNoLogs("absl", level="WARNING"):
      self._check(_pipeline_config(cache_length), hf_config)

  def test_reads_the_text_config_of_a_multimodal_model(self):
    # 8 query heads on 4 KV heads, one level down: 2 per KV head, so the
    # largest cache_length under the limit at prefill 128 is 1920.
    with self.assertLogs("absl", level="WARNING") as logs:
      self._check(_pipeline_config(4096), _hf_config(8, 4, nested=True))
    self.assertLen(logs.records, 1)
    self.assertIn("1920", logs.records[0].getMessage())

  def test_uses_the_largest_prefill_length(self):
    # A cache of 1024 is 0.258 MiB at prefill 32 and 1.125 MiB at prefill 128.
    cfg = _pipeline_config(1024, prefill_lengths=[32, 128])
    with self.assertLogs("absl", level="WARNING") as logs:
      self._check(cfg, _hf_config(4, 1))
    self.assertLen(logs.records, 1)
    self.assertIn("1.125 MiB", logs.records[0].getMessage())

  def test_warning_says_when_no_cache_length_fits(self):
    # 16 query heads on 1 KV head leave 256 positions for cache + prefill.
    with self.assertLogs("absl", level="WARNING") as logs:
      self._check(_pipeline_config(256), _hf_config(16, 1))
    self.assertLen(logs.records, 1)
    self.assertIn("no cache_length fits", logs.records[0].getMessage())

  def test_reads_the_hf_config_with_remote_code_off(self):
    from_pretrained = self._check(_pipeline_config(896), _hf_config(4, 1))
    from_pretrained.assert_called_once_with(
        "org/model", trust_remote_code=False
    )

  def test_stays_silent_when_the_hf_config_cannot_be_read(self):
    with self.assertNoLogs("absl", level="WARNING"):
      self._check(_pipeline_config(4096), error=OSError("offline"))

  def test_stays_silent_without_prefill_lengths(self):
    cfg = _pipeline_config(4096, prefill_lengths=[])
    with self.assertNoLogs("absl", level="WARNING"):
      from_pretrained = self._check(cfg, _hf_config(4, 1))
    from_pretrained.assert_not_called()

  def test_skips_other_backends_without_reading_the_hf_config(self):
    cfg = _pipeline_config(4096, backend="mediatek", soc_model="mt6993")
    with self.assertNoLogs("absl", level="WARNING"):
      from_pretrained = self._check(cfg, _hf_config(4, 1))
    from_pretrained.assert_not_called()

  def test_build_pipeline_config_runs_the_check(self):
    with mock.patch.object(
        transformers.AutoConfig,
        "from_pretrained",
        return_value=_hf_config(4, 1),
    ):
      with self.assertLogs("absl", level="WARNING") as logs:
        cfg = config_manager.build_pipeline_config(
            model_id="org/model",
            target_vendor="qualcomm",
            target_soc="SM8850",
            prefill_lengths=[128],
            cache_length=1024,
        )
    self.assertLen(logs.records, 1)
    self.assertEqual(cfg.cache_length, 1024)


if __name__ == "__main__":
  absltest.main()
