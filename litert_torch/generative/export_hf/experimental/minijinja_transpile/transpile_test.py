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
"""Tests for split cache layers."""

from litert_torch.generative.export_hf.experimental.minijinja_transpile import transpile as transpile_lib

from absl.testing import absltest as googletest


class TranspileTest(googletest.TestCase):

  def test_replace(self):
    self.assertEqual(
        transpile_lib.transpile_jinja2("{{ text.replace('a', 'b') }}"),
        "{{ (text | replace('a', 'b')) }}",
    )

  def test_strip_variants(self):
    self.assertEqual(
        transpile_lib.transpile_jinja2("{{ text.strip() }}"),
        "{{ (text | trim()) }}",
    )
    self.assertEqual(
        transpile_lib.transpile_jinja2("{{ text.lstrip('\\n') }}"),
        "{{ (text | trim('\\n')) }}",
    )
    self.assertEqual(
        transpile_lib.transpile_jinja2("{{ text.rstrip() }}"),
        "{{ (text | trim()) }}",
    )

  def test_split_and_list(self):
    self.assertEqual(
        transpile_lib.transpile_jinja2("{{ text.split('</think>')[-1] }}"),
        "{{ (text | split('</think>') | list)[-1] }}",
    )

  def test_len_function(self):
    self.assertEqual(
        transpile_lib.transpile_jinja2("{% if len(messages) > 0 %}"),
        "{% if (messages | length) > 0 %}",
    )

  def test_nested_parentheses_in_split(self):
    self.assertEqual(
        transpile_lib.transpile_jinja2("{{ text.split(get_separator())[0] }}"),
        "{{ (text | split(get_separator()) | list)[0] }}",
    )

  def test_ignores_strings(self):
    original = "{{ 'do not touch text.replace() me' }}"
    self.assertEqual(transpile_lib.transpile_jinja2(original), original)

  def test_complex_chaining(self):
    self.assertEqual(
        transpile_lib.transpile_jinja2(
            "{{ text.split('<think>')[-1].lstrip('\\n') }}"
        ),
        "{{ ((text | split('<think>') | list)[-1] | trim('\\n')) }}",
    )

  def test_strict_type_is_true(self):
    # Should drop 'is true' and leave truthiness check
    self.assertEqual(
        transpile_lib.transpile_jinja2("{% if enable_thinking is true %}"),
        "{% if enable_thinking  %}",
    )

  def test_strict_type_is_false(self):
    # Should wrap the variable in not()
    self.assertEqual(
        transpile_lib.transpile_jinja2("{% if enable_thinking is false %}"),
        "{% if not (enable_thinking ) %}",
    )

  def test_strict_type_complex_is_false(self):
    # Proves the backtracker safely finds the start of complex chains like messages[-1].content
    self.assertEqual(
        transpile_lib.transpile_jinja2(
            "{% if messages[-1].content is false %}"
        ),
        "{% if not (messages[-1].content ) %}",
    )

  def test_strict_type_with_and_operator(self):
    # Proves the backtracker knows to stop at logical operators ('and')
    self.assertEqual(
        transpile_lib.transpile_jinja2(
            "{% if defined and feature_flag is false %}"
        ),
        "{% if defined and not (feature_flag ) %}",
    )

  def test_keep_trailing_newline(self):
    self.assertEqual(
        transpile_lib.transpile_jinja2("{{ text }}\n"),
        "{{ text }}\n",
    )

  def test_user_loop_case(self):
    template = """        {%- for tool in tools %}
            {{- "\\n" }}
            {{- tool | tojson(ensure_ascii=False) }}
        {%- endfor %}"""
    self.assertEqual(
        transpile_lib.transpile_jinja2(template),
        template,
    )

  def test_qwen_reasoning_logic(self):
    self.assertEqual(
        transpile_lib.transpile_jinja2(
            "{% if loop.last or (not loop.last and reasoning_content) %}"
        ),
        "{% if reasoning_content %}",
    )


if __name__ == "__main__":
  googletest.main()
