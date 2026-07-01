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
"""Transpiles a jinja2 template to a minijinja compatible template."""

import jinja2
import jinja2.lexer


class PreserveWhitespaceLexer(jinja2.lexer.Lexer):
  """A custom Jinja2 lexer that disables whitespace stripping."""

  def __init__(self, environment: jinja2.Environment):
    super().__init__(environment)
    # Replace OptionalLStrip with normal tuples in rules to prevent whitespace
    # stripping
    for state, rule_list in self.rules.items():
      new_rules = []
      for rule in rule_list:
        if isinstance(rule.tokens, jinja2.lexer.OptionalLStrip):
          rule = jinja2.lexer._Rule(
              rule.pattern, tuple(rule.tokens), rule.command
          )
        new_rules.append(rule)
      self.rules[state] = new_rules


class PreserveWhitespaceEnvironment(jinja2.Environment):
  """A custom Jinja2 Environment that uses PreserveWhitespaceLexer."""

  @property
  def lexer(self) -> jinja2.lexer.Lexer:
    return PreserveWhitespaceLexer(self)


def find_expr_start(tokens: list, start_pos: int) -> int:
  """Walks backwards through the token stream to find the exact start of a variable,

  list, or parenthesized expression, ignoring things inside brackets/parens.
  """
  pos = start_pos
  # Skip trailing whitespace right before the dot
  while pos >= 0 and tokens[pos][1] == 'whitespace':
    pos -= 1

  bracket_level = 0
  paren_level = 0

  while pos >= 0:
    t_type = tokens[pos][1]
    t_val = tokens[pos][2]

    # Track if we are safely inside parentheses or brackets
    if t_type == 'operator' and t_val == ']':
      bracket_level += 1
    elif t_type == 'operator' and t_val == '[':
      bracket_level -= 1
    elif t_type == 'operator' and t_val == ')':
      paren_level += 1
    elif t_type == 'operator' and t_val == '(':
      paren_level -= 1

    # If we are at the top level
    elif bracket_level == 0 and paren_level == 0:
      # Check if it's a Jinja keyword/operator that acts as a boundary
      if t_type in ('variable_begin', 'block_begin') or (
          t_type == 'name'
          and str(t_val)
          in (
              'if',
              'elif',
              'else',
              'for',
              'set',
              'and',
              'or',
              'not',
              'in',
              'is',
              'return',
          )
      ):
        nxt = pos + 1
        while nxt <= start_pos and tokens[nxt][1] == 'whitespace':
          nxt += 1
        return nxt

      if t_type in ('name', 'string', 'integer', 'float'):
        prev = pos - 1
        while prev >= 0 and tokens[prev][1] == 'whitespace':
          prev -= 1
        # If the token before this is a dot or bracket, it's part of a chain (a.b.c)
        if (
            prev >= 0
            and tokens[prev][1] == 'operator'
            and tokens[prev][2] in ('.', '[', '(')
        ):
          pass
        else:
          return pos  # Found the start
      elif t_type == 'operator' and t_val != '.':
        # We hit an operator (+, -, =, etc.). The expression starts after it.
        nxt = pos + 1
        while nxt <= start_pos and tokens[nxt][1] == 'whitespace':
          nxt += 1
        return nxt
    pos -= 1
  return 0


def fix_python_methods(tokens: list) -> list:
  """Safely converts .replace(), .split(), .strip() etc.

  into parenthesized Jinja filters to completely prevent Precedence/Syntax
  errors.
  """
  new_tokens = list(tokens)
  i = 0
  while i < len(new_tokens):
    t_type = new_tokens[i][1]
    t_val = new_tokens[i][2]

    # Look for [dot] -> [name] -> [lparen]
    if (
        t_type == 'operator'
        and t_val == '.'
        and i + 2 < len(new_tokens)
        and new_tokens[i + 1][1] == 'name'
        and new_tokens[i + 2][1] == 'operator'
        and new_tokens[i + 2][2] == '('
    ):
      method_name = new_tokens[i + 1][2]

      if method_name in [
          'replace',
          'split',
          'lstrip',
          'rstrip',
          'strip',
          'lower',
          'upper',
      ]:
        # Find exactly where the variable/expression started
        start_idx = find_expr_start(new_tokens, i - 1)

        # Find the matching closing parenthesis for the method arguments
        lparen_idx = i + 2
        paren_count = 1
        j = lparen_idx + 1
        while paren_count > 0 and j < len(new_tokens):
          if new_tokens[j][1] == 'operator' and new_tokens[j][2] == '(':
            paren_count += 1
          elif new_tokens[j][1] == 'operator' and new_tokens[j][2] == ')':
            paren_count -= 1
          j += 1
        rparen_idx = j - 1

        args_tokens = new_tokens[lparen_idx + 1 : rparen_idx]
        filter_name = (
            'trim'
            if method_name in ['lstrip', 'rstrip', 'strip']
            else method_name
        )
        lineno = new_tokens[i][0]

        filter_call = (
            [
                (lineno, 'whitespace', ' '),
                (lineno, 'operator', '|'),
                (lineno, 'whitespace', ' '),
                (lineno, 'name', filter_name),
                (lineno, 'operator', '('),
            ]
            + args_tokens
            + [(lineno, 'operator', ')')]
        )

        if method_name == 'split':
          filter_call += [
              (lineno, 'whitespace', ' '),
              (lineno, 'operator', '|'),
              (lineno, 'whitespace', ' '),
              (lineno, 'name', 'list'),
          ]

        replacement = (
            [(lineno, 'operator', '(')]
            + new_tokens[start_idx:i]
            + filter_call
            + [(lineno, 'operator', ')')]
        )

        # Replace the old slice with the safely wrapped slice
        new_tokens = (
            new_tokens[:start_idx] + replacement + new_tokens[rparen_idx + 1 :]
        )

        # Advance pointer to end of our new replacement
        i = start_idx + len(replacement) - 1
    i += 1
  return new_tokens


def fix_len_calls(tokens: list) -> list:
  """Converts len(expr) into (expr | length)."""
  new_tokens = list(tokens)
  i = 0
  while i < len(new_tokens):
    if (
        new_tokens[i][1] == 'name'
        and new_tokens[i][2] == 'len'
        and i + 1 < len(new_tokens)
        and new_tokens[i + 1][1] == 'operator'
        and new_tokens[i + 1][2] == '('
    ):
      lparen_idx = i + 1
      paren_count = 1
      j = lparen_idx + 1
      while paren_count > 0 and j < len(new_tokens):
        if new_tokens[j][1] == 'operator' and new_tokens[j][2] == '(':
          paren_count += 1
        elif new_tokens[j][1] == 'operator' and new_tokens[j][2] == ')':
          paren_count -= 1
        j += 1
      rparen_idx = j - 1

      args_tokens = new_tokens[lparen_idx + 1 : rparen_idx]
      lineno = new_tokens[i][0]

      replacement = (
          [(lineno, 'operator', '(')]
          + args_tokens
          + [
              (lineno, 'whitespace', ' '),
              (lineno, 'operator', '|'),
              (lineno, 'whitespace', ' '),
              (lineno, 'name', 'length'),
              (lineno, 'operator', ')'),
          ]
      )

      new_tokens = new_tokens[:i] + replacement + new_tokens[rparen_idx + 1 :]
      i = i + len(replacement) - 1
    i += 1
  return new_tokens


def fix_strict_type_checks(tokens: list) -> list:
  """Fixes `is true` / `is false` strict checking."""
  new_tokens = list(tokens)
  i = 0
  while i < len(new_tokens):
    if new_tokens[i][1] == 'name' and new_tokens[i][2] == 'is':
      j = i + 1
      while j < len(new_tokens) and new_tokens[j][1] == 'whitespace':
        j += 1
      if (
          j < len(new_tokens)
          and new_tokens[j][1] == 'name'
          and new_tokens[j][2].lower() in ('true', 'false')
      ):
        is_false = new_tokens[j][2].lower() == 'false'
        end_idx = j
        if is_false:
          start_idx = find_expr_start(new_tokens, i - 1)
          lineno = new_tokens[start_idx][0]
          replacement = (
              [
                  (lineno, 'name', 'not'),
                  (lineno, 'whitespace', ' '),
                  (lineno, 'operator', '('),
              ]
              + new_tokens[start_idx:i]
              + [(lineno, 'operator', ')')]
          )
          new_tokens = (
              new_tokens[:start_idx] + replacement + new_tokens[end_idx + 1 :]
          )
          i = start_idx + len(replacement) - 1
        else:
          new_tokens = new_tokens[:i] + new_tokens[end_idx + 1 :]
          i = i - 1
    i += 1
  return new_tokens


class PatternToken:
  """A token pattern matcher for declarative template mutations."""

  def __init__(self, token_type: str, value=None, bind_to: str | None = None):
    self.type = token_type
    self.value = value
    self.bind_to = bind_to

  def match(self, token: tuple) -> bool:
    # token is (lineno, token_type, value)
    if token[1] != self.type:
      return False
    if self.value is None:
      return True
    if isinstance(self.value, set):
      return token[2] in self.value
    return token[2] == self.value


def next_non_ws(tokens: list, start: int) -> int:
  idx = start
  while idx < len(tokens) and tokens[idx][1] == 'whitespace':
    idx += 1
  return idx


def match_pattern_at(
    tokens: list, start_idx: int, pattern: list
) -> tuple | None:
  """Tries to match a pattern at start_idx, skipping whitespace between tokens.

  Returns (end_idx, bounds) if matched, where end_idx is the index after the
  last matched token, and bounds is a dict of bound tokens.
  Returns None if not matched.
  """
  idx = start_idx
  bounds = {}
  for pattern_idx, pattern_token in enumerate(pattern):
    if idx >= len(tokens):
      return None
    # Skip whitespace only between pattern tokens (not before the first one)
    if pattern_token.type != 'whitespace' and pattern_idx > 0:
      idx = next_non_ws(tokens, idx)
      if idx >= len(tokens):
        return None
    token = tokens[idx]
    if not pattern_token.match(token):
      return None
    if pattern_token.bind_to:
      bounds[pattern_token.bind_to] = token
    idx += 1
  return idx, bounds


REASONING_NAMES = {
    'reasoning_content',
    'thought',
    'thought_content',
    'reasoning',
}


def fix_qwen_reasoning_logic(tokens: list) -> list:
  """Simplifies 'loop.last or (not loop.last and reasoning_content)' to 'reasoning_content'."""
  pattern = [
      PatternToken('name', 'loop'),
      PatternToken('operator', '.'),
      PatternToken('name', 'last'),
      PatternToken('name', 'or'),
      PatternToken('operator', '('),
      PatternToken('name', 'not'),
      PatternToken('name', 'loop'),
      PatternToken('operator', '.'),
      PatternToken('name', 'last'),
      PatternToken('name', 'and'),
      PatternToken('name', REASONING_NAMES, bind_to='x'),
      PatternToken('operator', ')'),
  ]

  new_tokens = list(tokens)
  i = 0
  while i < len(new_tokens):
    match_result = match_pattern_at(new_tokens, i, pattern)
    if match_result:
      end_idx, bounds = match_result
      x_token = bounds['x']
      replacement = [x_token]
      new_tokens = new_tokens[:i] + replacement + new_tokens[end_idx:]
      i = i + len(replacement) - 1
    i += 1
  return new_tokens


def transpile_jinja2(template_str: str) -> str:
  """Transpiles a jinja2 template to a minijinja compatible template."""
  try:
    env = PreserveWhitespaceEnvironment(keep_trailing_newline=True)
    tokens = list(env.lex(template_str))

    # Run the stream mutations
    tokens = fix_strict_type_checks(tokens)
    tokens = fix_python_methods(tokens)
    tokens = fix_len_calls(tokens)
    tokens = fix_qwen_reasoning_logic(tokens)

    # Reconstruct the string perfectly
    output = [str(t[2]) for t in tokens]
    return ''.join(output)
  except Exception as e:  # pylint: disable=broad-except
    print('Failed to transpile template', e)
    return template_str
