# Copyright 2024 The LiteRT Torch Authors.
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

"""Progress tracking and logging library."""

import contextlib
import dataclasses
import time
from typing import Generator
from litert_torch import _config
from rich import console

__all__ = ["task", "console"]

config = _config.config
console = console.Console(color_system="auto")


@dataclasses.dataclass
class Task:
  name: str
  start_time: float = 0.0


_task_stack: list[Task] = []


def _fmt_elapsed_time(elapsed_time: float) -> str:
  minutes, seconds = divmod(elapsed_time, 60)
  return f"{int(minutes):02d}:{int(seconds):02d}"


def _style_elapsed_time(elapsed: float) -> str:
  formatted = _fmt_elapsed_time(elapsed)
  if elapsed < 5:
    return f"[dim default]{formatted}[/dim default]"
  elif elapsed < 60:
    return f"[yellow]{formatted}[/yellow]"
  elif elapsed < 60 * 10:
    return f"[bold yellow]{formatted}[/bold yellow]"
  else:
    return f"[bold red]{formatted}[/bold red]"


def _task_stack_repr() -> str:
  return " > ".join(
      f"[bold default]{task.name}[/bold default]" for task in _task_stack
  )


def _stack_elapsed_time() -> str:
  elapsed = (
      0 if not _task_stack else time.perf_counter() - _task_stack[0].start_time
  )
  return f"[dim default]({_fmt_elapsed_time(elapsed)})[/dim default]"


def log(message: str):
  """Log a message."""
  if not config.show_progress:
    return
  console.print(  # pyrefly: ignore[missing-attribute]
      f"{_stack_elapsed_time()} [bold blue][ INFO][/bold blue] {message}"
  )


@contextlib.contextmanager
def task(name: str) -> Generator[None, None, None]:
  """Context manager for tracking one task."""
  if not config.show_progress:
    yield
    return

  current_task = Task(name, time.perf_counter())
  _task_stack.append(current_task)

  stack_view = _task_stack_repr()
  console.print(  # pyrefly: ignore[missing-attribute]
      f"{_stack_elapsed_time()} [bold cyan][START][/bold cyan] {stack_view}"
  )

  try:
    yield
  except Exception:
    console.print(  # pyrefly: ignore[missing-attribute]
        f"{_stack_elapsed_time()} [bold red][ FAIL][/bold red] {stack_view}"
    )
    raise
  else:
    elapsed_time = time.perf_counter() - current_task.start_time
    console.print(  # pyrefly: ignore[missing-attribute]
        f"{_stack_elapsed_time()} [bold green][ DONE][/bold green]"
        f" [dim]{stack_view}[/dim]"
        f" [dim](+[/dim]{_style_elapsed_time(elapsed_time)}[dim])[/dim]"
    )
  finally:
    if _task_stack:
      _task_stack.pop()
