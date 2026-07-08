"""Small dependency-free terminal progress helpers for setup scripts."""
from __future__ import annotations

import sys
import time


class ProgressBar:
    """Render one-line setup progress without external packages."""

    def __init__(self, total, *, label="Progress", width=28, enabled=True, stream=None):
        self.total = max(1, int(total or 1))
        self.label = str(label)
        self.width = max(8, int(width))
        self.enabled = bool(enabled)
        self.stream = stream or sys.stderr
        self.current = 0
        self.started_at = time.monotonic()
        self._last_message = ""

    def update(self, current=None, *, message=""):
        if current is None:
            self.current += 1
        else:
            self.current = int(current)
        self.current = min(max(self.current, 0), self.total)
        self._last_message = str(message or self._last_message)
        if self.enabled:
            self._render(done=False)

    def step(self, *, message=""):
        self.update(self.current + 1, message=message)

    def finish(self, *, message="done"):
        self.current = self.total
        self._last_message = str(message or self._last_message)
        if self.enabled:
            self._render(done=True)

    def _render(self, *, done):
        fraction = self.current / self.total
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.monotonic() - self.started_at
        percent = int(round(fraction * 100))
        suffix = f" | {self._last_message}" if self._last_message else ""
        line = (
            f"\r{self.label}: [{bar}] {self.current}/{self.total} "
            f"{percent:3d}% {elapsed:5.1f}s{suffix}"
        )
        print(line, end="\n" if done else "", file=self.stream, flush=True)


def progress_enabled(json_output=False, quiet=False):
    """Return whether interactive progress should be shown."""
    return not json_output and not quiet and sys.stderr.isatty()
