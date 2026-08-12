"""Run clock: the agent's wall-clock time, infra pauses excluded."""

from __future__ import annotations

import time
from typing import Callable


class RunClock:
    def __init__(self, deadline_seconds: float, now_fn: Callable[[], float] = time.monotonic):
        self._deadline = deadline_seconds
        self._now = now_fn
        self._started_at = now_fn()
        self._paused_total = 0.0
        self._paused_at: float | None = None

    def pause(self) -> None:
        if self._paused_at is None:
            self._paused_at = self._now()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._paused_total += self._now() - self._paused_at
            self._paused_at = None

    def paused_total(self) -> float:
        extra = (self._now() - self._paused_at) if self._paused_at is not None else 0.0
        return self._paused_total + extra

    def elapsed(self) -> float:
        return self._now() - self._started_at - self.paused_total()

    def remaining(self) -> float:
        return max(0.0, self._deadline - self.elapsed())

    def pct_elapsed(self) -> float:
        return self.elapsed() / self._deadline
