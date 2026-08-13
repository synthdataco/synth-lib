"""Global GPU lock: only one GPU job at a time on the machine (runs AND verdict)."""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import IO

DEFAULT_LOCK_PATH = Path("/tmp/synth-bench-gpu.lock")


class GpuLockBusyError(RuntimeError):
    pass


class GpuLock:
    def __init__(self, path: Path = DEFAULT_LOCK_PATH):
        self._path = path
        self._fh: IO[str] | None = None

    def acquire(self, blocking: bool = True) -> "GpuLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w")
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(self._fh, flags)
        except BlockingIOError as exc:
            self._fh.close()
            self._fh = None
            raise GpuLockBusyError(f"GPU lock busy: {self._path}") from exc
        return self

    def release(self) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "GpuLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
