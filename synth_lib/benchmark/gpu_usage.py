"""GPU activity on the host during a leg, read from nvidia-smi at every driver poll."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

QUERY = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
# A poll counts as GPU time when any device is at least this busy.
BUSY_UTILIZATION_PCT = 10.0


@dataclass
class GpuUsage:
    """GPU-busy time and peak memory over one leg. Legs run one at a time, so the devices' activity
    is the leg's. The first read that fails (no nvidia-smi, no driver, an unreadable value) turns
    the meter off for the rest of the leg, with the reason in `error`."""

    busy_seconds: float = 0.0
    peak_memory_mib: float = 0.0
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None

    @property
    def busy_minutes(self) -> float:
        return self.busy_seconds / 60

    def sample(self, interval_seconds: float) -> None:
        """Credit `interval_seconds` of GPU time if a device is busy now."""
        if not self.available:
            return
        try:
            out = subprocess.run(QUERY, capture_output=True, text=True, timeout=10, check=True)
            readings = [[float(v) for v in line.split(",")] for line in out.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.error = repr(exc)
            return
        if any(utilization >= BUSY_UTILIZATION_PCT for utilization, _ in readings):
            self.busy_seconds += interval_seconds
        self.peak_memory_mib = max([self.peak_memory_mib, *(memory for _, memory in readings)])

    def line(self, required_minutes: float = 0.0) -> str:
        """The GPU line of BUDGET.md; empty when the meter is off."""
        if not self.available:
            return ""
        line = f"- GPU: {self.busy_minutes:.0f} min busy so far, peak memory {self.peak_memory_mib / 1024:.1f} GiB"
        if required_minutes:
            line += f". A nomination stands from {required_minutes:g} min"
        return line
