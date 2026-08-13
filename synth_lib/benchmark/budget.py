"""$ budget + clock tracking, phases (normal/landing/exhausted), BUDGET.md rendering for the agent to read."""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass
from pathlib import Path

import requests

from synth_lib.benchmark.clock import RunClock

LANDING_ORDER = (
    "## LANDING ORDER — NON-NEGOTIABLE\n\n"
    "Your envelope is nearly exhausted. STOP any new experiment NOW.\n"
    "1. Finalize `agent/journal.md` (including the last iteration).\n"
    "2. Write the complete `agent/report.md` (honest results, failures included).\n"
    "3. Nominate your champion: write the `agent/CHAMPION` file (sha, agent_dir, profiles, notes).\n"
    "   The path matters: the harness looks ONLY at `agent/CHAMPION`.\n"
    "4. Verify that the weights/artifacts required by `simulate()` are in the workspace.\n"
)


class Phase(enum.Enum):
    NORMAL = "normal"
    LANDING = "landing"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class BudgetStatus:
    spend_usd: float
    budget_usd: float
    pct_usd: float
    elapsed_s: float
    remaining_s: float
    pct_time: float
    phase: Phase
    infra_down: bool = False
    last_error: str | None = None


class BudgetTracker:
    def __init__(self, admin, virtual_key: str, budget_usd: float, soft_landing_pct: float, clock: RunClock):
        self._admin = admin
        self._key = virtual_key
        self._budget = budget_usd
        self._soft = soft_landing_pct
        self.clock = clock
        self._last_spend = 0.0
        self._infra_down = False
        self._last_error: str | None = None

    def status(self) -> BudgetStatus:
        try:
            self._last_spend = self._admin.key_info(self._key)["spend"]
            self._last_error = None
            if self._infra_down:
                self.clock.resume()
                self._infra_down = False
        except (OSError, requests.RequestException) as exc:
            self._last_error = repr(exc)
            if not self._infra_down:
                self.clock.pause()
                self._infra_down = True
        pct_usd = self._last_spend / self._budget
        pct_time = self.clock.pct_elapsed()
        worst = max(pct_usd, pct_time)
        # $ and time are both monotonically non-decreasing: a worst computed from stale data
        # (outage) remains a valid lower bound — landing must never fall back to normal during
        # an outage, and a stale worst>=1.0 already means it's genuinely exhausted.
        if worst >= 1.0:
            phase = Phase.EXHAUSTED
        elif worst >= self._soft:
            phase = Phase.LANDING
        else:
            phase = Phase.NORMAL
        return BudgetStatus(
            spend_usd=self._last_spend,
            budget_usd=self._budget,
            pct_usd=pct_usd,
            elapsed_s=self.clock.elapsed(),
            remaining_s=self.clock.remaining(),
            pct_time=pct_time,
            phase=phase,
            infra_down=self._infra_down,
            last_error=self._last_error,
        )

    def write_budget_file(self, workspace: Path, status: BudgetStatus, notes: str = "") -> None:
        lines = [
            "# BUDGET — maintained by the harness (read-only for you)",
            "",
            f"- Credits: {status.spend_usd:.2f} / {status.budget_usd:.2f} USD ({status.pct_usd:.0%})",
            f"- Time   : {status.elapsed_s / 3600:.1f} h elapsed, {status.remaining_s / 3600:.1f} h remaining"
            f" ({status.pct_time:.0%})",
            f"- Phase  : {status.phase.value}",
        ]
        if status.infra_down:
            note = "- NOTE: infra outage in progress — the clock is PAUSED, keep working if you can."
            if status.last_error:
                note += f" (cause: {status.last_error})"
            lines.append(note)
        if status.phase is Phase.LANDING:
            lines += ["", LANDING_ORDER]
        if notes:
            lines += ["", f"## Harness notes\n\n{notes}"]
        target = workspace / "BUDGET.md"
        tmp = workspace / "BUDGET.md.tmp"
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, target)
