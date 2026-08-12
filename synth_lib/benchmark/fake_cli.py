"""Simulated agent for testing the driver without an LLM.

Scenarios (env FAKE_CLI_SCENARIO):
  lands           : journals on each tick; on the landing order, writes report.md + CHAMPION then exit 0.
  ignores_landing : journals but never lands (tests kill + relaunch).
  crashes         : exit 1 after 2 ticks (tests the resume).
  silent          : produces nothing (tests the did-not-land).
  stops_early     : if BUDGET.md already orders landing, lands; otherwise journals once
                    and exits 0 (tests the voluntary stop, distinct from a crash).
Env: FAKE_CLI_AGENT_DIR (agent workspace), FAKE_CLI_TICK_SECONDS (default 0.1).
A prompt containing "LANDING ORDER" forces immediate landing mode (landing-only relaunch).
"""

import os
import sys
import time
from pathlib import Path


def main() -> int:
    scenario = os.environ.get("FAKE_CLI_SCENARIO", "lands")
    agent_dir = Path(os.environ["FAKE_CLI_AGENT_DIR"])
    tick = float(os.environ.get("FAKE_CLI_TICK_SECONDS", "0.1"))
    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    journal = agent_dir / "journal.md"
    forced_landing = "LANDING ORDER" in prompt.upper()

    def land() -> int:
        (agent_dir / "report.md").write_text("# Report\n\nHonest results.\n")
        (agent_dir / "CHAMPION").write_text(
            f"sha: deadbeef\nagent_dir: {agent_dir.name}\nprofiles: [low]\nnotes: fake\n"
        )
        with journal.open("a") as fh:
            fh.write("\n## Landing\ndone.\n")
        return 0

    if scenario == "silent":
        time.sleep(3600)
        return 0
    if forced_landing:
        return land()
    if scenario == "stops_early":
        budget = agent_dir / "BUDGET.md"
        if budget.exists() and "LANDING ORDER" in budget.read_text():
            return land()
        with journal.open("a") as fh:
            fh.write("\n## Iteration\nhypothesis/result.\n")
        time.sleep(2 * tick)
        return 0
    for i in range(10_000):
        time.sleep(tick)
        with journal.open("a") as fh:
            fh.write(f"\n## Iteration {i}\nhypothesis/result.\n")
        if scenario == "crashes" and i >= 1:
            return 1
        budget = agent_dir / "BUDGET.md"
        if scenario == "lands" and budget.exists() and "LANDING ORDER" in budget.read_text():
            return land()
    return 0


if __name__ == "__main__":
    sys.exit(main())
