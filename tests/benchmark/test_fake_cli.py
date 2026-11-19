import os
import subprocess
import sys

from synth_lib.benchmark.cli_adapters import FAKE_CLI_PATH as FAKE  # single source of truth
from synth_lib.benchmark.nomination import nominated_sha


def test_lands_when_budget_orders_it(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (agent_dir / "BUDGET.md").write_text("## LANDING ORDER — NON-NEGOTIABLE")
    env = {
        **os.environ,
        "FAKE_CLI_SCENARIO": "lands",
        "FAKE_CLI_AGENT_DIR": str(agent_dir),
        "FAKE_CLI_TICK_SECONDS": "0.01",
    }
    proc = subprocess.run([sys.executable, str(FAKE), "go"], env=env, timeout=10)
    assert proc.returncode == 0
    assert (agent_dir / "report.md").exists()
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True)
    assert nominated_sha(agent_dir / "CHAMPION") == head.stdout.strip()


def test_crashes_scenario_exits_1(tmp_path):
    env = {
        **os.environ,
        "FAKE_CLI_SCENARIO": "crashes",
        "FAKE_CLI_AGENT_DIR": str(tmp_path),
        "FAKE_CLI_TICK_SECONDS": "0.01",
    }
    proc = subprocess.run([sys.executable, str(FAKE), "go"], env=env, timeout=10)
    assert proc.returncode == 1
