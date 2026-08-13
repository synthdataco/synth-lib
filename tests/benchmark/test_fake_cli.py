import os
import subprocess
import sys

from synth_lib.benchmark.cli_adapters import FAKE_CLI_PATH as FAKE  # single source of truth


def test_lands_when_budget_orders_it(tmp_path):
    (tmp_path / "BUDGET.md").write_text("## LANDING ORDER — NON-NEGOTIABLE")
    env = {
        **os.environ,
        "FAKE_CLI_SCENARIO": "lands",
        "FAKE_CLI_AGENT_DIR": str(tmp_path),
        "FAKE_CLI_TICK_SECONDS": "0.01",
    }
    proc = subprocess.run([sys.executable, str(FAKE), "go"], env=env, timeout=10)
    assert proc.returncode == 0
    assert (tmp_path / "CHAMPION").exists() and (tmp_path / "report.md").exists()


def test_crashes_scenario_exits_1(tmp_path):
    env = {
        **os.environ,
        "FAKE_CLI_SCENARIO": "crashes",
        "FAKE_CLI_AGENT_DIR": str(tmp_path),
        "FAKE_CLI_TICK_SECONDS": "0.01",
    }
    proc = subprocess.run([sys.executable, str(FAKE), "go"], env=env, timeout=10)
    assert proc.returncode == 1
