"""End-to-end mini-campaign: setup -> sequential run (fake) -> collection + state.json."""

import sys
from pathlib import Path

import yaml

from synth_lib.benchmark.run_campaign import run_all, setup_campaign
from tests.benchmark.test_driver import HoldAdmin


def _write_campaign(tmp_path: Path, repo: Path) -> Path:
    cfg = {
        "name": "mini",
        "objective": "test",
        "models": [{"id": "f1", "cli": "fake", "model": "none"}, {"id": "f2", "cli": "fake", "model": "none"}],
        "budget_usd_per_model": 10.0,
        "deadline_hours_per_model": 1.0,
        "data_cutoff": "2026-07-15",
        "forward_window_days": 7,
        "baselines": [{"name": "b", "module": "x"}],
        "hardware": "h",
        "proxy_url": "http://x",
        "poll_seconds": 0.05,
        "landing_grace_seconds": 0.3,
    }
    p = tmp_path / "campaign.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_mini_campaign_end_to_end(tmp_path, monkeypatch):
    from tests.benchmark.test_workspace import _make_repo

    repo = _make_repo(tmp_path)
    # snapshot source minimal
    (repo / "market_data" / "pyth" / "BTC" / "1m").mkdir(parents=True)
    (repo / "market_data" / "pyth" / "BTC" / "1m" / "date=2026-07-10.parquet").write_bytes(b"x")
    campaign_file = _write_campaign(tmp_path, repo)

    state = setup_campaign(campaign_file, data_root=repo, campaigns_root=tmp_path / "campaigns")
    assert (state.dir / "snapshot" / "manifest.json").exists()
    assert (state.dir / "runs" / "f1").exists() and (state.dir / "runs" / "f2").exists()

    results = run_all(
        state,
        admin=HoldAdmin(spend=8.6),
        sandboxed=False,
        fake_env={"FAKE_CLI_TICK_SECONDS": "0.02"},
        python_exe=sys.executable,
    )
    assert [r.model_id for r in results] == ["f1", "f2"]  # sequential, yaml order
    assert all(r.landed for r in results)
    saved = yaml.safe_load((state.dir / "state.json").read_text())
    assert saved["runs_done"] == ["f1", "f2"] and "forward_window_start" in saved


def test_rerun_does_not_move_forward_window(tmp_path):
    from tests.benchmark.test_workspace import _make_repo

    repo = _make_repo(tmp_path)
    (repo / "market_data" / "pyth" / "BTC" / "1m").mkdir(parents=True)
    (repo / "market_data" / "pyth" / "BTC" / "1m" / "date=2026-07-10.parquet").write_bytes(b"x")
    campaign_file = _write_campaign(tmp_path, repo)

    state = setup_campaign(campaign_file, data_root=repo, campaigns_root=tmp_path / "campaigns")
    run_all(
        state,
        admin=HoldAdmin(spend=8.6),
        sandboxed=False,
        fake_env={"FAKE_CLI_TICK_SECONDS": "0.02"},
        python_exe=sys.executable,
    )
    first = yaml.safe_load((state.dir / "state.json").read_text())["forward_window_start"]

    results_rerun = run_all(
        state,
        admin=HoldAdmin(spend=8.6),
        sandboxed=False,
        fake_env={"FAKE_CLI_TICK_SECONDS": "0.02"},
        python_exe=sys.executable,
    )
    assert results_rerun == []  # all runs are already in runs_done, none re-executed
    second = yaml.safe_load((state.dir / "state.json").read_text())["forward_window_start"]
    assert second == first
