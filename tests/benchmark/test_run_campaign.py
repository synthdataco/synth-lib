"""Tests for the run_campaign.py orchestrator: sandbox-not-wired guard rail + state.json resilience."""

from datetime import date
from pathlib import Path

import pytest

from synth_lib.benchmark.campaign import BaselineSpec, CampaignConfig, ModelSpec
from synth_lib.benchmark.run_campaign import CampaignState


def _cfg(root: Path) -> CampaignConfig:
    return CampaignConfig(
        name="t",
        objective="o",
        models=(ModelSpec(id="f", cli="fake", model="none"),),
        budget_usd_per_model=10.0,
        deadline_hours_per_model=1.0,
        data_cutoff=date(2026, 7, 15),
        forward_window_days=7,
        baselines=(BaselineSpec(name="b", module="x"),),
        hardware="h",
        proxy_url="http://x",
        root=root,
    )


def test_containerize_url_rewrites_localhost_and_loopback():
    from synth_lib.benchmark.run_campaign import containerize_url

    assert containerize_url("http://localhost:4000") == "http://host.docker.internal:4000"
    assert containerize_url("http://127.0.0.1:4000/v1") == "http://host.docker.internal:4000/v1"
    assert containerize_url("http://example.com:4000") == "http://example.com:4000"  # unchanged


def test_offline_bundle_env_absent_when_no_bundle(tmp_path):
    """A campaign without a bundle must behave exactly as before: no env var at all."""
    from synth_lib.benchmark.run_campaign import offline_bundle_env

    (tmp_path / "snapshot").mkdir()
    assert offline_bundle_env(tmp_path / "snapshot") == {}


def test_offline_bundle_env_points_at_the_container_path(tmp_path):
    """The path must be the CONTAINER's view: the snapshot is mounted at /workspace/market_data,
    so a host path would be meaningless inside the sandbox."""
    from synth_lib.benchmark.run_campaign import OFFLINE_ROOT_ENV, offline_bundle_env

    snapshot = tmp_path / "snapshot"
    (snapshot / "offline_data").mkdir(parents=True)
    assert offline_bundle_env(snapshot) == {OFFLINE_ROOT_ENV: "/workspace/market_data/offline_data"}


def test_offline_bundle_env_ignores_a_file_of_that_name(tmp_path):
    from synth_lib.benchmark.run_campaign import offline_bundle_env

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "offline_data").write_text("not a directory")
    assert offline_bundle_env(snapshot) == {}


def test_sandboxed_run_exports_the_offline_root_when_a_bundle_exists(tmp_path, monkeypatch):
    """End to end: the env var must reach the actual `docker run` argv."""
    import synth_lib.benchmark.run_campaign as rc
    from synth_lib.benchmark.campaign import BaselineSpec, CampaignConfig, ModelSpec

    cfg = CampaignConfig(
        name="camp",
        objective="o",
        models=(ModelSpec(id="f", cli="claude-code", model="claude-3-5"),),
        budget_usd_per_model=10.0,
        deadline_hours_per_model=1.0,
        data_cutoff=date(2026, 7, 15),
        forward_window_days=7,
        baselines=(BaselineSpec(name="b", module="x"),),
        hardware="h",
        proxy_url="http://localhost:4000",
        root=tmp_path / "campaigns",
    )
    state = CampaignState(cfg=cfg, dir=cfg.dir)
    (state.dir / "snapshot" / "offline_data").mkdir(parents=True)
    state.save({"runs_done": []})

    captured: dict = {}

    class DummyRun:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def run(self):
            return None

    monkeypatch.setattr(rc, "ModelRun", DummyRun)
    rc.run_all(state, admin=None, sandboxed=True, python_exe="python")

    argv = " ".join(captured["kwargs"]["cmd_wrapper"](["echo", "hi"]))
    assert f"{rc.OFFLINE_ROOT_ENV}=/workspace/market_data/offline_data" in argv


def test_passthrough_legs_mint_uncapped_keys(tmp_path, monkeypatch):
    """gemini-cli reaches the proxy via the passthrough, where LiteLLM's budget reservation leaks
    exactly max_budget and any cap self-trips (PROXY_COMPAT "Phantom budget 429"). Its key must be
    minted with cap=None; /v1-routed legs keep the real cap. The driver enforces the budget either
    way, from cfg.budget_usd_per_model and true spend."""
    import synth_lib.benchmark.run_campaign as rc
    from synth_lib.benchmark.campaign import BaselineSpec, CampaignConfig, ModelSpec

    cfg = CampaignConfig(
        name="camp",
        objective="o",
        models=(
            ModelSpec(id="c", cli="claude-code", model="claude-model"),
            ModelSpec(id="g", cli="gemini-cli", model="gemini-3.1-pro-preview"),
        ),
        budget_usd_per_model=10.0,
        deadline_hours_per_model=1.0,
        data_cutoff=date(2026, 7, 15),
        forward_window_days=7,
        baselines=(BaselineSpec(name="b", module="x"),),
        hardware="h",
        proxy_url="http://localhost:4000",
        root=tmp_path / "campaigns",
    )
    state = CampaignState(cfg=cfg, dir=cfg.dir)
    state.dir.mkdir(parents=True, exist_ok=True)
    state.save({"runs_done": []})

    minted = {}

    class Admin:
        def generate_key(self, alias, max_budget):
            minted[alias] = max_budget
            return f"sk-{alias}"

    class DummyRun:
        def __init__(self, *a, **k):
            pass

        def run(self):
            return None

    monkeypatch.setattr(rc, "ModelRun", DummyRun)
    rc.run_all(state, admin=Admin(), sandboxed=False, python_exe="python")

    assert minted["camp-c"] == 10.0  # /v1 leg: real cap
    assert minted["camp-g"] is None  # passthrough leg: uncapped, driver-only enforcement


def test_run_all_sandboxed_wires_container_and_wrapper(tmp_path, monkeypatch):
    # sandboxed=True must build a container_name per run and a cmd_wrapper that wraps the
    # adapter's raw command via sandbox_cmd (proxy rewritten to host.docker.internal in-container).
    import synth_lib.benchmark.run_campaign as rc
    from synth_lib.benchmark.campaign import BaselineSpec, CampaignConfig, ModelSpec

    cfg = CampaignConfig(
        name="camp",
        objective="o",
        models=(ModelSpec(id="f", cli="claude-code", model="claude-3-5"),),
        budget_usd_per_model=10.0,
        deadline_hours_per_model=1.0,
        data_cutoff=date(2026, 7, 15),
        forward_window_days=7,
        baselines=(BaselineSpec(name="b", module="x"),),
        hardware="h",
        proxy_url="http://localhost:4000",
        root=tmp_path / "campaigns",
    )
    state = CampaignState(cfg=cfg, dir=cfg.dir)
    state.dir.mkdir(parents=True, exist_ok=True)
    state.save({"runs_done": []})

    captured: dict = {}

    class DummyRun:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def run(self):
            return None

    monkeypatch.setattr(rc, "ModelRun", DummyRun)
    rc.run_all(state, admin=None, sandboxed=True, python_exe="python")

    kwargs = captured["kwargs"]
    assert kwargs["container_name"] == "synthbench-camp-f"
    wrapped = kwargs["cmd_wrapper"](["echo", "hi"])
    assert wrapped[0] == "docker"
    assert "--name" in wrapped and "synthbench-camp-f" in wrapped
    joined = " ".join(wrapped)
    assert "host.docker.internal:4000" in joined  # ANTHROPIC_BASE_URL rewritten for in-container use
    assert "localhost" not in joined


def test_corrupt_state_raises_clear_error(tmp_path):
    cfg = _cfg(tmp_path / "campaigns")
    state = CampaignState(cfg=cfg, dir=cfg.dir)
    state.dir.mkdir(parents=True, exist_ok=True)
    state.state_file.write_text("{ not valid json !!")
    with pytest.raises(RuntimeError, match=str(state.state_file)):
        state.load()


def test_provision_home_writes_codex_config(tmp_path):
    from synth_lib.benchmark.campaign import ModelSpec
    from synth_lib.benchmark.cli_adapters import build_adapter
    from synth_lib.benchmark.run_campaign import provision_home

    adapter = build_adapter(
        ModelSpec(id="x", cli="codex", model="codex-model", wire_api="responses"),
        proxy_url="http://localhost:4000",
        virtual_key="sk-v",
    )
    home = tmp_path / "codex-home"
    provision_home(adapter, home)
    cfg_path = home / ".codex" / "config.toml"
    assert cfg_path.exists()
    assert 'wire_api = "responses"' in cfg_path.read_text()


def test_run_all_provisions_home_and_sets_home_env(tmp_path, monkeypatch):
    import synth_lib.benchmark.run_campaign as rc

    cfg = _cfg(tmp_path / "campaigns")
    state = CampaignState(cfg=cfg, dir=cfg.dir)
    state.dir.mkdir(parents=True, exist_ok=True)
    state.save({"runs_done": []})

    captured: dict = {}

    class DummyRun:
        def __init__(self, *args, **kwargs):
            captured["extra_env"] = kwargs.get("extra_env")

        def run(self):
            return None

    monkeypatch.setattr(rc, "ModelRun", DummyRun)
    rc.run_all(state, admin=None, sandboxed=False, python_exe="python")

    home = state.dir / "runs" / "f-home"
    assert home.is_dir()
    assert captured["extra_env"]["HOME"] == str(home)


def test_setup_accepts_campaign_file_already_in_place(tmp_path):
    """Smoke regression 07-24: campaign.yaml already living in campaigns/<name>/ => no SameFileError."""
    import yaml

    from synth_lib.benchmark.run_campaign import setup_campaign
    from tests.benchmark.test_workspace import _make_repo

    repo = _make_repo(tmp_path)
    md = repo / "market_data" / "pyth" / "BTC" / "1m"
    md.mkdir(parents=True)
    (md / "date=2026-07-10.parquet").write_bytes(b"x")
    camp_dir = tmp_path / "campaigns" / "inplace"
    camp_dir.mkdir(parents=True)
    cfg = {
        "name": "inplace",
        "objective": "t",
        "models": [{"id": "f1", "cli": "fake", "model": "none"}],
        "budget_usd_per_model": 1.0,
        "deadline_hours_per_model": 1.0,
        "data_cutoff": "2026-07-15",
        "forward_window_days": 1,
        "baselines": [{"name": "b", "module": "x"}],
        "hardware": "h",
        "proxy_url": "http://x",
    }
    f = camp_dir / "campaign.yaml"
    f.write_text(yaml.safe_dump(cfg))
    state = setup_campaign(f, data_root=repo, campaigns_root=tmp_path / "campaigns")
    assert (state.dir / "campaign.yaml").exists() and (state.dir / "snapshot" / "manifest.json").exists()


def test_virtual_key_persisted_and_reused_across_reruns(tmp_path):
    """Smoke regression 07-24: LiteLLM aliases are unique => regenerating on every run = 400.
    The key must be persisted in state.json and reused (same budget after a crash)."""
    import json

    from synth_lib.benchmark.run_campaign import run_all, setup_campaign
    from tests.benchmark.test_workspace import _make_repo

    class CountingAdmin:
        def __init__(self):
            self.generated = 0

        def generate_key(self, alias, max_budget_usd):
            self.generated += 1
            return f"sk-gen-{self.generated}"

        def key_info(self, key):
            return {"spend": 8.6, "max_budget": 10.0}  # direct LANDING => the fake lands quickly

    repo = _make_repo(tmp_path)
    md = repo / "market_data" / "pyth" / "BTC" / "1m"
    md.mkdir(parents=True)
    (md / "date=2026-07-10.parquet").write_bytes(b"x")
    import yaml as _yaml

    camp = {
        "name": "rerun",
        "objective": "t",
        "models": [{"id": "f1", "cli": "fake", "model": "none"}],
        "budget_usd_per_model": 10.0,
        "deadline_hours_per_model": 1.0,
        "data_cutoff": "2026-07-15",
        "forward_window_days": 1,
        "baselines": [{"name": "b", "module": "x"}],
        "hardware": "h",
        "proxy_url": "http://x",
        "poll_seconds": 0.05,
        "landing_grace_seconds": 0.3,
    }
    f = tmp_path / "campaign.yaml"
    f.write_text(_yaml.safe_dump(camp))
    state = setup_campaign(f, data_root=repo, campaigns_root=tmp_path / "campaigns")
    admin = CountingAdmin()
    import sys

    run_all(state, admin=admin, sandboxed=False, fake_env={"FAKE_CLI_TICK_SECONDS": "0.02"}, python_exe=sys.executable)
    assert admin.generated == 1
    saved = json.loads(state.state_file.read_text())
    assert saved["virtual_keys"] == {"f1": "sk-gen-1"}
    # simulates a resume: runs_done cleared, the key must be REUSED, not regenerated
    saved["runs_done"] = []
    state.save(saved)
    run_all(state, admin=admin, sandboxed=False, fake_env={"FAKE_CLI_TICK_SECONDS": "0.02"}, python_exe=sys.executable)
    assert admin.generated == 1  # pas de second generate_key
