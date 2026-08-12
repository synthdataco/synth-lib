import os
import subprocess
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from synth_lib.benchmark.campaign import BaselineSpec, CampaignConfig, ModelSpec
from synth_lib.benchmark.constitution import render_constitution
from synth_lib.benchmark.workspace import create_workspace


def _cfg(root: Path) -> CampaignConfig:
    return CampaignConfig(
        name="testcamp",
        objective="Improve the miner.",
        models=(ModelSpec(id="fake1", cli="fake", model="fake-model"),),
        budget_usd_per_model=10.0,
        deadline_hours_per_model=1.0,
        data_cutoff=date(2026, 7, 15),
        forward_window_days=7,
        baselines=(BaselineSpec(name="synth_default", module="synth_lib/benchmark/verdict/baseline_default.py"),),
        hardware="test-rig",
        proxy_url="http://localhost:4000",
        root=root,
    )


def _make_repo(tmp_path: Path) -> Path:
    """Minimal source repo for the setup_campaign/run_campaign tests: the agent workspace no
    longer depends on it (see create_workspace), only market_data is still extracted from it
    (build_snapshot). Kept here (imported by test_integration/test_run_campaign)."""
    repo = tmp_path / "repo"
    (repo / "market_data").mkdir(parents=True)
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base", "--allow-empty"],
        ["git", "tag", "campaign-base"],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    return repo


def test_create_workspace_builds_standalone_repo(tmp_path):
    cfg = _cfg(tmp_path / "campaigns")
    ws = create_workspace(cfg, cfg.models[0])
    assert ws == cfg.dir / "runs" / "fake1"
    assert (ws / "pyproject.toml").exists()
    assert "synth-lib" in (ws / "pyproject.toml").read_text()
    agent_dir = ws / "agent"
    assert (agent_dir / "modeling.py").exists()
    assert "def simulate" in (agent_dir / "modeling.py").read_text()
    assert (agent_dir / "journal.md").read_text() == ""
    assert (agent_dir / "suggestions.md").read_text() == ""
    # agent/predict.py must be a syntactically valid template (it runs inside the workspace
    # with synth-lib installed, not in this suite).
    predict_path = agent_dir / "predict.py"
    compile(predict_path.read_text(), str(predict_path), "exec")
    # CAMPAIGN.md must be visible at the root (agents look there first) AND in agent/.
    assert "testcamp" in (ws / "CAMPAIGN.md").read_text()
    assert "testcamp" in (agent_dir / "CAMPAIGN.md").read_text()
    # Standalone git repo with an initial commit — agents need git for the CHAMPION sha.
    rev = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert rev
    log = subprocess.run(["git", "-C", str(ws), "log", "--oneline"], check=True, capture_output=True, text=True).stdout
    assert len(log.strip().splitlines()) == 1


def test_gpu_campaign_preseeds_torch(tmp_path):
    """A GPU campaign must not make each agent discover `uv add torch` on its own budget."""
    cfg = replace(_cfg(tmp_path / "campaigns"), gpu=True)
    ws = create_workspace(cfg, cfg.models[0])
    deps = (ws / "pyproject.toml").read_text()
    assert "torch>=2.0" in deps
    # the pre-existing deps must survive the patch
    for kept in ("synth-lib", "numpy>=2.2", "pandas>=2.3"):
        assert kept in deps


def test_cpu_campaign_omits_torch(tmp_path):
    """A ~3GB CUDA wheel would eat a short CPU campaign's deadline for nothing (smoke)."""
    cfg = replace(_cfg(tmp_path / "campaigns"), gpu=False)
    ws = create_workspace(cfg, cfg.models[0])
    assert "torch" not in (ws / "pyproject.toml").read_text()


def test_workspace_gitignores_regenerable_bulk(tmp_path):
    """The agent is told to commit often; .venv and 1000-path predictions must stay out."""
    cfg = _cfg(tmp_path / "campaigns")
    ws = create_workspace(cfg, cfg.models[0])
    ignored = (ws / ".gitignore").read_text()
    for pattern in (".venv/", "miner_outputs/"):
        assert pattern in ignored


def test_recreate_workspace_fails_clearly(tmp_path):
    cfg = _cfg(tmp_path / "campaigns")
    create_workspace(cfg, cfg.models[0])
    ws = cfg.dir / "runs" / "fake1"
    with pytest.raises(ValueError, match=str(ws)):
        create_workspace(cfg, cfg.models[0])


def _commit(ws: Path, message: str, env: dict | None = None) -> subprocess.CompletedProcess:
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True, capture_output=True)
    return subprocess.run(
        ["git", "-c", "user.email=a@a", "-c", "user.name=a", "commit", "-qm", message],
        cwd=ws,
        capture_output=True,
        env={**os.environ, **(env or {})},
    )


def test_pre_commit_hook_installed_and_executable(tmp_path):
    ws = create_workspace(_cfg(tmp_path), ModelSpec(id="m1", cli="fake", model="none"))
    hook = ws / ".git" / "hooks" / "pre-commit"
    assert hook.exists() and os.access(hook, os.X_OK)
    # The baseline commit already passed through it — the scaffold is checked by the same rule.
    assert subprocess.run(["git", "log", "-1"], cwd=ws, capture_output=True).returncode == 0


def test_pre_commit_hook_rejects_oversized_blob(tmp_path):
    ws = create_workspace(_cfg(tmp_path), ModelSpec(id="m1", cli="fake", model="none"))
    (ws / "big.parquet").write_bytes(b"x" * 4096)
    done = _commit(ws, "add data", env={"CAMPAIGN_MAX_BLOB_BYTES": "1024"})
    assert done.returncode != 0
    err = done.stderr.decode()
    assert "big.parquet" in err and "--no-verify" in err


def test_pre_commit_hook_allows_small_blob_and_no_verify_escape(tmp_path):
    ws = create_workspace(_cfg(tmp_path), ModelSpec(id="m1", cli="fake", model="none"))
    (ws / "small.txt").write_bytes(b"x" * 16)
    assert _commit(ws, "small", env={"CAMPAIGN_MAX_BLOB_BYTES": "1024"}).returncode == 0
    # Weights must remain committable: the champion has to run offline (--network none).
    (ws / "weights.bin").write_bytes(b"x" * 4096)
    subprocess.run(["git", "add", "-A"], cwd=ws, check=True, capture_output=True)
    done = subprocess.run(
        ["git", "-c", "user.email=a@a", "-c", "user.name=a", "commit", "-qm", "weights", "--no-verify"],
        cwd=ws,
        capture_output=True,
        env={**os.environ, "CAMPAIGN_MAX_BLOB_BYTES": "1024"},
    )
    assert done.returncode == 0


def test_render_constitution_substitutes_all_vars(tmp_path):
    cfg = _cfg(tmp_path / "campaigns")
    text = render_constitution(cfg, cfg.models[0])
    assert "$" not in text  # no unsubstituted variable
    assert "10.0 USD" in text and "2026-07-15" in text
    assert "synth_lib" in text
    assert "test-rig" in text


def test_scaffolded_predict_anchors_paths_to_workspace_not_cwd(tmp_path, monkeypatch):
    """The price snapshot is mounted at the workspace ROOT, not inside agent/. If predict.py
    resolves `market_data/` relative to cwd (MinutePriceStore's default), the agent reads
    nothing as soon as it launches from agent/ — and burns its budget debugging OUR template.
    Regression 07-24: paths must be anchored to the file's location."""
    import importlib.util
    from datetime import date, datetime, timezone

    import pandas as pd

    from synth_lib.benchmark.campaign import BaselineSpec, CampaignConfig, ModelSpec

    cfg = CampaignConfig(
        name="cwdtest",
        objective="o",
        models=(ModelSpec(id="m1", cli="fake", model="none"),),
        budget_usd_per_model=1.0,
        deadline_hours_per_model=1.0,
        data_cutoff=date(2026, 7, 15),
        forward_window_days=1,
        baselines=(BaselineSpec(name="b", module="x"),),
        hardware="h",
        proxy_url="http://x",
        root=tmp_path / "campaigns",
    )
    ws = create_workspace(cfg, cfg.models[0])

    # Import from an ARBITRARY cwd (not the workspace): that's the point of the test.
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("scaffold_predict", ws / "agent" / "predict.py")
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)

    assert predict.PREDICTIONS_DIR.is_absolute()
    assert ws in predict.PREDICTIONS_DIR.parents, predict.PREDICTIONS_DIR

    captured: dict = {}

    class StubStore:
        def __init__(self, asset, root=None, client=None):
            captured["root"] = root

        def load_range(self, start, end):
            idx = pd.date_range(start, end, freq="1min", tz="UTC")
            return pd.DataFrame({"timestamp": idx, "close": 100.0})

    monkeypatch.setattr(predict, "MinutePriceStore", StubStore)
    start = datetime(2026, 7, 10, tzinfo=timezone.utc)
    end = datetime(2026, 7, 10, 3, tzinfo=timezone.utc)
    n = predict.generate("BTC", start, end, cadence_minutes=60)

    assert n == 3
    root = captured["root"]
    assert root is not None and Path(root).is_absolute(), f"cwd-relative root: {root}"
    assert ws in Path(root).parents, f"root outside the workspace: {root}"
    assert Path(root) == ws / "market_data" / "pyth" / "BTC" / "1m"


def _scaffold_and_import(tmp_path, monkeypatch):
    """Scaffolds a workspace and imports its agent/predict.py from an ARBITRARY cwd."""
    import importlib.util

    cfg = CampaignConfig(
        name="cwdtest2",
        objective="o",
        models=(ModelSpec(id="m1", cli="fake", model="none"),),
        budget_usd_per_model=1.0,
        deadline_hours_per_model=1.0,
        data_cutoff=date(2026, 7, 15),
        forward_window_days=1,
        baselines=(BaselineSpec(name="b", module="x"),),
        hardware="h",
        proxy_url="http://x",
        root=tmp_path / "campaigns",
    )
    ws = create_workspace(cfg, cfg.models[0])
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("scaffold_predict2", ws / "agent" / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return ws, module


def test_predict_forces_cwd_to_workspace_root(tmp_path, monkeypatch):
    """synth-lib's backtest() resolves market_data/ via cwd (_price_store without root): without
    a chdir, launching from agent/ triggers a network ingest + write against a RO mount."""
    import os

    ws, predict = _scaffold_and_import(tmp_path, monkeypatch)
    os.chdir(ws / "agent")
    predict.ensure_workspace_cwd()
    assert Path(os.getcwd()).resolve() == ws.resolve()


def test_predict_fails_clearly_outside_snapshot(tmp_path, monkeypatch):
    """A window outside the snapshot must fail EARLY with an actionable message, not trigger
    an unreadable ingest."""
    from datetime import datetime, timezone

    ws, predict = _scaffold_and_import(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        predict.check_snapshot_coverage(
            "BTC", datetime(2026, 7, 10, tzinfo=timezone.utc), datetime(2026, 7, 12, tzinfo=timezone.utc)
        )
    msg = str(exc.value)
    assert "outside snapshot" in msg and "2026-07-10" in msg and "data_cutoff" in msg


def test_predict_coverage_passes_when_partitions_present(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    ws, predict = _scaffold_and_import(tmp_path, monkeypatch)
    root = ws / "market_data" / "pyth" / "BTC" / "1m"
    root.mkdir(parents=True)
    for day in ("2026-07-10", "2026-07-11", "2026-07-12"):
        (root / f"date={day}.parquet").write_bytes(b"x")
    predict.check_snapshot_coverage(
        "BTC", datetime(2026, 7, 10, tzinfo=timezone.utc), datetime(2026, 7, 12, tzinfo=timezone.utc)
    )
