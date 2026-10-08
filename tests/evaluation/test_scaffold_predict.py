"""The agent's evaluator answers the prompts the validator scored, offline, one run at a time.

backtest() reads field scores and rewards past --eval-end, so it must read them from the offline
bundle; the prompts it scores are the bundle's, at the start times the validator used; and two runs
must never score each other's predictions.
"""

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SCAFFOLD_PREDICT = (
    Path(__file__).resolve().parents[2] / "synth_lib" / "benchmark" / "scaffold" / "workspace" / "agent" / "predict.py"
)
END = datetime(2026, 9, 14, tzinfo=timezone.utc)


@pytest.fixture
def predict(tmp_path, monkeypatch):
    """The scaffold's predict.py, its workspace in tmp_path, the offline bundle set."""
    spec = importlib.util.spec_from_file_location("scaffold_predict", SCAFFOLD_PREDICT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    monkeypatch.setattr(module, "PREDICTIONS_ROOT", tmp_path / "miner_outputs" / "campaign")
    monkeypatch.setenv(module.OFFLINE_ROOT_ENV, str(tmp_path / "offline_data"))
    return module


def test_refuses_to_run_without_the_offline_bundle(predict, monkeypatch):
    monkeypatch.delenv(predict.OFFLINE_ROOT_ENV)
    with pytest.raises(SystemExit, match="live API"):
        predict.require_offline_bundle()


def test_answers_the_start_times_the_validator_scored(predict, monkeypatch):
    """Scored in the --days before --eval-end, so started up to a horizon earlier."""
    asked = []
    starts = pd.to_datetime(["2026-09-10T23:17:00Z", "2026-09-11T00:43:00Z"], utc=True)

    def scores(start, end, asset, time_length, time_increment):
        asked.append((start, end))
        return pd.DataFrame({"start_time": list(starts) * 2})  # one row per miner

    monkeypatch.setattr(predict, "get_miner_scores", scores)

    assert predict.prompt_times("BTC", END, 3, 86_400, 300) == list(starts)
    assert asked == [(END - timedelta(days=3), END)]


def test_a_grid_spans_the_same_prompts_a_horizon_earlier(predict):
    times = predict.prompt_times("BTC", END, 1, 3_600, 60, cadence_minutes=30)
    assert times[0] == END - timedelta(days=1, hours=1)
    assert times[-1] == END - timedelta(hours=1, minutes=30)


def test_no_scored_prompt_in_the_bundle_says_so(predict, monkeypatch):
    monkeypatch.setattr(predict, "get_miner_scores", lambda *a, **k: pd.DataFrame())
    with pytest.raises(SystemExit, match="no scored BTC prompt"):
        predict.prompt_times("BTC", END, 3, 86_400, 300)


def test_each_run_scores_only_its_own_predictions(predict, monkeypatch):
    """A shared directory let a concurrent run of another model overwrite the files being scored."""
    t = pd.Timestamp("2026-09-12T10:17:00Z")
    scored_dirs: list[Path] = []

    def fake_generate(asset, times, out_dir, **kw):
        (out_dir / f"{t:%Y-%m-%d_%H:%M:%SZ}_{asset}_86400.json").write_text("{}")
        return 1

    def fake_backtest(*, predictions_dir, **kw):
        scored_dirs.append(predictions_dir)
        assert [p.name for p in predictions_dir.iterdir()] == [f"{t:%Y-%m-%d_%H:%M:%SZ}_BTC_86400.json"]
        return SimpleNamespace(summary={})

    monkeypatch.setattr(predict, "prompt_times", lambda *a, **k: [t])
    monkeypatch.setattr(predict, "check_snapshot_coverage", lambda *a: None)
    monkeypatch.setattr(predict, "ensure_workspace_cwd", lambda: None)
    monkeypatch.setattr(predict, "generate", fake_generate)
    monkeypatch.setattr(predict, "backtest", fake_backtest)
    for argv in (
        ["--eval-end", "2026-09-14"],
        ["--eval-end", "2026-09-14"],
        ["--eval-end", "2026-09-14", "--keep-predictions"],
    ):
        monkeypatch.setattr(sys, "argv", ["predict.py", *argv])
        predict.main()

    assert len(set(scored_dirs)) == 3
    assert [d.exists() for d in scored_dirs] == [False, False, True]


def test_a_grid_is_answered_from_the_grid_point_before_each_prompt(predict, monkeypatch):
    tolerances = []
    monkeypatch.setattr(predict, "prompt_times", lambda *a, **k: [pd.Timestamp("2026-09-12T10:00:00Z")])
    monkeypatch.setattr(predict, "check_snapshot_coverage", lambda *a: None)
    monkeypatch.setattr(predict, "ensure_workspace_cwd", lambda: None)
    monkeypatch.setattr(predict, "generate", lambda *a, **k: 1)
    monkeypatch.setattr(
        predict,
        "backtest",
        lambda *, match_tolerance_minutes, **kw: tolerances.append(match_tolerance_minutes)
        or SimpleNamespace(summary={}),
    )
    monkeypatch.setattr(sys, "argv", ["predict.py", "--eval-end", "2026-09-14", "--cadence-minutes", "60"])
    predict.main()

    assert tolerances == [60]
