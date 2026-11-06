"""The agent's evaluator answers the prompts the validator scored, offline, one run at a time.

backtest() reads field scores and rewards past --eval-end, so it must read them from the offline
bundle; the prompts it scores are the bundle's, at the start times the validator used; and two runs
must never score each other's predictions.
"""

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from synth_lib.benchmark.verdict.run_verdict import CADENCE_MINUTES
from synth_lib.preparation.config import OHLCV_COLUMNS
from synth_lib.preparation.event_calendar.events import CALENDAR_ENV, event_frame, write_atomic

SCAFFOLD_PREDICT = (
    Path(__file__).resolve().parents[2] / "synth_lib" / "benchmark" / "scaffold" / "workspace" / "agent" / "predict.py"
)
END = datetime(2026, 9, 14, tzinfo=timezone.utc)


def _load_predict(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("scaffold_predict", SCAFFOLD_PREDICT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    monkeypatch.setattr(module, "PREDICTIONS_ROOT", tmp_path / "miner_outputs" / "campaign")
    monkeypatch.setenv(module.OFFLINE_ROOT_ENV, str(tmp_path / "offline_data"))
    return module


@pytest.fixture
def predict(tmp_path, monkeypatch):
    """The scaffold's predict.py, its workspace in tmp_path, the offline bundle set."""
    return _load_predict(tmp_path, monkeypatch)


def _with_modeling(monkeypatch, **attributes) -> list:
    """Stands in for agent/modeling.py: a simulate() recording each context, plus `attributes`."""
    seen: list = []

    def simulate(asset, start_time, time_increment, time_length, num_simulations, context_prices):
        seen.append(context_prices)
        return (start_time, time_increment, *[[1.0] * (time_length // time_increment + 1)] * num_simulations)

    modeling = ModuleType("modeling")
    modeling.simulate = simulate
    for name, value in attributes.items():
        setattr(modeling, name, value)
    monkeypatch.setitem(sys.modules, "modeling", modeling)
    return seen


def _snapshot(tmp_path, asset: str, first: str, last: str) -> Path:
    root = tmp_path / "market_data" / "prices" / asset / "1m"
    root.mkdir(parents=True)
    for day in pd.date_range(first, last, freq="D", tz="UTC"):
        idx = pd.date_range(day, periods=1440, freq="1min", tz="UTC")
        pd.DataFrame({"timestamp": idx, **{c: 1.0 for c in OHLCV_COLUMNS}}).to_parquet(
            root / f"date={day.date()}.parquet"
        )
    return root


@pytest.mark.parametrize("attributes, days", [({"CONTEXT_DAYS": 14}, 14), ({}, 7)])
def test_backtests_hand_the_champion_its_context_days(tmp_path, monkeypatch, attributes, days):
    seen = _with_modeling(monkeypatch, **attributes)
    predict = _load_predict(tmp_path, monkeypatch)
    assert predict.CONTEXT_DAYS == days
    _snapshot(tmp_path, "BTC", "2026-08-01", "2026-08-20")
    t = pd.Timestamp("2026-08-20", tz="UTC")
    predict.generate("BTC", [t], tmp_path / "out", num_simulations=2)
    assert seen[0].index[0] == t - pd.Timedelta(days=days)
    assert len(seen[0]) == days * 1440 - 2  # the verdict's slice: up to the bar labelled t - 3 min


def test_a_context_reaching_before_the_snapshot_is_reported(tmp_path, monkeypatch, capsys):
    seen = _with_modeling(monkeypatch)
    predict = _load_predict(tmp_path, monkeypatch)
    _snapshot(tmp_path, "SPCX", "2026-08-10", "2026-08-20")
    times = [pd.Timestamp("2026-08-12", tz="UTC"), pd.Timestamp("2026-08-20", tz="UTC")]
    predict.generate("SPCX", times, tmp_path / "out", num_simulations=2)
    assert seen[0].loc[:"2026-08-09 23:59", "close"].isna().all()
    assert "warning: SPCX contexts start with up to 5.0 days of NaN" in capsys.readouterr().out


@pytest.mark.parametrize("value", [2, 31, True, 14.0])
def test_backtests_refuse_context_days_outside_the_bounds(tmp_path, monkeypatch, value):
    _with_modeling(monkeypatch, CONTEXT_DAYS=value)
    with pytest.raises(ValueError, match="from 3 to 30"):
        _load_predict(tmp_path, monkeypatch)


def test_a_listing_inside_the_context_is_not_missing_but_a_hole_is(predict, tmp_path):
    """The days before an asset's first partition are the context's NaN head; a missing day after it
    is a hole the run must not paper over."""
    root = _snapshot(tmp_path, "SPCX", "2026-08-10", "2026-08-20")
    start, end = datetime(2026, 8, 5, tzinfo=timezone.utc), datetime(2026, 8, 20, tzinfo=timezone.utc)
    first_prompt = start + timedelta(days=7)
    predict.check_snapshot_coverage("SPCX", start, end, first_prompt)
    (root / "date=2026-08-12.parquet").unlink()
    with pytest.raises(SystemExit, match=r"for SPCX: missing partitions \['2026-08-12'\]"):
        predict.check_snapshot_coverage("SPCX", start, end, first_prompt)


def test_prompts_before_the_first_partition_are_missing(predict, tmp_path):
    """Only the context's head is exempt: a prompt on a day the snapshot does not hold stops the run."""
    _snapshot(tmp_path, "SPCX", "2026-08-10", "2026-08-20")
    first_prompt, end = datetime(2026, 8, 4, tzinfo=timezone.utc), datetime(2026, 8, 20, tzinfo=timezone.utc)
    with pytest.raises(SystemExit, match=r"for SPCX: missing partitions \['2026-08-04', '2026-08-05'"):
        predict.check_snapshot_coverage("SPCX", first_prompt - timedelta(days=30), end, first_prompt)


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


def _run_main(predict, monkeypatch, argv):
    """main() up to scoring: the prompt_times call and the scorer it reached."""
    seen = {}

    def times(*a, **k):
        seen["times"] = k
        return [pd.Timestamp("2026-09-12T10:00:00Z")]

    monkeypatch.setattr(predict, "prompt_times", times)
    monkeypatch.setattr(predict, "check_snapshot_coverage", lambda *a: None)
    monkeypatch.setattr(predict, "ensure_workspace_cwd", lambda: None)
    monkeypatch.setattr(predict, "generate", lambda *a, **k: 1)

    def fake_backtest(**kw):
        seen["scorer"] = "backtest"
        return SimpleNamespace(summary={})

    def fake_score_crps(**kw):
        seen["scorer"] = "score_crps"
        return SimpleNamespace(summary={})

    monkeypatch.setattr(predict, "backtest", fake_backtest)
    monkeypatch.setattr(predict, "score_crps", fake_score_crps)
    monkeypatch.setattr(sys, "argv", ["predict.py", "--eval-end", "2026-09-14", *argv])
    predict.main()
    return seen


def test_crps_only_needs_no_field_bundle_and_answers_the_scored_density(predict, monkeypatch):
    monkeypatch.delenv(predict.OFFLINE_ROOT_ENV)
    seen = _run_main(predict, monkeypatch, ["--crps-only"])
    assert seen["scorer"] == "score_crps"
    assert seen["times"] == {"cadence_minutes": 60}
    one_hour = ["--crps-only", "--time-length", "3600", "--time-increment", "60"]
    assert _run_main(predict, monkeypatch, one_hour)["times"]["cadence_minutes"] == 10


def test_the_crps_only_grid_is_the_verdicts(predict):
    assert predict.CRPS_ONLY_CADENCE_MINUTES == CADENCE_MINUTES


def test_a_grid_spans_the_same_prompts_a_horizon_earlier(predict):
    times = predict.prompt_times("BTC", END, 1, 3_600, 60, cadence_minutes=30)
    assert times[0] == END - timedelta(days=1, hours=1)
    assert times[-1] == END - timedelta(hours=1, minutes=30)


def test_no_scored_prompt_in_the_bundle_says_so(predict, monkeypatch):
    monkeypatch.setattr(predict, "get_miner_scores", lambda *a, **k: pd.DataFrame())
    with pytest.raises(SystemExit, match="no scored BTC prompt"):
        predict.prompt_times("BTC", END, 3, 86_400, 300)


@pytest.mark.parametrize("scorer, flags", [("backtest", []), ("score_crps", ["--crps-only"])])
def test_each_run_scores_only_its_own_predictions(predict, monkeypatch, scorer, flags):
    """A shared directory let a concurrent run of another model overwrite the files being scored."""
    t = pd.Timestamp("2026-09-12T10:17:00Z")
    scored_dirs: list[Path] = []

    def fake_generate(asset, times, out_dir, **kw):
        (out_dir / f"{t:%Y-%m-%d_%H:%M:%SZ}_{asset}_86400.json").write_text("{}")
        return 1

    def fake_score(*, predictions_dir, **kw):
        scored_dirs.append(predictions_dir)
        assert [p.name for p in predictions_dir.iterdir()] == [f"{t:%Y-%m-%d_%H:%M:%SZ}_BTC_86400.json"]
        return SimpleNamespace(summary={})

    monkeypatch.setattr(predict, "prompt_times", lambda *a, **k: [t])
    monkeypatch.setattr(predict, "check_snapshot_coverage", lambda *a: None)
    monkeypatch.setattr(predict, "ensure_workspace_cwd", lambda: None)
    monkeypatch.setattr(predict, "generate", fake_generate)
    monkeypatch.setattr(predict, scorer, fake_score)
    for argv in (
        ["--eval-end", "2026-09-14", *flags],
        ["--eval-end", "2026-09-14", *flags],
        ["--eval-end", "2026-09-14", "--keep-predictions", *flags],
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


def test_the_event_split_compares_event_prompts_with_the_rest(predict, tmp_path, monkeypatch, capsys):
    events = [
        {"time_utc": pd.Timestamp("2026-07-14 12:30", tz="UTC"), "event": "CPI", "scope": "all", "source": "fred:CPI"},
        {"time_utc": pd.Timestamp("2026-07-14 12:40", tz="UTC"), "event": "EARNINGS", "scope": "TSLAX", "source": "x"},
    ]
    write_atomic(event_frame(events), tmp_path / "events.parquet")
    monkeypatch.setenv(CALENDAR_ENV, str(tmp_path / "events.parquet"))
    first, second = pd.Timestamp("2026-07-14 12:00", tz="UTC"), pd.Timestamp("2026-07-14 14:00", tz="UTC")
    prompts = pd.DataFrame(
        {
            "miner_id": [predict.CANDIDATE_ID, 7, predict.CANDIDATE_ID, 7],
            "asset": "BTC",
            "start_time": [first, first, second, second],
            "scored_time": [first + pd.Timedelta(hours=1)] * 2 + [second + pd.Timedelta(hours=1)] * 2,
            "crps": [12.0, 10.0, 10.5, 10.0],
        }
    )
    predict.print_event_split(prompts, "BTC", 3600)
    out = capsys.readouterr().out
    assert "no event" in out and "+5.00%" in out  # the 14:00 prompt holds no event
    assert "CPI" in out and "+20.00%" in out  # the 12:00 prompt holds the 12:30 release
    assert "EARNINGS" not in out  # TSLAX's event, not BTC's


def test_the_event_split_of_an_empty_calendar_is_all_no_event(predict, tmp_path, monkeypatch, capsys):
    write_atomic(event_frame([]), tmp_path / "events.parquet")
    monkeypatch.setenv(CALENDAR_ENV, str(tmp_path / "events.parquet"))
    start = pd.Timestamp("2026-07-14 12:00", tz="UTC")
    prompts = pd.DataFrame(
        {
            "miner_id": [predict.CANDIDATE_ID, 7],
            "asset": "BTC",
            "start_time": [start, start],
            "scored_time": [start + pd.Timedelta(hours=1)] * 2,
            "crps": [12.0, 10.0],
        }
    )
    predict.print_event_split(prompts, "BTC", 3600)
    assert "no event" in capsys.readouterr().out
