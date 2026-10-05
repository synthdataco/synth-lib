"""The agent's evaluator must not let backtest() read past the snapshot.

backtest() pads the window it is given: scores to the end of the --eval-end day, rewards a day
further, prices 26 h past the last prompt's start. Whatever the snapshot lacks it fetches live, so an
--eval-end at the data cutoff reaches into the evaluation window.
"""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCAFFOLD_PREDICT = (
    Path(__file__).resolve().parents[2] / "synth_lib" / "benchmark" / "scaffold" / "workspace" / "agent" / "predict.py"
)


@pytest.fixture
def predict(tmp_path, monkeypatch):
    """The scaffold's predict.py, reading a snapshot whose last day is 2026-09-14."""
    spec = importlib.util.spec_from_file_location("scaffold_predict", SCAFFOLD_PREDICT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    root = module.store_root("BTC")
    root.mkdir(parents=True)
    for day in ("2026-09-13", "2026-09-14"):
        (root / f"date={day}.parquet").write_bytes(b"")
    return module


@pytest.mark.parametrize("time_length, latest", [(86_400, "2026-09-11"), (3_600, "2026-09-12")])
def test_eval_end_stops_a_horizon_and_26h_before_the_snapshot_ends(predict, time_length, latest):
    allowed = datetime.fromisoformat(latest).replace(tzinfo=timezone.utc)
    predict.check_backtest_padding("BTC", allowed, time_length)
    with pytest.raises(SystemExit, match=f"Use --eval-end {latest} or earlier"):
        predict.check_backtest_padding("BTC", allowed + timedelta(days=1), time_length)


def test_eval_end_at_the_data_cutoff_is_refused(predict):
    """The argument the constitution used to call legal, and campaign-4's agents used."""
    with pytest.raises(SystemExit, match="too late"):
        predict.check_backtest_padding("BTC", datetime(2026, 9, 14, tzinfo=timezone.utc), 86_400)
