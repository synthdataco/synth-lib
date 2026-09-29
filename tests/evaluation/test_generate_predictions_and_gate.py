"""generate_predictions: the no-lookahead guarantee, file format, and the miner contract gate.

The gate (tier 1) runs a miner's raw simulate() output through the validator's OWN
response_validation_v2 — the function that accepts or rejects a live response — with a synthetic
context, no data files, no network. It is deliberately strict: it enforces the LIVE contract
(int metadata slots, 8-significant-digit prices), which is what the deployment wrapper must
produce, not what the campaign backtester tolerates.
"""

from datetime import timezone
from pathlib import Path

import json

import numpy as np
import pandas as pd
import pytest
from synth.simulation_input import SimulationInput  # type: ignore[import-untyped]
from synth.validator import response_validation_v2  # type: ignore[import-untyped]

from synth_lib.benchmark.generate_predictions import (
    context_end,
    CONTEXT_MINUTES,
    generate,
    load_minute_prices,
    load_simulate,
    prompt_grid,
    store_root,
)

UTC = timezone.utc
SCAFFOLD_MODELING = (
    Path(__file__).resolve().parents[2] / "synth_lib" / "benchmark" / "scaffold" / "workspace" / "agent" / "modeling.py"
)


def _series(start: str, days: int) -> pd.Series:
    idx = pd.date_range(start, periods=days * 24 * 60, freq="1min", tz="UTC")
    rng = np.random.default_rng(7)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(idx)))), index=idx, name="close")


def _frame(series: pd.Series) -> pd.DataFrame:
    """The OHLCV context the loader hands simulate(), from a close series."""
    closes = series.to_numpy()
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1.0, "trade_count": 1.0},
        index=series.index,
    )


# -- the no-lookahead guarantee ------------------------------------------------


def test_context_never_reaches_past_the_prompt(tmp_path):
    """THE property the verdict rests on: at prompt time t, the model sees only prices <= t,
    even though the loaded frame extends days past it."""
    series = _series("2026-07-23", 12)  # through 08-03, far beyond the window
    seen: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    def spy(asset, start_time, time_increment, time_length, num_simulations, context_prices):
        t = pd.Timestamp(start_time)
        seen.append((t, context_prices.index.min(), context_prices.index.max()))
        steps = time_length // time_increment + 1
        return (start_time, time_increment, *[[float(context_prices["close"].iloc[-1])] * steps] * num_simulations)

    n = generate(
        spy,
        "BTC",
        pd.Timestamp("2026-07-30", tz="UTC"),
        pd.Timestamp("2026-07-31", tz="UTC"),
        _frame(series),
        tmp_path,
        cadence_minutes=360,
        time_increment=300,
        time_length=86_400,
        num_simulations=3,
    )
    assert n == (4, 0) and len(seen) == 4  # 00:00, 06:00, 12:00, 18:00 — 24:00 excluded (t < window_end)
    for t, lo, hi in seen:
        # The bar labelled t closes at t + 60s and its close is the prompt's first scored point, so
        # "up to t" would already be a leak. A 24h prompt is issued 120s early, plus the bar width.
        assert hi <= t - pd.Timedelta(seconds=180), f"context leaked past the issuance: {hi} vs {t}"
        assert lo >= t - pd.Timedelta(minutes=CONTEXT_MINUTES)


def _generate_one_day(tmp_path, window_end="2026-07-30 01:00"):
    return generate(
        load_simulate(SCAFFOLD_MODELING),
        "BTC",
        pd.Timestamp("2026-07-30", tz="UTC"),
        pd.Timestamp(window_end, tz="UTC"),
        _frame(_series("2026-07-23", 9)),
        tmp_path,
        cadence_minutes=60,
        time_increment=300,
        time_length=86_400,
        num_simulations=5,
    )


def test_prediction_file_format(tmp_path):
    """One file per day, float32, shaped (prompts, simulations, steps), with an index naming the
    prompt start_times it holds — that index is what makes a re-run skippable."""
    assert _generate_one_day(tmp_path) == (1, 0)

    root = tmp_path / "BTC" / "86400_300"
    assert sorted(f.name for f in root.iterdir()) == ["date=2026-07-30.json", "date=2026-07-30.npy"]
    paths = np.load(root / "date=2026-07-30.npy")
    assert paths.shape == (1, 5, 289) and paths.dtype == np.float32
    index = json.loads((root / "date=2026-07-30.json").read_text())
    assert index["start_times"] == ["2026-07-30T00:00:00+00:00"]
    assert index["num_simulations"] == 5 and index["num_steps"] == 289


def test_a_day_already_generated_is_not_generated_again(tmp_path):
    """The point of the cache: a re-score of a window reuses the paths instead of paying for them."""
    assert _generate_one_day(tmp_path) == (1, 0)
    before = (tmp_path / "BTC" / "86400_300" / "date=2026-07-30.npy").stat().st_mtime_ns

    assert _generate_one_day(tmp_path) == (0, 1)
    assert (tmp_path / "BTC" / "86400_300" / "date=2026-07-30.npy").stat().st_mtime_ns == before


def test_a_day_missing_a_prompt_is_regenerated(tmp_path):
    """A wider window over the same day wants prompts the file does not hold; a partial day is
    never served, so an interrupted run cannot silently score fewer prompts."""
    assert _generate_one_day(tmp_path) == (1, 0)
    assert _generate_one_day(tmp_path, window_end="2026-07-30 03:00") == (3, 0)
    index = json.loads((tmp_path / "BTC" / "86400_300" / "date=2026-07-30.json").read_text())
    assert len(index["start_times"]) == 3


def test_prompt_grid_keeps_last_prompt_on_unaligned_end():
    grid = prompt_grid(pd.Timestamp("2026-07-30", tz="UTC"), pd.Timestamp("2026-07-30 02:30", tz="UTC"), 60)
    assert [t.hour for t in grid] == [0, 1, 2]  # 02:00 kept despite the unaligned end


def test_explicit_prompt_times_replace_the_grid_and_stay_inside_the_window():
    """The validator's kept requests sit at arbitrary minutes, so the list is not a grid — but it
    is still bounded by the window, and anything outside it has no realized path yet."""
    times = [pd.Timestamp(t, tz="UTC") for t in ("2026-07-29 23:50", "2026-07-30 00:04", "2026-07-30 02:31")]
    grid = prompt_grid(
        pd.Timestamp("2026-07-30", tz="UTC"), pd.Timestamp("2026-07-30 02:30", tz="UTC"), 60, prompt_times=times
    )
    assert [str(t) for t in grid] == ["2026-07-30 00:04:00+00:00"]


def test_store_root_resolves_under_prices(tmp_path):
    (tmp_path / "prices" / "BTC" / "1m").mkdir(parents=True)
    assert store_root(tmp_path, "BTC") == tmp_path / "prices" / "BTC" / "1m"
    with pytest.raises(FileNotFoundError):
        store_root(tmp_path, "XAU")


def test_load_minute_prices_raises_on_missing_partition(tmp_path):
    """A silent hole would shrink every context spanning it; the loader must refuse instead."""
    root = tmp_path / "prices" / "BTC" / "1m"
    root.mkdir(parents=True)
    idx = pd.date_range("2026-07-30", periods=1440, freq="1min", tz="UTC")
    pd.DataFrame(
        {"timestamp": idx, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "trade_count": 1.0}
    ).to_parquet(root / "date=2026-07-30.parquet")
    with pytest.raises(FileNotFoundError, match="2026-07-31"):
        load_minute_prices(
            tmp_path, "BTC", pd.Timestamp("2026-07-30", tz="UTC"), pd.Timestamp("2026-07-31 04:00", tz="UTC")
        )


# -- the miner contract gate (tier 1: synthetic context, validator's own validation) ------------


def _gate(response, sim_input: SimulationInput) -> str:
    return response_validation_v2.validate_responses(response, sim_input, process_time_str="1.0")


def test_gate_scaffold_starter_raw_output_fails_the_live_contract():
    """Documents two real gaps between the campaign contract and the LIVE one: the validator
    demands int metadata slots and prices that round-trip through 8 significant digits. Raw
    float64 output fails — which is precisely what the deployment wrapper must fix."""
    series = _series("2026-07-23", 8)
    t = series.index[-1]
    sim_input = SimulationInput(
        asset="BTC", start_time=t.isoformat(), time_increment=300, time_length=86_400, num_simulations=4
    )
    simulate = load_simulate(SCAFFOLD_MODELING)
    raw = simulate(
        asset="BTC",
        start_time=t.isoformat(),
        time_increment=300,
        time_length=86_400,
        num_simulations=4,
        context_prices=_frame(series),
    )
    verdict = _gate(raw, sim_input)
    assert verdict != "CORRECT" and "incorrect" in verdict  # iso-string start slot already fails


def test_gate_wrapped_output_passes_the_live_contract():
    """The deployment wrapper's exact obligations, proven sufficient: int-ify the two metadata
    slots and round every price to 8 significant digits."""
    series = _series("2026-07-23", 8)
    t = series.index[-1]
    sim_input = SimulationInput(
        asset="BTC", start_time=t.isoformat(), time_increment=300, time_length=86_400, num_simulations=4
    )
    simulate = load_simulate(SCAFFOLD_MODELING)
    raw = simulate(
        asset="BTC",
        start_time=t.isoformat(),
        time_increment=300,
        time_length=86_400,
        num_simulations=4,
        context_prices=_frame(series),
    )
    wrapped = (
        int(t.timestamp()),
        300,
        *[[float(f"{v:.7e}") for v in path] for path in raw[2:]],
    )
    assert _gate(wrapped, sim_input) == "CORRECT"


def test_the_context_stops_before_the_first_scored_point():
    """The store labels a bar by its OPEN time, so the bar labelled t closes at t+60s — and that
    close is real_prices[0], the first point the prompt is scored against. Ending the context at t
    hands the model the answer to its own first step. crypto-1h is issued 60s early, the 24h
    competitions 120s, and a miner cannot hold a bar that has not closed by then."""
    t = pd.Timestamp("2026-09-20T12:00:00Z")
    assert context_end(t, 3_600) == t - pd.Timedelta(seconds=120)
    assert context_end(t, 86_400) == t - pd.Timedelta(seconds=180)
    assert context_end(t, 3_600, issuance_lead=0) == t - pd.Timedelta(seconds=60)
    # never at or past t, whatever the format
    assert all(context_end(t, tl) < t for tl in (3_600, 86_400, 999))
