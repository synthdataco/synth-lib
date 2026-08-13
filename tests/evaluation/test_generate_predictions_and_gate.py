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
        return (start_time, time_increment, *[[float(context_prices.iloc[-1])] * steps] * num_simulations)

    n = generate(
        spy,
        "BTC",
        pd.Timestamp("2026-07-30", tz="UTC"),
        pd.Timestamp("2026-07-31", tz="UTC"),
        series,
        tmp_path,
        cadence_minutes=360,
        time_increment=300,
        time_length=86_400,
        num_simulations=3,
    )
    assert n == 4 == len(seen)  # 00:00, 06:00, 12:00, 18:00 — 24:00 excluded (t < window_end)
    for t, lo, hi in seen:
        assert hi <= t, f"context leaked past the prompt: {hi} > {t}"
        assert lo >= t - pd.Timedelta(minutes=CONTEXT_MINUTES)


def test_prediction_file_format(tmp_path):
    series = _series("2026-07-23", 9)
    simulate = load_simulate(SCAFFOLD_MODELING)
    generate(
        simulate,
        "BTC",
        pd.Timestamp("2026-07-30", tz="UTC"),
        pd.Timestamp("2026-07-30 01:00", tz="UTC"),
        series,
        tmp_path,
        cadence_minutes=60,
        time_increment=300,
        time_length=86_400,
        num_simulations=5,
    )
    files = list(tmp_path.glob("*.json"))
    assert [f.name for f in files] == ["2026-07-30_00:00:00Z_BTC_86400.json"]
    payload = json.loads(files[0].read_text())
    assert payload["num_steps"] == 289 and len(payload["paths"]) == 5
    assert all(len(p) == 289 for p in payload["paths"])


def test_prompt_grid_keeps_last_prompt_on_unaligned_end():
    grid = prompt_grid(pd.Timestamp("2026-07-30", tz="UTC"), pd.Timestamp("2026-07-30 02:30", tz="UTC"), 60)
    assert [t.hour for t in grid] == [0, 1, 2]  # 02:00 kept despite the unaligned end


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
    pd.DataFrame({"timestamp": idx, "close": 1.0}).to_parquet(root / "date=2026-07-30.parquet")
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
        context_prices=series,
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
        context_prices=series,
    )
    wrapped = (
        int(t.timestamp()),
        300,
        *[[float(f"{v:.7e}") for v in path] for path in raw[2:]],
    )
    assert _gate(wrapped, sim_input) == "CORRECT"
