"""Harness-owned generation of forward predictions: same code for champions and baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CONTEXT_MINUTES = 7 * 24 * 60


def prompt_grid(window_start: pd.Timestamp, window_end: pd.Timestamp, cadence_minutes: int = 60) -> list[pd.Timestamp]:
    """Prompts starting in [start, end) — the last one must be able to realize before window_end + 24h,
    since the realized data is downloaded afterward by the harness.
    Filters with `t < window_end` rather than `[:-1]`: a window_end not aligned to the cadence
    (e.g. 02:30 with an hourly cadence) must not drop the last valid prompt (02:00)."""
    grid = pd.date_range(window_start, window_end, freq=f"{cadence_minutes}min", tz="UTC")
    return [t for t in grid if t < window_end]


def generate_predictions(
    simulate_fn,
    asset: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    out_dir: Path,
    price_series: pd.Series,
    cadence_minutes: int = 60,
    # 1000 is the validator's PromptConfig.num_simulations: the Synth field's CRPS was computed
    # from 1000 sampled paths, and empirical CRPS is biased upward for small N, so a lower
    # default here would unfairly penalize the champion relative to the field in the verdict.
    num_simulations: int = 1000,
    time_increment: int = 300,
    time_length: int = 86_400,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for t in prompt_grid(window_start, window_end, cadence_minutes):
        context = price_series.loc[t - pd.Timedelta(minutes=CONTEXT_MINUTES) : t]
        out = simulate_fn(
            asset=asset,
            start_time=t.isoformat(),
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            context_prices=context,
        )
        paths = [list(map(float, p)) for p in out[2:]]
        payload = {
            "start_timestamp": t.isoformat(),
            "asset": asset,
            "time_increment": time_increment,
            "time_length": time_length,
            "num_simulations": num_simulations,
            "num_steps": len(paths[0]),
            "paths": paths,
        }
        name = t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{asset}_{time_length}.json"
        (out_dir / name).write_text(json.dumps(payload))
        count += 1
    return count
