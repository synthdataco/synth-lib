"""Turning raw scores and prices into per-prompt scoring inputs."""

from __future__ import annotations

import bisect
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synth_lib.backtester.config import (
    PREDICTION_MATCH_TOLERANCE_MINUTES,
    UTC,
    _offline_root,
)
from synth_lib.preparation.realized_path_store import (
    RealizedPathStore,
    prefetch_realized_paths,
)



def _slice_real_prices(
    prices: pd.DataFrame,
    start_time: Any,
    time_length: int,
    time_increment: int,
) -> list[float]:
    """Slice a prompt's realized prices out of the local minute frame.

    Raises on a length mismatch (window doesn't cover the prompt — unrepairable);
    missing minutes come back as NaN for _fill_gaps_from_realized_paths.
    """
    step_minutes = time_increment // 60
    window = prices.loc[
        start_time : start_time + pd.Timedelta(seconds=time_length) : step_minutes
    ].iloc[:, 0]
    # Coerce: older object-dtype partitions read back as None, which np.isfinite rejects.
    real_prices = pd.to_numeric(window, errors="coerce").tolist()
    expected_steps = (time_length // time_increment) + 1
    if len(real_prices) != expected_steps:
        raise ValueError(
            f"Price data length mismatch at {start_time}: "
            f"got {len(real_prices)} prices, expected {expected_steps}"
        )
    return real_prices


def _has_missing_prices(real_prices: list[float]) -> bool:
    """True when any price is missing or non-finite. Coerces None to NaN first."""
    coerced = pd.to_numeric(pd.Series(real_prices), errors="coerce")
    return not bool(np.isfinite(coerced).all())


def _fill_gaps_from_realized_paths(
    prompts: list[dict[str, Any]],
    log_prefix: str,
) -> int:
    """Score NaN-holed prompts from the validator's realized path instead.

    Hyperliquid retains ~3.5 days of minute candles, so older days ingest as NaN
    for HYPE and every commodity/equity. Mutates `prompts` in place, returns the
    number substituted. NaN inside a realized path is left as-is.
    """
    gaps = [
        prompt
        for prompt in prompts
        if prompt["real_prices"] and _has_missing_prices(prompt["real_prices"])
    ]
    if not gaps:
        return 0

    by_prompt_config: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for prompt in gaps:
        key = (prompt["asset"], prompt["time_length"], prompt["time_increment"])
        by_prompt_config.setdefault(key, []).append(prompt)

    filled = 0
    for (asset, time_length, time_increment), group in by_prompt_config.items():
        store = RealizedPathStore(asset, time_length, time_increment)
        if _offline_root() is None:
            prefetch_realized_paths(store, [p["start_time"] for p in group], verbose=False)
        expected_steps = (time_length // time_increment) + 1
        for prompt in group:
            series = store.get(prompt["start_time"])
            if series is None or len(series) != expected_steps:
                continue
            prompt["real_prices"] = series.to_list()
            filled += 1

    print(
        f"{log_prefix} filled {filled}/{len(gaps)} price-gap prompts "
        f"from validator realized paths",
        flush=True,
    )
    if filled < len(gaps):
        warnings.warn(
            f"{len(gaps) - filled} of {len(gaps)} price-gap prompts have no stored "
            "realized path either, and will score as NaN CRPS.",
            UserWarning,
            stacklevel=2,
        )
    return filled

def _parse_prediction_filename_time(path: Path) -> datetime | None:
    """Extract start_time from a prediction filename like 2026-03-23_00:01:00Z_BTC_86400.json."""
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    time_str = f"{parts[0]}_{parts[1]}"
    return datetime.strptime(time_str, "%Y-%m-%d_%H:%M:%SZ").replace(tzinfo=UTC)


@dataclass(frozen=True)
class PredictionRef:
    """Where one prompt's paths live. `row` indexes into a day file; None means the
    file is the prompt, which is how predictions were written before the day layout."""

    path: Path
    row: int | None = None


PredictionIndex = dict[tuple[str, int], dict[pd.Timestamp, PredictionRef]]


def build_prediction_index(root: Path) -> PredictionIndex:
    """Map (asset, time_length) -> {start_time: PredictionRef} over a predictions tree.

    Reads the day layout written by generate_predictions
    (`<asset>/<time_length>_<time_increment>/date=*.npy` beside a `.json` index) and
    the one-JSON-per-prompt trees that predate it, so an archived champion still scores.
    """
    index: PredictionIndex = {}
    for day_index in sorted(root.glob("*/*/date=*.json")):
        meta = json.loads(day_index.read_text())
        paths_file = day_index.with_suffix(".npy")
        if not paths_file.exists():
            continue  # an index without its paths is an interrupted write, not a prompt
        key = (meta["asset"], meta["time_length"])
        for row, start in enumerate(meta["start_times"]):
            index.setdefault(key, {})[pd.Timestamp(start)] = PredictionRef(
                paths_file, row
            )
    for prompt_file in sorted(root.glob("**/*.json")):
        start = _parse_prediction_filename_time(prompt_file)
        if start is None or prompt_file.name.startswith("_"):
            continue
        parts = prompt_file.stem.split("_")
        key = (parts[2], int(parts[3]))
        index.setdefault(key, {})[pd.Timestamp(start)] = PredictionRef(prompt_file)
    return index


def _find_prediction_file(
    index: PredictionIndex,
    start_time: datetime,
    asset: str,
    time_length: int,
    tolerance_minutes: int = PREDICTION_MATCH_TOLERANCE_MINUTES,
) -> PredictionRef | None:
    """The closest prediction to start_time for this (asset, time_length), or None.

    start_time is approximate (derived from scored_time - time_length), so the match is
    nearest within tolerance_minutes rather than exact. Bisects the sorted start_times:
    a window holds thousands of prompts and this runs once per prompt.
    """
    prompts = index.get((asset, time_length))
    if not prompts:
        return None
    starts = sorted(prompts)
    target = pd.Timestamp(start_time)
    position = bisect.bisect_left(starts, target)
    best: pd.Timestamp | None = None
    best_delta = timedelta(minutes=tolerance_minutes)
    for candidate in starts[max(0, position - 1) : position + 1]:
        delta = abs(target - candidate).to_pytimedelta()
        if delta < best_delta:
            best_delta, best = delta, candidate
    return prompts[best] if best is not None else None
