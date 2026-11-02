"""Day partitions of the field data an offline bundle holds, under its root:

    miner_scores/{slug}/{asset}/date=YYYY-MM-DD.parquet   # rows by scored_time
    rewards_history/{slug}/date=YYYY-MM-DD.parquet        # rows by updated_at
    miner_pool_usd/date=YYYY-MM-DD.parquet                # rows by date

A day fetched whole once it had settled is complete. A day fetched before that, or only up to the
end a run asked for, is kept as date=YYYY-MM-DD.partial.parquet and fetched again by the next run.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

# Imports nothing from synth-lib, so the campaign orchestrator can read the layout without pulling
# the backtester in.
UTC = timezone.utc

SCORES_DIR = "miner_scores"
REWARDS_DIR = "rewards_history"
POOL_DIR = "miner_pool_usd"
# How long after a day ends its scores and reward rounds are all published.
SETTLE = timedelta(hours=1)


def scores_dir(root: Path, slug: str, asset: str) -> Path:
    return root / SCORES_DIR / slug / asset


def rewards_dir(root: Path, slug: str) -> Path:
    return root / REWARDS_DIR / slug


def pool_dir(root: Path) -> Path:
    return root / POOL_DIR


def day_path(directory: Path, day: date, partial: bool = False) -> Path:
    return directory / f"date={day.isoformat()}{'.partial' if partial else ''}.parquet"


def _days(start: datetime, end: datetime) -> list[date]:
    """The days [start, end) touches."""
    first, last = start.date(), (end - timedelta(microseconds=1)).date()
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def read_days(directory: Path, start: datetime, end: datetime) -> pd.DataFrame | None:
    """The partitions of the days [start, end) touches, a complete one over a partial one.

    None when the directory does not exist; an empty frame when it holds none of those days.
    """
    if not directory.is_dir():
        return None
    frames = []
    for day in _days(start, end):
        complete, partial = day_path(directory, day), day_path(directory, day, partial=True)
        if complete.exists():
            frames.append(pd.read_parquet(complete))
        elif partial.exists():
            frames.append(pd.read_parquet(partial))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def refresh_days(
    directory: Path,
    start: datetime,
    end: datetime,
    fetch: Callable[[datetime, datetime], pd.DataFrame],
    time_column: str,
    force: bool = False,
    now: datetime | None = None,
) -> int:
    """Fetches each day [start, end) touches that has no complete partition, every one with `force`.

    A day's fetch stops at `end`. Returns how many days were fetched.
    """
    now = now or datetime.now(UTC)
    directory.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for day in _days(start, end):
        complete, partial = day_path(directory, day), day_path(directory, day, partial=True)
        if complete.exists() and not force:
            continue
        day_start = datetime.combine(day, time.min, tzinfo=UTC)
        day_end = day_start + timedelta(days=1)
        fetch_end = min(day_end, end)
        frame = fetch(day_start, fetch_end)
        if not frame.empty:
            stamps = pd.to_datetime(frame[time_column], utc=True)
            frame = frame.loc[(stamps >= day_start) & (stamps < fetch_end)]
        whole = fetch_end == day_end and now >= day_end + SETTLE
        frame.to_parquet(complete if whole else partial, index=False)
        if whole:
            partial.unlink(missing_ok=True)
        fetched += 1
    return fetched
