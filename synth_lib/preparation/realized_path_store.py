"""Prompt-keyed cache of the realized price paths the validator scored against."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from synth_lib.preparation.config import (
    PROMPT_START_MATCH_TOLERANCE_MINUTES,
    SYNTH_API_REQUEST_INTERVAL_SECONDS,
    UTC,
)
from synth_lib.preparation.validator_api import (
    get_prompt_start_times,
    get_realized_path,
    snap_to_prompt_start,
)


class RealizedPathStore:
    """Local cache of validator realized paths, keyed by prompt.

    Separate from MinutePriceStore and not a PriceClient: prompt start_times are
    irregular, so these cannot be folded into a minute grid. Partitioned by the
    prompt's start date under
    market_data/realized/{asset}/{time_length}_{time_increment}/.
    """

    def __init__(
        self,
        asset: str,
        time_length: int,
        time_increment: int,
        root: Path | None = None,
    ):
        self.asset = asset
        self.time_length = time_length
        self.time_increment = time_increment
        self.root = Path(
            root or Path(f"market_data/realized/{asset}/{time_length}_{time_increment}")
        ).expanduser()
        self._days: dict[date, dict[pd.Timestamp, list[float]]] = {}

    def day_path(self, day: date) -> Path:
        """Return the daily parquet path."""
        return self.root / f"date={day.isoformat()}.parquet"

    # -- storage ------------------------------------------------------------

    def _load_day(self, day: date) -> dict[pd.Timestamp, list[float]]:
        """Return the day's cached paths keyed by exact start_time."""
        if day in self._days:
            return self._days[day]
        entries: dict[pd.Timestamp, list[float]] = {}
        path = self.day_path(day)
        if path.exists():
            frame = pd.read_parquet(path)
            timestamps = pd.to_datetime(frame["start_time"], utc=True)
            for timestamp, prices in zip(timestamps, frame["prices"]):
                entries[timestamp] = list(prices)
        self._days[day] = entries
        return entries

    def _write_day(self, day: date, entries: dict[pd.Timestamp, list[float]]) -> Path:
        """Persist the day's paths, replacing the partition."""
        self.root.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            {
                "start_time": list(entries),
                "prices": [list(prices) for prices in entries.values()],
            }
        )
        frame["start_time"] = pd.to_datetime(frame["start_time"], utc=True)
        frame = frame.sort_values("start_time").reset_index(drop=True)
        frame["fetched_at"] = datetime.now(tz=UTC).replace(microsecond=0)
        path = self.day_path(day)
        frame.to_parquet(path, index=False)
        return path

    # -- ingestion ----------------------------------------------------------

    def prefetch(
        self,
        prompt_start_times: list[pd.Timestamp],
        verbose: bool = True,
    ) -> int:
        """Fetch and cache the paths for these exact prompt starts.

        Skips already-cached prompts so an interrupted backfill resumes. Returns
        the number newly written. Paced under the endpoint's rate limit.
        """
        by_day: dict[date, list[pd.Timestamp]] = {}
        for start_time in prompt_start_times:
            by_day.setdefault(pd.Timestamp(start_time).date(), []).append(start_time)

        fetched = 0
        for day in sorted(by_day):
            entries = self._load_day(day)
            missing = [t for t in sorted(by_day[day]) if t not in entries]
            if not missing:
                continue
            if verbose:
                print(
                    f"Fetching {len(missing)} realized paths for {self.asset} "
                    f"{self.time_length}s on {day.isoformat()}",
                    flush=True,
                )
            added = 0
            for start_time in missing:
                series = get_realized_path(
                    self.asset, start_time, self.time_length, self.time_increment
                )
                time.sleep(SYNTH_API_REQUEST_INTERVAL_SECONDS)
                if series is None:
                    continue
                entries[start_time] = series.to_list()
                added += 1
            if added:
                self._write_day(day, entries)
            fetched += added
        return fetched

    # -- loading ------------------------------------------------------------

    def get(
        self,
        approx_start_time: datetime,
        tolerance_minutes: int = PROMPT_START_MATCH_TOLERANCE_MINUTES,
    ) -> pd.Series | None:
        """Return the cached path for the prompt nearest approx_start_time.

        Local-only, never fetches. Searches the previous day's partition too: an
        approximation just past midnight belongs to a prompt started before it.
        """
        target = pd.Timestamp(approx_start_time)
        if target.tzinfo is None:
            target = target.tz_localize("UTC")
        candidates: dict[pd.Timestamp, list[float]] = {}
        for day in (target.date() - timedelta(days=1), target.date()):
            candidates.update(self._load_day(day))
        match = snap_to_prompt_start(target, sorted(candidates), tolerance_minutes)
        if match is None:
            return None
        prices = candidates[match]
        index = pd.date_range(
            start=match, periods=len(prices), freq=pd.Timedelta(seconds=self.time_increment)
        )
        return pd.Series(prices, index=index, dtype=float, name="close")


def prefetch_realized_paths(
    store: RealizedPathStore,
    approx_start_times: list[datetime],
    tolerance_minutes: int = PROMPT_START_MATCH_TOLERANCE_MINUTES,
    verbose: bool = True,
) -> dict[pd.Timestamp, pd.Timestamp]:
    """Resolve approximate prompt starts to exact ones and cache their paths.

    Fetches only the prompts asked for — the issued set is ~10x larger. Returns
    the approximate -> exact mapping, omitting starts with no match in tolerance.
    """
    approximations = sorted({pd.Timestamp(t) for t in approx_start_times})
    if not approximations:
        return {}
    pad = timedelta(minutes=tolerance_minutes)
    prompt_start_times = get_prompt_start_times(
        store.asset,
        store.time_length,
        store.time_increment,
        approximations[0] - pad,
        approximations[-1] + pad,
    )
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for approximation in approximations:
        match = snap_to_prompt_start(approximation, prompt_start_times, tolerance_minutes)
        if match is not None:
            mapping[approximation] = match
    store.prefetch(sorted(set(mapping.values())), verbose=verbose)
    return mapping
