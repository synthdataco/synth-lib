"""Append-only local minute-price store backed by daily parquet partitions."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from synth_lib.preparation.config import (
    CONTEXT_WINDOW_MINUTES,
    MINUTES_PER_DAY,
    UTC,
    default_store_root,
    utc_datetime,
)
from synth_lib.preparation.price_client import PriceClient, build_price_client

# How far back from now the current day's fetch window has to end.
#
# Both venue providers prove a window is settled by finding a candle whose open time is strictly
# GREATER than the requested end (`saw_settled_witness` in synth's price_data_provider); without one
# they raise, and the clients turn that into "no data". Asking for the current day as
# 00:00..23:59 therefore returns NOTHING — the end is in the future, so no witness can exist — and
# the whole day persists as NaN. Two minutes guarantees at least one fully-closed candle after the
# window, one to close the last requested minute and one to witness it.
CURRENT_DAY_SETTLE_MARGIN_MINUTES = 2


class MinutePriceStore:
    """Append-only local price store backed by daily parquet partitions."""

    def __init__(self, asset: str, root: Path | None = None, client: PriceClient | None = None):
        self.asset = asset
        self.root = Path(root or default_store_root(asset)).expanduser()
        # Route by asset rather than defaulting to one venue: a store built with the wrong
        # client silently fetches nothing for assets that venue does not serve.
        self.client = client or build_price_client(asset)

    def day_path(self, day: date) -> Path:
        """Return the daily parquet path."""
        return self.root / f"date={day.isoformat()}.parquet"

    def ensure_root(self) -> None:
        """Ensure the root directory exists."""
        self.root.mkdir(parents=True, exist_ok=True)

    # -- ingestion ----------------------------------------------------------

    def ingest_range(
        self, start_day: date, end_day: date, force_refresh: bool = False, verbose: bool = True
    ) -> list[Path]:
        """Ingest an inclusive date range from the configured client."""
        self.ensure_root()
        total_days = (end_day - start_day).days + 1
        paths: list[Path] = []
        cursor = start_day
        day_num = 0
        while cursor <= end_day:
            day_num += 1
            if verbose:
                print(f"Ingesting {self.asset}: {cursor.isoformat()} (day {day_num}/{total_days})")
            paths.append(self.ingest_day(cursor, force_refresh=force_refresh))
            cursor += timedelta(days=1)
        return paths

    def ingest_day(self, day: date, force_refresh: bool = False) -> Path:
        """Fetch and persist one day of minute prices."""
        self.ensure_root()
        path = self.day_path(day)
        today = datetime.now(tz=UTC).date()
        is_final = day < today
        if path.exists() and is_final and not force_refresh:
            return path

        day_start = datetime.combine(day, time.min, tzinfo=UTC)
        day_end = datetime.combine(day, time.max, tzinfo=UTC).replace(second=0, microsecond=0)
        if not is_final:
            # The current day is still running: clamp the window to the last settled minute rather
            # than asking for 23:59, which no venue can witness yet. Without this the day comes back
            # empty and persists as 1440 NaN rows — invisible in a backtest (which reads settled days
            # only) and fatal when serving, where it silently ages every context by up to 24 hours.
            settled_end = datetime.now(tz=UTC).replace(second=0, microsecond=0) - timedelta(
                minutes=CURRENT_DAY_SETTLE_MARGIN_MINUTES
            )
            day_end = min(day_end, settled_end)

        if day_end < day_start:
            # Within the settle margin of midnight: nothing has settled today yet. Persist the empty
            # grid so the partition exists, and let the next refresh fill it.
            fetched = pd.DataFrame(columns=["timestamp", "close"])
        else:
            fetched = self.client.fetch_range(self.asset, day_start, day_end)
        expected_index = pd.date_range(day_start, periods=MINUTES_PER_DAY, freq="1min", tz="UTC")
        if not fetched.empty:
            fetched = fetched.set_index("timestamp")
        frame = pd.DataFrame(index=expected_index)
        # float NaN, not pd.NA: a scalar pd.NA makes the column object dtype,
        # which reads back as None and breaks float()/np.isfinite() consumers.
        frame["close"] = fetched["close"].reindex(expected_index) if not fetched.empty else float("nan")
        frame["source"] = getattr(self.client, "source_name", "unknown")
        frame["ingested_at"] = datetime.now(tz=UTC).replace(microsecond=0)
        frame["is_final"] = bool(is_final)
        frame = frame.reset_index(names="timestamp")
        frame.to_parquet(path, index=False)
        return path

    def refresh_recent(self, days: int = 8) -> list[Path]:
        """Refresh recent days, including the current day."""
        today = datetime.now(tz=UTC).date()
        start_day = today - timedelta(days=max(1, days))
        return self.ingest_range(start_day, today, force_refresh=True)

    # -- loading ------------------------------------------------------------

    def load_range(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Load a contiguous inclusive range from local storage."""
        start_time = utc_datetime(start_time)
        end_time = utc_datetime(end_time)
        frames: list[pd.DataFrame] = []
        cursor = start_time.date()
        while cursor <= end_time.date():
            path = self.day_path(cursor)
            if not path.exists():
                raise FileNotFoundError(f"Missing local partition: {path}")
            frames.append(pd.read_parquet(path))
            cursor += timedelta(days=1)
        frame = pd.concat(frames, ignore_index=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
        window = frame.loc[(frame["timestamp"] >= start_time) & (frame["timestamp"] <= end_time)].copy()
        if window.empty:
            raise ValueError("Requested range is empty in the local store.")
        if window["close"].isna().any():
            missing = int(window["close"].isna().sum())
            raise ValueError(f"Local store has {missing} missing minute prices in the requested range.")
        return window.reset_index(drop=True)

    def validate_range(self, start_time: datetime, end_time: datetime) -> dict[str, Any]:
        """Validate continuity and gap counts for a range."""
        frame = self.load_range(start_time, end_time)
        expected_rows = int((end_time - start_time).total_seconds() // 60) + 1
        duplicate_count = int(frame["timestamp"].duplicated().sum())
        missing_count = int(frame["close"].isna().sum())
        return {
            "expected_rows": expected_rows,
            "actual_rows": int(len(frame)),
            "duplicate_rows": duplicate_count,
            "missing_rows": missing_count,
            "is_contiguous": len(frame) == expected_rows and duplicate_count == 0,
        }

    def get_context_window(self, start_time: datetime) -> pd.Series:
        """Return the 7-day minute context ending at start_time."""
        start_time = utc_datetime(start_time)
        context_start = start_time - timedelta(minutes=CONTEXT_WINDOW_MINUTES)
        frame = self.load_range(context_start, start_time)
        expected_rows = CONTEXT_WINDOW_MINUTES + 1
        if len(frame) != expected_rows:
            raise ValueError(f"Expected {expected_rows} context rows, got {len(frame)}.")
        return pd.Series(
            frame["close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(frame["timestamp"], tz="UTC"),
            name="close",
        )

    def get_real_price_path(self, start_time: datetime) -> pd.Series:
        """Return the true 24-hour path at 5-minute resolution."""
        start_time = utc_datetime(start_time)
        frame = self.load_range(start_time, start_time + timedelta(hours=24))
        step_minutes = 5
        aligned = frame.iloc[::step_minutes].copy()
        expected_rows = (24 * 60 // step_minutes) + 1
        if len(aligned) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows in real path, got {len(aligned)}.")
        return pd.Series(
            aligned["close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(aligned["timestamp"], tz="UTC"),
            name="close",
        )

    @staticmethod
    def _day_from_path(path: Path) -> date:
        """Parse the partition date from its path."""
        return date.fromisoformat(path.stem.split("=", maxsplit=1)[1])
