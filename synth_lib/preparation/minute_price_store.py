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
    utc_datetime,
)
from synth_lib.preparation.price_client import PriceClient
from synth_lib.preparation.pyth_client import PythHistoryClient


class MinutePriceStore:
    """Append-only local price store backed by daily parquet partitions."""

    def __init__(self, asset: str, root: Path | None = None, client: PriceClient | None = None):
        self.asset = asset
        self.root = Path(root or Path(f"market_data/pyth/{asset}/1m")).expanduser()
        self.client = client or PythHistoryClient()

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
        fetched = self.client.fetch_range(self.asset, day_start, day_end)
        expected_index = pd.date_range(day_start, periods=MINUTES_PER_DAY, freq="1min", tz="UTC")
        if not fetched.empty:
            fetched = fetched.set_index("timestamp")
        frame = pd.DataFrame(index=expected_index)
        # float NaN, not pd.NA: a scalar pd.NA makes the column object dtype,
        # which reads back as None and breaks float()/np.isfinite() consumers.
        frame["close"] = (
            fetched["close"].reindex(expected_index) if not fetched.empty else float("nan")
        )
        frame["source"] = getattr(self.client, "source_name", "pyth")
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
