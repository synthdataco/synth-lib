"""Multi-asset Pyth data ingestion and local storage for Synth Subnet miners."""

from __future__ import annotations

import hashlib
import time as _time
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Pyth symbol mapping — pulled dynamically from synth-subnet
# ---------------------------------------------------------------------------
PYTH_HISTORY_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"

SYNTHDATA_API_BASE = "https://api.synthdata.co"

from synth.validator.price_data_provider import PriceDataProvider

PYTH_SYMBOLS: dict[str, str] = dict(PriceDataProvider.PYTH_SYMBOL_MAP)
HYPERLIQUID_SYMBOLS: dict[str, str] = dict(PriceDataProvider.HYPERLIQUID_SYMBOL_MAP)
ALL_SYMBOLS: dict[str, str] = {**PYTH_SYMBOLS, **HYPERLIQUID_SYMBOLS}

MINUTES_PER_DAY = 24 * 60
CONTEXT_WINDOW_MINUTES = 7 * 24 * 60
DEFAULT_TOTAL_MONTHS = 15
DEFAULT_HELDOUT_MONTHS = 0


def _utc_datetime(value: datetime) -> datetime:
    """Return a UTC datetime with second precision."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Price client protocol
# ---------------------------------------------------------------------------


class PriceClient(Protocol):
    """Structural interface for minute-price fetchers."""

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Pyth API client
# ---------------------------------------------------------------------------


class PythHistoryClient:
    """Thin client for the Pyth minute-price endpoint with retry logic."""

    def __init__(self, timeout_seconds: int = 30, max_retries: int = 3):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch a range of minute closes from Pyth with exponential backoff."""
        start_time = _utc_datetime(start_time)
        end_time = _utc_datetime(end_time)
        if asset not in PYTH_SYMBOLS:
            raise ValueError(f"Unsupported asset: {asset}. Supported: {list(PYTH_SYMBOLS.keys())}")
        params = {
            "symbol": PYTH_SYMBOLS[asset],
            "resolution": 1,
            "from": int(start_time.timestamp()),
            "to": int(end_time.timestamp()),
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(PYTH_HISTORY_URL, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                timestamps = payload.get("t", [])
                closes = payload.get("c", [])
                if len(timestamps) != len(closes):
                    raise ValueError("Pyth response has mismatched timestamps and closes.")
                if not timestamps:
                    return pd.DataFrame(columns=["timestamp", "close"])
                frame = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                        "close": pd.Series(closes, dtype="float64"),
                    }
                )
                return frame.sort_values("timestamp").drop_duplicates("timestamp")
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    delay = 2 ** (attempt + 1)
                    _time.sleep(delay)
        raise RuntimeError(f"Pyth API failed after {self.max_retries} attempts for {asset}") from last_exc


# ---------------------------------------------------------------------------
# Hyperliquid API client (delegates to synth.validator.price_data_provider)
# ---------------------------------------------------------------------------


class HyperliquidClient:
    """Wrapper around synth's PriceDataProvider matching PythHistoryClient's interface."""

    def __init__(self) -> None:
        self._provider = PriceDataProvider()

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        start_time = _utc_datetime(start_time)
        end_time = _utc_datetime(end_time)
        if asset not in HYPERLIQUID_SYMBOLS:
            raise ValueError(f"Unsupported Hyperliquid asset: {asset}")

        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        closes = self._provider.download_hyperliquid_price_data(
            beginning=start_ts,
            end=end_ts,
            symbol=asset,
            time_increment=60,
        )
        if not closes:
            return pd.DataFrame(columns=["timestamp", "close"])

        timestamps = pd.date_range(start_time, end_time, freq="1min", tz="UTC")
        if len(timestamps) != len(closes):
            raise RuntimeError(
                f"Hyperliquid timestamp/close length mismatch for {asset}: "
                f"{len(timestamps)} vs {len(closes)}"
            )
        frame = pd.DataFrame({"timestamp": timestamps, "close": pd.Series(closes, dtype="float64")})
        return frame.dropna(subset=["close"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# MinutePriceStore — asset-agnostic daily parquet store
# ---------------------------------------------------------------------------


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
        """Ingest an inclusive date range from Pyth."""
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
        frame["close"] = fetched["close"].reindex(expected_index) if not fetched.empty else pd.NA
        frame["source"] = "pyth"
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
        start_time = _utc_datetime(start_time)
        end_time = _utc_datetime(end_time)
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
        start_time = _utc_datetime(start_time)
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
        start_time = _utc_datetime(start_time)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_date_range(total_months: int, heldout_months: int) -> tuple[date, date]:
    """Compute (start_day, end_day) for the download window."""
    utc_now = datetime.now(tz=UTC)
    anchored = pd.Timestamp(utc_now).floor("D") - pd.DateOffset(months=heldout_months) - pd.Timedelta(days=1)
    end_day = anchored.date()
    start_day = (pd.Timestamp(end_day, tz="UTC") + pd.Timedelta(days=1) - pd.DateOffset(months=total_months)).date()
    return start_day, end_day


def download_market_data(
    asset: str = "BTC",
    total_months: int = DEFAULT_TOTAL_MONTHS,
    heldout_months: int = DEFAULT_HELDOUT_MONTHS,
    force_refresh: bool = False,
) -> Path:
    """Download minute-level price data for the given asset.

    Downloads a ``total_months`` window ending at (today - heldout_months).
    Stores data as daily parquet partitions.
    Skips days that already exist unless force_refresh=True.

    Returns the root directory containing the parquet files.
    """
    if asset not in ALL_SYMBOLS:
        raise ValueError(f"Unsupported asset: {asset}. Supported: {list(ALL_SYMBOLS.keys())}")
    start_day, end_day = _compute_date_range(total_months, heldout_months)
    print(f"Downloading {asset} data: {start_day} to {end_day} ({total_months} months, {heldout_months} held out)")
    client = HyperliquidClient() if asset in HYPERLIQUID_SYMBOLS else PythHistoryClient()
    store = MinutePriceStore(asset, client=client)
    store.ingest_range(start_day, end_day, force_refresh=force_refresh)
    print(f"Done. Data stored in {store.root}")
    return store.root


def download_all_assets(
    total_months: int = DEFAULT_TOTAL_MONTHS,
    heldout_months: int = DEFAULT_HELDOUT_MONTHS,
    force_refresh: bool = False,
    assets: list[str] | None = None,
    days: int | None = None,
) -> dict[str, Path]:
    """Download data for Synth Subnet assets.

    Parameters
    ----------
    assets : list of asset names to download. Defaults to all PYTH_SYMBOLS.
    days : if set, download this many days ending today (ignores
        total_months / heldout_months).
    """
    asset_list = assets if assets is not None else list(ALL_SYMBOLS.keys())

    if days is not None:
        utc_now = datetime.now(tz=UTC)
        end_day = utc_now.date()
        start_day = end_day - timedelta(days=days)
    else:
        start_day, end_day = _compute_date_range(total_months, heldout_months)

    print(f"Downloading {len(asset_list)} assets: {start_day} to {end_day}")
    hyperliquid_client = HyperliquidClient()
    results: dict[str, Path] = {}
    for asset in asset_list:
        if asset in HYPERLIQUID_SYMBOLS:
            store = MinutePriceStore(asset, client=hyperliquid_client)
        elif asset in PYTH_SYMBOLS:
            store = MinutePriceStore(asset)
        else:
            print(f"  Skipping unsupported asset: {asset}")
            continue
        store.ingest_range(start_day, end_day, force_refresh=force_refresh)
        results[asset] = store.root
    print("Done.")
    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Pyth minute-level price data for Synth Subnet assets.")
    parser.add_argument("--asset", default=None, help="Single asset to download (default: all)")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="If set, download this many days ending today (includes today; ignores --months/--heldout-months)",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Re-download existing partitions")
    args = parser.parse_args()

    download_all_assets(
        assets=[args.asset] if args.asset else None,
        days=args.days,
        force_refresh=args.force_refresh,
    )
