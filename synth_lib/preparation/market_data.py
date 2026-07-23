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
# Symbol mapping — pulled dynamically from synth-subnet
# ---------------------------------------------------------------------------
PYTH_HISTORY_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"

SYNTHDATA_API_BASE = "https://api.synthdata.co"

from synth.validator.price_data_provider import PriceDataProvider

# Provider symbol maps, mirrored 1:1 from synth-subnet. The validator's
# fetch_data routes each asset by precedence Binance -> Hyperliquid -> Pyth, and
# build_price_client() below mirrors that. As of the PR #304 feed migration the
# split is: crypto majors (BTC/ETH/SOL/XRP) from Binance; HYPE plus every
# commodity/equity — XAU, NVDAX, TSLAX, AAPLX, GOOGLX, WTIOIL, SPCX and SP500
# (coin xyz:SP500, which replaces the tokenized SPY) — from Hyperliquid; only the
# deprecated SPYX rollout tail from Pyth.
BINANCE_SYMBOLS: dict[str, str] = dict(PriceDataProvider.BINANCE_ASSET_MAP)
HYPERLIQUID_SYMBOLS: dict[str, str] = dict(PriceDataProvider.HYPERLIQUID_ASSET_MAP)
PYTH_SYMBOLS: dict[str, str] = dict(PriceDataProvider.PYTH_SYMBOL_MAP)
ALL_SYMBOLS: dict[str, str] = {**BINANCE_SYMBOLS, **HYPERLIQUID_SYMBOLS, **PYTH_SYMBOLS}

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
    """Thin client for the Pyth minute-price endpoint with retry logic.

    The TradingView shim rejects ranges over ~10,000 one-minute bars with
    ``s=error`` (a 7-day request fails while 5 days succeeds), so fetch_range
    paginates internally in MAX_DAYS_PER_REQUEST slices — mirroring the
    Binance/Hyperliquid providers, which paginate in their own download
    methods. Callers can request any range in one call (e.g. a live miner's
    full 7-day context).
    """

    MAX_DAYS_PER_REQUEST = 5

    source_name = "pyth"

    def __init__(self, timeout_seconds: int = 30, max_retries: int = 3):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch a range of minute closes from Pyth, paginated, with backoff."""
        start_time = _utc_datetime(start_time)
        end_time = _utc_datetime(end_time)
        if asset not in PYTH_SYMBOLS:
            raise ValueError(f"Unsupported asset: {asset}. Supported: {list(PYTH_SYMBOLS.keys())}")
        frames: list[pd.DataFrame] = []
        cursor = start_time
        chunk = timedelta(days=self.MAX_DAYS_PER_REQUEST)
        while cursor < end_time:
            chunk_end = min(cursor + chunk, end_time)
            frames.append(self._fetch_chunk(asset, cursor, chunk_end))
            cursor = chunk_end
        if not frames:
            return pd.DataFrame(columns=["timestamp", "close"])
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )

    def _fetch_chunk(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """One shim request (must stay under the shim's bar cap) with backoff."""
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
                # TradingView UDF status: "ok" (data), "no_data" (legitimately
                # empty, e.g. closed market), "error" (bad request — e.g. the
                # range cap). Only no_data may return empty silently; an error
                # must raise, or callers mistake a failed fetch for a data gap.
                status = payload.get("s")
                if status == "error":
                    raise ValueError(f"Pyth shim error for {asset}: {payload.get('errmsg', 'no errmsg')}")
                timestamps = payload.get("t", []) or []
                closes = payload.get("c", []) or []
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

    source_name = "hyperliquid"

    def __init__(self) -> None:
        self._provider = PriceDataProvider()

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        start_time = _utc_datetime(start_time)
        end_time = _utc_datetime(end_time)
        if asset not in HYPERLIQUID_SYMBOLS:
            raise ValueError(f"Unsupported Hyperliquid asset: {asset}")

        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        try:
            closes = self._provider.download_hyperliquid_price_data(
                beginning=start_ts,
                end=end_ts,
                symbol=asset,
                time_increment=60,
            )
        except ValueError:
            # No settled Hyperliquid candles for this window — e.g. the asset was
            # not yet listed (SPCX before ~2026-07-19) or a market-closure gap.
            # Treat as "no data" so the day ingests as NaN (downstream scoring
            # tolerates NaN per-prompt) instead of crashing the backtest.
            return pd.DataFrame(columns=["timestamp", "close"])
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
# Binance API client (delegates to synth.validator.price_data_provider)
# ---------------------------------------------------------------------------


class BinanceClient:
    """Wrapper around synth's PriceDataProvider matching PythHistoryClient's interface."""

    source_name = "binance"

    def __init__(self) -> None:
        self._provider = PriceDataProvider()

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        start_time = _utc_datetime(start_time)
        end_time = _utc_datetime(end_time)
        if asset not in BINANCE_SYMBOLS:
            raise ValueError(f"Unsupported Binance asset: {asset}")

        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        try:
            closes = self._provider.download_binance_price_data(
                beginning=start_ts,
                end=end_ts,
                symbol=asset,
                time_increment=60,
            )
        except ValueError:
            # No settled Binance candles for this window (unsettled current day or
            # a gap) — treat as "no data" so the day ingests as NaN instead of
            # crashing, matching the Hyperliquid path.
            return pd.DataFrame(columns=["timestamp", "close"])
        if not closes:
            return pd.DataFrame(columns=["timestamp", "close"])

        timestamps = pd.date_range(start_time, end_time, freq="1min", tz="UTC")
        if len(timestamps) != len(closes):
            raise RuntimeError(
                f"Binance timestamp/close length mismatch for {asset}: "
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


def build_price_client(asset: str) -> PriceClient:
    """Return the price client for an asset, mirroring the validator's routing.

    Precedence matches PriceDataProvider.fetch_data: Binance, then Hyperliquid,
    then Pyth.
    """
    if asset in BINANCE_SYMBOLS:
        return BinanceClient()
    if asset in HYPERLIQUID_SYMBOLS:
        return HyperliquidClient()
    if asset in PYTH_SYMBOLS:
        return PythHistoryClient()
    raise ValueError(f"Unsupported asset: {asset}. Supported: {list(ALL_SYMBOLS.keys())}")


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
    start_day, end_day = _compute_date_range(total_months, heldout_months)
    print(f"Downloading {asset} data: {start_day} to {end_day} ({total_months} months, {heldout_months} held out)")
    client = build_price_client(asset)
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
    assets : list of asset names to download. Defaults to all supported assets.
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
    results: dict[str, Path] = {}
    for asset in asset_list:
        try:
            client = build_price_client(asset)
        except ValueError:
            print(f"  Skipping unsupported asset: {asset}")
            continue
        store = MinutePriceStore(asset, client=client)
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
