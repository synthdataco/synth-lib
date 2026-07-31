"""Pyth Benchmarks minute-price client."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from synth_lib.preparation.config import PYTH_HISTORY_URL, PYTH_SYMBOLS, utc_datetime


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
        start_time = utc_datetime(start_time)
        end_time = utc_datetime(end_time)
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
                    time.sleep(delay)
        raise RuntimeError(f"Pyth API failed after {self.max_retries} attempts for {asset}") from last_exc
