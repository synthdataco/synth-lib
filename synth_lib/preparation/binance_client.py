"""Binance minute-OHLCV client."""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd

from synth_lib.preparation.config import BINANCE_SYMBOLS, OHLCV_COLUMNS, utc_datetime, venue_session

# BINANCE_API_HOST is a process-env escape hatch, read at import time: api.binance.com answers
# HTTP 451 from geo-restricted regions (US-hosted CI runners use data-api.binance.vision).
BINANCE_SPOT_URL = os.environ.get("BINANCE_API_HOST", "https://api.binance.com") + "/api/v3/klines"

MINUTE_MS = 60_000
MAX_KLINES_PER_REQUEST = 1000
REQUEST_SPACING_SECONDS = 0.2

# Kline array positions. The endpoint returns 12 fields per candle.
OPEN_TIME, OPEN, HIGH, LOW, CLOSE, VOLUME = 0, 1, 2, 3, 4, 5
TRADE_COUNT = 8

EMPTY = pd.DataFrame(columns=["timestamp", *OHLCV_COLUMNS])


class BinanceClient:
    """Minute OHLCV from Binance spot, implementing the PriceClient protocol."""

    source_name = "binance"

    def __init__(self) -> None:
        self._session = venue_session()

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        start_time = utc_datetime(start_time)
        end_time = utc_datetime(end_time)
        if asset not in BINANCE_SYMBOLS:
            raise ValueError(f"Unsupported Binance asset: {asset}")

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        klines, settled = self._download(BINANCE_SYMBOLS[asset], start_ms, end_ms)
        # No candle opens after the window, so nothing proves the last requested minute has closed.
        # Report "no data" rather than a half-formed tail: the store persists that as NaN, which is
        # recoverable, whereas a partial minute written as final is not.
        if not settled or not klines:
            return EMPTY.copy()

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([k[OPEN_TIME] for k in klines], unit="ms", utc=True),
                "open": [float(k[OPEN]) for k in klines],
                "high": [float(k[HIGH]) for k in klines],
                "low": [float(k[LOW]) for k in klines],
                "close": [float(k[CLOSE]) for k in klines],
                "volume": [float(k[VOLUME]) for k in klines],
                "trade_count": [float(k[TRADE_COUNT]) for k in klines],
            }
        )
        return frame.dropna(subset=["close"]).drop_duplicates("timestamp").reset_index(drop=True)

    def _download(self, symbol: str, start_ms: int, end_ms: int) -> tuple[list, bool]:
        """Klines covering [start_ms, end_ms], plus whether a candle opens strictly after it."""
        klines: list = []
        settled = False
        cursor = start_ms
        while cursor <= end_ms:
            response = self._session.get(
                BINANCE_SPOT_URL,
                params={
                    "symbol": symbol,
                    "interval": "1m",
                    "startTime": cursor,
                    # One minute past the window, so the settlement witness is in the same response.
                    "endTime": end_ms + MINUTE_MS,
                    "limit": MAX_KLINES_PER_REQUEST,
                },
                timeout=30,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            for kline in batch:
                opened = int(kline[OPEN_TIME])
                if opened > end_ms:
                    settled = True
                else:
                    klines.append(kline)
            last_opened = int(batch[-1][OPEN_TIME])
            if last_opened < cursor:  # no forward progress; the venue has nothing more
                break
            cursor = last_opened + MINUTE_MS
            time.sleep(REQUEST_SPACING_SECONDS)
        return klines, settled
