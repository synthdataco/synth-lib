"""Hyperliquid minute-OHLCV client."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from synth_lib.preparation.config import HYPERLIQUID_SYMBOLS, OHLCV_COLUMNS, utc_datetime, venue_session

HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"

# The venue serves at most this many candles per request, which is what bounds a Hyperliquid-routed
# asset's history to roughly three and a half days at one-minute resolution.
MAX_CANDLES = 5000

EMPTY = pd.DataFrame(columns=["timestamp", *OHLCV_COLUMNS])


class HyperliquidClient:
    """Minute OHLCV from Hyperliquid, implementing the PriceClient protocol."""

    source_name = "hyperliquid"

    def __init__(self) -> None:
        self._session = venue_session()

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        start_time = utc_datetime(start_time)
        end_time = utc_datetime(end_time)
        if asset not in HYPERLIQUID_SYMBOLS:
            raise ValueError(f"Unsupported Hyperliquid asset: {asset}")

        end_ms = int(end_time.timestamp() * 1000)
        response = self._session.post(
            HYPERLIQUID_INFO_URL,
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": HYPERLIQUID_SYMBOLS[asset],
                    "interval": "1m",
                    "startTime": int(start_time.timestamp() * 1000),
                    # One minute past the window, so the settlement witness is in the same response.
                    "endTime": end_ms + 60_000,
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        candles = response.json()
        if not candles:
            return EMPTY.copy()

        # No candle opens after the window, so nothing proves the last requested minute has closed.
        # Report "no data" rather than a half-formed tail: the store persists that as NaN, which is
        # recoverable, whereas a partial minute written as final is not.
        if not any(int(candle["t"]) > end_ms for candle in candles):
            return EMPTY.copy()
        inside = [candle for candle in candles if int(candle["t"]) <= end_ms]
        if not inside:
            return EMPTY.copy()

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([c["t"] for c in inside], unit="ms", utc=True),
                "open": [float(c["o"]) for c in inside],
                "high": [float(c["h"]) for c in inside],
                "low": [float(c["l"]) for c in inside],
                "close": [float(c["c"]) for c in inside],
                "volume": [float(c["v"]) for c in inside],
                "trade_count": [float(c["n"]) for c in inside],
            }
        )
        return frame.dropna(subset=["close"]).drop_duplicates("timestamp").reset_index(drop=True)
