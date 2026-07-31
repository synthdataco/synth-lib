"""Binance minute-price client (delegates to synth.validator.price_data_provider)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from synth.validator.price_data_provider import PriceDataProvider

from synth_lib.preparation.config import BINANCE_SYMBOLS, utc_datetime


class BinanceClient:
    """Wrapper around synth's PriceDataProvider matching PythHistoryClient's interface."""

    source_name = "binance"

    def __init__(self) -> None:
        self._provider = PriceDataProvider()

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        start_time = utc_datetime(start_time)
        end_time = utc_datetime(end_time)
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
