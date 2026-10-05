"""Price-client protocol and the validator's asset -> venue routing."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from synth_lib.preparation.binance_client import BinanceClient
from synth_lib.preparation.config import (
    ALL_SYMBOLS,
    BINANCE_SYMBOLS,
    HYPERLIQUID_SYMBOLS,
)
from synth_lib.preparation.hyperliquid_client import HyperliquidClient


class PriceClient(Protocol):
    # Minutes of history the venue still serves, or None when it serves its whole history.
    # Callers that re-fetch settled days must not ask beyond it.
    retention_minutes: int | None

    """Structural interface for minute-price fetchers."""

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame: ...


def build_price_client(asset: str, **client_kwargs) -> PriceClient:
    """Return the price client for an asset, mirroring the validator's routing.

    Precedence matches PriceDataProvider.fetch_data: Binance, then Hyperliquid. `client_kwargs`
    (`session`, `timeout`) reach the client unchanged.
    """
    if asset in BINANCE_SYMBOLS:
        return BinanceClient(**client_kwargs)
    if asset in HYPERLIQUID_SYMBOLS:
        return HyperliquidClient(**client_kwargs)
    raise ValueError(f"Unsupported asset: {asset}. Supported: {list(ALL_SYMBOLS.keys())}")
