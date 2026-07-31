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
    PYTH_SYMBOLS,
)
from synth_lib.preparation.hyperliquid_client import HyperliquidClient
from synth_lib.preparation.pyth_client import PythHistoryClient


class PriceClient(Protocol):
    """Structural interface for minute-price fetchers."""

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame: ...


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
