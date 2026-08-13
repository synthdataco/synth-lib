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
    RETIRED_SYMBOLS,
)
from synth_lib.preparation.hyperliquid_client import HyperliquidClient


class PriceClient(Protocol):
    """Structural interface for minute-price fetchers."""

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame: ...


def build_price_client(asset: str) -> PriceClient:
    """Return the price client for an asset, mirroring the validator's routing.

    Precedence matches PriceDataProvider.fetch_data: Binance, then Hyperliquid.
    """
    if asset in BINANCE_SYMBOLS:
        return BinanceClient()
    if asset in HYPERLIQUID_SYMBOLS:
        return HyperliquidClient()
    if asset in RETIRED_SYMBOLS:
        raise ValueError(
            f"{asset} was served by Pyth, which is retired: no client can fetch it. Supply its "
            f"history from an archive, or exclude it."
        )
    raise ValueError(f"Unsupported asset: {asset}. Supported: {list(ALL_SYMBOLS.keys())}")
