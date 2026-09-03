"""Shared constants for the price-preparation layer: endpoints, symbol maps, windows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from synth.validator.price_data_provider import PriceDataProvider

UTC = timezone.utc

SYNTHDATA_API_BASE = "https://api.synthdata.co"

# Provider symbol maps, mirrored 1:1 from synth-subnet. The validator's fetch_data routes each
# asset by precedence Binance -> Hyperliquid, and build_price_client() mirrors that: crypto majors
# (BTC/ETH/SOL/XRP) from Binance, HYPE plus every commodity/equity from Hyperliquid.
BINANCE_SYMBOLS: dict[str, str] = dict(PriceDataProvider.BINANCE_ASSET_MAP)
HYPERLIQUID_SYMBOLS: dict[str, str] = dict(PriceDataProvider.HYPERLIQUID_ASSET_MAP)
ALL_SYMBOLS: dict[str, str] = {**BINANCE_SYMBOLS, **HYPERLIQUID_SYMBOLS}

# Local price store layout. `prices` is the canonical directory; `pyth` is what it was called
# before the Pyth exit, and the name never meant anything about the source anyway — each partition's
# `source` column records provenance (binance / hyperliquid / an archive / ...). Kept as a fallback
# so an existing store does not have to be moved; see default_store_root.
MARKET_DATA_DIR = Path("market_data")
STORE_SUBDIR = "prices"
LEGACY_STORE_SUBDIR = "pyth"

MINUTES_PER_DAY = 24 * 60
CONTEXT_WINDOW_MINUTES = 7 * 24 * 60
DEFAULT_TOTAL_MONTHS = 15
DEFAULT_HELDOUT_MONTHS = 0

# -- Synth validator API --
PROMPTS_PAGE_SIZE_DAYS = 50  # endpoint caps at 60 days; `to` is inflated +24h server-side
SYNTH_API_REQUEST_INTERVAL_SECONDS = 0.6  # no-auth endpoints allow ~2 req/s per IP
SYNTH_API_MAX_RETRIES = 3
SYNTH_API_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
PROMPT_START_MATCH_TOLERANCE_MINUTES = 30


def utc_datetime(value: datetime) -> datetime:
    """Return a UTC datetime with second precision."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0)


def default_store_root(asset: str) -> Path:
    """Default MinutePriceStore root for `asset`: `market_data/prices/{asset}/1m`.

    Falls back to the legacy `market_data/pyth/` tree when that is what exists on disk, so an
    existing store keeps working without being moved. The choice is made at the market_data level,
    not per asset, so a store never ends up split across both names.
    """
    base = MARKET_DATA_DIR
    if not (base / STORE_SUBDIR).exists() and (base / LEGACY_STORE_SUBDIR).exists():
        return base / LEGACY_STORE_SUBDIR / asset / "1m"
    return base / STORE_SUBDIR / asset / "1m"
