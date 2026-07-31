"""Shared constants for the price-preparation layer: endpoints, symbol maps, windows."""

from __future__ import annotations

from datetime import datetime, timezone

from synth.validator.price_data_provider import PriceDataProvider

UTC = timezone.utc

PYTH_HISTORY_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
SYNTHDATA_API_BASE = "https://api.synthdata.co"

# Provider symbol maps, mirrored 1:1 from synth-subnet. The validator's fetch_data
# routes each asset by precedence Binance -> Hyperliquid -> Pyth, and
# build_price_client() mirrors that. Crypto majors (BTC/ETH/SOL/XRP) come from
# Binance; HYPE plus every commodity/equity from Hyperliquid; only the deprecated
# SPYX tail from Pyth.
BINANCE_SYMBOLS: dict[str, str] = dict(PriceDataProvider.BINANCE_ASSET_MAP)
HYPERLIQUID_SYMBOLS: dict[str, str] = dict(PriceDataProvider.HYPERLIQUID_ASSET_MAP)
PYTH_SYMBOLS: dict[str, str] = dict(PriceDataProvider.PYTH_SYMBOL_MAP)
ALL_SYMBOLS: dict[str, str] = {**BINANCE_SYMBOLS, **HYPERLIQUID_SYMBOLS, **PYTH_SYMBOLS}

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
