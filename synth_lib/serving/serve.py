"""Serving core for a benchmark champion: data routing, warm-up, live context, contract adapter.

A champion's `simulate()` speaks the CAMPAIGN contract: it returns
`(start_time_iso, time_increment, *paths)` with raw float64 prices. The LIVE validator
(`synth.validator.response_validation_v2`) demands more: both metadata slots must be ints and
every price must round-trip through 8 significant digits. `wrap_output` is that adapter; its
sufficiency is proven by the contract-gate tests in tests/serving/.

Data routing goes through `synth_lib.preparation.build_price_client`: Binance for the cryptos,
Hyperliquid for the tokenised equity/commodity perps. Pyth is retired, so a Pyth-only asset is
NOT servable and `venue_store` refuses it rather than routing to a dead feed — a validator prompt
for such an asset then raises in the miner, which is loud in monitoring, by design.

There are NO defensive guards in this module: a hole in the data or an exploding path must crash
the request, not be papered over silently.

Warm-up: `warm_up(...)` fills the local minute store from each asset's own venue before serving.
Be aware of the retention asymmetry — Binance serves deep minute history, while Hyperliquid's
candle endpoint keeps roughly the last 5000 minutes (~3.5 days). A freshly-started miner therefore
has a full 7-day context for the cryptos and a shorter one for the HL-routed assets, which fills
in as the background refresh accumulates days. Run the miner a few days before you care about its
com-equ scores, or pre-populate the store from your own archive.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Callable, Sequence

import pandas as pd
from synth.simulation_input import SimulationInput  # type: ignore[import-untyped]
from synth.validator.competition_config import ALL_COMPETITIONS  # type: ignore[import-untyped]

from synth_lib.preparation.config import BINANCE_SYMBOLS, HYPERLIQUID_SYMBOLS
from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.preparation.price_client import build_price_client

logger = logging.getLogger(__name__)

WARMUP_DAYS = 8  # 7-day context + 1 day of slack
CONTEXT_MINUTES = 7 * 24 * 60
MIN_REAL_BARS = 60  # below this a trimmed context is worse than a slightly stale one


def wrap_output(raw: Sequence, start_time: datetime, time_increment: int) -> tuple:
    """Campaign-contract simulate() output -> live-contract response."""
    return (
        int(start_time.timestamp()),
        int(time_increment),
        *[[float(f"{v:.7e}") for v in path] for path in raw[2:]],
    )


def venue_store(asset: str) -> MinutePriceStore:
    """A store bound to the venue the asset is actually SCORED against."""
    if asset not in BINANCE_SYMBOLS and asset not in HYPERLIQUID_SYMBOLS:
        raise ValueError(f"no live venue for {asset}: it has no Binance/Hyperliquid market (Pyth is retired)")
    return MinutePriceStore(asset, client=build_price_client(asset))


def servable_assets() -> list[str]:
    """Every competition asset that has a live venue, in competition order."""
    assets: list[str] = []
    for comp in ALL_COMPETITIONS:
        for asset in comp.asset_list:
            if asset in assets:
                continue
            if asset in BINANCE_SYMBOLS or asset in HYPERLIQUID_SYMBOLS:
                assets.append(asset)
            else:
                logger.warning("excluding %s from serving: no live venue (Pyth is retired)", asset)
    return assets


def warm_up(assets: Sequence[str], days: int = WARMUP_DAYS) -> None:
    """Fill the local minute store from each asset's own venue before serving.

    Idempotent across restarts: `ingest_range` skips complete final partitions, so only genuinely
    missing days are fetched. Hyperliquid's ~3.5-day retention caps how far back HL-routed assets
    can reach (see the module docstring); those days simply come back empty and accumulate forward.
    """
    today = datetime.now(tz=UTC).date()
    for asset in assets:
        try:
            venue_store(asset).ingest_range(today - timedelta(days=days), today, verbose=False)
            logger.info("warm-up complete for %s", asset)
        except Exception as exc:  # a venue outage must not stop the miner from starting
            logger.warning("warm-up incomplete for %s: %s", asset, exc)


def build_context(store: MinutePriceStore, start_time: datetime, window_minutes: int = CONTEXT_MINUTES) -> pd.Series:
    """The 7-day minute context ending at `start_time`, lenient enough to serve from.

    Deliberately not `MinutePriceStore.get_context_window`, which is strict (it raises on a missing
    day or a short window) — correct for backtesting, fatal on the request path, where a single
    feed gap would cost a whole response. Here: skip missing day partitions, reindex onto the full
    minute grid, fill internal gaps, and DROP a trailing run of missing bars so the series ends on
    the last real print. Ending on ffilled bars would show the model zero recent volatility and
    collapse its fan.
    """
    context_start = start_time - timedelta(minutes=window_minutes)
    frames = []
    day = context_start.date()
    while day <= start_time.date():
        path = store.day_path(day)
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["timestamp", "close"]))
        day += timedelta(days=1)
    if not frames:
        raise ValueError(f"no partitions for {store.asset} in {context_start.isoformat()}..{start_time.isoformat()}")

    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    grid = pd.date_range(context_start, start_time, freq="1min", tz="UTC")
    raw = pd.to_numeric(frame.set_index("timestamp")["close"], errors="coerce").reindex(grid)

    coverage = float(raw.notna().mean())
    if coverage < 0.70:
        logger.warning("low real-bar coverage for %s: %.1f%% — CRPS quality degraded", store.asset, coverage * 100)
    last_real = raw.last_valid_index()
    trimmed = raw if last_real is None else raw.loc[:last_real]
    if len(trimmed) < MIN_REAL_BARS:
        trimmed = raw
    close = trimmed.ffill().bfill()
    if close.isna().any():
        raise ValueError(f"no usable closes for {store.asset} ending {start_time.isoformat()}")
    close.name = "close"
    return close


def serve_request(simulate_fn: Callable, store: MinutePriceStore, simulation_input: SimulationInput) -> tuple:
    """Build the live context, run the champion, adapt the output to the live contract."""
    start_time = datetime.fromisoformat(simulation_input.start_time)
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=UTC)
    context = build_context(store, start_time)
    raw = simulate_fn(
        asset=simulation_input.asset,
        start_time=simulation_input.start_time,
        time_increment=simulation_input.time_increment,
        time_length=simulation_input.time_length,
        num_simulations=simulation_input.num_simulations,
        context_prices=context,
    )
    return wrap_output(raw, start_time, simulation_input.time_increment)
