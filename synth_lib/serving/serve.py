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

Warm-up: `warm_up(...)` fills the local minute store from each asset's own venue before it is served.
Be aware of the retention asymmetry — Binance serves deep minute history, while Hyperliquid's
candle endpoint keeps roughly the last 5000 minutes (~3.5 days). A freshly-started miner therefore
has a full 7-day context for the cryptos and a shorter one for the HL-routed assets, which fills
in as the background refresh accumulates days. Run the miner a few days before you care about its
com-equ scores, or pre-populate the store from your own archive.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta
from typing import Callable, Sequence

import pandas as pd
import requests
from synth.simulation_input import SimulationInput  # type: ignore[import-untyped]
from synth.validator.competition_config import ALL_COMPETITIONS  # type: ignore[import-untyped]

from synth_lib.benchmark.generate_predictions import context_end
from synth_lib.preparation.config import (
    BINANCE_SYMBOLS,
    HYPERLIQUID_SYMBOLS,
    OHLCV_COLUMNS,
    legacy_partition_error,
)
from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.preparation.price_client import build_price_client

logger = logging.getLogger(__name__)

WARMUP_DAYS = 8  # 7-day context + 1 day of slack
CONTEXT_MINUTES = 7 * 24 * 60
MIN_REAL_BARS = 60  # below this a trimmed context is worse than a slightly stale one
# Prompts whose context is topped up at request time (crypto-1h); the 24 h formats are not.
FRESH_TAIL_TIME_LENGTH = 3_600
FRESH_TAIL_MINUTES = 15
# The request path's venue call: no retries, and far inside the response deadline.
FRESH_TAIL_TIMEOUT_SECONDS = 1.5
EMPTY_TAIL = pd.DataFrame(columns=["timestamp", *OHLCV_COLUMNS])


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


class FreshTail:
    """One asset's newest bars, fetched at request time up to the bar the verdict's context ends on.

    One fetch per minute, shared by every prompt for that minute: several validators send the same
    asset at the same minute boundary. Any failure yields no bars, so the context falls back to the
    store."""

    def __init__(self, asset: str, client=None):
        self.asset = asset
        self._client = client or build_price_client(
            asset, session=requests.Session(), timeout=FRESH_TAIL_TIMEOUT_SECONDS
        )
        self._lock = threading.Lock()
        self._end: datetime | None = None
        self._bars = EMPTY_TAIL

    def bars(self, start_time: datetime, now: datetime | None = None) -> pd.DataFrame:
        now = now or datetime.now(tz=UTC)
        # The newest closed bar, never past the label the verdict ends this prompt's context on.
        newest_closed = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
        end = min(context_end(start_time, FRESH_TAIL_TIME_LENGTH), newest_closed)
        with self._lock:
            if end != self._end:
                self._end = end
                try:
                    start = end - timedelta(minutes=FRESH_TAIL_MINUTES - 1)
                    self._bars = self._client.fetch_range(self.asset, start, end)
                except Exception as exc:
                    logger.warning("fresh tail unavailable for %s at %s: %s", self.asset, end.isoformat(), exc)
                    self._bars = EMPTY_TAIL
            return self._bars


def build_context(
    store: MinutePriceStore,
    start_time: datetime,
    window_minutes: int = CONTEXT_MINUTES,
    tail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The 7-day minute OHLCV context ending at `start_time`, lenient enough to serve from.

    Deliberately not `MinutePriceStore.get_context_window`, which is strict (it raises on a missing
    day or a short window) — correct for backtesting, fatal on the request path, where a single
    feed gap would cost a whole response. Here: skip missing day partitions, reindex onto the full
    minute grid, fill internal gaps, and DROP a trailing run of missing bars so the series ends on
    the last real print. Ending on ffilled bars would show the model zero recent volatility and
    collapse its fan.

    Only `close` is filled. A carried-forward high/low would assert a range that no candle traded,
    so the other columns stay NaN on minutes the venue did not report.

    `tail` (FreshTail's bars) wins over the store for the same minute: today's partition holds a
    NaN row for every minute its last refresh had not settled yet.
    """
    context_start = start_time - timedelta(minutes=window_minutes)
    frames = []
    day = context_start.date()
    while day <= start_time.date():
        path = store.day_path(day)
        if path.exists():
            try:
                frames.append(pd.read_parquet(path, columns=["timestamp", *OHLCV_COLUMNS]))
            except ValueError as exc:  # pyarrow's ArrowInvalid, on a column the file lacks
                raise ValueError(legacy_partition_error(path)) from exc
        day += timedelta(days=1)
    if tail is not None and not tail.empty:
        frames.append(tail[["timestamp", *OHLCV_COLUMNS]])
    if not frames:
        raise ValueError(f"no partitions for {store.asset} in {context_start.isoformat()}..{start_time.isoformat()}")

    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    # stable sort + keep="last": the tail, appended last, wins a minute the store also holds
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    grid = pd.date_range(context_start, start_time, freq="1min", tz="UTC")
    raw = frame.set_index("timestamp")[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce").reindex(grid)

    coverage = float(raw["close"].notna().mean())
    if coverage < 0.70:
        logger.warning("low real-bar coverage for %s: %.1f%% — CRPS quality degraded", store.asset, coverage * 100)
    last_real = raw["close"].last_valid_index()
    trimmed = raw if last_real is None else raw.loc[:last_real]
    if len(trimmed) < MIN_REAL_BARS:
        trimmed = raw
    trimmed = trimmed.copy()
    trimmed["close"] = trimmed["close"].ffill().bfill()
    if trimmed["close"].isna().any():
        raise ValueError(f"no usable closes for {store.asset} ending {start_time.isoformat()}")
    return trimmed


def start_time_of(simulation_input: SimulationInput) -> datetime:
    start_time = datetime.fromisoformat(simulation_input.start_time)
    return start_time if start_time.tzinfo is not None else start_time.replace(tzinfo=UTC)


def serve_request(
    simulate_fn: Callable,
    store: MinutePriceStore,
    simulation_input: SimulationInput,
    tail: pd.DataFrame | None = None,
) -> tuple:
    """Build the live context, run the champion, adapt the output to the live contract."""
    start_time = start_time_of(simulation_input)
    context = build_context(store, start_time, tail=tail)
    raw = simulate_fn(
        asset=simulation_input.asset,
        start_time=simulation_input.start_time,
        time_increment=simulation_input.time_increment,
        time_length=simulation_input.time_length,
        num_simulations=simulation_input.num_simulations,
        context_prices=context,
    )
    return wrap_output(raw, start_time, simulation_input.time_increment)
