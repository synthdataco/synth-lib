"""ChampionMiner: a generic Bittensor SN50 miner that serves one benchmark champion.

Subclasses set `simulate_fn` and nothing else — `unpack_champion.py` generates that subclass.
Startup does not wait for data: a background thread warms each venue-routed minute store up, asset
by asset, then keeps them all fresh, and an asset is served as soon as its own warm-up is done. Each
request slices the trailing 7-day context — a crypto-1h one topped up with the bars that closed
since the last refresh — and adapts the champion's output to the live validator contract.

No guards: an unservable asset, a data hole, or an exploding path crashes the request so it
shows up in monitoring instead of silently degrading.

Deliberately NO `from __future__ import annotations` here. bittensor's `axon.attach` discovers the
request type by reading the first parameter's annotation off `forward_miner` and calling
`issubclass()` on it. Under PEP 563 that annotation is the *string* `"Simulation"`, so attach dies
with `TypeError: issubclass() arg 1 must be a class` and the miner cannot start at all. Keep the
annotations in this module evaluated.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from neurons.miner import Miner  # type: ignore[import-untyped]
from synth.protocol import Simulation  # type: ignore[import-untyped]

from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.serving.serve import (
    FRESH_TAIL_TIME_LENGTH,
    WARMUP_DAYS,
    FreshTail,
    serve_request,
    servable_assets,
    start_time_of,
    venue_store,
    warm_up,
)

logger = logging.getLogger(__name__)

# The champion reads its volatility off the most recent bars, so the age of the newest bar at
# request time is a scoring cost, not just a cosmetic one. Today's partition is a single venue
# call, cheap enough to top up every minute; the deep pass that force-refetches WARMUP_DAYS and
# repairs older gaps stays on its slower cadence.
#
# Neither runs on the request path. The one venue call there is FreshTail's, for crypto-1h prompts
# only: no retries, a short timeout, and the stored context when it fails.
CURRENT_DAY_REFRESH_SECONDS = 60
REFRESH_INTERVAL_SECONDS = 5 * 60


class ChampionMiner(Miner):
    """Serves one champion's simulate() over all venue-routed competition assets."""

    simulate_fn: Callable | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(self).simulate_fn is None:
            raise TypeError("subclass must set simulate_fn (see synth_lib/serving/unpack_champion.py)")
        self._start(servable_assets())

    def _start(self, assets: list[str]) -> None:
        """Binds the stores and starts the background thread. Returns before any venue call, so the
        axon comes up while the data is still being fetched."""
        self._stores: dict[str, MinutePriceStore] = {asset: venue_store(asset) for asset in assets}
        self._tails: dict[str, FreshTail] = {asset: FreshTail(asset) for asset in assets}
        # An asset joins once its whole warm-up window is in: a partly warmed store can end days back.
        self._ready: set[str] = set()
        threading.Thread(target=self._warm_up_then_refresh, daemon=True).start()
        logger.info("warming up %d assets in the background", len(assets))

    def _warm_up_then_refresh(self) -> None:
        for asset in self._stores:
            warm_up([asset])
            self._ready.add(asset)
        logger.info("warm-up complete; refreshing every %d s", CURRENT_DAY_REFRESH_SECONDS)
        self._background_refresh()

    def _refresh_once(self, deep: bool) -> None:
        """One pass over every store. A venue outage must not stop the others being refreshed."""
        today = datetime.now(tz=timezone.utc).date()
        for asset, store in self._stores.items():
            try:
                if deep:
                    store.refresh_recent(days=WARMUP_DAYS)
                else:
                    store.ingest_day(today, force_refresh=True)
            except Exception as exc:
                logger.warning("refresh failed for %s: %s", asset, exc)

    def _background_refresh(self) -> None:
        deep_every = max(1, REFRESH_INTERVAL_SECONDS // CURRENT_DAY_REFRESH_SECONDS)
        tick = 0
        while True:
            self._refresh_once(deep=tick % deep_every == 0)
            tick += 1
            time.sleep(CURRENT_DAY_REFRESH_SECONDS)

    async def forward_miner(self, synapse: Simulation) -> Simulation:
        simulation_input = synapse.simulation_input
        store = self._stores[simulation_input.asset]
        if simulation_input.asset not in self._ready:
            raise RuntimeError(f"{simulation_input.asset} is still warming up")
        tail = None
        if simulation_input.time_length == FRESH_TAIL_TIME_LENGTH:
            # Only the venue call leaves the event loop: champions keep module state, so simulate()
            # must not run on two threads at once.
            fresh_tail = self._tails[simulation_input.asset]
            tail = await asyncio.to_thread(fresh_tail.bars, start_time_of(simulation_input))
        synapse.simulation_output = serve_request(type(self).simulate_fn, store, simulation_input, tail=tail)
        return synapse
