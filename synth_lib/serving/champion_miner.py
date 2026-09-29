"""ChampionMiner: a generic Bittensor SN50 miner that serves one benchmark champion.

Subclasses set `simulate_fn` and nothing else — `unpack_champion.py` generates that subclass.
Startup blocks on a venue warm-up (see serve.py), then a background thread keeps every
venue-routed minute store fresh; each request slices the trailing 7-day context and adapts the
champion's output to the live validator contract.

No guards: an unservable asset, a data hole, or an exploding path crashes the request so it
shows up in monitoring instead of silently degrading.

Deliberately NO `from __future__ import annotations` here. bittensor's `axon.attach` discovers the
request type by reading the first parameter's annotation off `forward_miner` and calling
`issubclass()` on it. Under PEP 563 that annotation is the *string* `"Simulation"`, so attach dies
with `TypeError: issubclass() arg 1 must be a class` and the miner cannot start at all. Keep the
annotations in this module evaluated.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from neurons.miner import Miner  # type: ignore[import-untyped]
from synth.protocol import Simulation  # type: ignore[import-untyped]

from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.serving.serve import WARMUP_DAYS, serve_request, servable_assets, venue_store, warm_up

logger = logging.getLogger(__name__)

# The champion reads its volatility off the most recent bars, so the age of the newest bar at
# request time is a scoring cost, not just a cosmetic one. Today's partition is a single venue
# call, cheap enough to top up every minute; the deep pass that force-refetches WARMUP_DAYS and
# repairs older gaps stays on its slower cadence.
#
# Neither runs on the request path on purpose. A venue round-trip there would risk the response
# deadline, and a missed prompt is filled at the field's 95th percentile — far more expensive than
# a context that is a minute old.
CURRENT_DAY_REFRESH_SECONDS = 60
REFRESH_INTERVAL_SECONDS = 5 * 60


class ChampionMiner(Miner):
    """Serves one champion's simulate() over all venue-routed competition assets."""

    simulate_fn: Callable | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(self).simulate_fn is None:
            raise TypeError("subclass must set simulate_fn (see synth_lib/serving/unpack_champion.py)")
        assets = servable_assets()
        logger.info("warming up %d assets from their venues...", len(assets))
        warm_up(assets)
        self._stores: dict[str, MinutePriceStore] = {asset: venue_store(asset) for asset in assets}
        self._refresh_started = False
        self._refresh_lock = threading.Lock()

    def _ensure_refresh_thread(self) -> None:
        with self._refresh_lock:
            if self._refresh_started:
                return
            threading.Thread(target=self._background_refresh, daemon=True).start()
            self._refresh_started = True
            logger.info("ChampionMiner refresh thread started")

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
        self._ensure_refresh_thread()
        simulation_input = synapse.simulation_input
        synapse.simulation_output = serve_request(
            type(self).simulate_fn, self._stores[simulation_input.asset], simulation_input
        )
        return synapse
