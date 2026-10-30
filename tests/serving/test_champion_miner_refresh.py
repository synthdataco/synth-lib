"""The serving store's freshness is a scoring cost, not a cosmetic one.

The champion reads its volatility off the most recent bars, so the age of the newest bar at
request time shows up in CRPS. Measured on the campaign-3 champion, a context carrying a serving
store's staleness scored worse than one cut at the issuance instant, and the gap widened with the
refresh interval. This pins the cadence and what each pass asks the venue for.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from synth_lib.preparation.config import OHLCV_COLUMNS
from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.serving.champion_miner import REFRESH_SECONDS, ChampionMiner
from synth_lib.serving.serve import WARMUP_DAYS


class _Store:
    def __init__(self):
        self.refreshed: list[int] = []

    def refresh_recent(self, days):
        self.refreshed.append(days)


class _Venue:
    source_name = "test"
    retention_minutes = None

    def __init__(self):
        self.days: list = []

    def fetch_range(self, asset, start_time, end_time):
        self.days.append(start_time.date())
        index = pd.date_range(start_time, end_time, freq="1min", tz="UTC")
        return pd.DataFrame({"timestamp": index, **{c: 1.0 for c in OHLCV_COLUMNS}})


@pytest.fixture
def midday(monkeypatch):
    """Freeze the store's clock mid-day, so today has settled minutes to fetch."""
    now = datetime(2026, 10, 29, 9, 30, 5, tzinfo=timezone.utc)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr("synth_lib.preparation.minute_price_store.datetime", FrozenDatetime)
    return now


def _miner(stores):
    miner = object.__new__(ChampionMiner)  # Miner.__init__ needs a wallet, a subtensor and a chain
    miner._stores = stores
    return miner


def test_each_pass_refreshes_the_warm_up_window():
    store = _Store()
    _miner({"BTC": store})._refresh_once()
    assert store.refreshed == [WARMUP_DAYS]


def test_one_venue_outage_does_not_stop_the_others():
    class _Broken(_Store):
        def refresh_recent(self, days):
            raise RuntimeError("venue down")

    ok = _Store()
    _miner({"HYPE": _Broken(), "BTC": ok})._refresh_once()
    assert ok.refreshed == [WARMUP_DAYS]


def test_after_the_warm_up_a_pass_fetches_only_today(tmp_path, midday):
    """Settled days are never asked for again, so the pass is cheap enough to run every minute."""
    venue = _Venue()
    store = MinutePriceStore("BTC", root=tmp_path, client=venue)
    today = midday.date()
    store.ingest_range(today - timedelta(days=WARMUP_DAYS), today, verbose=False)
    venue.days.clear()

    _miner({"BTC": store})._refresh_once()
    assert venue.days == [today]


def test_the_store_is_topped_up_every_minute():
    assert REFRESH_SECONDS <= 60
