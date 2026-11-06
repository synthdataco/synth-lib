"""The serving store's freshness is a scoring cost, not a cosmetic one.

The champion reads its volatility off the most recent bars, so the age of the newest bar at
request time shows up in CRPS. Measured on the campaign-3 champion, a context carrying a serving
store's staleness scored worse than one cut at the issuance instant, and the gap widened with the
refresh interval. This pins the cadence and what each pass asks the venue for.
"""

import logging
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import synth_lib.serving.serve as serve
from synth_lib.benchmark.generate_predictions import DEFAULT_CONTEXT_DAYS
from synth_lib.preparation.config import OHLCV_COLUMNS
from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.serving.champion_miner import REFRESH_SECONDS, ChampionMiner
from synth_lib.serving.serve import ContextCache


class _Store:
    def __init__(self):
        self.refreshed: list[int] = []

    def refresh_recent(self, days):
        self.refreshed.append(days)


class _Venue:
    source_name = "test"

    def __init__(self, retention_minutes=None):
        self.retention_minutes = retention_minutes
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
    monkeypatch.setattr("synth_lib.serving.serve.datetime", FrozenDatetime)
    return now


def _miner(stores, miner_class=ChampionMiner):
    miner = object.__new__(miner_class)  # Miner.__init__ needs a wallet, a subtensor and a chain
    miner._stores = stores
    miner._contexts = ContextCache()
    return miner


def test_each_pass_refreshes_the_contexts_date_partitions():
    """A D-day context touches D + 1 date partitions; a miner.py that does not set context_days
    serves the default."""
    store = _Store()
    _miner({"BTC": store})._refresh_once()
    assert store.refreshed == [DEFAULT_CONTEXT_DAYS + 1]

    class FourteenDays(ChampionMiner):
        context_days = 14

    store = _Store()
    _miner({"BTC": store}, FourteenDays)._refresh_once()
    assert store.refreshed == [15]


def test_each_pass_rebuilds_the_cached_contexts():
    miner = _miner({"BTC": _Store()})
    t = datetime(2026, 10, 29, 9, 30, tzinfo=timezone.utc)
    frame = miner._contexts.get(("BTC", 3_600, t), pd.DataFrame)
    miner._refresh_once()
    assert miner._contexts.get(("BTC", 3_600, t), pd.DataFrame) is not frame


def test_one_venue_outage_does_not_stop_the_others():
    class _Broken(_Store):
        def refresh_recent(self, days):
            raise RuntimeError("venue down")

    ok = _Store()
    _miner({"HYPE": _Broken(), "BTC": ok})._refresh_once()
    assert ok.refreshed == [DEFAULT_CONTEXT_DAYS + 1]


def test_the_assets_already_served_are_refreshed_while_the_others_warm_up(monkeypatch):
    """Each asset's warm-up is followed by a refresh of today's partition of the assets ready before it,
    never of one still warming up."""
    warmed: list[str] = []
    monkeypatch.setattr("synth_lib.serving.champion_miner.warm_up", lambda assets, context_days: warmed.extend(assets))
    monkeypatch.setattr(ChampionMiner, "_background_refresh", lambda self: None)

    class _Broken(_Store):
        def refresh_recent(self, days):
            raise RuntimeError("venue down")

    stores = {"XAU": _Broken(), "NVDAX": _Store(), "BTC": _Store(), "HYPE": _Store()}
    miner = _miner(stores)
    miner._ready = set()
    miner._calendar_ready = threading.Event()
    miner._calendar_ready.set()
    miner._warm_up_then_refresh()
    assert warmed == ["XAU", "NVDAX", "BTC", "HYPE"]
    assert (stores["NVDAX"].refreshed, stores["BTC"].refreshed, stores["HYPE"].refreshed) == ([1, 1], [1], [])
    assert miner._ready == set(stores)


def test_after_the_warm_up_a_pass_fetches_only_today(tmp_path, midday):
    """Settled days are never asked for again, so the pass is cheap enough to run every minute."""
    venue = _Venue()
    store = MinutePriceStore("BTC", root=tmp_path, client=venue)
    today = midday.date()
    store.ingest_range(today - timedelta(days=DEFAULT_CONTEXT_DAYS + 1), today, verbose=False)
    venue.days.clear()

    _miner({"BTC": store})._refresh_once()
    assert venue.days == [today]


@pytest.mark.parametrize("retention_minutes, oldest", [(5_000, 3), (None, 31)])
def test_the_warm_up_stops_at_the_venues_retention(tmp_path, midday, monkeypatch, caplog, retention_minutes, oldest):
    """A 30-day context on a venue keeping ~5000 minutes: asking past them would only be told nothing."""
    venue = _Venue(retention_minutes)
    monkeypatch.setattr(serve, "venue_store", lambda asset: MinutePriceStore(asset, root=tmp_path, client=venue))
    with caplog.at_level(logging.INFO, logger=serve.__name__):
        serve.warm_up(["XAU"], 30)
    assert min(venue.days) == midday.date() - timedelta(days=oldest)
    assert "XAU" in caplog.text and "of its 30-day context holds a real close" in caplog.text


def test_the_store_is_topped_up_every_minute():
    assert REFRESH_SECONDS <= 60
