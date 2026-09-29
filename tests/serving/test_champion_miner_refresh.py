"""The serving store's freshness is a scoring cost, not a cosmetic one.

The champion reads its volatility off the most recent bars, so the age of the newest bar at
request time shows up in CRPS. Measured on the campaign-3 champion, a context carrying a serving
store's staleness scored worse than one cut at the issuance instant, and the gap widened with the
refresh interval. This pins the two cadences and the split between them.
"""

from datetime import date

from synth_lib.serving.champion_miner import (
    CURRENT_DAY_REFRESH_SECONDS,
    REFRESH_INTERVAL_SECONDS,
    ChampionMiner,
)


class _Store:
    def __init__(self):
        self.deep = 0
        self.days: list[date] = []

    def refresh_recent(self, days):
        self.deep += 1

    def ingest_day(self, day, force_refresh=False):
        assert force_refresh, "a top-up that does not force a refetch re-reads the same partition"
        self.days.append(day)


def _miner(stores):
    miner = object.__new__(ChampionMiner)  # Miner.__init__ needs a wallet, a subtensor and a chain
    miner._stores = stores
    return miner


def test_the_cheap_pass_tops_up_only_today():
    store = _Store()
    _miner({"BTC": store})._refresh_once(deep=False)
    assert store.days == [date.today()] and store.deep == 0


def test_the_deep_pass_repairs_older_days():
    store = _Store()
    _miner({"BTC": store})._refresh_once(deep=True)
    assert store.deep == 1 and store.days == []


def test_one_venue_outage_does_not_stop_the_others():
    class _Broken(_Store):
        def ingest_day(self, day, force_refresh=False):
            raise RuntimeError("venue down")

    ok = _Store()
    _miner({"HYPE": _Broken(), "BTC": ok})._refresh_once(deep=False)
    assert ok.days == [date.today()]


def test_today_is_topped_up_far_more_often_than_the_deep_pass():
    """The deep pass force-refetches WARMUP_DAYS for every asset; running it at the top-up cadence
    would multiply venue traffic for days that are already settled."""
    assert CURRENT_DAY_REFRESH_SECONDS < REFRESH_INTERVAL_SECONDS
    assert REFRESH_INTERVAL_SECONDS % CURRENT_DAY_REFRESH_SECONDS == 0
