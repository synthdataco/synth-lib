"""A re-fetch must not trade real prices for nothing, and a settled day is not fetched again.

Venues serve a bounded window of history — Hyperliquid about 5000 minutes. A serving miner that
force-refreshes its last N days therefore asks for days the venue has dropped, gets an empty
frame, and writes it over a good partition: 1440 NaN rows marked final, and the champion's context
loses those days. Observed live as nine consecutive empty WTIOIL days in a miner's store.

What a day needs is read from its partition: one written while it was still today, or from an empty
fetch, is fetched again; one fetched complete after it ended never is.
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from synth_lib.preparation.config import MINUTES_PER_DAY, OHLCV_COLUMNS
from synth_lib.preparation.minute_price_store import MinutePriceStore

DAY = date(2026, 9, 25)


class _Client:
    source_name = "test"
    retention_minutes = None

    def __init__(self, rows: int):
        self.rows = rows
        self.calls: list[date] = []

    def fetch_range(self, asset, start_time, end_time):
        self.calls.append(start_time.date())
        if not self.rows:
            return pd.DataFrame(columns=["timestamp", *OHLCV_COLUMNS])
        index = pd.date_range(start_time, periods=self.rows, freq="1min", tz="UTC")
        return pd.DataFrame({"timestamp": index, **{c: 1.0 for c in OHLCV_COLUMNS}})


def _store(tmp_path, client):
    return MinutePriceStore("WTIOIL", root=tmp_path / "WTIOIL" / "1m", client=client)


def _closes(path):
    return int(pd.read_parquet(path, columns=["close"])["close"].notna().sum())


def test_an_aged_out_day_does_not_erase_the_prices_already_stored(tmp_path):
    good = _store(tmp_path, _Client(rows=MINUTES_PER_DAY))
    path = good.ingest_day(DAY, force_refresh=True)
    assert _closes(path) == MINUTES_PER_DAY

    # the same day re-fetched after the venue dropped it
    empty = _store(tmp_path, _Client(rows=0))
    assert _closes(empty.ingest_day(DAY, force_refresh=True)) == MINUTES_PER_DAY


def test_a_refetch_that_recovers_minutes_still_wins(tmp_path):
    partial = _store(tmp_path, _Client(rows=100))
    assert _closes(partial.ingest_day(DAY, force_refresh=True)) == 100
    full = _store(tmp_path, _Client(rows=MINUTES_PER_DAY))
    assert _closes(full.ingest_day(DAY, force_refresh=True)) == MINUTES_PER_DAY


def test_refresh_recent_does_not_ask_past_the_venues_retention(tmp_path):
    class _Bounded(_Client):
        retention_minutes = 5000  # Hyperliquid: about 3.5 days

    client = _Bounded(rows=10)
    _store(tmp_path, client).refresh_recent(days=8)
    oldest = min(client.calls)
    today = datetime.now(tz=timezone.utc).date()
    assert today - oldest <= timedelta(days=3), f"asked for {oldest}, beyond what the venue serves"


def test_an_unbounded_venue_is_still_refreshed_in_full(tmp_path):
    client = _Client(rows=10)
    _store(tmp_path, client).refresh_recent(days=8)
    today = datetime.now(tz=timezone.utc).date()
    assert today - min(client.calls) == timedelta(days=8)


def test_a_day_written_while_it_was_today_is_completed_once_it_ends(tmp_path, monkeypatch):
    """The skip reads the partition's own flag: a day stored partial is fetched again, once."""
    clock = {"now": datetime(2026, 9, 25, 9, 0, tzinfo=timezone.utc)}

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"]

    monkeypatch.setattr("synth_lib.preparation.minute_price_store.datetime", Clock)
    client = _Client(rows=500)
    store = _store(tmp_path, client)
    store.ingest_day(DAY)
    assert not bool(pd.read_parquet(store.day_path(DAY))["is_final"].iloc[0])

    clock["now"] = datetime(2026, 9, 26, 0, 5, tzinfo=timezone.utc)
    client.rows = MINUTES_PER_DAY
    assert _closes(store.ingest_day(DAY)) == MINUTES_PER_DAY
    store.ingest_day(DAY)
    assert client.calls == [DAY, DAY], "a settled day was fetched again"


def test_an_empty_fetch_of_a_past_day_is_asked_again(tmp_path):
    client = _Client(rows=0)
    store = _store(tmp_path, client)
    store.ingest_day(DAY)
    client.rows = MINUTES_PER_DAY
    assert _closes(store.ingest_day(DAY)) == MINUTES_PER_DAY


def test_stored_zero_trade_minutes_do_not_block_a_refetch_that_leaves_them_out(tmp_path):
    """A partition holding flat zero-trade candles has fewer real prices than its closes suggest."""
    path = _store(tmp_path, _Client(rows=MINUTES_PER_DAY)).ingest_day(DAY, force_refresh=True)
    frame = pd.read_parquet(path)
    frame.loc[frame.index[::10], "trade_count"] = 0.0
    frame.to_parquet(path, index=False)

    traded = _Client(rows=MINUTES_PER_DAY - MINUTES_PER_DAY // 10)
    written = _store(tmp_path, traded).ingest_day(DAY, force_refresh=True)
    assert pd.read_parquet(written)["trade_count"].ne(0).all()
