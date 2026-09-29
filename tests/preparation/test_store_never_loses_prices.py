"""A forced re-fetch must not trade real prices for nothing.

Venues serve a bounded window of history — Hyperliquid about 5000 minutes. A serving miner that
force-refreshes its last N days therefore asks for days the venue has dropped, gets an empty
frame, and writes it over a good partition. The result is 1440 NaN rows marked final, which
build_context then forward-fills into a flat line: the model reads zero volatility off it and its
fan collapses. Observed live as nine consecutive empty WTIOIL days in a miner's store.
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
