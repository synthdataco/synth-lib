"""The current day must ingest real prices, not an all-NaN grid.

Both venue providers refuse a window they cannot prove is settled: they look for a candle whose
open time is strictly greater than the requested end (`saw_settled_witness`), and raise when there
is none. The clients translate that into "no data". So asking for the current day as 00:00..23:59 —
an end in the future — returns nothing, and the day persists as 1440 NaN rows.

That is invisible to the backtester, which reads settled days only, and fatal to the miner:
`serve.build_context` drops the trailing NaN run, so every response is generated from a context
ending at the previous midnight — up to 24 hours stale. A model asked to predict the next hour from
a price hours old cannot score, however good it is, and nothing in the pipeline reports an error:
the partition is present, correctly shaped, and refreshed on schedule.

`FakeVenue` below reproduces the witness rule rather than the symptom, so these tests fail against
the unclamped code for the same reason the real venues do.
"""

from datetime import UTC, datetime, time, timedelta

import pandas as pd
import pytest

from synth_lib.preparation.minute_price_store import (
    CURRENT_DAY_SETTLE_MARGIN_MINUTES,
    MinutePriceStore,
)


class FakeVenue:
    """A venue that serves closed minutes only and demands a settled witness, as the real ones do."""

    source_name = "fake"

    def __init__(self, now: datetime):
        self.now = now
        self.requests: list[tuple[datetime, datetime]] = []

    def fetch_range(self, asset: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        self.requests.append((start_time, end_time))
        # The newest candle that exists is the one that opened a minute ago; a witness must open
        # strictly after end_time. No witness -> the provider raises and the client reports no data.
        newest_open = self.now.replace(second=0, microsecond=0) - timedelta(minutes=1)
        if end_time >= newest_open:
            return pd.DataFrame(columns=["timestamp", "close"])
        index = pd.date_range(start_time, min(end_time, newest_open), freq="1min", tz="UTC")
        return pd.DataFrame({"timestamp": index, "close": [100.0 + i for i in range(len(index))]})


@pytest.fixture
def frozen_now(monkeypatch):
    """Freeze the store's clock at a mid-morning minute so 'today' is genuinely partial."""
    now = datetime(2026, 8, 17, 8, 53, 5, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr("synth_lib.preparation.minute_price_store.datetime", FrozenDatetime)
    return now


def _store(tmp_path, venue) -> MinutePriceStore:
    return MinutePriceStore("BTC", root=tmp_path, client=venue)


def test_current_day_partition_has_real_prices(tmp_path, frozen_now):
    """The regression: before the fix this partition was 1440 rows of NaN."""
    venue = FakeVenue(frozen_now)
    path = _store(tmp_path, venue).ingest_day(frozen_now.date())

    frame = pd.read_parquet(path)
    assert len(frame) == 1440, "the full-grid partition contract must survive the clamp"
    real = int(frame["close"].notna().sum())
    assert real > 0, "current day ingested as an all-NaN grid — the miner would serve a stale context"
    # 00:00 through the last settled minute, inclusive.
    assert real == 8 * 60 + 53 - CURRENT_DAY_SETTLE_MARGIN_MINUTES + 1


def test_current_day_window_is_clamped_to_a_settled_minute(tmp_path, frozen_now):
    venue = FakeVenue(frozen_now)
    _store(tmp_path, venue).ingest_day(frozen_now.date())

    _, end = venue.requests[-1]
    assert end == frozen_now.replace(second=0, microsecond=0) - timedelta(minutes=CURRENT_DAY_SETTLE_MARGIN_MINUTES)
    assert end < frozen_now, "asked for a window ending in the future — no venue can witness it"


def test_settled_day_still_requests_the_whole_day(tmp_path, frozen_now):
    """The clamp must not touch finalised days: a past partition is still 00:00..23:59."""
    venue = FakeVenue(frozen_now)
    yesterday = frozen_now.date() - timedelta(days=1)
    path = _store(tmp_path, venue).ingest_day(yesterday)

    start, end = venue.requests[-1]
    assert start == datetime.combine(yesterday, time.min, tzinfo=UTC)
    assert end == datetime.combine(yesterday, time.max, tzinfo=UTC).replace(second=0, microsecond=0)

    frame = pd.read_parquet(path)
    assert len(frame) == 1440
    assert bool(frame["is_final"].iloc[0]) is True
    assert int(frame["close"].notna().sum()) == 1440


def test_just_after_midnight_writes_an_empty_grid_without_calling_the_venue(tmp_path, monkeypatch):
    """Inside the settle margin of midnight nothing has settled yet — persist the grid, do not fetch."""
    now = datetime(2026, 8, 17, 0, 1, 30, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr("synth_lib.preparation.minute_price_store.datetime", FrozenDatetime)

    venue = FakeVenue(now)
    path = _store(tmp_path, venue).ingest_day(now.date())

    assert venue.requests == [], "must not ask for a window that ends before the day begins"
    frame = pd.read_parquet(path)
    assert len(frame) == 1440
    assert int(frame["close"].notna().sum()) == 0
    assert bool(frame["is_final"].iloc[0]) is False


def test_serving_context_reaches_the_current_minute(tmp_path, frozen_now):
    """End to end: what build_context would hand the champion must not be a day stale.

    build_context drops the trailing NaN run, so an all-NaN current day silently rolls the context
    back to the previous midnight. This asserts the property the miner actually depends on.
    """
    venue = FakeVenue(frozen_now)
    store = _store(tmp_path, venue)
    store.ingest_day(frozen_now.date() - timedelta(days=1))
    store.ingest_day(frozen_now.date())

    frame = pd.concat([pd.read_parquet(store.day_path(frozen_now.date() - timedelta(days=i))) for i in (1, 0)])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    last_real = frame.dropna(subset=["close"])["timestamp"].max()

    staleness = frozen_now - last_real
    assert staleness <= timedelta(minutes=CURRENT_DAY_SETTLE_MARGIN_MINUTES + 1), (
        f"context ends {staleness} before now; the champion would predict from a stale price"
    )
