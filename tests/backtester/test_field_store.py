"""Day partitions of an offline bundle: which days a refresh fetches, and what a reader gets back."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from synth_lib.backtester.field_store import day_path, read_days, refresh_days

UTC = timezone.utc
DAY = datetime(2026, 9, 15, tzinfo=UTC)
LATER = DAY + timedelta(days=30)


class _Fetch:
    """One row per hour of the requested range, recording each request."""

    def __init__(self):
        self.calls: list[tuple[datetime, datetime]] = []

    def __call__(self, start, end):
        self.calls.append((start, end))
        return pd.DataFrame({"scored_time": pd.date_range(start, end, freq="1h", inclusive="left"), "crps": 1.0})


def test_each_day_lands_in_its_own_partition(tmp_path):
    fetch = _Fetch()
    assert refresh_days(tmp_path, DAY, DAY + timedelta(days=2), fetch, "scored_time", now=LATER) == 2
    assert day_path(tmp_path, date(2026, 9, 15)).exists() and day_path(tmp_path, date(2026, 9, 16)).exists()
    assert len(pd.read_parquet(day_path(tmp_path, date(2026, 9, 16)))) == 24


def test_complete_days_are_not_fetched_again(tmp_path):
    fetch = _Fetch()
    refresh_days(tmp_path, DAY, DAY + timedelta(days=2), fetch, "scored_time", now=LATER)
    assert refresh_days(tmp_path, DAY, DAY + timedelta(days=3), fetch, "scored_time", now=LATER) == 1
    assert fetch.calls[-1][0] == DAY + timedelta(days=2)


def test_force_fetches_every_day_again(tmp_path):
    fetch = _Fetch()
    refresh_days(tmp_path, DAY, DAY + timedelta(days=2), fetch, "scored_time", now=LATER)
    assert refresh_days(tmp_path, DAY, DAY + timedelta(days=2), fetch, "scored_time", force=True, now=LATER) == 2


def test_a_day_cut_at_the_requested_end_is_partial_and_fetched_again(tmp_path):
    fetch = _Fetch()
    refresh_days(tmp_path, DAY, DAY + timedelta(hours=25), fetch, "scored_time", now=LATER)
    partial = day_path(tmp_path, date(2026, 9, 16), partial=True)
    assert len(pd.read_parquet(partial)) == 1 and not day_path(tmp_path, date(2026, 9, 16)).exists()

    refresh_days(tmp_path, DAY, DAY + timedelta(days=2), fetch, "scored_time", now=LATER)
    assert len(pd.read_parquet(day_path(tmp_path, date(2026, 9, 16)))) == 24 and not partial.exists()


def test_a_day_that_has_not_settled_is_partial(tmp_path):
    refresh_days(
        tmp_path, DAY, DAY + timedelta(days=1), _Fetch(), "scored_time", now=DAY + timedelta(days=1, minutes=5)
    )
    assert day_path(tmp_path, date(2026, 9, 15), partial=True).exists()
    assert not day_path(tmp_path, date(2026, 9, 15)).exists()


def test_a_settled_day_that_came_back_empty_is_partial_and_fetched_again(tmp_path):
    def empty(start, end):
        return pd.DataFrame(columns=["scored_time", "crps"])

    refresh_days(tmp_path, DAY, DAY + timedelta(days=1), empty, "scored_time", now=LATER)
    assert day_path(tmp_path, date(2026, 9, 15), partial=True).exists()
    assert refresh_days(tmp_path, DAY, DAY + timedelta(days=1), _Fetch(), "scored_time", now=LATER) == 1
    assert len(pd.read_parquet(day_path(tmp_path, date(2026, 9, 15)))) == 24


def test_an_interrupted_write_leaves_no_partition_behind(tmp_path, monkeypatch):
    def cut_short(self, path, **kwargs):
        Path(path).write_bytes(b"PAR1")
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_parquet", cut_short)
    with pytest.raises(KeyboardInterrupt):
        refresh_days(tmp_path, DAY, DAY + timedelta(days=1), _Fetch(), "scored_time", now=LATER)
    assert list(tmp_path.iterdir()) == []


def test_rows_outside_the_day_are_dropped(tmp_path):
    """A range query's `to` names a whole day: the neighbour's rows must not land twice."""

    def spill(start, end):
        return pd.DataFrame({"scored_time": [start - timedelta(minutes=1), start, end], "crps": 1.0})

    refresh_days(tmp_path, DAY, DAY + timedelta(days=1), spill, "scored_time", now=LATER)
    assert list(pd.read_parquet(day_path(tmp_path, date(2026, 9, 15)))["scored_time"]) == [pd.Timestamp(DAY)]


def test_a_reader_prefers_the_complete_day_and_reads_only_the_range(tmp_path):
    fetch = _Fetch()
    refresh_days(tmp_path, DAY, DAY + timedelta(days=3), fetch, "scored_time", now=LATER)
    pd.DataFrame({"scored_time": [DAY], "crps": 9.0}).to_parquet(day_path(tmp_path, date(2026, 9, 15), partial=True))
    frame = read_days(tmp_path, DAY, DAY + timedelta(days=2))
    assert len(frame) == 48 and (frame["crps"] == 1.0).all()


def test_a_reader_tells_a_missing_store_from_an_empty_range(tmp_path):
    assert read_days(tmp_path / "absent", DAY, DAY + timedelta(days=1)) is None
    assert read_days(tmp_path, DAY, DAY + timedelta(days=1)).empty
