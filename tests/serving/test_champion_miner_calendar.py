"""The serving miner refreshes the event calendar and points the champion at it, as its campaign did."""

import threading
import time
from pathlib import Path

import pandas as pd
import pytest

import synth_lib.serving.champion_miner as cm
from synth_lib.preparation.event_calendar import build as calendar_build
from synth_lib.serving.champion_miner import ChampionMiner


@pytest.fixture
def miner(monkeypatch):
    """A miner whose first calendar refresh is held until the test releases it."""
    release = threading.Event()
    monkeypatch.setattr(cm, "warm_up", lambda assets: None)
    monkeypatch.setattr(cm, "venue_store", lambda asset: f"store:{asset}")
    monkeypatch.setattr(cm, "FreshTail", lambda asset: None)
    monkeypatch.setattr(ChampionMiner, "_background_refresh", lambda self: None)
    monkeypatch.setattr(ChampionMiner, "_refresh_calendar", lambda self: release.wait(timeout=10))
    monkeypatch.setenv(cm.CALENDAR_ENV, "")  # _start sets it; monkeypatch removes it after the test
    miner = object.__new__(ChampionMiner)  # Miner.__init__ needs a wallet, a subtensor and a chain
    miner.release = release
    return miner


def test_start_points_the_champion_at_the_served_calendar(miner):
    miner._start(["BTC"])
    assert Path(cm.os.environ[cm.CALENDAR_ENV]) == (Path("market_data") / "calendar" / "events.parquet").resolve()
    miner.release.set()


def test_an_asset_waits_for_the_first_calendar_refresh(miner):
    miner._start(["BTC"])
    time.sleep(0.2)
    assert "BTC" not in miner._ready  # warmed up, but its calendar is not written yet
    miner.release.set()
    deadline = time.monotonic() + 5
    while "BTC" not in miner._ready:
        assert time.monotonic() < deadline, "the asset never joined"
        time.sleep(0.01)


def test_a_calendar_that_never_arrives_delays_the_assets_once(miner, monkeypatch):
    monkeypatch.setattr(cm, "CALENDAR_WAIT_SECONDS", 0.5)
    assets = ["BTC", "ETH", "SOL", "XRP"]
    miner._start(assets)
    deadline = time.monotonic() + 1.5  # one wait, not one per asset
    while len(miner._ready) < len(assets):
        assert time.monotonic() < deadline, f"only {sorted(miner._ready)} joined"
        time.sleep(0.01)
    miner.release.set()


def test_the_refresh_asks_for_recent_history_and_the_days_ahead(monkeypatch):
    calls = []
    monkeypatch.setattr(cm.calendar_build, "update", lambda root, start, end, now: calls.append((start, end, now)))
    miner = object.__new__(ChampionMiner)
    miner._calendar_root = Path("market_data")
    miner._refresh_calendar()
    start, end, now = calls[0]
    assert isinstance(now, pd.Timestamp) and now.tzinfo is not None
    assert (now.date() - start).days == cm.CALENDAR_HISTORY_DAYS
    assert (end - now.date()).days == calendar_build.DAYS_AHEAD


def test_a_failed_refresh_never_stops_serving(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cm.calendar_build, "update", broken)
    miner = object.__new__(ChampionMiner)
    miner._calendar_root = Path("market_data")
    miner._refresh_calendar()  # logged, not raised
