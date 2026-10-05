"""The miner serves while its price store warms up, one asset at a time.

A synchronous warm-up kept the axon down until every asset was fetched, and each missed prompt is
filled at the field's 95th percentile. An asset is served once its own warm-up window is in, never
from a partly warmed store, whose newest bar can be days old.
"""

import asyncio
import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from synth.simulation_input import SimulationInput  # type: ignore[import-untyped]

import synth_lib.serving.champion_miner as cm
from synth_lib.serving.champion_miner import ChampionMiner


@pytest.fixture
def miner(monkeypatch):
    """A miner whose XAU warm-up is held until the test releases it."""
    release_xau = threading.Event()
    warmed: list[str] = []

    def warm_up(assets):
        if assets == ["XAU"]:
            release_xau.wait(timeout=10)
        warmed.extend(assets)

    monkeypatch.setattr(cm, "warm_up", warm_up)
    monkeypatch.setattr(cm, "venue_store", lambda asset: f"store:{asset}")
    monkeypatch.setattr(cm, "FreshTail", lambda asset: None)
    monkeypatch.setattr(ChampionMiner, "_background_refresh", lambda self: None)
    monkeypatch.setattr(cm, "serve_request", lambda fn, store, sim_input, tail: (store,))
    miner = object.__new__(ChampionMiner)  # Miner.__init__ needs a wallet, a subtensor and a chain
    miner.release_xau, miner.warmed = release_xau, warmed
    return miner


def _wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the background warm-up never got there"
        time.sleep(0.01)


def _forward(miner, asset):
    sim_input = SimulationInput(
        asset=asset,
        start_time=datetime(2026, 9, 30, 12, tzinfo=UTC).isoformat(),
        time_increment=300,
        time_length=86_400,
        num_simulations=3,
    )
    return asyncio.run(miner.forward_miner(SimpleNamespace(simulation_input=sim_input, simulation_output=None)))


def test_start_returns_before_the_warm_up_finishes(miner):
    miner._start(["BTC", "XAU"])
    assert "XAU" not in miner._ready
    miner.release_xau.set()
    _wait_until(lambda: miner._ready == {"BTC", "XAU"})
    assert miner.warmed == ["BTC", "XAU"]


def test_an_asset_is_served_as_soon_as_its_own_warm_up_is_done(miner):
    miner._start(["BTC", "XAU"])
    _wait_until(lambda: "BTC" in miner._ready)
    assert _forward(miner, "BTC").simulation_output == ("store:BTC",)
    with pytest.raises(RuntimeError, match="XAU is still warming up"):
        _forward(miner, "XAU")
    miner.release_xau.set()
    _wait_until(lambda: "XAU" in miner._ready)
    assert _forward(miner, "XAU").simulation_output == ("store:XAU",)


def test_an_asset_with_no_live_venue_still_fails_loudly(miner):
    miner._start(["BTC"])
    with pytest.raises(KeyError):
        _forward(miner, "SPYX")
