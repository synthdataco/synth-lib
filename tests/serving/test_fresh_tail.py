"""A crypto-1h context ends on the bar the verdict's does, not on the store's last refresh.

The store settles today's partition two minutes behind its refresh, so between refreshes the newest
stored bar trails the prompt's start by four minutes or more, where the verdict hands the champion
the bar labelled start - 2 min. FreshTail fetches what closed in between, at request time.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
from synth.simulation_input import SimulationInput  # type: ignore[import-untyped]

import synth_lib.serving.champion_miner as cm
from synth_lib.preparation.config import OHLCV_COLUMNS
from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.serving.champion_miner import ChampionMiner
from synth_lib.serving.serve import EMPTY_TAIL, FreshTail, build_context

START = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
ISSUED = START - timedelta(seconds=59)  # the validator sends a crypto-1h prompt a minute ahead


def _bars(first: datetime, n: int, close: float = 100.0) -> pd.DataFrame:
    index = pd.date_range(first, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"timestamp": index, **{c: close + np.arange(n) for c in OHLCV_COLUMNS}})


class _Client:
    def __init__(self, fail: bool = False):
        self.calls: list[tuple[datetime, datetime]] = []
        self.fail = fail

    def fetch_range(self, asset, start, end):
        self.calls.append((start, end))
        if self.fail:
            raise TimeoutError("venue too slow")
        return _bars(start, int((end - start) / timedelta(minutes=1)) + 1)


def test_the_tail_ends_on_the_bar_the_verdict_ends_on():
    client = _Client()
    bars = FreshTail("BTC", client=client).bars(START, now=ISSUED)
    assert client.calls == [(START - timedelta(minutes=16), START - timedelta(minutes=2))]
    assert bars["timestamp"].iloc[-1] == START - timedelta(minutes=2)


def test_a_late_clock_never_reaches_past_the_verdicts_bar():
    client = _Client()
    FreshTail("BTC", client=client).bars(START, now=START + timedelta(seconds=30))
    assert client.calls[0][1] == START - timedelta(minutes=2)


def test_one_fetch_per_minute_shared_by_every_prompt():
    client = _Client()
    tail = FreshTail("BTC", client=client)
    tail.bars(START, now=ISSUED)
    tail.bars(START, now=ISSUED + timedelta(seconds=3))  # a second validator, same minute
    assert len(client.calls) == 1
    tail.bars(START + timedelta(minutes=1), now=ISSUED + timedelta(minutes=1))
    assert len(client.calls) == 2


def test_a_failed_fetch_yields_no_bars():
    assert FreshTail("BTC", client=_Client(fail=True)).bars(START, now=ISSUED).empty


def _store_settled_until(tmp_path, last_settled: datetime) -> MinutePriceStore:
    """Today's partition as ingest_day writes it: every minute, NaN past the last settled one."""
    root = tmp_path / "BTC" / "1m"
    root.mkdir(parents=True)
    for day in pd.date_range(START.date() - timedelta(days=7), START.date(), freq="D"):
        frame = _bars(day.to_pydatetime().replace(tzinfo=UTC), 1440)
        frame.loc[frame["timestamp"] > last_settled, OHLCV_COLUMNS] = np.nan
        frame.to_parquet(root / f"date={day.date().isoformat()}.parquet", index=False)
    return MinutePriceStore("BTC", root=root)


def test_tail_bars_replace_the_minutes_the_store_has_not_settled(tmp_path):
    store = _store_settled_until(tmp_path, START - timedelta(minutes=5))
    assert build_context(store, START).index[-1] == START - timedelta(minutes=5)
    tail = _bars(START - timedelta(minutes=16), 15, close=7.0)
    context = build_context(store, START, tail=tail)
    assert context.index[-1] == START - timedelta(minutes=2)
    assert context["close"].iloc[-1] == tail["close"].iloc[-1]


def test_no_tail_leaves_the_stored_context_unchanged(tmp_path):
    store = _store_settled_until(tmp_path, START - timedelta(minutes=5))
    pd.testing.assert_frame_equal(build_context(store, START, tail=EMPTY_TAIL), build_context(store, START))


def _synapse(time_length: int):
    sim_input = SimulationInput(
        asset="BTC", start_time=START.isoformat(), time_increment=60, time_length=time_length, num_simulations=3
    )
    return SimpleNamespace(simulation_input=sim_input, simulation_output=None)


def test_only_crypto_1h_prompts_fetch_a_tail(monkeypatch):
    class _Tail:
        def __init__(self):
            self.calls = 0

        def bars(self, start_time):
            self.calls += 1
            return EMPTY_TAIL

    served = []
    monkeypatch.setattr(cm, "serve_request", lambda fn, store, sim_input, tail: served.append(tail) or ("ok",))
    miner = object.__new__(ChampionMiner)  # Miner.__init__ needs a wallet, a subtensor and a chain
    miner._stores, miner._tails, miner._ready = {"BTC": None}, {"BTC": _Tail()}, {"BTC"}

    asyncio.run(miner.forward_miner(_synapse(86_400)))
    assert miner._tails["BTC"].calls == 0 and served == [None]
    asyncio.run(miner.forward_miner(_synapse(3_600)))
    assert miner._tails["BTC"].calls == 1 and served[-1] is EMPTY_TAIL
