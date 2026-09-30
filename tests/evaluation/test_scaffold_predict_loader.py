"""The agent's evaluator must read every asset the verdict will score it on.

MinutePriceStore.load_range raises on a single NaN close. An untraded minute is normal on the
thin Hyperliquid-routed markets — 42% of AAPLX minutes never trade, and 291 of its 298 days
contain at least one — so that loader cannot read 9 of the 13 assets at all. Agents in two
consecutive campaigns hit this and each wrote their own harness around it, which also puts them
outside the rule that says every CRPS number comes from synth-lib.

predict.py loads through the same path the scored evaluation uses, so NaN reaches simulate()
instead of stopping the run.
"""

import numpy as np
import pandas as pd
import pytest

from synth_lib.benchmark.generate_predictions import load_minute_prices
from synth_lib.preparation.config import OHLCV_COLUMNS
from synth_lib.preparation.minute_price_store import MinutePriceStore

LO = pd.Timestamp("2026-09-05", tz="UTC")
HI = pd.Timestamp("2026-09-08", tz="UTC")


@pytest.fixture
def thin_store(tmp_path):
    """A snapshot shaped like a thin equity: most minutes never trade."""
    root = tmp_path / "market_data" / "prices" / "AAPLX" / "1m"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for day in pd.date_range("2026-09-04", "2026-09-09", freq="D"):
        index = pd.date_range(day, periods=1440, freq="1min", tz="UTC")
        close = 200 + np.cumsum(rng.normal(0, 0.02, 1440))
        close[rng.random(1440) < 0.40] = np.nan
        frame = pd.DataFrame({"timestamp": index, **{c: close for c in OHLCV_COLUMNS}})
        frame.to_parquet(root / f"date={day.date()}.parquet", index=False)
    return tmp_path / "market_data", root


def test_the_strict_loader_cannot_read_a_thin_asset(thin_store):
    """Pins why predict.py must not use it — if this ever stops raising, the workaround can go."""
    _, root = thin_store
    with pytest.raises(ValueError, match="missing minute prices"):
        MinutePriceStore("AAPLX", root=root).load_range(LO, HI)


def test_the_evaluator_reads_it_and_hands_the_gaps_to_the_model(thin_store):
    data_root, _ = thin_store
    prices = load_minute_prices(data_root, "AAPLX", LO, HI)
    assert len(prices) == 3 * 1440 + 1
    assert prices["close"].isna().any(), "the gaps must survive to simulate(), not be filled"
    assert 0.3 < prices["close"].isna().mean() < 0.5


def test_a_missing_day_still_fails(thin_store):
    """Leniency is about untraded minutes, not about a hole in the snapshot: a missing partition
    would silently shorten every context that spans it."""
    data_root, root = thin_store
    (root / "date=2026-09-06.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="missing partition"):
        load_minute_prices(data_root, "AAPLX", LO, HI)
