"""Tests for synth_lib.preparation.market_data — SPCX Hyperliquid routing.

SPCX (tokenized SpaceX) is only served with real minute data on Hyperliquid
(coin ``xyz:SPCX``, listed ~2026-07-19). The free Pyth benchmarks shim the
backtester uses returns nothing for its ``Pyth.HL.SPCX/USDC`` feed, so SPCX must
route to Hyperliquid like WTIOIL — and windows before it was listed (no candles)
must ingest as NaN rather than crash.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from synth.validator.price_data_provider import PriceDataProvider
from synth_lib.preparation.market_data import (
    HYPERLIQUID_SYMBOLS,
    PYTH_SYMBOLS,
    HyperliquidClient,
)

UTC = timezone.utc


class TestSpcxRouting:
    def test_spcx_routed_to_hyperliquid_not_pyth(self) -> None:
        assert HYPERLIQUID_SYMBOLS.get("SPCX") == "xyz:SPCX"
        assert "SPCX" not in PYTH_SYMBOLS

    def test_synth_provider_resolves_spcx_coin(self) -> None:
        # The delegated HL fetch maps asset -> coin via synth's own map, so it
        # must know SPCX or download_hyperliquid_price_data would KeyError.
        assert PriceDataProvider.HYPERLIQUID_SYMBOL_MAP.get("SPCX") == "xyz:SPCX"

    def test_backtest_price_store_uses_hyperliquid_for_spcx(self) -> None:
        from synth_lib.backtester.backtest import _price_store

        store = _price_store("SPCX")
        assert isinstance(store.client, HyperliquidClient)


class TestHyperliquidNoDataGraceful:
    """A no-data / not-yet-listed window must yield an empty frame, not crash,
    so pre-listing days ingest as NaN (downstream scoring tolerates NaN)."""

    def test_not_yet_settled_returns_empty(self) -> None:
        with patch.object(
            PriceDataProvider,
            "download_hyperliquid_price_data",
            side_effect=ValueError("realized path not yet settled for asset SPCX"),
        ):
            df = HyperliquidClient().fetch_range(
                "SPCX",
                datetime(2026, 7, 17, tzinfo=UTC),
                datetime(2026, 7, 17, 23, 59, tzinfo=UTC),
            )
        assert df.empty
        assert list(df.columns) == ["timestamp", "close"]

    def test_empty_candles_returns_empty(self) -> None:
        with patch.object(
            PriceDataProvider,
            "download_hyperliquid_price_data",
            return_value=[],
        ):
            df = HyperliquidClient().fetch_range(
                "SPCX",
                datetime(2026, 7, 17, tzinfo=UTC),
                datetime(2026, 7, 17, 23, 59, tzinfo=UTC),
            )
        assert df.empty


class TestIngestSourceLabel:
    """ingest_day must label the parquet 'source' by the actual client, so
    HL-routed assets (WTIOIL, SPCX) are not mislabeled 'pyth'."""

    def test_hyperliquid_client_labels_source_hyperliquid(self, tmp_path) -> None:
        from datetime import date

        from synth_lib.preparation.market_data import MinutePriceStore

        class _StubHL:
            source_name = "hyperliquid"

            def fetch_range(self, asset, start, end):  # noqa: ANN001
                idx = pd.date_range(start, periods=3, freq="1min", tz="UTC")
                return pd.DataFrame({"timestamp": idx, "close": [1.0, 2.0, 3.0]})

        store = MinutePriceStore("SPCX", root=tmp_path, client=_StubHL())
        df = pd.read_parquet(store.ingest_day(date(2026, 7, 20)))
        assert set(df["source"].unique()) == {"hyperliquid"}
