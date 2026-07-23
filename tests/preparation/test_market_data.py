"""Tests for synth_lib.preparation.market_data — price routing + Pyth pagination.

Since the PR #304 feed migration the validator sources prices from Binance
(crypto majors), Hyperliquid (HYPE + every commodity/equity, incl. SP500 and
SPCX) and Pyth (deprecated SPYX tail). market_data mirrors that routing, and
windows with no settled candles (unlisted asset / gap) must ingest as NaN
rather than crash.

The Pyth path also paginates: the Benchmarks TradingView shim rejects ranges
over ~10k one-minute bars with ``s=error`` and an empty ``t``. The client used
to return that as an empty DataFrame, so a live miner requesting its 7-day
context in one call (SPYX — the only Pyth-routed asset post feed-migration)
silently got no data and never answered SPYX prompts.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from synth.validator.price_data_provider import PriceDataProvider
from synth_lib.preparation.market_data import (
    ALL_SYMBOLS,
    BINANCE_SYMBOLS,
    HYPERLIQUID_SYMBOLS,
    PYTH_SYMBOLS,
    BinanceClient,
    HyperliquidClient,
    MinutePriceStore,
    PythHistoryClient,
    build_price_client,
)

UTC = timezone.utc


class TestAssetRouting:
    def test_sp500_routed_to_hyperliquid(self) -> None:
        assert HYPERLIQUID_SYMBOLS.get("SP500") == "xyz:SP500"
        assert "SP500" not in BINANCE_SYMBOLS
        assert "SP500" not in PYTH_SYMBOLS
        assert isinstance(build_price_client("SP500"), HyperliquidClient)

    def test_backtest_price_store_uses_hyperliquid_for_sp500(self) -> None:
        from synth_lib.backtester.backtest import _price_store

        assert isinstance(_price_store("SP500").client, HyperliquidClient)

    def test_spcx_still_hyperliquid(self) -> None:
        assert HYPERLIQUID_SYMBOLS.get("SPCX") == "xyz:SPCX"
        assert isinstance(build_price_client("SPCX"), HyperliquidClient)

    def test_crypto_majors_routed_to_binance(self) -> None:
        assert BINANCE_SYMBOLS.get("BTC") == "BTCUSDT"
        for asset in ("BTC", "ETH", "SOL", "XRP"):
            assert isinstance(build_price_client(asset), BinanceClient)

    def test_spyx_routed_to_pyth(self) -> None:
        assert PYTH_SYMBOLS.get("SPYX") == "Crypto.SPYX/USD"
        assert isinstance(build_price_client("SPYX"), PythHistoryClient)

    def test_maps_are_disjoint_and_union_is_all(self) -> None:
        b, h, p = set(BINANCE_SYMBOLS), set(HYPERLIQUID_SYMBOLS), set(PYTH_SYMBOLS)
        assert b.isdisjoint(h) and b.isdisjoint(p) and h.isdisjoint(p)
        assert set(ALL_SYMBOLS) == b | h | p

    def test_unknown_asset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported asset"):
            build_price_client("DOGE")

    def test_pinned_provider_is_migrated(self) -> None:
        # Sanity that the installed synth-subnet is the post-#304 migration:
        # SP500 lives on Hyperliquid and the old HYPERLIQUID_SYMBOL_MAP is gone.
        assert PriceDataProvider.HYPERLIQUID_ASSET_MAP.get("SP500") == "xyz:SP500"
        assert not hasattr(PriceDataProvider, "HYPERLIQUID_SYMBOL_MAP")


class TestNoDataGraceful:
    """A no-data / not-yet-settled window must yield an empty frame, not crash,
    so unlisted-asset and gap days ingest as NaN (downstream tolerates NaN)."""

    def test_hyperliquid_not_yet_settled_returns_empty(self) -> None:
        with patch.object(
            PriceDataProvider,
            "download_hyperliquid_price_data",
            side_effect=ValueError("realized path not yet settled for asset SP500"),
        ):
            df = HyperliquidClient().fetch_range(
                "SP500",
                datetime(2026, 7, 10, tzinfo=UTC),
                datetime(2026, 7, 10, 23, 59, tzinfo=UTC),
            )
        assert df.empty
        assert list(df.columns) == ["timestamp", "close"]

    def test_binance_not_yet_settled_returns_empty(self) -> None:
        with patch.object(
            PriceDataProvider,
            "download_binance_price_data",
            side_effect=ValueError("realized path not yet settled for asset BTC"),
        ):
            df = BinanceClient().fetch_range(
                "BTC",
                datetime(2026, 7, 10, tzinfo=UTC),
                datetime(2026, 7, 10, 23, 59, tzinfo=UTC),
            )
        assert df.empty
        assert list(df.columns) == ["timestamp", "close"]


class TestIngestSourceLabel:
    """ingest_day labels the parquet 'source' by the actual client, so
    HL/Binance-routed assets are not mislabeled 'pyth'."""

    @staticmethod
    def _stub(source: str):
        class _Stub:
            source_name = source

            def fetch_range(self, asset, start, end):  # noqa: ANN001
                idx = pd.date_range(start, periods=3, freq="1min", tz="UTC")
                return pd.DataFrame({"timestamp": idx, "close": [1.0, 2.0, 3.0]})

        return _Stub()

    def test_hyperliquid_source_label(self, tmp_path) -> None:
        store = MinutePriceStore("SP500", root=tmp_path, client=self._stub("hyperliquid"))
        df = pd.read_parquet(store.ingest_day(date(2026, 7, 21)))
        assert set(df["source"].unique()) == {"hyperliquid"}

    def test_binance_source_label(self, tmp_path) -> None:
        store = MinutePriceStore("BTC", root=tmp_path, client=self._stub("binance"))
        df = pd.read_parquet(store.ingest_day(date(2026, 7, 21)))
        assert set(df["source"].unique()) == {"binance"}


def _response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    return resp


def _ok_payload(start_ts: int, n_bars: int) -> dict:
    ts = [start_ts + 60 * i for i in range(n_bars)]
    return {"s": "ok", "t": ts, "c": [100.0 + i for i in range(n_bars)]}


class TestPythHistoryClientPagination:
    @patch("synth_lib.preparation.market_data.requests.get")
    def test_seven_day_range_is_paginated_under_the_shim_cap(self, mock_get: MagicMock) -> None:
        """A 7-day request must split into <= MAX_DAYS_PER_REQUEST slices (the
        shim errors above ~10k bars) and concat to one contiguous frame."""
        start = datetime(2026, 7, 15, tzinfo=UTC)
        end = datetime(2026, 7, 22, tzinfo=UTC)

        def per_chunk(url, params, timeout):
            n_bars = (params["to"] - params["from"]) // 60
            assert n_bars <= PythHistoryClient.MAX_DAYS_PER_REQUEST * 24 * 60, "chunk exceeds the shim cap"
            return _response(_ok_payload(params["from"], n_bars))

        mock_get.side_effect = per_chunk
        frame = PythHistoryClient().fetch_range("SPYX", start, end)

        assert mock_get.call_count == 2  # 5d + 2d
        assert len(frame) == 7 * 24 * 60
        assert frame["timestamp"].is_monotonic_increasing
        assert not frame["timestamp"].duplicated().any()

    @patch("synth_lib.preparation.market_data.requests.get")
    def test_boundary_duplicates_are_dropped(self, mock_get: MagicMock) -> None:
        """Chunks sharing a boundary timestamp must not duplicate that bar."""
        start = datetime(2026, 7, 15, tzinfo=UTC)
        end = datetime(2026, 7, 21, tzinfo=UTC)

        def per_chunk(url, params, timeout):
            # Emit one extra bar at the chunk end == next chunk's first bar.
            n_bars = (params["to"] - params["from"]) // 60 + 1
            return _response(_ok_payload(params["from"], n_bars))

        mock_get.side_effect = per_chunk
        frame = PythHistoryClient().fetch_range("SPYX", start, end)
        assert not frame["timestamp"].duplicated().any()

    @patch("synth_lib.preparation.market_data._time.sleep")
    @patch("synth_lib.preparation.market_data.requests.get")
    def test_shim_error_status_raises_instead_of_returning_empty(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """s=error with empty t must raise (after retries), never masquerade
        as a data gap — the silent-empty behavior is what broke SPYX serving."""
        mock_get.return_value = _response({"s": "error", "errmsg": "range too large", "t": [], "c": []})
        with pytest.raises(RuntimeError, match="Pyth API failed"):
            PythHistoryClient(max_retries=2).fetch_range(
                "SPYX", datetime(2026, 7, 21, tzinfo=UTC), datetime(2026, 7, 22, tzinfo=UTC)
            )

    @patch("synth_lib.preparation.market_data.requests.get")
    def test_no_data_status_returns_empty_frame(self, mock_get: MagicMock) -> None:
        """s=no_data (e.g. closed market) is a legitimate empty response."""
        mock_get.return_value = _response({"s": "no_data", "t": [], "c": []})
        frame = PythHistoryClient().fetch_range(
            "SPYX", datetime(2026, 7, 21, tzinfo=UTC), datetime(2026, 7, 22, tzinfo=UTC)
        )
        assert isinstance(frame, pd.DataFrame)
        assert frame.empty
        assert mock_get.call_count == 1  # no retries for a legitimate empty
