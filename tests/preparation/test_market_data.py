"""Tests for the synth_lib.preparation package — price routing, Pyth pagination,
validator-API clients and the realized-path store.

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
from synth_lib.preparation.binance_client import BinanceClient
from synth_lib.preparation.config import (
    ALL_SYMBOLS,
    BINANCE_SYMBOLS,
    HYPERLIQUID_SYMBOLS,
    RETIRED_SYMBOLS,
)
from synth_lib.preparation.hyperliquid_client import HyperliquidClient
from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.preparation.price_client import build_price_client
from synth_lib.preparation.realized_path_store import (
    RealizedPathStore,
    prefetch_realized_paths,
)
from synth_lib.preparation.validator_api import (
    get_prompt_start_times,
    get_realized_path,
    snap_to_prompt_start,
)

UTC = timezone.utc


class TestAssetRouting:
    def test_sp500_routed_to_hyperliquid(self) -> None:
        assert HYPERLIQUID_SYMBOLS.get("SP500") == "xyz:SP500"
        assert "SP500" not in BINANCE_SYMBOLS
        assert "SP500" not in RETIRED_SYMBOLS
        assert isinstance(build_price_client("SP500"), HyperliquidClient)

    def test_backtest_price_store_uses_hyperliquid_for_sp500(self) -> None:
        from synth_lib.backtester.loading import _price_store

        assert isinstance(_price_store("SP500").client, HyperliquidClient)

    def test_spcx_still_hyperliquid(self) -> None:
        assert HYPERLIQUID_SYMBOLS.get("SPCX") == "xyz:SPCX"
        assert isinstance(build_price_client("SPCX"), HyperliquidClient)

    def test_crypto_majors_routed_to_binance(self) -> None:
        assert BINANCE_SYMBOLS.get("BTC") == "BTCUSDT"
        for asset in ("BTC", "ETH", "SOL", "XRP"):
            assert isinstance(build_price_client(asset), BinanceClient)

    def test_retired_feed_asset_raises_instead_of_routing(self) -> None:
        """SPYX was Pyth-only and Pyth is gone. Routing must refuse rather than hand back a client
        that fetches nothing: empty partitions look exactly like real coverage downstream."""
        assert RETIRED_SYMBOLS.get("SPYX") == "Crypto.SPYX/USD"
        with pytest.raises(ValueError, match="retired"):
            build_price_client("SPYX")

    def test_maps_are_disjoint_and_fetchable_union_excludes_retired(self) -> None:
        b, h, r = set(BINANCE_SYMBOLS), set(HYPERLIQUID_SYMBOLS), set(RETIRED_SYMBOLS)
        assert b.isdisjoint(h) and b.isdisjoint(r) and h.isdisjoint(r)
        assert set(ALL_SYMBOLS) == b | h, "ALL_SYMBOLS is what can be fetched; retired feeds are not"

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


def _api_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    return resp


class TestGetRealizedPath:
    """GET /validation/realized-path — per-prompt realized paths."""

    @patch("synth_lib.preparation.validator_api.requests.get")
    def test_returns_indexed_series(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _api_response(
            {
                "start_time": "2026-07-22T17:05:00Z",
                "asset": "HYPE",
                "time_increment": 60,
                "time_length": 180,
                "real_prices": [58.1, 58.2, 58.3, 58.4],
            }
        )
        series = get_realized_path("HYPE", datetime(2026, 7, 22, 17, 5, tzinfo=UTC), 180, 60)

        assert series is not None
        assert len(series) == 4
        assert series.index[0] == pd.Timestamp("2026-07-22T17:05:00Z")
        assert series.index[-1] == pd.Timestamp("2026-07-22T17:08:00Z")
        assert float(series.iloc[-1]) == 58.4

    @patch("synth_lib.preparation.validator_api.requests.get")
    def test_not_found_returns_none_without_raising(self, mock_get: MagicMock) -> None:
        """The endpoint answers 404 for a prompt it has no path for — a normal
        outcome, not an error, so raise_for_status must not be reached."""
        mock_get.return_value = _api_response({"message": "No realized path available"}, status_code=404)

        assert get_realized_path("HYPE", datetime(2026, 3, 25, 2, 39, tzinfo=UTC), 3600, 60) is None

    @patch("synth_lib.preparation.validator_api.requests.get")
    def test_venue_gaps_come_back_as_nan(self, mock_get: MagicMock) -> None:
        """real_prices is nullable: the validator stores None where the venue
        had no candle. Those must survive as NaN, not blow up the parse."""
        mock_get.return_value = _api_response({"real_prices": [10.0, None, 12.0]})
        series = get_realized_path("WTIOIL", datetime(2026, 7, 22, tzinfo=UTC), 120, 60)

        assert series is not None
        assert series.isna().tolist() == [False, True, False]


class TestGetPromptStartTimes:
    @patch("synth_lib.preparation.validator_api.requests.get")
    def test_paginates_and_dedupes(self, mock_get: MagicMock) -> None:
        """The endpoint caps ranges at 60 days, so a 120-day window must page,
        and overlapping pages must not duplicate a start_time."""
        mock_get.return_value = _api_response({"start_times": ["2026-06-01T00:01:00Z", "2026-06-01T01:03:00Z"]})
        starts = get_prompt_start_times(
            "XAU", 86_400, 300, datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 7, 30, tzinfo=UTC)
        )

        assert mock_get.call_count == 3  # 50 + 50 + 20 days
        assert starts == [pd.Timestamp("2026-06-01T00:01:00Z"), pd.Timestamp("2026-06-01T01:03:00Z")]

    @patch("synth_lib.preparation.validator_api.requests.get")
    def test_empty_range_returns_empty_list(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _api_response({"start_times": None})

        assert (
            get_prompt_start_times(
                "XAU", 86_400, 300, datetime(2026, 7, 1, tzinfo=UTC), datetime(2026, 7, 2, tzinfo=UTC)
            )
            == []
        )


class TestSnapToPromptStart:
    """scored_time - time_length overshoots the real start by the scoring delay,
    so the exact prompt at or just before the approximation is the right match."""

    EXACT = [
        pd.Timestamp("2026-07-20T00:01:00Z"),
        pd.Timestamp("2026-07-20T01:03:00Z"),
        pd.Timestamp("2026-07-20T02:07:00Z"),
    ]

    def test_prefers_the_earlier_candidate(self) -> None:
        # 01:05 sits 2 min after 01:03 and 62 min after 00:01 — and 62 min
        # before 02:07, which would be the wrong direction anyway.
        assert snap_to_prompt_start(pd.Timestamp("2026-07-20T01:05:00Z"), self.EXACT) == self.EXACT[1]

    def test_accepts_a_later_candidate_when_no_earlier_one_is_in_range(self) -> None:
        assert snap_to_prompt_start(pd.Timestamp("2026-07-19T23:59:00Z"), self.EXACT) == self.EXACT[0]

    def test_returns_none_beyond_tolerance(self) -> None:
        assert snap_to_prompt_start(pd.Timestamp("2026-07-20T05:00:00Z"), self.EXACT) is None

    def test_naive_input_is_treated_as_utc(self) -> None:
        assert snap_to_prompt_start(datetime(2026, 7, 20, 1, 5), self.EXACT) == self.EXACT[1]

    def test_empty_prompt_list_returns_none(self) -> None:
        assert snap_to_prompt_start(pd.Timestamp("2026-07-20T01:05:00Z"), []) is None


class TestRealizedPathStore:
    START = pd.Timestamp("2026-05-02T01:03:00Z")

    @staticmethod
    def _store(tmp_path) -> RealizedPathStore:  # noqa: ANN001
        return RealizedPathStore("XAU", 86_400, 300, root=tmp_path)

    @patch("synth_lib.preparation.realized_path_store.time.sleep")
    @patch("synth_lib.preparation.realized_path_store.get_realized_path")
    def test_prefetch_persists_and_get_matches_an_approximate_start(
        self, mock_fetch: MagicMock, mock_sleep: MagicMock, tmp_path
    ) -> None:
        mock_fetch.return_value = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.date_range(self.START, periods=3, freq="5min"),
        )
        assert self._store(tmp_path).prefetch([self.START]) == 1

        # A fresh store proves it round-tripped through parquet, and the lookup
        # is by approximate start (here 4 minutes late, as scoring delay makes it).
        series = self._store(tmp_path).get(self.START + pd.Timedelta(minutes=4))
        assert series is not None
        assert series.to_list() == [1.0, 2.0, 3.0]
        assert series.index[0] == self.START

    @patch("synth_lib.preparation.realized_path_store.time.sleep")
    @patch("synth_lib.preparation.realized_path_store.get_realized_path")
    def test_prefetch_skips_cached_prompts(self, mock_fetch: MagicMock, mock_sleep: MagicMock, tmp_path) -> None:
        """Already-cached prompts are not re-fetched, so a long backfill resumes."""
        mock_fetch.return_value = pd.Series([1.0], index=[self.START])
        self._store(tmp_path).prefetch([self.START])

        assert self._store(tmp_path).prefetch([self.START]) == 0
        assert mock_fetch.call_count == 1

    @patch("synth_lib.preparation.realized_path_store.time.sleep")
    @patch("synth_lib.preparation.realized_path_store.get_realized_path")
    def test_unavailable_prompt_is_not_cached(self, mock_fetch: MagicMock, mock_sleep: MagicMock, tmp_path) -> None:
        mock_fetch.return_value = None
        store = self._store(tmp_path)

        assert store.prefetch([self.START]) == 0
        assert store.get(self.START) is None

    def test_get_never_hits_the_network(self, tmp_path) -> None:
        """Reads are local-only so offline backtests work off a warmed store."""
        with patch("synth_lib.preparation.validator_api.requests.get") as mock_get:
            assert self._store(tmp_path).get(self.START) is None
        mock_get.assert_not_called()

    @patch("synth_lib.preparation.realized_path_store.time.sleep")
    @patch("synth_lib.preparation.realized_path_store.get_realized_path")
    def test_get_searches_the_previous_day_partition(
        self, mock_fetch: MagicMock, mock_sleep: MagicMock, tmp_path
    ) -> None:
        """An approximation just past midnight must still find its prompt, which
        is partitioned under the previous day."""
        exact = pd.Timestamp("2026-05-01T23:58:00Z")
        mock_fetch.return_value = pd.Series([7.0], index=[exact])
        self._store(tmp_path).prefetch([exact])

        assert self._store(tmp_path).get(pd.Timestamp("2026-05-02T00:04:00Z")) is not None

    @patch("synth_lib.preparation.realized_path_store.time.sleep")
    @patch("synth_lib.preparation.realized_path_store.get_realized_path")
    @patch("synth_lib.preparation.realized_path_store.get_prompt_start_times")
    def test_prefetch_realized_paths_fetches_only_snapped_prompts(
        self, mock_prompts: MagicMock, mock_fetch: MagicMock, mock_sleep: MagicMock, tmp_path
    ) -> None:
        """Density tapering leaves far fewer scored prompts than issued ones, so
        only the prompts we actually need are fetched — not the whole listing."""
        wanted = pd.Timestamp("2026-05-02T01:03:00Z")
        mock_prompts.return_value = [
            pd.Timestamp("2026-05-02T00:58:00Z"),
            wanted,
            pd.Timestamp("2026-05-02T01:08:00Z"),
        ]
        mock_fetch.return_value = pd.Series([1.0], index=[wanted])

        store = self._store(tmp_path)
        mapping = prefetch_realized_paths(store, [wanted + pd.Timedelta(minutes=2)])

        assert list(mapping.values()) == [wanted]
        assert mock_fetch.call_count == 1
        assert mock_fetch.call_args.args[1] == wanted

    def test_no_start_times_makes_no_requests(self, tmp_path) -> None:
        with patch("synth_lib.preparation.realized_path_store.get_prompt_start_times") as mock_prompts:
            assert prefetch_realized_paths(self._store(tmp_path), []) == {}
        mock_prompts.assert_not_called()
