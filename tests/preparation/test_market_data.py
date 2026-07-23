"""Tests for synth_lib.preparation.market_data.

PythHistoryClient pagination + status handling. Regression context: the Pyth
Benchmarks TradingView shim rejects ranges over ~10k one-minute bars with
``s=error`` and an empty ``t``. The client used to return that as an empty
DataFrame, so a live miner requesting its 7-day context in one call (SPYX —
the only Pyth-routed asset post feed-migration) silently got no data and never
answered SPYX prompts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from synth_lib.preparation.market_data import PythHistoryClient

UTC = timezone.utc


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
