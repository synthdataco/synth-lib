"""The smoothing replay follows the live validator's window and miner identity.

Both rules were measured against /rewards/scores on 352 crypto-24h updates (2026-09-12 -> 09-23).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from synth_lib.backtester.scoring import calculate_smoothed_scores

U = pd.Timestamp(datetime(2026, 9, 20, 12, 0, tzinfo=UTC))


def _rows(uid: int, offsets_and_scores: list[tuple[timedelta, float]]) -> list[dict]:
    return [
        {
            "scored_time": U - off,
            "miner_uid": uid,
            "asset": "BTC",
            "new_prompt_scores": score,
            "percentile95": 50.0,
            "lowest_score": 0.0,
            "time_length": 86400,
        }
        for off, score in offsets_and_scores
    ]


def _smoothed(scores: list[dict], **kw) -> pd.Series:
    out = calculate_smoothed_scores(pd.DataFrame(scores), pd.DataFrame({"updated_at": [U]}), cutoff_days=10, **kw)
    return out.set_index("miner_uid")["new_smoothed_score"]


class TestLiveWindow:
    def test_both_ends_are_open_and_the_last_minute_is_not_yet_counted(self) -> None:
        offsets = [timedelta(days=10), timedelta(days=9), timedelta(minutes=2), timedelta(minutes=1), timedelta(0)]
        scores = _rows(1, list(zip(offsets, [1000.0, 10.0, 20.0, 3000.0, 5000.0])))
        scores += _rows(2, [(off, 1.0) for off in offsets])
        # Only the rows scored at u - 9 d and u - 2 min are inside (u - 10 d, u - 1 min).
        assert _smoothed(scores)[1] == pytest.approx((10.0 + 20.0) / 2)
