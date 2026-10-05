"""The smoothing replay follows the live validator's window and miner identity.

Both rules were measured against /rewards/scores on 352 crypto-24h updates (2026-09-12 -> 09-23).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from synth.validator.competition_config import SMOOTHED_SCORE_COEFFICIENT

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


class TestReregisteredUid:
    """uid 5 changes hands at R = u - 3 d; uid 1 is present throughout and sets the field's timestamps."""

    R = U - timedelta(days=3)
    TIMES = [timedelta(days=d) for d in (8, 6, 4, 2, 1)]

    def _field(self) -> list[dict]:
        scores = _rows(1, [(off, 1.0) for off in self.TIMES])
        # Previous occupant: prompts started before R (scored up to R + 24 h), all scored 7.
        # New occupant: prompts started from R, first scored at u - 2 d, both scored 3.
        scores += _rows(5, [(timedelta(days=8), 7.0), (timedelta(days=6), 7.0), (timedelta(days=4), 7.0)])
        scores += _rows(5, [(timedelta(days=2), 3.0), (timedelta(days=1), 3.0)])
        return scores

    def test_merged_without_registrations(self) -> None:
        assert _smoothed(self._field())[5] == pytest.approx((7.0 * 3 + 3.0 * 2) / 5)

    def test_new_occupant_is_a_late_joiner_and_the_previous_one_left_the_softmax(self) -> None:
        out = calculate_smoothed_scores(
            pd.DataFrame(self._field()),
            pd.DataFrame({"updated_at": [U]}),
            cutoff_days=10,
            registrations={5: self.R.to_pydatetime()},
        )
        # One row per registered miner: the previous occupant is not paid at u >= R.
        assert sorted(out["miner_uid"]) == [1, 5]
        # Backfilled with percentile95 - lowest_score = 50 at the three earlier field timestamps.
        assert out.set_index("miner_uid").loc[5, "new_smoothed_score"] == pytest.approx((50.0 * 3 + 3.0 * 2) / 5)
        assert out["reward_weight"].sum() == pytest.approx(SMOOTHED_SCORE_COEFFICIENT)

    def test_previous_occupant_is_paid_until_it_is_replaced(self) -> None:
        before = self.R - timedelta(hours=1)
        out = calculate_smoothed_scores(
            pd.DataFrame(self._field()),
            pd.DataFrame({"updated_at": [before]}),
            cutoff_days=10,
            registrations={5: self.R.to_pydatetime()},
        )
        # Only the rows scored before that update (u - 8 d, u - 6 d, u - 4 d) are in its window.
        assert out.set_index("miner_uid").loc[5, "new_smoothed_score"] == pytest.approx(7.0)

    def test_new_occupant_is_not_paid_before_its_first_scored_prompt(self) -> None:
        # Between R and the new occupant's first scored prompt live has no rows for it, and the
        # previous occupant is already deregistered: nobody holds uid 5.
        out = calculate_smoothed_scores(
            pd.DataFrame(self._field()),
            pd.DataFrame({"updated_at": [self.R + timedelta(hours=12)]}),
            cutoff_days=10,
            registrations={5: self.R.to_pydatetime()},
        )
        assert list(out["miner_uid"]) == [1]


class TestUidThatChangedHandsTwice:
    """uid 5 changes hands at R1 = u - 5.5 d and again at R2 = u - 3.5 d: three miners live."""

    R1 = U - timedelta(days=5, hours=12)
    R2 = U - timedelta(days=3, hours=12)

    def _field(self) -> list[dict]:
        scores = _rows(1, [(timedelta(days=d), 1.0) for d in (8, 6, 4, 2, 1)])
        scores += _rows(5, [(timedelta(days=8), 7.0), (timedelta(days=6), 7.0)])  # started before R1
        scores += _rows(5, [(timedelta(days=4), 5.0)])  # started at u - 5 d: between R1 and R2
        scores += _rows(5, [(timedelta(days=2), 3.0), (timedelta(days=1), 3.0)])  # started after R2
        return scores

    def _smoothed_at(self, updated_at: pd.Timestamp) -> pd.Series:
        out = calculate_smoothed_scores(
            pd.DataFrame(self._field()),
            pd.DataFrame({"updated_at": [updated_at]}),
            cutoff_days=10,
            registrations={5: [self.R2.to_pydatetime(), self.R1.to_pydatetime()]},
        )
        return out.set_index("miner_uid")["new_smoothed_score"]

    def test_the_middle_occupant_holds_the_uid_between_the_two_registrations(self) -> None:
        # Its one row (u - 4 d) is counted; the two field timestamps before it are backfilled at 50.
        assert self._smoothed_at(U - timedelta(days=3, hours=18))[5] == pytest.approx((50.0 * 2 + 5.0) / 3)

    def test_the_current_occupant_holds_it_after_the_second(self) -> None:
        # Backfilled at the three field timestamps before its first row, not paid the middle one's 5.
        assert self._smoothed_at(U)[5] == pytest.approx((50.0 * 3 + 3.0 * 2) / 5)
