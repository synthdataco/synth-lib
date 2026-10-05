"""The smoothing replay follows the live validator's window, miner identity and save order."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from synth.validator.competition_config import CRYPTO_24H, SMOOTHED_SCORE_COEFFICIENT
from synth.validator.moving_average import prepare_df_for_moving_average

from synth_lib.backtester.scoring import (
    _first_saved_first,
    calculate_smoothed_scores,
    compute_combined_smoothed_scores,
)

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

    def test_the_combined_field_names_each_occupant(self) -> None:
        """Both rounds report uid 5; miner_id says which occupant held it."""
        before = self.R - timedelta(hours=1)
        results = [
            SimpleNamespace(
                prompt_df=pd.DataFrame(self._field()),
                smoothed_scores=pd.DataFrame({"updated_at": [before, U]}),
            )
        ]
        out = compute_combined_smoothed_scores(
            results, competition=CRYPTO_24H, registrations={5: self.R.to_pydatetime()}
        )
        held = out.loc[out["miner_uid"] == 5].set_index("updated_at")["miner_id"]
        assert held.to_dict() == {before: 1_000_005, U: 5}


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


class TestSaveOrder:
    """BTC and ETH are both scored at T; the validator saved ETH's scores first. uid 9 joins later."""

    T = U - timedelta(days=2)

    def _frame(self) -> pd.DataFrame:
        rows = []
        for asset, p95 in (("BTC", 40.0), ("ETH", 60.0)):  # the frame lists BTC first
            for scored, uid in ((self.T, 1), (U - timedelta(days=1), 1), (U - timedelta(days=1), 9)):
                rows.append(
                    {
                        "scored_time": scored,
                        "miner_id": uid,
                        "asset": asset,
                        "prompt_score_v3": 1.0,
                        "percentile95": p95,
                        "lowest_score": 0.0,
                    }
                )
        return pd.DataFrame(rows)

    def _backfill_at_t(self, frame: pd.DataFrame) -> list[tuple[str, float]]:
        prepared = prepare_df_for_moving_average(frame)
        rows = prepared.loc[(prepared["miner_id"] == 9) & (prepared["scored_time"] == self.T)]
        return list(zip(rows["asset"], rows["prompt_score_v3"]))

    def test_a_late_joiner_is_backfilled_from_the_prompt_saved_first(self) -> None:
        ordered = _first_saved_first(self._frame(), {self.T: ["ETH", "BTC"]})
        assert self._backfill_at_t(ordered) == [("ETH", 60.0)]

    def test_without_the_order_the_frame_order_decides(self) -> None:
        assert self._backfill_at_t(self._frame()) == [("BTC", 40.0)]

    def test_the_combined_field_is_ordered_before_the_backfill(self, monkeypatch) -> None:
        import synth_lib.backtester.scoring as scoring

        seen: list[pd.DataFrame] = []

        def spy(df):
            seen.append(df)
            return prepare_df_for_moving_average(df)

        monkeypatch.setattr(scoring, "prepare_df_for_moving_average", spy)
        frame = self._frame().rename(columns={"miner_id": "miner_uid", "prompt_score_v3": "new_prompt_scores"})
        results = [
            SimpleNamespace(
                prompt_df=frame.loc[frame["asset"] == asset],
                smoothed_scores=pd.DataFrame({"updated_at": [U]}),
            )
            for asset in ("BTC", "ETH")
        ]
        scoring.compute_combined_smoothed_scores(results, competition=CRYPTO_24H, save_order={self.T: ["ETH", "BTC"]})
        first_at_t = seen[0].loc[seen[0]["scored_time"] == self.T].iloc[0]
        assert first_at_t["asset"] == "ETH"
