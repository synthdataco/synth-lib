"""CRPS scoring and the validator's smoothed-score / reward-weight replay."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synth.validator.competition_config import (
    ALL_COMPETITIONS,
    CRYPTO_24H,
    SMOOTHED_SCORE_COEFFICIENT,
    CompetitionConfig,
)
from synth.validator.crps_calculation import (
    calculate_crps_for_miner,
    calculate_total_score_for_miner,
)
from synth.validator.moving_average import (
    compute_smoothed_score,
    prepare_df_for_moving_average,
)
from synth.validator.reward import compute_prompt_scores

from synth_lib.backtester.config import (
    _COMBINED_EMPTY_COLS,
    _LEGACY_FALLBACK_WINDOW_DAYS,
    UTC,
    OUTLIER_CAP_DATE,
    VOL_CRPS_1H_DATE,
    competition_for,
    slug_for,
)
from synth_lib.backtester.loading import load_prediction
from synth_lib.backtester.miner_data_handler import _BACKTEST_MDH, _BacktestMinerDataHandler
from synth_lib.backtester.prompt_scores_legacy import compute_prompt_scores_pre_cap
from synth_lib.backtester.result import BacktestResult


def _compute_prompt_scores_for_group(crps: pd.Series) -> pd.Series:
    """Wrapper around synth's compute_prompt_scores for use in groupby.apply."""
    result = compute_prompt_scores(crps.values)
    if result[0] is None:
        return pd.Series(0.0, index=crps.index)
    return pd.Series(result[0], index=crps.index)


def _compute_prompt_score_stats_for_group(crps: pd.Series, capped_era: bool = True) -> pd.DataFrame:
    """Like _compute_prompt_scores_for_group but also returns percentile95 and
    lowest_score so synth.prepare_df_for_moving_average can apply its worst-score
    backfill rule to new miners. The name must match what that function reads.

    `capped_era` selects the validator's scoring as of the group's scored_time: the outlier clip
    has only existed since OUTLIER_CAP_DATE.
    """
    # The validator also returns a per-row was_capped flag, which it persists so the outlier
    # clip rate stays monitorable. Nothing downstream here reads it.
    scorer = compute_prompt_scores if capped_era else compute_prompt_scores_pre_cap
    capped, p95, low, _was_capped = scorer(crps.values)
    n = len(crps)
    if capped is None:
        return pd.DataFrame(
            {
                "new_prompt_scores": [0.0] * n,
                "percentile95": [0.0] * n,
                "lowest_score": [0.0] * n,
            },
            index=crps.index,
        )
    return pd.DataFrame(
        {
            "new_prompt_scores": capped,
            "percentile95": [float(p95)] * n,
            "lowest_score": [float(low)] * n,
        },
        index=crps.index,
    )


# A prompt scored in the minute before an update is not yet counted by it. Measured on 352 live
# crypto-24h updates (2026-09-12 -> 09-23): every update with a prompt in that minute pointed there.
_UPDATE_LAG = pd.Timedelta(minutes=1)


def _live_window(prepared: pd.DataFrame, updated_at: Any, cutoff_days: int) -> pd.DataFrame:
    """The rows the validator's update at `updated_at` averages: scored in (u - cutoff_days, u - 1 min).

    The lower bound is the validator's SQL (`scored_time > :min_scored_time`); the upper one is
    behavioural, see _UPDATE_LAG. Both ends are open.
    """
    scored = prepared["scored_time"]
    return prepared.loc[(scored > updated_at - pd.Timedelta(days=cutoff_days)) & (scored < updated_at - _UPDATE_LAG)]


# miner_id given to the previous occupant of a re-registered uid: uid + this, far above any uid.
_PREVIOUS_OCCUPANT = 1_000_000


def _split_reregistered(
    df: pd.DataFrame, registrations: Mapping[int, datetime] | None, time_length: int
) -> pd.DataFrame:
    """Give the previous occupant of each re-registered uid its own miner_id.

    Live keys a miner by hotkey, the public scores by uid. A uid re-registered at R is two miners
    live: the rows of prompts started before R belong to the previous occupant, the rest to the new
    one, which `prepare_df_for_moving_average` then backfills as a late joiner. `df` carries
    `miner_id` (= uid) and `scored_time`; a prompt starts `time_length` seconds before it is scored.
    Only the latest registration of a uid is known here, so an earlier one inside the span stays
    merged.
    """
    if not registrations:
        return df
    df = df.copy()
    start = pd.to_datetime(df["scored_time"], utc=True) - pd.Timedelta(seconds=time_length)
    for uid, registered_at in registrations.items():
        previous = (df["miner_id"] == uid) & (start < pd.Timestamp(registered_at))
        df.loc[previous, "miner_id"] = uid + _PREVIOUS_OCCUPANT
    return df


class _RegisteredAtUpdate(_BacktestMinerDataHandler):
    """miner_id -> uid only for miners registered at the update being computed.

    `compute_smoothed_score` drops a miner whose uid resolves to None, which is how the validator
    leaves a deregistered hotkey out of the softmax. Set `updated_at` before each call.
    """

    def __init__(self, registrations: Mapping[int, datetime], scores: pd.DataFrame) -> None:
        super().__init__()
        self.registrations = {uid: pd.Timestamp(t) for uid, t in registrations.items()}
        # The new occupant's first real row. Before it is in a window, live has no rows for the
        # new occupant, so it is absent; the whole-frame backfill would otherwise pay it the worst
        # score from R on. `scores` is the split frame, before prepare_df_for_moving_average.
        mine = scores.loc[scores["miner_id"].isin(self.registrations)]
        self.first_scored = pd.to_datetime(mine["scored_time"], utc=True).groupby(mine["miner_id"]).min().to_dict()
        self.updated_at: pd.Timestamp | None = None

    def populate_miner_uid_in_miner_data(self, miner_data: list[dict]) -> list[dict]:
        u = pd.Timestamp(self.updated_at)
        for row in miner_data:
            miner_id = int(row["miner_id"])
            if miner_id >= _PREVIOUS_OCCUPANT:
                uid = miner_id - _PREVIOUS_OCCUPANT
                row["miner_uid"] = uid if u < self.registrations[uid] else None
            else:
                registered_at = self.registrations.get(miner_id)
                first = self.first_scored.get(miner_id)
                counted = first is not None and first < u - _UPDATE_LAG
                row["miner_uid"] = miner_id if registered_at is None or (u >= registered_at and counted) else None
        return miner_data


def _miner_data_handler(
    registrations: Mapping[int, datetime] | None, scores: pd.DataFrame
) -> _BacktestMinerDataHandler:
    return _RegisteredAtUpdate(registrations, scores) if registrations else _BACKTEST_MDH


def _smoothed_at(
    mdh: _BacktestMinerDataHandler,
    prepared: pd.DataFrame,
    updated_at: Any,
    cutoff_days: int,
    competition: CompetitionConfig,
) -> list[dict] | None:
    if isinstance(mdh, _RegisteredAtUpdate):
        mdh.updated_at = pd.Timestamp(updated_at)
    return compute_smoothed_score(mdh, _live_window(prepared, updated_at, cutoff_days), updated_at, competition)


def calculate_smoothed_scores(
    all_scores: pd.DataFrame,
    rewards_history: pd.DataFrame,
    cutoff_days: int = 10,
    scores_column: str = "new_prompt_scores",
    competition: CompetitionConfig = CRYPTO_24H,
    registrations: Mapping[int, datetime] | None = None,
) -> pd.DataFrame:
    """Compute smoothed scores and reward weights using synth's compute_smoothed_score.

    `registrations` maps a uid to the chain time its current occupant registered. With it, a uid
    re-registered inside the data is scored as two miners, as live does (see _split_reregistered);
    without it, every uid is one miner.

    Delegates to synth.validator.moving_average.compute_smoothed_score for each
    rewards_history timestamp, using a fake MinerDataHandler (miner_uid == miner_id).

    Returns DataFrame with columns: updated_at, miner_uid, new_smoothed_score, reward_weight.
    """
    # Adapt column names to what synth expects: miner_id, prompt_score_v3
    input_df = all_scores.copy()
    # Use miner_uid as miner_id for synth; drop any existing miner_id to avoid duplication
    if "miner_id" in input_df.columns:
        input_df = input_df.drop(columns=["miner_id"])
    input_df = input_df.rename(columns={"miner_uid": "miner_id", scores_column: "prompt_score_v3"})
    input_df["scored_time"] = pd.to_datetime(input_df["scored_time"])
    input_df = _split_reregistered(input_df, registrations, competition.time_length)

    # Prepare the df (backfill new miners, etc.)
    prepared = prepare_df_for_moving_average(input_df)

    mdh = _miner_data_handler(registrations, input_df)
    result_rows = []
    for updated_at in rewards_history["updated_at"].sort_values().unique():
        rewards = _smoothed_at(mdh, prepared, updated_at, cutoff_days, competition)
        if rewards is None:
            continue

        for r in rewards:
            result_rows.append(
                {
                    "updated_at": pd.Timestamp(r["updated_at"]),
                    "miner_uid": r["miner_uid"],
                    "new_smoothed_score": r["smoothed_score"],
                    "reward_weight": r["reward_weight"],
                }
            )

    return pd.DataFrame(result_rows)


def compute_combined_smoothed_scores(
    results: list[BacktestResult],
    competition: CompetitionConfig = CRYPTO_24H,
    cutoff_days: int | None = None,
    simulate_registration: datetime | None = None,
    registrations: Mapping[int, datetime] | None = None,
) -> pd.DataFrame:
    """Real-validator-equivalent cross-asset smoothed scores.

    Concatenates per-asset CRPS frames into one multi-asset frame, then calls
    synth's compute_smoothed_score once per rewards round (union of updated_at
    timestamps across all per-asset smoothed_scores). That function applies
    ASSET_COEFFICIENTS, per-miner coefficient-sum normalization, and a single
    softmax across all miners — matching the real validator rather than our
    previous un-weighted hand-rolled aggregation.

    Returns DataFrame with columns: updated_at, miner_uid, new_smoothed_score,
    reward_weight. reward_weight sums to SMOOTHED_SCORE_COEFFICIENT (1/3)
    across miners per timestamp. `registrations`: as in calculate_smoothed_scores.
    """
    if not results:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)

    # Default the leaderboard window to the competition's own setting.
    if cutoff_days is None:
        cutoff_days = competition.window_days

    # Concat per-asset prompt_df frames. percentile95 and lowest_score must be
    # carried through so synth's prepare_df_for_moving_average can backfill new
    # miners (it silently skips backfill when those columns are absent).
    cols = [
        "scored_time",
        "miner_uid",
        "asset",
        "new_prompt_scores",
        "percentile95",
        "lowest_score",
    ]
    frames = []
    for r in results:
        if r.prompt_df.empty:
            continue
        df = r.prompt_df[[c for c in cols if c in r.prompt_df.columns]].copy()
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)
    combined_crps = pd.concat(frames, ignore_index=True)

    # Adapt column names to what synth expects: miner_id, prompt_score_v3
    if "miner_id" in combined_crps.columns:
        combined_crps = combined_crps.drop(columns=["miner_id"])
    combined_crps = combined_crps.rename(columns={"miner_uid": "miner_id", "new_prompt_scores": "prompt_score_v3"})
    combined_crps["scored_time"] = pd.to_datetime(combined_crps["scored_time"])
    combined_crps = _split_reregistered(combined_crps, registrations, competition.time_length)

    prepared = prepare_df_for_moving_average(combined_crps)

    # Union of rewards-round timestamps across all per-asset smoothed_scores
    timestamps: set[pd.Timestamp] = set()
    for r in results:
        if r.smoothed_scores.empty:
            continue
        for t in pd.to_datetime(r.smoothed_scores["updated_at"]).unique():
            timestamps.add(pd.Timestamp(t))

    mdh = _miner_data_handler(registrations, combined_crps)
    result_rows = []
    for updated_at in sorted(timestamps):
        rewards = _smoothed_at(mdh, prepared, updated_at, cutoff_days, competition)
        if rewards is None:
            continue
        for row in rewards:
            result_rows.append(
                {
                    "updated_at": pd.Timestamp(row["updated_at"]),
                    "miner_uid": row["miner_uid"],
                    "new_smoothed_score": row["smoothed_score"],
                    "reward_weight": row["reward_weight"],
                }
            )

    if not result_rows:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)
    if simulate_registration is not None:
        # Simulating registration → show the backfilled onboarding period as-is.
        return pd.DataFrame(result_rows)
    return _trim_warmup(
        pd.DataFrame(result_rows),
        combined_crps,
        warmup_days=cutoff_days,
    )


def _score_single_prompt(
    file_path: Path | None,
    start_time: Any,
    asset_val: str,
    scored_time: Any,
    time_len: int,
    time_incr: int,
    real_prices: list[float],
    scoring_intervals: dict[str, int],
    vol_scoring_blocks: dict[str, tuple[int, float]],
    miner_id: int,
) -> dict:
    """Score a single prompt's prediction against real prices. Runs in a worker process.

    Returns a dict with scoring results, or a dict with crps=-1 for missing predictions.
    """
    if file_path is None:
        return {
            "miner_uid": miner_id,
            "scored_time": scored_time,
            "crps": -1,
            "asset": asset_val,
            "start_time": start_time,
            "time_increment": time_incr,
            "time_length": time_len,
            "miner_id": miner_id,
        }

    llm_predictions_raw = load_prediction(file_path)
    simulation_runs = np.asarray(llm_predictions_raw["paths"], dtype=float)
    # A non-finite path scores as a miss, the same as no prediction at all: compute_prompt_scores
    # fills -1 with the prompt's 95th percentile. Scoring it would give a NaN CRPS, and the
    # percentile is taken over every miner on that prompt, so one NaN drops the prompt from the
    # whole field instead of costing the champion anything.
    if not np.isfinite(simulation_runs).all():
        return {
            "miner_uid": miner_id,
            "scored_time": scored_time,
            "crps": -1,
            "asset": asset_val,
            "start_time": start_time,
            "time_increment": time_incr,
            "time_length": time_len,
            "miner_id": miner_id,
        }
    real_price_array = np.asarray(real_prices, dtype=float)
    if vol_scoring_blocks and pd.Timestamp(scored_time) >= VOL_CRPS_1H_DATE:
        total_crps, _ = calculate_total_score_for_miner(
            simulation_runs, real_price_array, time_incr, scoring_intervals, vol_scoring_blocks
        )
    else:
        total_crps, _ = calculate_crps_for_miner(simulation_runs, real_price_array, time_incr, scoring_intervals)

    return {
        "miner_uid": miner_id,
        "scored_time": scored_time,
        "crps": float(total_crps),
        "asset": asset_val,
        "start_time": start_time,
        "time_increment": time_incr,
        "time_length": time_len,
        "miner_id": miner_id,
    }


def _trim_warmup(
    smoothed_scores: pd.DataFrame,
    scored: pd.DataFrame,
    warmup_days: int = _LEGACY_FALLBACK_WINDOW_DAYS,
    warmup_anchor: datetime | None = None,
) -> pd.DataFrame:
    """Drop smoothed_scores rows whose `updated_at` is in the first `warmup_days`
    after `warmup_anchor` (or after `scored["scored_time"].min()` if no anchor).

    During that window the moving average is dominated by synth's new-miner
    worst-score backfill, so ranks are not directly comparable.
    No-op if trimming would leave the frame empty (short-window backtests / tests).
    """
    if smoothed_scores.empty or scored.empty:
        return smoothed_scores
    anchor = pd.Timestamp(warmup_anchor) if warmup_anchor is not None else pd.Timestamp(scored["scored_time"].min())
    warmup_end = anchor + pd.Timedelta(days=warmup_days)
    trimmed = smoothed_scores.loc[smoothed_scores["updated_at"] >= warmup_end]
    if trimmed.empty:
        return smoothed_scores
    return trimmed.reset_index(drop=True)
