"""CRPS scoring and the validator's smoothed-score / reward-weight replay."""

from __future__ import annotations

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
from synth.validator.crps_calculation import calculate_crps_for_miner
from synth.validator.moving_average import (
    compute_smoothed_score,
    prepare_df_for_moving_average,
)
from synth.validator.reward import compute_prompt_scores

from synth_lib.backtester.config import (
    _COMBINED_EMPTY_COLS,
    _LEGACY_FALLBACK_WINDOW_DAYS,
    UTC,
    competition_for,
    slug_for,
)
from synth_lib.backtester.loading import load_prediction
from synth_lib.backtester.miner_data_handler import _BACKTEST_MDH
from synth_lib.backtester.result import BacktestResult



def _compute_prompt_scores_for_group(crps: pd.Series) -> pd.Series:
    """Wrapper around synth's compute_prompt_scores for use in groupby.apply."""
    result = compute_prompt_scores(crps.values)
    if result[0] is None:
        return pd.Series(0.0, index=crps.index)
    return pd.Series(result[0], index=crps.index)


def _compute_prompt_score_stats_for_group(crps: pd.Series) -> pd.DataFrame:
    """Like _compute_prompt_scores_for_group but also returns percentile90 and
    lowest_score so synth.prepare_df_for_moving_average can apply its worst-score
    backfill rule to new miners.
    """
    capped, p90, low = compute_prompt_scores(crps.values)
    n = len(crps)
    if capped is None:
        return pd.DataFrame(
            {
                "new_prompt_scores": [0.0] * n,
                "percentile90": [0.0] * n,
                "lowest_score": [0.0] * n,
            },
            index=crps.index,
        )
    return pd.DataFrame(
        {
            "new_prompt_scores": capped,
            "percentile90": [float(p90)] * n,
            "lowest_score": [float(low)] * n,
        },
        index=crps.index,
    )

def calculate_smoothed_scores(
    all_scores: pd.DataFrame,
    rewards_history: pd.DataFrame,
    cutoff_days: int = 10,
    scores_column: str = "new_prompt_scores",
    competition: CompetitionConfig = CRYPTO_24H,
) -> pd.DataFrame:
    """Compute smoothed scores and reward weights using synth's compute_smoothed_score.

    Delegates to synth.validator.moving_average.compute_smoothed_score for each
    rewards_history timestamp, using a fake MinerDataHandler (miner_uid == miner_id).

    Returns DataFrame with columns: updated_at, miner_uid, new_smoothed_score, reward_weight.
    """
    # Adapt column names to what synth expects: miner_id, prompt_score_v3
    input_df = all_scores.copy()
    # Use miner_uid as miner_id for synth; drop any existing miner_id to avoid duplication
    if "miner_id" in input_df.columns:
        input_df = input_df.drop(columns=["miner_id"])
    input_df = input_df.rename(
        columns={"miner_uid": "miner_id", scores_column: "prompt_score_v3"}
    )
    input_df["scored_time"] = pd.to_datetime(input_df["scored_time"])

    # Prepare the df (backfill new miners, etc.)
    prepared = prepare_df_for_moving_average(input_df)

    result_rows = []
    for updated_at in rewards_history["updated_at"].sort_values().unique():
        cutoff = updated_at - pd.Timedelta(days=cutoff_days)
        window_df = prepared.loc[
            (prepared["scored_time"] >= cutoff)
            & (prepared["scored_time"] <= updated_at)
        ]
        rewards = compute_smoothed_score(
            _BACKTEST_MDH, window_df, updated_at, competition
        )
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
    across miners per timestamp.
    """
    if not results:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)

    # Default the leaderboard window to the competition's own setting.
    if cutoff_days is None:
        cutoff_days = competition.window_days

    # Concat per-asset prompt_df frames. percentile90 and lowest_score must be
    # carried through so synth's prepare_df_for_moving_average can backfill new
    # miners (it silently skips backfill when those columns are absent).
    cols = [
        "scored_time",
        "miner_uid",
        "asset",
        "new_prompt_scores",
        "percentile90",
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
    combined_crps = combined_crps.rename(
        columns={"miner_uid": "miner_id", "new_prompt_scores": "prompt_score_v3"}
    )
    combined_crps["scored_time"] = pd.to_datetime(combined_crps["scored_time"])

    prepared = prepare_df_for_moving_average(combined_crps)

    # Union of rewards-round timestamps across all per-asset smoothed_scores
    timestamps: set[pd.Timestamp] = set()
    for r in results:
        if r.smoothed_scores.empty:
            continue
        for t in pd.to_datetime(r.smoothed_scores["updated_at"]).unique():
            timestamps.add(pd.Timestamp(t))

    result_rows = []
    for updated_at in sorted(timestamps):
        cutoff = updated_at - pd.Timedelta(days=cutoff_days)
        window_df = prepared.loc[
            (prepared["scored_time"] >= cutoff)
            & (prepared["scored_time"] <= updated_at)
        ]
        rewards = compute_smoothed_score(
            _BACKTEST_MDH, window_df, updated_at, competition
        )
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
    real_price_array = np.asarray(real_prices, dtype=float)
    total_crps, _ = calculate_crps_for_miner(
        simulation_runs, real_price_array, time_incr, scoring_intervals
    )

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
    anchor = (
        pd.Timestamp(warmup_anchor)
        if warmup_anchor is not None
        else pd.Timestamp(scored["scored_time"].min())
    )
    warmup_end = anchor + pd.Timedelta(days=warmup_days)
    trimmed = smoothed_scores.loc[smoothed_scores["updated_at"] >= warmup_end]
    if trimmed.empty:
        return smoothed_scores
    return trimmed.reset_index(drop=True)
