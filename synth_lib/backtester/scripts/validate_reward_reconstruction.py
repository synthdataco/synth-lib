"""Validate the backtester's smoothing/aggregation pipeline against the API.

This is a MEASUREMENT script, independent of the price source. It does not
re-score predictions; instead it feeds the API's real per-prompt CRPS through
the exact same smoothing/aggregation code the backtester uses
(`compute_combined_smoothed_scores`) and compares the reconstructed per-round
reward_weights against the real on-chain reward_weights from /rewards/scores.

For each asset in the competition it builds a minimal per-asset BacktestResult
(mirroring how backtest() builds prompt_df at its "Step 12"): the API CRPS is
grouped per (scored_time, asset, time_length, time_increment) through
_compute_prompt_score_stats_for_group to produce new_prompt_scores /
percentile90 / lowest_score. The union of scored_times drives the reconstruction
rounds. compute_combined_smoothed_scores then applies ASSET_COEFFICIENTS,
per-miner normalization and the single cross-asset softmax — exactly the real
validator path.

Reconstructed weights (at scored_times) are aligned to real weights (at reward
rounds) per miner via a nearest merge_asof within a small tolerance, and the
absolute weight error is summarized. No hard threshold is asserted.

Note: CRPS is fetched only over [--from, --to]; the moving-average lookback is
`competition.window_days`, so the earliest reconstruction rounds see a truncated
history window and their weights are expected to drift from on-chain values.
Prefer reading the later-round behavior / interpret early-round error with that
in mind.

Usage:
    uv run synth_lib/backtester/scripts/validate_reward_reconstruction.py \\
        --competition crypto-24h --from 2026-07-02 --to 2026-07-05
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from synth_lib.backtester.backtest import (
    SLUG_TO_COMPETITION,
    BacktestResult,
    _compute_prompt_score_stats_for_group,
    compute_combined_smoothed_scores,
    get_miner_scores,
    get_rewards_history,
)

UTC = timezone.utc

# Align reconstruction rounds (at scored_times) to on-chain reward rounds (at
# updated_at). A round-cadence-scale tolerance for the nearest match.
ALIGN_TOLERANCE = pd.Timedelta(hours=1)
WEIGHT_MATCH_TOL = 1e-3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--competition", required=True,
                        choices=sorted(SLUG_TO_COMPETITION.keys()),
                        help="Single competition slug (required)")
    parser.add_argument("--from", dest="from_date", required=True,
                        help="Window start, YYYY-MM-DD (UTC midnight, inclusive)")
    parser.add_argument("--to", dest="to_date", required=True,
                        help="Window end, YYYY-MM-DD (UTC midnight, exclusive)")
    args = parser.parse_args(argv)
    args.from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=UTC)
    args.to_dt = datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=UTC)
    if args.to_dt <= args.from_dt:
        parser.error("--to must be strictly after --from")
    return args


def build_asset_result(scores: pd.DataFrame, asset: str) -> BacktestResult:
    """Build a minimal per-asset BacktestResult from raw API CRPS.

    Mirrors backtest()'s "Step 12": groups crps per
    (scored_time, asset, time_length, time_increment) through
    _compute_prompt_score_stats_for_group to attach new_prompt_scores,
    percentile90 and lowest_score. smoothed_scores carries only the unique
    scored_times as `updated_at` so compute_combined_smoothed_scores has
    timestamps to iterate.
    """
    df = scores.copy()
    stats = df.groupby(
        ["scored_time", "asset", "time_length", "time_increment"], group_keys=False
    )["crps"].apply(_compute_prompt_score_stats_for_group)
    df["new_prompt_scores"] = stats["new_prompt_scores"]
    df["percentile90"] = stats["percentile90"]
    df["lowest_score"] = stats["lowest_score"]

    smoothed = pd.DataFrame(
        {"updated_at": pd.to_datetime(df["scored_time"].unique(), utc=True)}
    ).sort_values("updated_at").reset_index(drop=True)

    return BacktestResult(
        miner_name=f"reconstruction_{asset}",
        prompt_df=df,
        smoothed_scores=smoothed,
        summary={"asset": asset, "n_rows": int(len(df))},
    )


def align_weights(
    recon: pd.DataFrame,
    real: pd.DataFrame,
    tolerance: pd.Timedelta = ALIGN_TOLERANCE,
) -> pd.DataFrame:
    """Align reconstructed and real reward_weights per miner via nearest merge_asof.

    Both frames must have columns updated_at, miner_uid, reward_weight.
    Returns the aligned frame with reward_weight_recon, reward_weight_real and
    abs_err, keeping only rows that matched a real round within `tolerance`.
    """
    left = (
        recon[["updated_at", "miner_uid", "reward_weight"]]
        .rename(columns={"reward_weight": "reward_weight_recon"})
        .sort_values("updated_at")
        .reset_index(drop=True)
    )
    right = (
        real[["updated_at", "miner_uid", "reward_weight"]]
        .rename(columns={"reward_weight": "reward_weight_real"})
        .sort_values("updated_at")
        .reset_index(drop=True)
    )
    left["miner_uid"] = left["miner_uid"].astype(int)
    right["miner_uid"] = right["miner_uid"].astype(int)

    merged = pd.merge_asof(
        left,
        right,
        on="updated_at",
        by="miner_uid",
        direction="nearest",
        tolerance=tolerance,
    )
    merged = merged.dropna(subset=["reward_weight_real"])
    merged["abs_err"] = (
        merged["reward_weight_recon"] - merged["reward_weight_real"]
    ).abs()
    return merged


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    competition = SLUG_TO_COMPETITION[args.competition]

    print(f"Competition: {args.competition} ({competition.label})")
    print(f"Window: {args.from_date} → {args.to_date} "
          f"({(args.to_dt - args.from_dt).days} days)")
    print(f"Assets: {competition.asset_list}")
    print(f"window_days (moving-average lookback): {competition.window_days}")
    print()

    results: list[BacktestResult] = []
    for asset in competition.asset_list:
        print(f"Fetching /validation/scores/historical for {asset} ...", flush=True)
        scores = get_miner_scores(
            args.from_dt,
            args.to_dt,
            asset,
            competition.time_length,
            competition.time_increment,
        )
        if scores.empty:
            print(f"  (no scores for {asset}, skipping)")
            continue
        print(f"  {len(scores)} score rows, {scores['scored_time'].nunique()} prompts")
        results.append(build_asset_result(scores, asset))

    if not results:
        print("ERROR: no CRPS data for any asset in this window.", file=sys.stderr)
        return 1

    print("\nReconstructing combined smoothed scores / reward weights ...", flush=True)
    recon = compute_combined_smoothed_scores(results, competition)
    if recon.empty:
        print("ERROR: reconstruction produced no rows.", file=sys.stderr)
        return 1
    recon["updated_at"] = pd.to_datetime(recon["updated_at"], utc=True)

    print(f"Fetching real /rewards/scores (prompt_name={args.competition}) ...",
          flush=True)
    real = get_rewards_history(args.from_dt, args.to_dt, prompt_name=args.competition)
    if real.empty:
        print("ERROR: /rewards/scores returned no rows for this window.",
              file=sys.stderr)
        return 1
    real["updated_at"] = pd.to_datetime(real["updated_at"], utc=True)

    # Per-round sum of reconstructed weights (sanity: should be ≈ 1/3).
    round_sums = recon.groupby("updated_at")["reward_weight"].sum()

    aligned = align_weights(recon, real)

    print("\n=== Reconstruction vs on-chain reward_weight ===")
    print(f"Reconstruction rounds:     {recon['updated_at'].nunique()}")
    print(f"On-chain reward rounds:    {real['updated_at'].nunique()}")
    print(f"Aligned rows (miner×round): {len(aligned)}")
    if aligned.empty:
        print("No aligned rows within tolerance "
              f"{ALIGN_TOLERANCE} — cannot measure weight error.")
    else:
        mean_err = float(aligned["abs_err"].mean())
        max_err = float(aligned["abs_err"].max())
        frac_within = float((aligned["abs_err"] <= WEIGHT_MATCH_TOL).mean())
        print(f"Mean abs weight error:     {mean_err:.6e}")
        print(f"Max abs weight error:      {max_err:.6e}")
        print(f"Fraction within {WEIGHT_MATCH_TOL:g}:     {frac_within:.4f}")
    print("\n=== Reconstructed per-round weight-sum sanity (target ≈ 1/3 = "
          f"{1/3:.6f}) ===")
    print(f"Rounds:      {len(round_sums)}")
    print(f"Mean sum:    {float(round_sums.mean()):.6f}")
    print(f"Min/Max sum: {float(round_sums.min()):.6f} / {float(round_sums.max()):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
