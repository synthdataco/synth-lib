"""Validate backtester earnings formula against on-chain leaderboard actuals.

For a user-specified window, compares the backtester's USD earnings formula
(reward_weight × daily_pool_usd / rounds_per_day, from backtest._compute_earnings_df)
against actual on-chain USD earnings derived from /leaderboard/historical
emissions proportionally applied to /v1/miners/rewards/pool USD pool.

The comparison is per the 3-competition model (crypto-1h / crypto-24h /
com-equ-24h). With --competition all (default) the three competitions'
reward_weights are summed per miner on the backtester side, matching the
competition-agnostic /leaderboard actual side.

Picks the top-N miners by actual_usd_total and prints a comparison table.

Usage:
    uv run synth_lib/backtester/scripts/validate_earnings_formula.py \\
        --from 2026-07-01 --to 2026-07-06 [--top-n 10] \\
        [--competition crypto-1h|crypto-24h|com-equ-24h|all] \\
        [--emission-factor 1.0]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from synth_lib.backtester.backtest import (
    API_SCORES_PAGE_SIZE_DAYS,
    EMISSION_NORMALIZATION_FACTOR,
    SYNTHDATA_API_BASE,
    _http_get,
    get_daily_miner_pool_usd,
    get_rewards_history,
)

UTC = timezone.utc

# The three live competition slugs accepted by /rewards/scores?prompt_name=.
# Each competition's reward_weight sums to 1/3; summing all three per miner
# reconstructs the full on-chain weight, matching the competition-agnostic
# /leaderboard actual side.
COMPETITION_SLUGS = ["crypto-1h", "crypto-24h", "com-equ-24h"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="from_date", required=True,
                        help="Window start, YYYY-MM-DD (UTC midnight, inclusive)")
    parser.add_argument("--to", dest="to_date", required=True,
                        help="Window end, YYYY-MM-DD (UTC midnight, exclusive)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top miners to compare (default 10)")
    parser.add_argument("--competition", default="all",
                        choices=[*COMPETITION_SLUGS, "all"],
                        help="Competition slug for /rewards/scores, or 'all' to "
                             "sum the three competitions per miner (default 'all')")
    parser.add_argument("--emission-factor", type=float, default=None,
                        help="Scale backtester USD by this factor. Default None "
                             "uses backtest.EMISSION_NORMALIZATION_FACTOR "
                             f"(={EMISSION_NORMALIZATION_FACTOR}). Pass 1.0 to see "
                             "the raw uncalibrated ratio.")
    args = parser.parse_args(argv)
    args.from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=UTC)
    args.to_dt = datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=UTC)
    if args.to_dt <= args.from_dt:
        parser.error("--to must be strictly after --from")
    return args


def fetch_leaderboard_historical(
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """Fetch per-epoch per-neuron emissions from /leaderboard/historical.

    Paginated in API_SCORES_PAGE_SIZE_DAYS (6-day) chunks to match the Synth
    API's behavior for other endpoints and avoid gateway timeouts on long
    windows. Returns a DataFrame with columns:
        updated_at (UTC tz-aware), neuron_uid (int), emission (float),
        incentive (float).
    Extra fields (stake, rank, pruning_score, coldkey, ip_address) are dropped.
    """
    start_time = start_time.replace(microsecond=0)
    end_time = end_time.replace(microsecond=0)
    chunks: list[pd.DataFrame] = []
    cursor = start_time
    while cursor < end_time:
        chunk_end = min(cursor + timedelta(days=API_SCORES_PAGE_SIZE_DAYS), end_time)
        if cursor >= chunk_end:
            break
        resp = _http_get(
            f"{SYNTHDATA_API_BASE}/leaderboard/historical",
            params={
                "start_time": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_time": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            df = pd.DataFrame(data)[["updated_at", "neuron_uid", "emission", "incentive"]]
            chunks.append(df)
        cursor = chunk_end

    if not chunks:
        return pd.DataFrame(columns=["updated_at", "neuron_uid", "emission", "incentive"])

    df = pd.concat(chunks, ignore_index=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df["neuron_uid"] = df["neuron_uid"].astype(int)
    df["emission"] = df["emission"].astype(float)
    df["incentive"] = df["incentive"].astype(float)
    df = df.drop_duplicates(subset=["updated_at", "neuron_uid"])
    return df


def compute_actual_usd(
    leaderboard_df: pd.DataFrame,
    daily_pool_usd: pd.Series,
    partial_coverage_threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute per-miner per-day actual USD earnings via proportional share.

    For each epoch snapshot, the miner's share of the miner pool is
        share = miner_emission / Σ_active_miners emission
    (active = incentive > 0). The miner's daily share is the mean share
    across that day's snapshots in which they were active. USD is that
    daily share × that day's USD pool.

    Days missing from daily_pool_usd are dropped (same behavior as the
    backtester side).

    Args:
        leaderboard_df: output of fetch_leaderboard_historical.
        daily_pool_usd: date → USD from get_daily_miner_pool_usd.
        partial_coverage_threshold: miners present in fewer than this
            fraction of the day's snapshots get `partial=True` flagged.

    Returns DataFrame with columns:
        neuron_uid, date (UTC midnight tz-aware), miner_daily_share,
        actual_usd, snapshots_in_day, snapshots_active, partial (bool).
    """
    if leaderboard_df.empty:
        return pd.DataFrame(columns=[
            "neuron_uid", "date", "miner_daily_share", "actual_usd",
            "snapshots_in_day", "snapshots_active", "partial",
        ])

    df = leaderboard_df.copy()
    df = df[df["incentive"] > 0]  # miners only
    if df.empty:
        return pd.DataFrame(columns=[
            "neuron_uid", "date", "miner_daily_share", "actual_usd",
            "snapshots_in_day", "snapshots_active", "partial",
        ])

    df["date"] = df["updated_at"].dt.floor("D")

    # share[uid, snapshot] = emission / total emission of active miners in that snapshot
    snapshot_totals = df.groupby("updated_at")["emission"].transform("sum")
    df["share"] = df["emission"] / snapshot_totals

    # For each miner × day: mean share across snapshots where they appeared
    agg = (
        df.groupby(["neuron_uid", "date"])
          .agg(miner_daily_share=("share", "mean"),
               snapshots_active=("share", "size"))
          .reset_index()
    )

    # Total snapshots per day (across ALL miners) to flag partial coverage
    snapshots_in_day = (
        df.groupby("date")["updated_at"].nunique().rename("snapshots_in_day")
    )
    agg = agg.join(snapshots_in_day, on="date")
    agg["partial"] = agg["snapshots_active"] <= partial_coverage_threshold * agg["snapshots_in_day"]

    # Apply USD pool; drop days missing from pool
    agg["actual_usd"] = agg["date"].map(daily_pool_usd) * agg["miner_daily_share"]
    missing_days = agg["actual_usd"].isna()
    if missing_days.any():
        dropped = sorted({d.date().isoformat() for d in agg.loc[missing_days, "date"]})
        print(f"  Warning: dropping {int(missing_days.sum())} miner-day rows on "
              f"{len(dropped)} day(s) with no /v1/miners/rewards/pool data: {dropped}")
        agg = agg.loc[~missing_days].copy()

    return agg


def compute_backtester_usd(
    rewards_df: pd.DataFrame,
    daily_pool_usd: pd.Series,
    emission_factor: float = 1.0,
) -> pd.DataFrame:
    """Compute per-miner per-day backtester-formula USD.

    Mirrors synth_lib.backtester.backtest._compute_earnings_df exactly,
    including the per-miner `rounds_per_day` grouping (backtest.py:1446),
    but for ALL miners in one pass.

    Formula per round:
        usd = reward_weight × daily_pool_usd[date] / rounds_per_day[uid, date]
              × emission_factor

    Args:
        rewards_df: output of get_rewards_history — must contain
            miner_uid, updated_at, reward_weight.
        daily_pool_usd: date → USD.
        emission_factor: on-chain emission normalization factor applied to the
            raw formula USD (see backtest.EMISSION_NORMALIZATION_FACTOR). 1.0 =
            raw uncalibrated USD.

    Returns DataFrame with columns:
        neuron_uid, date, bt_usd, rounds_in_day, mean_reward_weight.
    Rows on dates missing from daily_pool_usd are dropped with a warning.
    """
    if rewards_df.empty:
        return pd.DataFrame(columns=[
            "neuron_uid", "date", "bt_usd", "rounds_in_day", "mean_reward_weight",
        ])

    df = rewards_df[["miner_uid", "updated_at", "reward_weight", "prompt_name"]].copy()
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df["date"] = df["updated_at"].dt.floor("D")

    missing = ~df["date"].isin(daily_pool_usd.index)
    if missing.any():
        dropped = sorted({d.date().isoformat() for d in df.loc[missing, "date"]})
        print(f"  Warning: dropping {int(missing.sum())} reward rows on "
              f"{len(dropped)} day(s) with no /v1/miners/rewards/pool data: {dropped}")
        df = df.loc[~missing].copy()
        if df.empty:
            return pd.DataFrame(columns=[
                "neuron_uid", "date", "bt_usd", "rounds_in_day", "mean_reward_weight",
            ])

    # rounds_per_day MUST be per-(miner, competition): each competition has its
    # own reward-round cadence (crypto-1h ~720/day vs 24h ~288/day) and its
    # reward_weight sums to 1/3 within that competition. Grouping rounds across
    # concatenated competitions would compute a cadence-weighted AVERAGE instead
    # of the SUM of per-competition USD, halving multi-competition miners. So we
    # divide each competition's rounds separately, then sum per (miner, date).
    rounds = (
        df.groupby(["miner_uid", "date", "prompt_name"]).size().rename("rounds_in_day")
    )
    df = df.join(rounds, on=["miner_uid", "date", "prompt_name"])
    df["usd_per_round"] = (
        df["reward_weight"] * df["date"].map(daily_pool_usd) / df["rounds_in_day"]
        * emission_factor
    )

    agg = (
        df.groupby(["miner_uid", "date"])
          # bt_usd correctly sums usd_per_round across all competitions; the
          # rounds_in_day column is diagnostic only ("first" = one competition's
          # per-day round count for a multi-competition miner) and is not rendered
          # by format_table.
          .agg(bt_usd=("usd_per_round", "sum"),
               rounds_in_day=("rounds_in_day", "first"),
               mean_reward_weight=("reward_weight", "mean"))
          .reset_index()
          .rename(columns={"miner_uid": "neuron_uid"})
    )
    return agg


def format_table(
    bt_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    top_n: int,
) -> str:
    """Merge both sides on (neuron_uid, date), aggregate to totals per miner,
    rank by actual_usd_total desc, return a printable table string.
    """
    bt_totals = (
        bt_df.groupby("neuron_uid")
             .agg(bt_usd_total=("bt_usd", "sum"),
                  mean_rw=("mean_reward_weight", "mean"))
    )
    actual_totals = (
        actual_df.groupby("neuron_uid")
                 .agg(actual_usd_total=("actual_usd", "sum"),
                      any_partial=("partial", "any"))
    )
    merged = bt_totals.join(actual_totals, how="outer").fillna(
        {"bt_usd_total": 0.0, "actual_usd_total": 0.0, "mean_rw": 0.0, "any_partial": False}
    )
    merged["diff_usd"] = merged["bt_usd_total"] - merged["actual_usd_total"]
    merged["diff_pct"] = merged.apply(
        lambda r: (r["diff_usd"] / r["actual_usd_total"] * 100.0) if r["actual_usd_total"] > 0 else float("nan"),
        axis=1,
    )
    merged = merged.sort_values("actual_usd_total", ascending=False).head(top_n)

    lines = []
    lines.append("uid   | mean_rw   | bt_usd_total  | actual_usd_total | diff_$       | diff_%")
    lines.append("------+-----------+---------------+------------------+--------------+--------")
    for uid, row in merged.iterrows():
        flag = "*" if row["any_partial"] else " "
        pct = f"{row['diff_pct']:+7.2f}%" if pd.notna(row["diff_pct"]) else "    n/a"
        lines.append(
            f"{int(uid):>5}{flag}| {row['mean_rw']:.6f}  | {row['bt_usd_total']:>12,.2f}  | {row['actual_usd_total']:>15,.2f}  | {row['diff_usd']:>+11,.2f}  | {pct}"
        )
    lines.append("------+-----------+---------------+------------------+--------------+--------")
    total_bt = merged["bt_usd_total"].sum()
    total_actual = merged["actual_usd_total"].sum()
    total_diff = total_bt - total_actual
    total_pct = (total_diff / total_actual * 100.0) if total_actual > 0 else float("nan")
    total_pct_str = f"{total_pct:+7.2f}%" if pd.notna(total_pct) else "    n/a"
    lines.append(
        f"  SUM |           | {total_bt:>12,.2f}  | {total_actual:>15,.2f}  | {total_diff:>+11,.2f}  | {total_pct_str}"
    )
    if (merged["any_partial"] == True).any():
        lines.append("")
        lines.append("* = miner had partial snapshot coverage on at least one day (≤50% of day's snapshots)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    emission_factor = (
        args.emission_factor
        if args.emission_factor is not None
        else EMISSION_NORMALIZATION_FACTOR
    )

    print(f"Window: {args.from_date} → {args.to_date} "
          f"({(args.to_dt - args.from_dt).days} days)")
    print(f"Competition: {args.competition}")
    print(f"Emission factor: {emission_factor}")

    slugs = COMPETITION_SLUGS if args.competition == "all" else [args.competition]
    reward_frames = []
    for slug in slugs:
        print(f"Fetching /rewards/scores (prompt_name={slug}) ...")
        part = get_rewards_history(args.from_dt, args.to_dt, prompt_name=slug)
        if not part.empty:
            reward_frames.append(part)
    rewards_df = (
        pd.concat(reward_frames, ignore_index=True) if reward_frames else pd.DataFrame()
    )
    if rewards_df.empty:
        print("ERROR: /rewards/scores returned no rows for this window/competition.",
              file=sys.stderr)
        return 1

    print("Fetching /v1/miners/rewards/pool (daily USD pool) ...")
    daily_pool = get_daily_miner_pool_usd(args.from_dt, args.to_dt)
    if daily_pool.empty:
        print("ERROR: /v1/miners/rewards/pool returned no pool data for this window.", file=sys.stderr)
        return 1

    print("Fetching /leaderboard/historical ...")
    lb_df = fetch_leaderboard_historical(args.from_dt, args.to_dt)
    if lb_df.empty:
        print("ERROR: /leaderboard/historical returned no rows for this window.", file=sys.stderr)
        return 1

    n_snapshots = lb_df["updated_at"].nunique()
    n_miners = lb_df.loc[lb_df["incentive"] > 0, "neuron_uid"].nunique()
    print(f"Daily pool USD: mean ${float(daily_pool.mean()):,.2f}, "
          f"total ${float(daily_pool.sum()):,.2f}")
    print(f"Leaderboard snapshots: {n_snapshots:,}   Miners with incentive>0: {n_miners}")
    print()

    bt_df = compute_backtester_usd(rewards_df, daily_pool, emission_factor=emission_factor)
    actual_df = compute_actual_usd(lb_df, daily_pool)

    if bt_df.empty or actual_df.empty:
        print("ERROR: no overlapping miner-day rows between backtester and actual sides.",
              file=sys.stderr)
        return 1

    print(format_table(bt_df, actual_df, args.top_n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
