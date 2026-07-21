"""Generate a scatter plot: backtester vs on-chain actual earnings per miner.

Fetches reward-weights for each of the three live competitions (crypto-1h,
crypto-24h, com-equ-24h) and sums them per miner (matches the backtester's
grand-total convention — each competition's reward_weight sums to 1/3, the
three summing to the full on-chain weight), then plots against on-chain actuals
from /leaderboard/historical.

Note: this diagnostic intentionally plots the RAW backtester USD
(emission_factor=1.0), NOT the calibrated estimate — the point is to expose the
raw reward_weight-vs-realized-emission gap that backtest.EMISSION_NORMALIZATION_FACTOR
corrects (~+44% overestimate). For the calibrated per-miner estimate use
validate_earnings_formula.py / the earnings charts instead.

Usage:
    uv run synth_lib/backtester/scripts/plot_earnings_comparison.py \\
        --from 2026-07-01 --to 2026-07-06 [--top-n 10] [--out <path>]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from synth_lib.backtester.backtest import get_daily_miner_pool_usd, get_rewards_history
from synth_lib.backtester.scripts.validate_earnings_formula import (
    COMPETITION_SLUGS,
    compute_actual_usd,
    compute_backtester_usd,
    fetch_leaderboard_historical,
)

UTC = timezone.utc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD (UTC inclusive)")
    parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD (UTC exclusive)")
    parser.add_argument("--top-n", type=int, default=10, help="Top N miners by actual USD (default 10)")
    parser.add_argument("--out", type=Path, default=None, help="Output path (default: cwd/earnings_comparison.png)")
    args = parser.parse_args(argv)
    args.from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=UTC)
    args.to_dt = datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=UTC)
    if args.to_dt <= args.from_dt:
        parser.error("--to must be strictly after --from")
    if args.out is None:
        args.out = Path.cwd() / "earnings_comparison.png"
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    days = (args.to_dt - args.from_dt).days

    print(f"Window: {args.from_date} → {args.to_date} ({days} days)")

    print("Fetching /v1/miners/rewards/pool (daily USD pool) ...")
    daily_pool = get_daily_miner_pool_usd(args.from_dt, args.to_dt)
    print("Fetching /leaderboard/historical ...")
    lb_df = fetch_leaderboard_historical(args.from_dt, args.to_dt)

    if daily_pool.empty or lb_df.empty:
        print("ERROR: one or more data sources returned empty.", file=sys.stderr)
        return 1

    per_competition_totals = []
    for slug in COMPETITION_SLUGS:
        print(f"Fetching /rewards/scores ({slug}) ...")
        rewards = get_rewards_history(args.from_dt, args.to_dt, prompt_name=slug)
        if rewards.empty:
            print(f"  (no rewards for {slug})")
            continue
        bt = compute_backtester_usd(rewards, daily_pool)
        per_competition_totals.append(
            bt.groupby("neuron_uid")["bt_usd"].sum().rename(slug)
        )

    if not per_competition_totals:
        print("ERROR: /rewards/scores returned no rows for any competition.",
              file=sys.stderr)
        return 1

    actual = compute_actual_usd(lb_df, daily_pool)
    actual_totals = actual.groupby("neuron_uid")["actual_usd"].sum().rename("actual")

    merged = pd.concat([*per_competition_totals, actual_totals], axis=1).fillna(0.0)
    merged["bt_total"] = merged[[s for s in COMPETITION_SLUGS if s in merged.columns]].sum(axis=1)
    top = merged.sort_values("actual", ascending=False).head(args.top_n).copy()

    pct_bias = (top["bt_total"].sum() / top["actual"].sum() - 1) * 100
    max_usd = max(top["bt_total"].max(), top["actual"].max())
    lim = max_usd * 1.08

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, lim], [0, lim], ls="--", color="gray", lw=1.2, label="y = x (perfect match)")
    ax.scatter(top["actual"], top["bt_total"], s=120, color="tab:blue", edgecolor="white", zorder=3)
    for uid, row in top.iterrows():
        ax.annotate(
            f"uid {int(uid)}",
            xy=(row["actual"], row["bt_total"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual earnings (USD) — from on-chain /leaderboard/historical")
    ax.set_ylabel("Backtester earnings (USD) — 3 competitions summed")
    ax.set_title(
        f"Backtester vs on-chain actual earnings — top {args.top_n} miners, {days}-day window\n"
        f"{args.from_date} → {args.to_date}   |   aggregate bias: {pct_bias:+.1f}%",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to: {args.out}")
    print(f"Aggregate bias (sum bt_total / sum actual - 1): {pct_bias:+.2f}%")
    print("\nPer-miner data:")
    print(top.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
