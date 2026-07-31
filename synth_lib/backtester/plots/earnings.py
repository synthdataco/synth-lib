"""Estimated-earnings charts, per competition and grand total."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Charts are rendered inside worker threads; GUI backends (the macOS default)
# can only run on the main thread, so force the non-interactive Agg backend.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from synth.validator.competition_config import SMOOTHED_SCORE_COEFFICIENT, CompetitionConfig  # noqa: E402

from synth_lib.backtester.config import (  # noqa: E402
    DEFAULT_MINER_OUTPUT_ROOT,
    EMISSION_NORMALIZATION_FACTOR,
    slug_for,
)
from synth_lib.backtester.earnings import (  # noqa: E402
    _compute_earnings_df,
    _compute_grand_total_weights,
)
from synth_lib.backtester.loading import get_daily_miner_pool_usd  # noqa: E402
from synth_lib.backtester.result import BacktestResult  # noqa: E402



def plot_estimated_earnings(
    results: list[BacktestResult],
    competition: CompetitionConfig,
    combined: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Three-panel estimated USD earnings chart for our miner across all assets.

    Panel 1: Share of round (%) line, window-mean reference.
    Panel 2: Daily earnings ($) bars, mean-daily reference.
    Panel 3: Cumulative earnings ($) line, total annotated.

    Uses live subnet daily miner-pool USD via /v1/miners/rewards/pool. Attributes the
    full daily pool to this profile; if the subnet splits the pool across high/low
    frequency prompts, the $ here is an upper bound for a single-profile run.
    """
    if not results:
        raise RuntimeError("No backtest results to plot.")
    if combined.empty:
        raise RuntimeError("No combined smoothed scores — cannot compute earnings.")

    miner_name = results[0].miner_name
    miner_id = results[0].summary["miner_id"]

    first_ts = pd.to_datetime(combined["updated_at"].min())
    last_ts = pd.to_datetime(combined["updated_at"].max())
    daily_pool = get_daily_miner_pool_usd(
        first_ts.floor("D").to_pydatetime(),
        (last_ts.floor("D") + pd.Timedelta(days=1)).to_pydatetime(),
    )
    if daily_pool.empty:
        raise RuntimeError("No daily pool data available from /v1/miners/rewards/pool.")

    earnings = _compute_earnings_df(combined, miner_id, daily_pool)
    if earnings.empty:
        raise RuntimeError(f"Miner {miner_id} not found in combined smoothed scores.")

    daily = (
        earnings.set_index("updated_at")
        .resample("1D")
        .agg(usd=("usd_per_round", "sum"), share_mean=("share_of_round", "mean"))
        .dropna()
    )

    total_usd = float(earnings["usd_cumulative"].iloc[-1])
    mean_share_pct = float(earnings["share_of_round"].mean() * 100)
    mean_daily_usd = float(daily["usd"].mean()) if not daily.empty else 0.0
    dur_days = (last_ts - first_ts).total_seconds() / 86400
    n_assets = len(results)
    total_assets = len(competition.asset_list)
    partial_coverage = n_assets < total_assets

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.1, 1.3]},
    )

    # Panel 1: share of round (%)
    ax = axes[0]
    ax.plot(
        earnings["updated_at"],
        earnings["share_of_round"] * 100,
        color="tab:blue",
        lw=0.9,
    )
    ax.fill_between(
        earnings["updated_at"],
        0,
        earnings["share_of_round"] * 100,
        color="tab:blue",
        alpha=0.15,
    )
    ax.axhline(
        mean_share_pct,
        color="tab:orange",
        ls="--",
        lw=0.8,
        label=f"window mean {mean_share_pct:.2f}%",
    )
    ax.set_ylabel("Our share of round (%)")
    ax.legend(loc="upper right", fontsize=9)

    # Panel 2: daily earnings ($)
    ax = axes[1]
    ax.bar(daily.index, daily["usd"], color="tab:green", width=0.85, alpha=0.8)
    ax.axhline(
        mean_daily_usd,
        color="black",
        ls="--",
        lw=0.8,
        label=f"mean daily ${mean_daily_usd:,.0f}",
    )
    ax.set_ylabel("Daily earnings ($)")
    ax.legend(loc="upper right", fontsize=9)

    # Panel 3: cumulative earnings ($)
    ax = axes[2]
    ax.plot(
        earnings["updated_at"], earnings["usd_cumulative"], color="tab:purple", lw=1.8
    )
    ax.fill_between(
        earnings["updated_at"],
        0,
        earnings["usd_cumulative"],
        color="tab:purple",
        alpha=0.15,
    )
    ax.set_ylabel("Cumulative earnings ($)")
    ax.set_xlabel("Date (UTC)")
    ax.annotate(
        f"${total_usd:,.0f}",
        xy=(earnings["updated_at"].iloc[-1], total_usd),
        xytext=(-80, -10),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="tab:purple",
    )

    partial_line = (
        f"PARTIAL COVERAGE ({n_assets}/{total_assets} assets) — "
        "earnings overestimate real deployment. "
        if partial_coverage
        else ""
    )
    profile_pct = SMOOTHED_SCORE_COEFFICIENT * 100
    fig.suptitle(
        f"Estimated earnings — {miner_name} — {competition.label} "
        f"({n_assets}/{total_assets} assets, {dur_days:.1f}d)\n"
        f"{partial_line}"
        f"reward_weight × live daily miner-pool USD × {EMISSION_NORMALIZATION_FACTOR:.4g} "
        f"emission-normalization (realized-emission estimate; each competition's "
        f"weight sums to {profile_pct:.0f}%, sum all competition charts for total).",
        fontsize=11,
        y=0.995,
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"estimated_earnings_{slug_for(competition)}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    partial_tag = (
        f"  PARTIAL COVERAGE ({n_assets}/{total_assets} assets)"
        if partial_coverage
        else ""
    )
    print(
        f"\n  EARNINGS [{competition.label}] total: ${total_usd:,.0f} "
        f"over {dur_days:.1f}d (mean share {mean_share_pct:.2f}%){partial_tag}"
    )
    return chart_path


def plot_grand_total_earnings(
    results_by_profile: dict[str, list[BacktestResult]],
    combined_by_profile: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> Path:
    """Three-panel estimated earnings chart using grand-total reward_weight.

    Uses the full daily miner-pool USD (no × smoothed_score_coefficient factor),
    applied to each miner's grand-total share (reward_weight sums to 1.0 across
    miners per timestamp once both profiles are active).
    """
    grand = _compute_grand_total_weights(combined_by_profile)
    if grand.empty:
        raise RuntimeError("No grand-total rows — cannot compute earnings.")

    first_result = next(
        (rs[0] for rs in results_by_profile.values() if rs),
        None,
    )
    if first_result is None:
        raise RuntimeError("No backtest results to plot.")
    miner_name = first_result.miner_name
    miner_id = first_result.summary["miner_id"]

    first_ts = pd.to_datetime(grand["updated_at"].min())
    last_ts = pd.to_datetime(grand["updated_at"].max())
    daily_pool = get_daily_miner_pool_usd(
        first_ts.floor("D").to_pydatetime(),
        (last_ts.floor("D") + pd.Timedelta(days=1)).to_pydatetime(),
    )
    if daily_pool.empty:
        raise RuntimeError("No daily pool data available from /v1/miners/rewards/pool.")

    # Grand-total reward_weight already sums to ~1.0 across the 3 competitions,
    # so share_of_round divides by 1.0 (not the per-competition 1/3).
    earnings = _compute_earnings_df(
        grand, miner_id, daily_pool, share_denominator=1.0,
    )
    if earnings.empty:
        raise RuntimeError(f"Miner {miner_id} not found in grand-total weights.")

    daily = (
        earnings.set_index("updated_at")
        .resample("1D")
        .agg(usd=("usd_per_round", "sum"), share_mean=("share_of_round", "mean"))
        .dropna()
    )

    total_usd = float(earnings["usd_cumulative"].iloc[-1])
    mean_share_pct = float(earnings["share_of_round"].mean() * 100)
    mean_daily_usd = float(daily["usd"].mean()) if not daily.empty else 0.0
    dur_days = (last_ts - first_ts).total_seconds() / 86400

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.1, 1.3]},
    )

    ax = axes[0]
    ax.plot(
        earnings["updated_at"],
        earnings["share_of_round"] * 100,
        color="tab:blue",
        lw=0.9,
    )
    ax.fill_between(
        earnings["updated_at"],
        0,
        earnings["share_of_round"] * 100,
        color="tab:blue",
        alpha=0.15,
    )
    ax.axhline(
        mean_share_pct,
        color="tab:orange",
        ls="--",
        lw=0.8,
        label=f"window mean {mean_share_pct:.2f}%",
    )
    ax.set_ylabel("Our share of subnet (%)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.bar(daily.index, daily["usd"], color="tab:green", width=0.85, alpha=0.8)
    ax.axhline(
        mean_daily_usd,
        color="black",
        ls="--",
        lw=0.8,
        label=f"mean daily ${mean_daily_usd:,.0f}",
    )
    ax.set_ylabel("Daily earnings ($)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    ax.plot(
        earnings["updated_at"], earnings["usd_cumulative"], color="tab:purple", lw=1.8
    )
    ax.fill_between(
        earnings["updated_at"],
        0,
        earnings["usd_cumulative"],
        color="tab:purple",
        alpha=0.15,
    )
    ax.set_ylabel("Cumulative earnings ($)")
    ax.set_xlabel("Date (UTC)")
    ax.annotate(
        f"${total_usd:,.0f}",
        xy=(earnings["updated_at"].iloc[-1], total_usd),
        xytext=(-80, -10),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="tab:purple",
    )

    total_assets = sum(len(rs) for rs in results_by_profile.values())
    fig.suptitle(
        f"Estimated grand-total earnings — {miner_name} — "
        f"{len(results_by_profile)} competitions, {total_assets} assets, {dur_days:.1f}d\n"
        f"grand-total reward_weight (summed across competitions, ~1.0) × live daily "
        f"miner-pool USD × {EMISSION_NORMALIZATION_FACTOR:.4g} emission-normalization "
        f"(realized-emission estimate).",
        fontsize=11,
        y=0.995,
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "estimated_earnings_GRAND_TOTAL.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(
        f"\n  GRAND TOTAL EARNINGS total: ${total_usd:,.0f} "
        f"over {dur_days:.1f}d (mean share {mean_share_pct:.2f}%)"
    )
    return chart_path
