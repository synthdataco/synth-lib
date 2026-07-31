"""Rank-evolution charts: per asset, per competition, and the grand total."""

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

from synth.validator.competition_config import (  # noqa: E402
    ALL_COMPETITIONS,
    CRYPTO_24H,
    CompetitionConfig,
)

from synth_lib.backtester.config import DEFAULT_MINER_OUTPUT_ROOT, slug_for  # noqa: E402
from synth_lib.backtester.earnings import _compute_grand_total_weights  # noqa: E402
from synth_lib.backtester.result import BacktestResult  # noqa: E402
from synth_lib.backtester.scoring import compute_combined_smoothed_scores  # noqa: E402



def plot_rank_evolution(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot the miner's rank over time based on reward_weight and save the chart.

    Rank 1 = highest reward_weight at that timestamp.
    Returns the path to the saved PNG.
    """
    miner_id = result.summary["miner_id"]
    df = result.smoothed_scores.copy()
    df["updated_at"] = pd.to_datetime(df["updated_at"])

    # Derive asset and time_length from the prompt data for labelling
    assets = result.prompt_df["asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    time_lengths = result.prompt_df["time_length"].unique()
    tl_label = str(int(time_lengths[0])) if len(time_lengths) == 1 else "mixed"

    # Compute rank per timestamp (highest reward_weight = rank 1)
    df["rank"] = (
        df.groupby("updated_at")["reward_weight"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    miner_df = df.loc[df["miner_uid"] == miner_id].sort_values("updated_at")

    if miner_df.empty:
        raise RuntimeError(f"Miner {miner_id} not found in smoothed scores.")

    total_miners = df.groupby("updated_at")["miner_uid"].nunique()

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    sns.lineplot(
        data=miner_df,
        x="updated_at",
        y="rank",
        marker="o",
        markersize=5,
        linewidth=2,
        ax=ax,
    )

    # Shade the total miners range
    ax.fill_between(
        total_miners.index,
        1,
        total_miners.values,
        alpha=0.08,
        color="grey",
        label="Total miners",
    )

    ax.invert_yaxis()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(f"Rank evolution — {result.miner_name} — {asset_label} ({tl_label}s)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    # Save
    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"rank_evolution_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return chart_path


def plot_total_rank_evolution(
    results: list[BacktestResult],
    combined: pd.DataFrame,
    profile_label: str,
    output_dir: Path | None = None,
) -> Path:
    """Plot aggregate rank evolution across all assets.

    Ranks miners per `updated_at` timestamp by `reward_weight` (descending, 1 = best).
    Expects `combined` as the output of `compute_combined_smoothed_scores` — a single
    real-validator-equivalent smoothed-score DataFrame covering all assets.
    """
    if not results:
        raise RuntimeError("No backtest results to aggregate.")
    if combined.empty:
        raise RuntimeError(
            "No combined smoothed scores available for total rank chart."
        )

    miner_name = results[0].miner_name
    miner_id = results[0].summary["miner_id"]

    df = combined.copy()
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df["rank"] = (
        df.groupby("updated_at")["reward_weight"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    miner_df = df.loc[df["miner_uid"] == miner_id].sort_values("updated_at")
    if miner_df.empty:
        raise RuntimeError(f"Miner {miner_id} not found in combined smoothed scores.")

    total_miners = df.groupby("updated_at")["miner_uid"].nunique()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=miner_df,
        x="updated_at",
        y="rank",
        marker="o",
        markersize=5,
        linewidth=2,
        ax=ax,
    )
    ax.fill_between(
        total_miners.index,
        1,
        total_miners.values,
        alpha=0.08,
        color="grey",
        label="Total miners",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(f"Total rank evolution — {miner_name} — {profile_label} (all assets)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"rank_evolution_TOTAL_{profile_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    # Print final total rank
    last_ts = miner_df["updated_at"].max()
    last_slice = df.loc[df["updated_at"] == last_ts]
    miner_row = last_slice.loc[last_slice["miner_uid"] == miner_id]
    if not miner_row.empty:
        rank = int(miner_row["rank"].iloc[0])
        n = len(last_slice)
        rw = float(miner_row["reward_weight"].iloc[0])
        print(f"\n  TOTAL [{profile_label}] rank: {rank}/{n}  reward_weight: {rw:.6f}")

    return chart_path


def plot_grand_total_rank_evolution(
    results_by_profile: dict[str, list[BacktestResult]],
    combined_by_profile: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> Path:
    """Rank evolution across ALL assets from ALL profiles.

    Merges per-profile `combined` frames via `_compute_grand_total_weights`
    (forward-fill each profile onto the union of timestamps, then sum per
    miner). Ranks miners descending by grand-total reward_weight at each
    timestamp.
    """
    grand = _compute_grand_total_weights(combined_by_profile)
    if grand.empty:
        raise RuntimeError("No grand-total rows available for rank chart.")

    # miner_name / miner_id from any profile's first result
    first_result = next(
        (rs[0] for rs in results_by_profile.values() if rs),
        None,
    )
    if first_result is None:
        raise RuntimeError("No backtest results to aggregate.")
    miner_name = first_result.miner_name
    miner_id = first_result.summary["miner_id"]

    df = grand.copy()
    df["rank"] = (
        df.groupby("updated_at")["reward_weight"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    miner_df = df.loc[df["miner_uid"] == miner_id].sort_values("updated_at")
    if miner_df.empty:
        raise RuntimeError(f"Miner {miner_id} not found in grand-total weights.")

    total_miners = df.groupby("updated_at")["miner_uid"].nunique()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=miner_df,
        x="updated_at",
        y="rank",
        marker="o",
        markersize=5,
        linewidth=2,
        ax=ax,
    )
    ax.fill_between(
        total_miners.index,
        1,
        total_miners.values,
        alpha=0.08,
        color="grey",
        label="Total miners",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(
        f"Grand-total rank evolution — {miner_name} — all profiles, all assets"
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "rank_evolution_GRAND_TOTAL.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    # Print final grand-total rank
    last_ts = miner_df["updated_at"].max()
    last_slice = df.loc[df["updated_at"] == last_ts]
    miner_row = last_slice.loc[last_slice["miner_uid"] == miner_id]
    if not miner_row.empty:
        rank = int(miner_row["rank"].iloc[0])
        n = len(last_slice)
        rw = float(miner_row["reward_weight"].iloc[0])
        print(f"\n  GRAND TOTAL rank: {rank}/{n}  reward_weight: {rw:.6f}")

    return chart_path
