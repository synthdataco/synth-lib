"""CRPS charts: over time, by hour, by weekday, ratio distribution, weekly percentile."""

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

from synth_lib.backtester.config import DEFAULT_MINER_OUTPUT_ROOT  # noqa: E402
from synth_lib.backtester.result import BacktestResult  # noqa: E402


def _compute_relative_crps(result: BacktestResult) -> pd.DataFrame:
    """Compute per-prompt CRPS ratio and percentile rank vs other miners.

    Returns DataFrame with one row per scored prompt: scored_time, our_crps,
    others_median, crps_ratio, percentile_rank, hour, day_of_week, iso_week.
    """
    miner_id = result.summary["miner_id"]
    df = result.prompt_df.copy()
    df["scored_time"] = pd.to_datetime(df["scored_time"])
    df = df.loc[df["crps"] != -1]
    if df.empty:
        raise RuntimeError("No valid CRPS data for relative analysis.")

    # Median CRPS of other miners at each scored_time
    others = df.loc[df["miner_uid"] != miner_id]
    others_median = (
        others.groupby("scored_time")["crps"].median().rename("others_median")
    )

    # Our miner's CRPS
    ours = df.loc[df["miner_uid"] == miner_id, ["scored_time", "crps"]].rename(
        columns={"crps": "our_crps"}
    )
    if ours.empty:
        raise RuntimeError(f"Miner {miner_id} has no valid CRPS data.")

    merged = ours.merge(others_median, on="scored_time", how="inner")
    merged["crps_ratio"] = merged["our_crps"] / merged["others_median"]

    # Percentile rank: what fraction of miners we beat at each scored_time
    # Lower CRPS = better, so rank ascending. percentile = (rank-1)/(n-1)*100
    df = df.copy()
    df["_rank"] = df.groupby("scored_time")["crps"].rank(method="min", ascending=True)
    group_sizes = df.groupby("scored_time")["crps"].size().rename("_n")
    ours_rank = df.loc[df["miner_uid"] == miner_id, ["scored_time", "_rank"]]
    merged = (
        merged.merge(ours_rank, on="scored_time", how="left")
        .merge(group_sizes, on="scored_time", how="left")
    )
    merged["percentile_rank"] = np.where(
        merged["_n"] > 1,
        (merged["_rank"] - 1) / (merged["_n"] - 1) * 100,
        0.0,
    )
    merged = merged.drop(columns=["_rank", "_n"])

    # Time-derived columns
    merged["hour"] = merged["scored_time"].dt.hour
    merged["day_of_week"] = merged["scored_time"].dt.dayofweek
    merged["iso_week"] = merged["scored_time"].dt.isocalendar().week.astype(int)

    return merged


def plot_crps_over_time(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot our miner's CRPS over time vs the distribution of other miners.

    Shows: our miner as a line, others as median + shaded IQR and 10th-90th bands.
    Returns the path to the saved PNG.
    """
    miner_id = result.summary["miner_id"]
    df = result.prompt_df.copy()
    df["scored_time"] = pd.to_datetime(df["scored_time"])

    # Filter out missing predictions
    df = df.loc[df["crps"] != -1]
    if df.empty:
        raise RuntimeError("No valid CRPS data to plot.")

    # Derive labels
    assets = df["asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    time_lengths = df["time_length"].unique()
    tl_label = str(int(time_lengths[0])) if len(time_lengths) == 1 else "mixed"

    # Split our miner vs others
    miner_df = df.loc[df["miner_uid"] == miner_id].sort_values("scored_time")
    others_df = df.loc[df["miner_uid"] != miner_id]

    if miner_df.empty:
        raise RuntimeError(f"Miner {miner_id} not found in CRPS data.")

    # Compute percentile stats for other miners at each scored_time
    others_stats = (
        others_df.groupby("scored_time")["crps"]
        .agg(
            median="median",
            p10=lambda x: np.percentile(x, 10),
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
            p90=lambda x: np.percentile(x, 90),
        )
        .sort_index()
    )

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    # 10th-90th band (light)
    ax.fill_between(
        others_stats.index,
        others_stats["p10"],
        others_stats["p90"],
        alpha=0.08,
        color="steelblue",
        label="Range (10th\u201390th)",
    )
    # IQR band (darker)
    ax.fill_between(
        others_stats.index,
        others_stats["p25"],
        others_stats["p75"],
        alpha=0.2,
        color="steelblue",
        label="IQR (25th\u201375th)",
    )
    # Median line
    ax.plot(
        others_stats.index,
        others_stats["median"],
        linestyle="--",
        color="grey",
        linewidth=1.5,
        label="Median (others)",
    )
    # Our miner
    ax.plot(
        miner_df["scored_time"].values,
        miner_df["crps"].values,
        marker="o",
        markersize=5,
        linewidth=2,
        label=result.miner_name,
    )

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("CRPS")
    ax.set_title(
        f"CRPS over time \u2014 {result.miner_name} \u2014 {asset_label} ({tl_label}s)"
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    # Save
    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"crps_over_time_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return chart_path


def plot_crps_by_hour(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot mean/median CRPS ratio by hour of day. Returns path to saved PNG."""
    rel = _compute_relative_crps(result)

    assets = result.prompt_df.loc[result.prompt_df["crps"] != -1, "asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    tls = result.prompt_df.loc[result.prompt_df["crps"] != -1, "time_length"].unique()
    tl_label = str(int(tls[0])) if len(tls) == 1 else "mixed"

    hourly = (
        rel.groupby("hour")["crps_ratio"].agg(["mean", "median"]).reindex(range(24))
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hourly.index, hourly["mean"], color="salmon", label="Mean", zorder=2)
    ax.plot(
        hourly.index,
        hourly["median"],
        marker="o",
        markersize=5,
        linewidth=2,
        color="steelblue",
        label="Median",
        zorder=3,
    )
    ax.axhline(
        y=1.0, linestyle="--", color="grey", linewidth=1, label="Even with median miner"
    )
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("CRPS Ratio (ours / others median)")
    ax.set_title(
        f"Performance by Hour of Day \u2014 {result.miner_name} \u2014 {asset_label} ({tl_label}s)"
    )
    ax.set_xticks(range(24))
    ax.legend()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"crps_by_hour_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plot_crps_by_day(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot mean/median CRPS ratio by day of week. Returns path to saved PNG."""
    rel = _compute_relative_crps(result)

    assets = result.prompt_df.loc[result.prompt_df["crps"] != -1, "asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    tls = result.prompt_df.loc[result.prompt_df["crps"] != -1, "time_length"].unique()
    tl_label = str(int(tls[0])) if len(tls) == 1 else "mixed"

    daily = (
        rel.groupby("day_of_week")["crps_ratio"]
        .agg(["mean", "median"])
        .reindex(range(7))
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(7), daily["mean"], color="salmon", label="Mean", zorder=2)
    ax.plot(
        range(7),
        daily["median"],
        marker="o",
        markersize=5,
        linewidth=2,
        color="steelblue",
        label="Median",
        zorder=3,
    )
    ax.axhline(
        y=1.0, linestyle="--", color="grey", linewidth=1, label="Even with median miner"
    )
    ax.set_xlabel("Day of Week (UTC)")
    ax.set_ylabel("CRPS Ratio (ours / others median)")
    ax.set_title(
        f"Performance by Day of Week \u2014 {result.miner_name} \u2014 {asset_label} ({tl_label}s)"
    )
    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_LABELS)
    ax.legend()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"crps_by_day_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def plot_crps_ratio_distribution(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot histogram of CRPS ratio distribution. Returns path to saved PNG."""
    rel = _compute_relative_crps(result)

    assets = result.prompt_df.loc[result.prompt_df["crps"] != -1, "asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    tls = result.prompt_df.loc[result.prompt_df["crps"] != -1, "time_length"].unique()
    tl_label = str(int(tls[0])) if len(tls) == 1 else "mixed"

    our_median = float(rel["crps_ratio"].median())

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(rel["crps_ratio"], bins=50, color="salmon", edgecolor="white", zorder=2)
    ax.axvline(
        x=1.0,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Even with median miner",
    )
    ax.axvline(
        x=our_median,
        linestyle="--",
        color="steelblue",
        linewidth=1.5,
        label=f"Our median: {our_median:.2f}",
    )
    ax.set_xlabel("CRPS Ratio (ours / others median)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Distribution of Relative Performance \u2014 {result.miner_name} \u2014 {asset_label} ({tl_label}s)"
    )
    ax.legend()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"crps_ratio_dist_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def plot_weekly_percentile(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot mean percentile rank by ISO week. Returns path to saved PNG."""
    rel = _compute_relative_crps(result)

    assets = result.prompt_df.loc[result.prompt_df["crps"] != -1, "asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    tls = result.prompt_df.loc[result.prompt_df["crps"] != -1, "time_length"].unique()
    tl_label = str(int(tls[0])) if len(tls) == 1 else "mixed"

    weekly = rel.groupby("iso_week")["percentile_rank"].mean()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(weekly)), weekly.values, color="salmon", zorder=2)
    ax.axhline(y=50, linestyle="--", color="grey", linewidth=1, label="50th pctile")
    ax.set_xlabel("ISO Week")
    ax.set_ylabel("Mean Percentile Rank")
    ax.set_title(
        f"Weekly Percentile Trend \u2014 {result.miner_name} \u2014 {asset_label} ({tl_label}s)"
    )
    ax.set_xticks(range(len(weekly)))
    ax.set_xticklabels([str(w) for w in weekly.index])
    ax.legend()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"weekly_percentile_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path
