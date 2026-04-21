"""Run backtests for a miner against Synth subnet scoring data.

Select which profile(s) and asset(s) to run via --profile and --asset.
Falls back to random predictions if no prediction files are found.

Usage:
    uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --days 2 --profile all --asset ALL
    uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --days 2 --profile low --asset BTC TSLAX
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from synth.validator.prompt_config import HIGH_FREQUENCY, LOW_FREQUENCY, PromptConfig

UTC = timezone.utc
NUM_SIMULATIONS = 100

PROFILES: list[PromptConfig] = [LOW_FREQUENCY, HIGH_FREQUENCY]
PROFILE_MAP: dict[str, PromptConfig] = {"low": LOW_FREQUENCY, "high": HIGH_FREQUENCY}


def _generate_random_prediction(
    start_time: datetime,
    current_price: float,
    asset: str,
    time_length: int,
    time_increment: int,
) -> dict:
    """Generate random-walk price paths from current_price."""
    import numpy as np

    num_steps = time_length // time_increment
    returns = np.random.normal(0, 0.001, size=(NUM_SIMULATIONS, num_steps))
    cumulative = np.cumsum(returns, axis=1)
    paths = current_price * np.exp(cumulative)
    paths = np.column_stack([np.full(NUM_SIMULATIONS, current_price), paths])
    return {
        "start_timestamp": int(start_time.timestamp()),
        "asset": asset,
        "time_increment": time_increment,
        "time_length": time_length,
        "num_simulations": NUM_SIMULATIONS,
        "num_steps": num_steps,
        "paths": paths.tolist(),
    }


def _generate_random_predictions(
    days: int,
    output_dir: Path,
    asset: str,
    time_length: int,
    time_increment: int,
) -> None:
    """Fetch scored prompt times from the API, generate random predictions into output_dir."""
    from app.lib.backtester.backtest import download_price_data, get_miner_scores

    now = datetime.now(UTC)
    query_start = now - timedelta(days=days)

    print(f"Fetching miner scores for {asset} (time_length={time_length}s) over last {days} days...")
    try:
        scores = get_miner_scores(query_start, now, asset, time_length, time_increment=time_increment)
    except Exception as e:
        print(f"  Failed to fetch scores for {asset}/{time_length}: {e} — skipping.")
        return
    if scores.empty:
        print(f"  No scores returned for {asset}/{time_length} — skipping.")
        return

    start_times = sorted(scores["start_time"].unique())
    print(f"  Found {len(start_times)} scored prompts. Generating random predictions...")

    price_start = min(start_times) - timedelta(hours=1)
    price_end = max(start_times) + timedelta(seconds=time_length) + timedelta(hours=1)
    prices = download_price_data(price_start, price_end, asset, freq="1min")

    generated = 0
    for st in start_times:
        ts = st.to_pydatetime().replace(tzinfo=UTC)
        try:
            current_price = float(prices.loc[:ts].iloc[-1]["close"])
        except (IndexError, KeyError):
            continue

        filename = ts.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{asset}_{time_length}.json"
        pred = _generate_random_prediction(ts, current_price, asset, time_length, time_increment)
        (output_dir / filename).write_text(json.dumps(pred))
        generated += 1

    print(f"  Generated {generated} random prediction files for {asset}/{time_length}")


def _resolve_targets(profile_arg: str, asset_args: list[str]) -> list[tuple[PromptConfig, str]]:
    """Build the list of (profile, asset) pairs to backtest.

    Assets not in a given profile's asset_list are silently skipped for that profile.
    """
    profiles = list(PROFILES) if profile_arg == "all" else [PROFILE_MAP[profile_arg]]
    requested = {a.upper() for a in asset_args}
    include_all = "ALL" in requested

    targets: list[tuple[PromptConfig, str]] = []
    for profile in profiles:
        for asset in profile.asset_list:
            if include_all or asset in requested:
                targets.append((profile, asset))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtests against Synth subnet data")
    parser.add_argument("--miner-name", default="btc_research", help="Miner name (default: btc_research)")
    parser.add_argument("--days", type=int, default=2, help="Number of backtest days (default: 2)")
    parser.add_argument(
        "--profile",
        choices=["low", "high", "all"],
        default="low",
        help="Frequency profile to run: low, high, or all (default: low)",
    )
    parser.add_argument(
        "--asset",
        nargs="+",
        default=["BTC"],
        help="One or more asset symbols (e.g. BTC TSLAX), or ALL for every asset in the selected profile(s). Default: BTC",
    )
    parser.add_argument("--predictions-dir", type=str, default=None, help="Path to predictions directory")
    args = parser.parse_args()

    targets = _resolve_targets(args.profile, args.asset)
    if not targets:
        print(f"No matching (profile, asset) pairs for profile={args.profile} asset={args.asset}")
        return

    predictions_dir = Path(args.predictions_dir) if args.predictions_dir else None

    # Fall back to random predictions if no prediction files exist
    use_tmp = False
    if predictions_dir is None:
        default_dir = Path(f"miner_outputs/{args.miner_name}/predictions")
        has_predictions = default_dir.exists() and any(default_dir.glob("**/*.json"))
        if not has_predictions:
            print(f"No predictions found in {default_dir}, falling back to random predictions.")
            use_tmp = True

    if use_tmp:
        with tempfile.TemporaryDirectory(prefix="backtest_preds_") as tmpdir:
            predictions_dir = Path(tmpdir)
            for profile, asset in targets:
                _generate_random_predictions(
                    args.days,
                    predictions_dir,
                    asset,
                    profile.time_length,
                    profile.time_increment,
                )
            _run_targets(args.miner_name, args.days, predictions_dir, targets)
    else:
        _run_targets(args.miner_name, args.days, predictions_dir, targets)


def _run_single_backtest(
    miner_name: str,
    asset: str,
    time_length: int,
    time_increment: int,
    days: int,
    predictions_dir: Path | None,
) -> None:
    from app.lib.backtester.backtest import (
        backtest, plot_crps_by_day, plot_crps_by_hour, plot_crps_over_time,
        plot_crps_ratio_distribution, plot_rank_evolution, plot_weekly_percentile,
    )

    print(f"\nRunning {days}-day backtest for '{miner_name}' on {asset} (time_length={time_length}s)...")
    try:
        result = backtest(
            miner_name=miner_name,
            asset=asset,
            time_length=time_length,
            time_increment=time_increment,
            n_backtest_days=days,
            predictions_dir=predictions_dir,
        )
    except (RuntimeError, FileNotFoundError) as e:
        print(f"  FAILED: {e}")
        return

    summary = result.summary
    print(f"  Prompts scored:       {summary['num_prompts']}")
    print(f"  Mean CRPS:            {summary['mean_crps']:.6f}")
    print(f"  Final smoothed score: {summary['final_smoothed_score']}")

    chart_path = plot_rank_evolution(result)
    print(f"  Rank chart saved to:  {chart_path}")

    crps_chart = plot_crps_over_time(result)
    print(f"  CRPS chart saved to:  {crps_chart}")

    for plot_fn, chart_name in [
        (plot_crps_by_hour, "CRPS-by-hour"),
        (plot_crps_by_day, "CRPS-by-day"),
        (plot_crps_ratio_distribution, "CRPS-ratio-dist"),
        (plot_weekly_percentile, "Weekly-percentile"),
    ]:
        try:
            path = plot_fn(result)
            print(f"  {chart_name} chart saved to: {path}")
        except (RuntimeError, Exception) as e:
            print(f"  {chart_name} chart generation failed: {e}")


def _run_targets(
    miner_name: str,
    days: int,
    predictions_dir: Path | None,
    targets: list[tuple[PromptConfig, str]],
) -> None:
    """Run backtest for each (profile, asset) target and print a summary table."""
    for profile, asset in targets:
        print(f"\n{'─'*60}")
        print(f"[{profile.label}/{asset}] time_length={profile.time_length}s")
        print(f"{'─'*60}")
        try:
            _run_single_backtest(
                miner_name,
                asset,
                profile.time_length,
                profile.time_increment,
                days,
                predictions_dir,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\n{'='*60}")
    print("ALL BACKTESTS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
