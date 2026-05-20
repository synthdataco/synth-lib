"""Run backtests for a miner against Synth subnet scoring data.

Pick assets with --asset (one or more symbols, or "ALL") and profiles with
--profile (low, high, or all). When both profiles produce ≥2 assets' worth of
data, also emits grand-total rank + earnings charts across profiles.

Falls back to random predictions if no prediction files are found.

Usage:
    uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --days 2
    uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --asset BTC --profile low
    uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --asset BTC TSLAX
    uv run app/lib/backtester/scripts/run_backtest.py --miner-name gbm_agent --asset ALL --profile high
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from synth.validator.prompt_config import HIGH_FREQUENCY, LOW_FREQUENCY, PromptConfig

UTC = timezone.utc
NUM_SIMULATIONS = 100
PROFILES_BY_NAME: dict[str, list[PromptConfig]] = {
    "low": [LOW_FREQUENCY],
    "high": [HIGH_FREQUENCY],
    "all": [LOW_FREQUENCY, HIGH_FREQUENCY],
}


# ---------------------------------------------------------------------------
# Random-predictions fallback (used when no real predictions exist on disk)
# ---------------------------------------------------------------------------


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

    print(
        f"Fetching miner scores for {asset} (time_length={time_length}s) over last {days} days..."
    )
    try:
        scores = get_miner_scores(
            query_start, now, asset, time_length, time_increment=time_increment
        )
    except Exception as e:
        print(f"  Failed to fetch scores for {asset}/{time_length}: {e} — skipping.")
        return
    if scores.empty:
        print(f"  No scores returned for {asset}/{time_length} — skipping.")
        return

    start_times = sorted(scores["start_time"].unique())
    print(
        f"  Found {len(start_times)} scored prompts. Generating random predictions..."
    )

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
        pred = _generate_random_prediction(
            ts, current_price, asset, time_length, time_increment
        )
        (output_dir / filename).write_text(json.dumps(pred))
        generated += 1

    print(f"  Generated {generated} random prediction files for {asset}/{time_length}")


def _populate_random_dir(
    filtered_profiles: list[PromptConfig], days: int, output_dir: Path
) -> None:
    """Populate `output_dir` with random predictions for every asset in each filtered profile."""
    for profile in filtered_profiles:
        for asset in profile.asset_list:
            _generate_random_predictions(
                days, output_dir, asset, profile.time_length, profile.time_increment
            )


# ---------------------------------------------------------------------------
# Selection parsing
# ---------------------------------------------------------------------------


def _parse_asset_selection(raw: list[str]) -> list[str] | None:
    """Normalize --asset tokens. Returns None for 'ALL', else a deduplicated list of symbols.

    Accepts any of: `--asset BTC`, `--asset BTC TSLAX`, `--asset "BTC TSLAX"`,
    `--asset BTC,TSLAX`. Case-insensitive `ALL` (alone or mixed) means every asset.
    """
    tokens: list[str] = []
    for piece in raw:
        for sub in piece.replace(",", " ").split():
            tokens.append(sub)
    if any(t.upper() == "ALL" for t in tokens):
        return None
    # Preserve order, dedupe
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _build_filtered_profiles(
    profiles: list[PromptConfig],
    asset_selection: list[str] | None,
) -> list[PromptConfig]:
    """Intersect each profile's asset_list with the user selection; drop profiles with no match.

    `asset_selection=None` means keep every asset in each profile (the "ALL" case).
    """
    out: list[PromptConfig] = []
    for profile in profiles:
        if asset_selection is None:
            kept = list(profile.asset_list)
        else:
            kept = [a for a in asset_selection if a in profile.asset_list]
        if kept:
            out.append(dataclasses.replace(profile, asset_list=kept))
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _run(
    miner_name: str,
    filtered_profiles: list[PromptConfig],
    days: int,
    predictions_dir: Path | None,
    scoring_executor: ProcessPoolExecutor,
    eval_end: datetime | None = None,
    simulate_registration: datetime | None = None,
    simulate_deregistration: datetime | None = None,
) -> None:
    """Run each filtered profile (parallel if >1) and emit grand-total charts when both produced data."""
    import pandas as pd
    from app.lib.backtester.backtest import (
        BacktestResult,
        plot_grand_total_earnings,
        plot_grand_total_rank_evolution,
        run_backtest,
    )

    results_by_profile: dict[str, list[BacktestResult]] = {}
    combined_by_profile: dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=max(len(filtered_profiles), 1)) as executor:
        futures = {
            executor.submit(
                run_backtest,
                miner_name=miner_name,
                prompt_config=profile,
                n_backtest_days=days,
                predictions_dir=predictions_dir,
                scoring_executor=scoring_executor,
                eval_end=eval_end,
                simulate_registration=simulate_registration,
                simulate_deregistration=simulate_deregistration,
            ): profile
            for profile in filtered_profiles
        }
        for future in as_completed(futures):
            profile = futures[future]
            try:
                results, combined = future.result()
                results_by_profile[profile.label] = results
                combined_by_profile[profile.label] = combined
            except (RuntimeError, FileNotFoundError) as e:
                print(f"  [{profile.label}] BACKTEST FAILED: {e}")

    print(f"\n{'='*60}\nALL BACKTESTS COMPLETE\n{'='*60}")

    # Grand-total charts require both profiles to have combined frames with data.
    if len(combined_by_profile) >= 2 and all(
        not c.empty for c in combined_by_profile.values()
    ):
        try:
            path = plot_grand_total_rank_evolution(
                results_by_profile, combined_by_profile
            )
            print(f"  Grand-total rank chart saved to: {path}")
        except (RuntimeError, FileNotFoundError, KeyError) as e:
            print(f"  Grand-total rank chart failed: {e}")

        try:
            path = plot_grand_total_earnings(results_by_profile, combined_by_profile)
            print(f"  Grand-total earnings chart saved to: {path}")
        except (RuntimeError, FileNotFoundError, KeyError) as e:
            print(f"  Grand-total earnings chart failed: {e}")
    else:
        print("  Skipping grand-total charts: need both profiles with ≥2 assets each.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run backtests against Synth subnet data"
    )
    parser.add_argument(
        "--miner-name",
        default="btc_research",
        help="Miner name (default: btc_research)",
    )
    parser.add_argument(
        "--days", type=int, default=2, help="Number of backtest days (default: 2)"
    )
    parser.add_argument(
        "--asset",
        nargs="+",
        default=["ALL"],
        metavar="ASSET",
        help='Assets to backtest: one or more symbols (e.g. BTC, or BTC TSLAX), or "ALL" (default: ALL)',
    )
    parser.add_argument(
        "--profile",
        choices=["low", "high", "all"],
        default="all",
        help="Profile to backtest: low, high, or all (default: all)",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Path to predictions directory",
    )
    parser.add_argument(
        "--eval-end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Pin the evaluation window end date (default: now)",
    )
    parser.add_argument(
        "--simulate-registration",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Simulate the miner registering on this date. Drops our CRPS rows "
        "before this date so synth's worst-score backfill applies symmetrically "
        "with real late-joining miners.",
    )
    parser.add_argument(
        "--simulate-deregistration",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Simulate the miner leaving on this date. Drops our CRPS rows after "
        "this date.",
    )
    args = parser.parse_args()

    def _parse_iso(s: str | None) -> datetime | None:
        if s is None:
            return None
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)

    eval_end = _parse_iso(args.eval_end)
    simulate_registration = _parse_iso(args.simulate_registration)
    simulate_deregistration = _parse_iso(args.simulate_deregistration)

    asset_selection = _parse_asset_selection(args.asset)
    profiles = PROFILES_BY_NAME[args.profile]
    filtered_profiles = _build_filtered_profiles(profiles, asset_selection)

    if not filtered_profiles:
        sel = "ALL" if asset_selection is None else " ".join(asset_selection)
        print(
            f"ERROR: no assets match --asset '{sel}' in --profile {args.profile}. "
            f"Available LOW assets: {LOW_FREQUENCY.asset_list}. "
            f"Available HIGH assets: {HIGH_FREQUENCY.asset_list}.",
            file=sys.stderr,
        )
        sys.exit(2)

    asset_summary = ", ".join(f"{p.label}={p.asset_list}" for p in filtered_profiles)
    print(f"Running backtests for {args.miner_name}: {asset_summary}")

    predictions_dir = Path(args.predictions_dir) if args.predictions_dir else None

    # Decide whether to fall back to random predictions.
    use_tmp = False
    if predictions_dir is None:
        default_dir = Path(f"miner_outputs/{args.miner_name}/predictions")
        has_predictions = default_dir.exists() and any(default_dir.glob("**/*.json"))
        if not has_predictions:
            print(
                f"No predictions found in {default_dir}, falling back to random predictions."
            )
            use_tmp = True

    max_workers = max((os.cpu_count() or 4) - 2, 1)

    def _dispatch(pred_dir: Path | None) -> None:
        with ProcessPoolExecutor(max_workers=max_workers) as scoring_executor:
            _run(
                args.miner_name,
                filtered_profiles,
                args.days,
                pred_dir,
                scoring_executor,
                eval_end=eval_end,
                simulate_registration=simulate_registration,
                simulate_deregistration=simulate_deregistration,
            )

    if use_tmp:
        with tempfile.TemporaryDirectory(prefix="backtest_preds_") as tmpdir:
            tmp_dir = Path(tmpdir)
            _populate_random_dir(filtered_profiles, args.days, tmp_dir)
            _dispatch(tmp_dir)
    else:
        _dispatch(predictions_dir)


if __name__ == "__main__":
    main()
