"""Backtest orchestration: one asset (backtest) and a whole competition (run_backtest)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from synth.validator.competition_config import ALL_COMPETITIONS, CompetitionConfig

from synth_lib.backtester.caveats import (
    _maybe_warn_competition_split,
    _maybe_warn_hf_crps_formula_change,
)
from synth_lib.backtester.config import (
    DEFAULT_MINER_OUTPUT_ROOT,
    PREDICTION_MATCH_TOLERANCE_MINUTES,
    SCORING_INTERVALS,
    UTC,
    _COMBINED_EMPTY_COLS,
    competition_for,
    slug_for,
)
from synth_lib.backtester.loading import (
    download_price_data,
    get_miner_scores,
    get_rewards_history,
)
from synth_lib.backtester.preparation import (
    _fill_gaps_from_realized_paths,
    _find_prediction_file,
    _parse_prediction_filename_time,
    _slice_real_prices,
)
from synth_lib.backtester.plots.crps import (
    plot_crps_by_day,
    plot_crps_by_hour,
    plot_crps_over_time,
    plot_crps_ratio_distribution,
    plot_weekly_percentile,
)
from synth_lib.backtester.plots.earnings import (
    plot_estimated_earnings,
    plot_grand_total_earnings,
)
from synth_lib.backtester.plots.rank import (
    plot_grand_total_rank_evolution,
    plot_rank_evolution,
    plot_total_rank_evolution,
)
from synth_lib.backtester.result import BacktestResult, NoScoresAvailable
from synth_lib.backtester.scoring import (
    _compute_prompt_score_stats_for_group,
    _score_single_prompt,
    _trim_warmup,
    calculate_smoothed_scores,
    compute_combined_smoothed_scores,
)


def backtest(
    miner_name: str = "initial_research",
    asset: str = "BTC",
    *,
    time_length: int,
    time_increment: int = None,
    n_backtest_days: int = 15,
    miner_id: int = 999,
    predictions_dir: Path | None = None,
    scoring_intervals: dict[str, int] | None = None,
    competition: CompetitionConfig | None = None,
    scoring_executor: ProcessPoolExecutor | None = None,
    eval_end: datetime | None = None,
    simulate_registration: datetime | None = None,
    simulate_deregistration: datetime | None = None,
) -> BacktestResult:
    """Backtest a local miner against Synth subnet scoring data.
    Loads predictions from predictions_dir (or miner_outputs/{miner_name}/predictions/).
    """
    if competition is None:
        competition = competition_for(asset, time_length)
    if scoring_intervals is None:
        scoring_intervals = competition.scoring_intervals

    # Discover prediction files and their date range
    predictions_root = predictions_dir or (
        DEFAULT_MINER_OUTPUT_ROOT / miner_name / "predictions"
    )
    prediction_files = sorted(
        p for p in predictions_root.glob("**/*.json") if not p.name.startswith("_")
    )
    pred_times = [_parse_prediction_filename_time(p) for p in prediction_files]
    pred_times = [t for t in pred_times if t is not None]
    if not pred_times:
        raise FileNotFoundError(
            f"No valid prediction files found in {predictions_root}. "
            "Expected filenames like 2026-03-28_00:00:00Z_BTC_86400.json"
        )
    pred_start = min(pred_times)
    pred_end = max(pred_times)

    # Constrain backtest window to prediction coverage. The buffer before pred_start matches the
    # file-matching tolerance, so every returned prompt can resolve to a prediction file.
    # Anchor priority: simulate_deregistration > eval_end > now. When the user passes
    # simulate_registration, the window's lower bound shifts to it (minus 10d for the
    # smoothed-score moving-average lookback) instead of `anchor − n_backtest_days`,
    # so the API actually fetches scored prompts in the simulated mining period.
    if simulate_deregistration is not None:
        anchor = pd.Timestamp(simulate_deregistration).to_pydatetime()
    elif eval_end is not None:
        anchor = eval_end
    else:
        anchor = datetime.now(UTC)
    if simulate_registration is not None:
        window_start = pd.Timestamp(simulate_registration).to_pydatetime() - timedelta(
            days=competition.window_days
        )
    else:
        window_start = anchor - timedelta(days=n_backtest_days)
    query_start = max(
        pred_start - timedelta(minutes=PREDICTION_MATCH_TOLERANCE_MINUTES),
        window_start,
    )
    query_end = pred_end + timedelta(seconds=time_length) + timedelta(hours=1)

    # Step 2: fetch existing miner scores
    asset_backtest = asset
    time_length_backtest = time_length
    prompt_label = slug_for(competition)
    log_prefix = f"  [{prompt_label}/{asset_backtest}]"
    t_start = perf_counter()

    print(
        f"{log_prefix} fetching scored prompts ({query_start.date()} → {min(query_end, anchor).date()})...",
        flush=True,
    )
    asset_scores = get_miner_scores(
        start_time=query_start,
        end_time=min(query_end, anchor),
        asset=asset_backtest,
        time_length=time_length_backtest,
        time_increment=time_increment,
    )
    if asset_scores.empty:
        raise NoScoresAvailable(
            f"no scored prompts for asset={asset_backtest}, "
            f"time_length={time_length_backtest}, range=[{query_start}, {query_end}]"
        )
    print(
        f"{log_prefix} {len(asset_scores)} score rows ({asset_scores['scored_time'].nunique()} prompts) "
        f"in {perf_counter()-t_start:.1f}s",
        flush=True,
    )

    # Step 3: fetch rewards history
    prompt_name = None if not time_length_backtest else prompt_label
    t_rh = perf_counter()
    print(
        f"{log_prefix} fetching rewards history (prompt_name={prompt_name})...",
        flush=True,
    )
    rewards_history = get_rewards_history(
        asset_scores["scored_time"].min() - pd.Timedelta(hours=24),
        asset_scores["scored_time"].max() + pd.Timedelta(hours=24),
        prompt_name=prompt_name,
    )
    if rewards_history.empty:
        raise RuntimeError(
            f"Synth API returned no rewards history for prompt_name={prompt_name}, "
            f"range=[{asset_scores['scored_time'].min()}, {asset_scores['scored_time'].max()}]"
        )
    print(
        f"{log_prefix} {len(rewards_history)} rewards rows "
        f"({rewards_history['updated_at'].nunique()} update rounds) in {perf_counter()-t_rh:.1f}s",
        flush=True,
    )

    # Step 4: align scores and rewards history time ranges
    if asset_backtest is not None:
        asset_scores = asset_scores.sort_values("scored_time")
        rewards_history = rewards_history.sort_values("updated_at")

        result = pd.merge_asof(
            asset_scores[["scored_time"]].drop_duplicates(),
            rewards_history[["updated_at"]].drop_duplicates(),
            left_on="scored_time",
            right_on="updated_at",
            direction="forward",
        )
        result = result[~result["updated_at"].isna()]
        rewards_history = rewards_history.loc[
            rewards_history["updated_at"].isin(result["updated_at"])
        ].copy()
        asset_scores = asset_scores.loc[
            asset_scores["scored_time"].isin(result["scored_time"])
        ].copy()
    else:
        if rewards_history["updated_at"].max() < asset_scores["scored_time"].max():
            max_scored_time = asset_scores.loc[
                asset_scores["scored_time"] <= rewards_history["updated_at"].max(),
                "scored_time",
            ].max()
            asset_scores = asset_scores.loc[
                asset_scores["scored_time"] <= max_scored_time
            ].copy()
        elif rewards_history["updated_at"].max() > asset_scores["scored_time"].max():
            max_upd_at = rewards_history.loc[
                rewards_history["updated_at"] >= asset_scores["scored_time"].max(),
                "updated_at",
            ].min()
            rewards_history = rewards_history.loc[
                rewards_history["updated_at"] <= max_upd_at
            ].copy()

    # Step 5: download asset price data
    asset_prices: dict[str, pd.DataFrame] = {}
    for asset_ in asset_scores["asset"].unique():
        st_ = asset_scores["start_time"].min() - pd.Timedelta(hours=2)
        en_ = asset_scores["start_time"].max() + pd.Timedelta(hours=26)
        asset_prices[asset_] = download_price_data(st_, en_, asset_, freq="1min")

    # Step 6: score LLM miner predictions
    start_times = asset_scores["start_time"].sort_values().unique()
    t_score = perf_counter()
    print(
        f"{log_prefix} scoring {len(start_times)} prompts against local predictions...",
        flush=True,
    )

    # Pre-build work items: resolve file paths and slice prices upfront
    prompts: list[dict[str, Any]] = []

    for start_time in start_times:
        scores_subset = asset_scores.loc[
            asset_scores["start_time"] == start_time
        ].copy()

        for _, asset_val, scored_time, time_len, time_incr in (
            scores_subset[["asset", "scored_time", "time_length", "time_increment"]]
            .drop_duplicates()
            .itertuples()
        ):
            file_path = _find_prediction_file(
                prediction_files, start_time, asset_val, time_len
            )

            # Pre-slice real prices in the main process (avoids pickling DataFrames)
            real_prices: list[float] = []
            if file_path is not None:
                real_prices = _slice_real_prices(
                    asset_prices[asset_val], start_time, time_len, time_incr
                )

            prompts.append(
                {
                    "file_path": file_path,
                    "start_time": start_time,
                    "asset": asset_val,
                    "scored_time": scored_time,
                    "time_length": time_len,
                    "time_increment": time_incr,
                    "real_prices": real_prices,
                }
            )

    _fill_gaps_from_realized_paths(prompts, log_prefix)

    work_items = [
        (
            prompt["file_path"],
            prompt["start_time"],
            prompt["asset"],
            prompt["scored_time"],
            prompt["time_length"],
            prompt["time_increment"],
            prompt["real_prices"],
            scoring_intervals,
            miner_id,
        )
        for prompt in prompts
    ]

    # Dispatch scoring — parallel if executor provided, else sequential
    if scoring_executor is not None:
        futures = [
            scoring_executor.submit(_score_single_prompt, *item) for item in work_items
        ]
        raw_results = [f.result() for f in futures]
    else:
        raw_results = [_score_single_prompt(*item) for item in work_items]

    scores_extensions = [pd.DataFrame(r, index=[0]) for r in raw_results]

    # Step 7: concatenate LLM miner scores
    if not scores_extensions:
        raise RuntimeError(
            "No prediction files matched any scored prompt. "
            "Check that prediction filenames align with the scored time range."
        )
    new_scores = pd.concat(scores_extensions, ignore_index=True)
    print(
        f"{log_prefix} scored {len(new_scores)} prompts in {perf_counter()-t_score:.1f}s",
        flush=True,
    )

    # Step 8: trim to valid range
    new_scores = new_scores.sort_values("scored_time")
    valid_scores = new_scores.loc[new_scores["crps"] != -1]
    if valid_scores.empty:
        raise RuntimeError(
            "All matched predictions resulted in crps=-1 (scoring failures). "
            "No valid backtest results to report."
        )
    end_backtest_time = valid_scores["scored_time"].max()
    new_scores = new_scores.loc[new_scores["scored_time"] <= end_backtest_time]
    asset_scores = asset_scores.loc[asset_scores["scored_time"] <= end_backtest_time]

    # Optional: simulate the miner registering on a specific date by dropping our
    # CRPS rows before that date. synth.validator.prepare_df_for_moving_average
    # then sees our miner's first scored_time > global_min and applies the same
    # worst-score backfill it gives real late-joining miners — making rank
    # comparisons symmetric instead of giving us an unfair "from-the-start" boost.
    if simulate_registration is not None:
        sim_reg = pd.Timestamp(simulate_registration)
        new_scores = new_scores.loc[new_scores["scored_time"] >= sim_reg]
        if new_scores.empty:
            raise RuntimeError(
                f"simulate_registration={sim_reg} drops all of our miner's CRPS rows; "
                f"no valid backtest results."
            )
    if simulate_deregistration is not None:
        sim_dereg = pd.Timestamp(simulate_deregistration)
        new_scores = new_scores.loc[new_scores["scored_time"] <= sim_dereg]
        if new_scores.empty:
            raise RuntimeError(
                f"simulate_deregistration={sim_dereg} drops all of our miner's CRPS rows; "
                f"no valid backtest results."
            )

    # Step 9: merge LLM scores with all miner scores
    all_scores = pd.concat([asset_scores, new_scores], ignore_index=True)
    all_scores = all_scores.sort_values(["scored_time", "miner_uid"])

    if rewards_history["updated_at"].max() < all_scores["scored_time"].max():
        max_scored_time = all_scores.loc[
            all_scores["scored_time"] <= rewards_history["updated_at"].max(),
            "scored_time",
        ].max()
        all_scores = all_scores.loc[all_scores["scored_time"] <= max_scored_time].copy()
    elif rewards_history["updated_at"].max() > all_scores["scored_time"].max():
        max_upd_at = rewards_history.loc[
            rewards_history["updated_at"] >= all_scores["scored_time"].max(),
            "updated_at",
        ].min()
        rewards_history = rewards_history.loc[
            rewards_history["updated_at"] <= max_upd_at
        ].copy()

    # Step 12: apply prompt score calculation across all scores. We also persist
    # percentile90 and lowest_score per group because prepare_df_for_moving_average
    # uses them to compute the worst-score backfill for new miners (silently
    # skips backfill when those columns are absent).
    _stats = all_scores.groupby(
        ["scored_time", "asset", "time_length", "time_increment"], group_keys=False
    )["crps"].apply(_compute_prompt_score_stats_for_group)
    all_scores["new_prompt_scores"] = _stats["new_prompt_scores"]
    all_scores["percentile90"] = _stats["percentile90"]
    all_scores["lowest_score"] = _stats["lowest_score"]

    # Step 13: build rewards history entries for LLM miner
    new_miner_rewards_list = []
    for prompt_name_item in rewards_history["prompt_name"].unique():
        new_miner_rewards_prompt = pd.DataFrame(
            {
                "updated_at": rewards_history.loc[
                    rewards_history["prompt_name"] == prompt_name_item, "updated_at"
                ]
                .sort_values()
                .unique(),
                "miner_uid": miner_id,
                "prompt_name": prompt_name_item,
            }
        )
        new_miner_rewards_list.append(new_miner_rewards_prompt)
    new_miner_rewards = pd.concat(
        new_miner_rewards_list, ignore_index=True
    ).sort_values("updated_at")

    # Step 14: merge LLM miner rewards into full rewards history
    rewards_history = pd.concat(
        [
            rewards_history,
            new_miner_rewards,
        ]
    ).sort_values(["updated_at", "prompt_name", "miner_uid"])

    # Step 15: recalculate smoothed scores including LLM miner
    n_rounds = rewards_history["updated_at"].nunique()
    t_smooth = perf_counter()
    print(
        f"{log_prefix} computing smoothed scores over {n_rounds} reward rounds...",
        flush=True,
    )
    recalculated_smoothed_scores = calculate_smoothed_scores(
        all_scores,
        rewards_history,
        cutoff_days=competition.window_days,
        scores_column="new_prompt_scores",
        competition=competition,
    )
    print(
        f"{log_prefix} smoothed scores done in {perf_counter()-t_smooth:.1f}s "
        f"(total backtest: {perf_counter()-t_start:.1f}s)",
        flush=True,
    )

    # Trim the first `competition.window_days` of smoothed_scores when no
    # simulate_registration is set. In that case the backfill artifact applies to
    # real late-joining miners and inflates our rank — we want to hide it.
    # When simulate_registration IS set, the backfill applies to OUR miner and
    # the warmup ramp-up is the whole point of the simulation — don't trim it.
    if simulate_registration is None:
        recalculated_smoothed_scores = _trim_warmup(
            recalculated_smoothed_scores,
            all_scores,
            warmup_days=competition.window_days,
        )

    miner_smoothed = recalculated_smoothed_scores.loc[
        recalculated_smoothed_scores["miner_uid"] == miner_id, "new_smoothed_score"
    ]
    summary: dict[str, Any] = {
        "miner_name": miner_name,
        "miner_id": miner_id,
        "num_prompts": int((new_scores["crps"] != -1).sum()),
        "mean_crps": float(new_scores.loc[new_scores["crps"] != -1, "crps"].mean()),
        "final_smoothed_score": (
            float(miner_smoothed.iloc[-1]) if not miner_smoothed.empty else None
        ),
    }

    return BacktestResult(
        miner_name=miner_name,
        prompt_df=all_scores,
        smoothed_scores=recalculated_smoothed_scores,
        summary=summary,
    )

def run_backtest(
    miner_name: str,
    competition: CompetitionConfig,
    n_backtest_days: int,
    predictions_dir: Path | None = None,
    miner_id: int = 999,
    scoring_executor: ProcessPoolExecutor | None = None,
    eval_end: datetime | None = None,
    simulate_registration: datetime | None = None,
    simulate_deregistration: datetime | None = None,
) -> tuple[list[BacktestResult], pd.DataFrame]:
    """Run the full backtest for all assets in a prompt config.

    For each asset: runs backtest, prints rank summary, saves per-asset rank chart.
    After all assets: saves total rank evolution chart + estimated earnings chart
    when ≥2 assets succeed.

    Returns (results, combined) where:
      - results: list of successful BacktestResult (one per asset)
      - combined: per-profile combined smoothed-scores DataFrame (empty when <2 assets)
    """
    _maybe_warn_hf_crps_formula_change(
        competition=competition,
        n_backtest_days=n_backtest_days,
        eval_end=eval_end,
        simulate_registration=simulate_registration,
        simulate_deregistration=simulate_deregistration,
    )
    _maybe_warn_competition_split(
        competition=competition,
        n_backtest_days=n_backtest_days,
        eval_end=eval_end,
        simulate_registration=simulate_registration,
        simulate_deregistration=simulate_deregistration,
    )

    assets = competition.asset_list
    max_workers = min(len(assets), 6)
    print(f"  Dispatching {len(assets)} assets to {max_workers} backtest workers...")

    # Phase 1: run backtest() for each asset in parallel (I/O-bound: API calls + disk)
    asset_results: dict[str, BacktestResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                backtest,
                miner_name=miner_name,
                asset=asset,
                time_length=competition.time_length,
                time_increment=competition.time_increment,
                n_backtest_days=n_backtest_days,
                miner_id=miner_id,
                predictions_dir=predictions_dir,
                scoring_intervals=competition.scoring_intervals,
                competition=competition,
                scoring_executor=scoring_executor,
                eval_end=eval_end,
                simulate_registration=simulate_registration,
                simulate_deregistration=simulate_deregistration,
            ): asset
            for asset in assets
        }
        for future in as_completed(futures):
            asset = futures[future]
            try:
                asset_results[asset] = future.result()
            except NoScoresAvailable as e:
                # Not a failure — the validator isn't prompting this asset here.
                print(f"\n--- BACKTEST [{competition.label}/{asset}] ---")
                print(f"  SKIPPED: {e}")
            except Exception as e:
                print(f"\n--- BACKTEST [{competition.label}/{asset}] ---")
                print(f"  FAILED: {e}")

    # Phase 2: print summaries + plot charts sequentially (matplotlib is not thread-safe)
    results: list[BacktestResult] = []
    for asset in assets:
        if asset not in asset_results:
            continue
        result = asset_results[asset]
        results.append(result)
        summary = result.summary

        print(f"\n--- BACKTEST [{competition.label}/{asset}] ---")
        ss = result.smoothed_scores.copy()
        if not ss.empty:
            last_ts = ss["updated_at"].max()
            last_slice = ss.loc[ss["updated_at"] == last_ts]
            total_miners = len(last_slice)
            miner_rw = last_slice.loc[last_slice["miner_uid"] == miner_id]
            if not miner_rw.empty:
                rank = int(
                    last_slice["reward_weight"]
                    .rank(ascending=False, method="min")
                    .loc[miner_rw.index[0]]
                )
                rw = float(miner_rw["reward_weight"].iloc[0])
                sm = float(summary.get("final_smoothed_score", 0) or 0)
                print(
                    f"  rank: {rank}/{total_miners}  reward_weight: {rw:.6f}  smoothed_score: {sm:.2f}"
                )
            else:
                print(f"  Miner {miner_id} not found in final smoothed scores.")
        else:
            print(f"  No smoothed scores computed.")

        print(
            f"  Prompts scored: {summary['num_prompts']}  Mean CRPS: {summary['mean_crps']:.6f}"
        )

        try:
            chart_path = plot_rank_evolution(result)
            print(f"  Rank chart saved to: {chart_path}")
        except (RuntimeError, Exception) as e:
            print(f"  Chart generation failed: {e}")

        try:
            crps_chart = plot_crps_over_time(result)
            print(f"  CRPS chart saved to: {crps_chart}")
        except (RuntimeError, Exception) as e:
            print(f"  CRPS chart generation failed: {e}")

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

    # Cross-asset charts: compute combined smoothed scores once, share across both charts.
    combined = pd.DataFrame(columns=_COMBINED_EMPTY_COLS)
    if len(results) >= 2:
        try:
            combined = compute_combined_smoothed_scores(
                results,
                competition,
                simulate_registration=simulate_registration,
            )
        except Exception as e:
            print(f"  Combined smoothed scores failed: {e}")

        if not combined.empty:
            try:
                total_chart = plot_total_rank_evolution(
                    results,
                    combined=combined,
                    profile_label=slug_for(competition),
                )
                print(f"  Total rank chart saved to: {total_chart}")
            except Exception as e:
                print(f"  Total rank chart failed: {e}")

            try:
                earnings_chart = plot_estimated_earnings(
                    results,
                    competition=competition,
                    combined=combined,
                )
                print(f"  Earnings chart saved to: {earnings_chart}")
            except Exception as e:
                print(f"  Earnings chart failed: {e}")

    return results, combined
