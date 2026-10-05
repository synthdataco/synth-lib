"""Verdict evaluation via synth-lib

For each competition: backtest each asset (per-asset fault isolation preserved — an
asset that fails, e.g. missing SPYX data / SPCX store quirks, must not fail the whole
evaluation), pass the list of successful `BacktestResult`s to
`compute_combined_smoothed_scores` (WITHOUT cutoff_days — letting the function use the
competition's own window, `competition.window_days`, like the real validator), then compute
the candidate's (uid 999) rank/percentile in this combined field. Each competition rank
already weights its assets by the validator's coefficients; the cross-competition aggregate
is therefore an UNWEIGHTED AVERAGE of the three percentiles (one third each).

`reward_metrics` therefore also simulates the candidate's EMISSIONS
over the window from those same reward_weights, and the headline verdict Score becomes
100 x mean over competitions of min(1, reward_vs_top) (candidate's summed reward_weight /
best other miner's, capped per competition so the Score is bounded 0-100 like the standard
LLM benchmarks; beats_field flags when the cap binds and the raw ratio stays reported).

Integrity guard: a competition where some assets failed produces a combined score
computed on FEWER assets than the validator would actually use. `assets_failed` must
therefore appear in the verdict report, and a candidate with failures is not strictly
comparable to a candidate without failures — comparability is judged at the meta-report
level, not here.

Each competition also writes into `out_dir`, beside the verdict JSON: a rank-evolution chart (the
candidate's rank per scoring round against the field it faced — the verdict reports only the first
and last round), the per-prompt CRPS frame, and the recomputed smoothed scores and reward weights
of the whole field. The last two carry every miner, not just the candidate, so a champion that is
also mining live can be read twice out of one file: uid 999 as the backtester scored it, and its
live uid as the validator did.

Validator realized sourcing (Binance/HL), scores API paginated 1 day, CRPS = the
validator's official function — all handled by synth-lib.
Window > 3 days: prepare the offline bundle FIRST (prepare_offline_bundle), otherwise the API rejects it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd

from synth.validator.competition_config import COM_EQU_24H, CRYPTO_1H, CRYPTO_24H  # type: ignore[import-untyped]
from synth_lib.backtester.config import _OFFLINE_ENV_VAR, slug_for  # type: ignore[import-untyped]
from synth_lib.backtester.loading import get_rewards_history  # type: ignore[import-untyped]
from synth_lib.backtester.orchestration import backtest  # type: ignore[import-untyped]
from synth_lib.backtester.plots.rank import plot_total_rank_evolution  # type: ignore[import-untyped]
from synth_lib.backtester.scoring import (  # type: ignore[import-untyped]
    Registrations,
    SaveOrder,
    compute_combined_smoothed_scores,
)
from synth_lib.backtester.scripts.build_offline_bundle import (  # type: ignore[import-untyped]
    build_bundle,
    coerce_numeric_columns,
)

COMPETITIONS = (CRYPTO_24H, COM_EQU_24H, CRYPTO_1H)
MINER_ID = 999  # uid used to inject the candidate into the field (synth-lib default)


def reward_metrics(smoothed_scores: pd.DataFrame, miner_id: int = MINER_ID) -> dict | None:
    """Simulated-emissions metrics over the window

    - reward_share: candidate's fraction of the competition's emission pool over the window.
    - reward_vs_top: candidate total / best OTHER miner's total. The candidate is excluded from
      the denominator so a champion that beats the whole field reads as > 1.0, not capped at 1.
    - reward_rank: 1 + number of other miners whose window total beats the candidate's.
    """
    # miner_id tells a uid's successive occupants apart; frames written before it carry only miner_uid.
    key = "miner_id" if "miner_id" in smoothed_scores.columns else "miner_uid"
    totals = smoothed_scores.groupby(key)["reward_weight"].sum()
    if miner_id not in totals.index:
        return None
    candidate = float(totals.loc[miner_id])
    others = totals.drop(miner_id)
    if others.empty:
        return None
    vs_top = candidate / float(others.max())
    return {
        "reward_share": candidate / float(totals.sum()),
        "reward_vs_top": vs_top,
        "reward_rank": int((others > candidate).sum()) + 1,
        # beats_field kept separate from the capped Score: matching the field's best caps at
        # 100, but BEATING the whole field is too interesting to lose to the cap.
        "beats_field": vs_top > 1.0,
    }


def asset_rank_stats(smoothed_scores: pd.DataFrame, miner_id: int = MINER_ID) -> dict | None:
    """Candidate's best and mean rank across ONE asset's scoring rounds (1 = best reward_weight).

    Complements the competition-level final_rank: the per-asset trajectory shows WHERE a
    champion earns its competition rank (e.g. rank 1 on two assets, invisible on the rest)."""
    if smoothed_scores.empty or not bool((smoothed_scores["miner_uid"] == miner_id).any()):
        return None
    ranks = smoothed_scores.groupby("updated_at")["reward_weight"].rank(ascending=False, method="min")
    mine = ranks[smoothed_scores["miner_uid"] == miner_id]
    last = smoothed_scores[smoothed_scores["updated_at"] == smoothed_scores["updated_at"].max()]
    return {
        "best_rank": int(mine.min()),
        "mean_rank": round(float(mine.mean()), 1),
        "field_size": int(len(last)),
    }


def final_rank(smoothed_scores: pd.DataFrame, miner_id: int = MINER_ID) -> tuple[int, int]:
    """Rank (1 = best reward_weight) at the last updated_at + field size.
    Same logic as the synth-lib runner (run_backtest.py)."""
    last = smoothed_scores[smoothed_scores["updated_at"] == smoothed_scores["updated_at"].max()]
    ranks = last["reward_weight"].rank(ascending=False, method="min")
    rank = int(ranks[last["miner_uid"] == miner_id].iloc[0])
    return rank, int(len(last))


def _rank_chart(results: list, combined: pd.DataFrame, slug: str, out_dir: Path) -> str | None:
    """Write the competition's rank-evolution chart and return its filename.

    A chart is a report, not a result: a failure here must not lose a scoring run that has already
    spent its sandbox time, so it is reported and the verdict is written without it.
    """
    try:
        return plot_total_rank_evolution(results, combined, slug, out_dir).name
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"  [{slug}] rank chart failed: {exc}")
        return None


def _live_rewards(combined: pd.DataFrame, slug: str) -> pd.DataFrame | None:
    """The validator's own smoothed_score and reward_weight over the rounds the backtest recomputed.

    Read from the same offline bundle, so it costs nothing and cannot drift from what was scored.
    It is the reference the recomputed field_rewards is checked against: a miner present in both
    should hold the same weight in the same round, and the candidate's rank is only meaningful if
    the rest of the field reconstructs.
    """
    if combined.empty:
        return None
    rounds = pd.to_datetime(combined["updated_at"], utc=True)
    live = get_rewards_history(
        rounds.min().to_pydatetime(), (rounds.max() + pd.Timedelta(seconds=1)).to_pydatetime(), prompt_name=slug
    )
    return None if live.empty else live


def _frames(results: list, combined: pd.DataFrame, slug: str, out_dir: Path) -> dict[str, str]:
    """Write the competition's per-prompt CRPS, recomputed field rewards and the validator's own,
    and name what landed.

    `prompt_scores` is every BacktestResult.prompt_df concatenated — one row per (prompt, miner),
    carrying crps, new_prompt_scores, percentile95 and lowest_score. `field_rewards` is the frame
    the rank and the Score are read off: updated_at, miner_uid, new_smoothed_score, reward_weight.
    `live_rewards` is the same two quantities as the validator published them.
    Reported like the chart rather than raised: a write failure must not lose a scored run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "prompt_scores": pd.concat([r.prompt_df for r in results], ignore_index=True) if results else None,
        "field_rewards": combined if not combined.empty else None,
        "live_rewards": _live_rewards(combined, slug),
    }
    written: dict[str, str] = {}
    for kind, frame in frames.items():
        if frame is None or frame.empty:
            continue
        name = f"{kind}_{slug}.parquet"
        try:
            # A miner posting a score beyond int64 types the column `object`, which pyarrow
            # refuses mid-write.
            coerce_numeric_columns(frame).to_parquet(out_dir / name, index=False)
        except (OSError, ValueError, KeyError) as exc:
            print(f"  [{slug}] {name} failed: {exc}")
            continue
        written[kind] = name
    return written


def evaluate_candidate(
    name: str,
    predictions_dir: Path,
    window_end: pd.Timestamp,
    window_days: int,
    out_dir: Path,
    simulate_registration: datetime | None = None,
    registrations: Registrations | None = None,
    save_order: Mapping[int, SaveOrder] | None = None,
) -> dict:
    """Evaluates a candidate on the three competitions (CRYPTO_24H, COM_EQU_24H, CRYPTO_1H).

    For each competition: backtest each asset via synth_lib.backtest() (one asset at a
    time, GPU lock not required: CPU scoring) with the usual per-asset fault isolation, then
    aggregate the successful BacktestResults with compute_combined_smoothed_scores(competition=comp) —
    WITHOUT cutoff_days, to let the function use the competition's own window
    (competition.window_days). This is exactly the real validator's aggregation: it applies
    ASSET_COEFFICIENTS, normalizes per miner, then a single softmax — not an average of asset-
    percentiles. The candidate's (uid 999) rank/percentile in this combined field is computed by
    final_rank(). Percentile = 1 - (rank-1)/field_size.

    Per competition, `out_dir` receives a rank-evolution chart, the per-prompt CRPS frame and the
    recomputed field rewards, each named for the competition.

    `simulate_registration` scores the champion as a miner that registered that day: its CRPS rows
    before it are dropped, the validator's moving average classifies it as a late joiner, and the
    preceding `competition.window_days` are backfilled at the field's worst score. That is what a
    champion actually earns in its first days live, and it is NOT comparable to a Score without it.

    `registrations` (uid -> registration times of its occupants inside the scored data, see
    `get_registrations`) scores each field uid that changed hands as the miners live sees.
    `save_order` (time_length -> `get_scores_save_order`) backfills late joiners from each minute's
    first-saved prompt, as live does.
    """
    per_competition: dict[str, dict] = {}
    competition_percentiles: list[float] = []
    competition_reward_ratios: list[float] = []
    for comp in COMPETITIONS:
        slug = slug_for(comp)
        results = []
        assets_failed: list[str] = []
        per_asset: dict[str, dict] = {}
        for asset in comp.asset_list:
            try:
                result = backtest(
                    miner_name=name,
                    asset=asset,
                    time_length=comp.time_length,
                    time_increment=comp.time_increment,
                    n_backtest_days=window_days,
                    predictions_dir=predictions_dir,
                    eval_end=window_end.to_pydatetime(),
                    competition=comp,
                    simulate_registration=simulate_registration,
                    registrations=registrations,
                )
                results.append(result)
                per_asset[asset] = {
                    "mean_crps": result.summary["mean_crps"],
                    "num_prompts": result.summary["num_prompts"],
                    # Beside mean_crps on purpose: dropped NaN points shrink the CRPS sum, so a
                    # thin realized path flatters the champion. Two verdicts compare only at
                    # equal coverage.
                    "realized_coverage": result.summary.get("realized_coverage"),
                    **(asset_rank_stats(result.smoothed_scores) or {}),
                }
            except Exception as exc:
                assets_failed.append(asset)
                per_asset[asset] = {"error": repr(exc)}

        # Also here, not only per asset: without it _trim_warmup drops the first
        # competition.window_days of rounds — exactly the backfilled onboarding period the
        # flag exists to show — and the headline rank, rewards and charts all read this frame.
        combined = compute_combined_smoothed_scores(
            results,
            competition=comp,
            simulate_registration=simulate_registration,
            registrations=registrations,
            save_order=save_order.get(comp.time_length) if save_order else None,
        )
        if simulate_registration is not None and not combined.empty:
            # The lookback is needed to compute the moving average, but the candidate did not exist
            # in it. Its rows there carry the new-miner backfill's weight, and summing them would
            # credit the champion with emissions from before it registered.
            rounds = pd.to_datetime(combined["updated_at"], utc=True)
            combined = combined.loc[rounds >= pd.Timestamp(simulate_registration)].reset_index(drop=True)
        has_candidate = not combined.empty and bool((combined["miner_uid"] == MINER_ID).any())
        if results and has_candidate:
            rank, field_size = final_rank(combined)
            percentile: float | None = 1.0 - (rank - 1) / field_size
            rewards = reward_metrics(combined)
            # rank/percentile are the FINAL round's position (what the validator pays on);
            # rank_over_rounds averages the candidate's rank across every scoring round of the
            # window — the requested per-competition average rank, robust to a lucky last round.
            rank_over_rounds = asset_rank_stats(combined)
            rank_chart = _rank_chart(results, combined, slug, out_dir)
        else:
            rank = field_size = None
            percentile = None
            rewards = None
            rank_over_rounds = None
            rank_chart = None
        frames = _frames(results, combined, slug, out_dir)

        per_competition[slug] = {
            "rank": rank,
            "field_size": field_size,
            "percentile": percentile,
            "rank_over_rounds": rank_over_rounds,
            "rank_chart": rank_chart,
            "frames": frames,
            "rewards": rewards,
            "assets_failed": assets_failed,
            "per_asset": per_asset,
        }
        if percentile is not None:
            competition_percentiles.append(percentile)
        if rewards is not None:
            # capped per competition BEFORE the mean, so one beats-the-field competition
            # cannot mask weakness in another (0-100 Score contract; see reward_metrics)
            competition_reward_ratios.append(min(1.0, rewards["reward_vs_top"]))

    mean_competition_percentile = (
        sum(competition_percentiles) / len(competition_percentiles) if competition_percentiles else None
    )
    mean_reward_vs_top = (
        sum(competition_reward_ratios) / len(competition_reward_ratios) if competition_reward_ratios else None
    )
    return {
        "name": name,
        "per_competition": per_competition,
        "mean_competition_percentile": mean_competition_percentile,
        "mean_reward_vs_top": mean_reward_vs_top,
    }


def write_verdict(candidates: list[dict], out_path: Path) -> None:
    ranked = sorted(
        candidates,
        key=lambda c: -(c["mean_competition_percentile"] if c["mean_competition_percentile"] is not None else -1.0),
    )
    out_path.write_text(json.dumps({"ranking": [c["name"] for c in ranked], "candidates": ranked}, indent=2))


def prepare_offline_bundle(
    window_end: pd.Timestamp,
    window_days: int,
    out_root: Path | None = None,
    simulate_registration: datetime | None = None,
) -> Path:
    """Builds the synth-lib offline bundle for the three competitions (mandatory for a
    window > ~3 days, since the live scores API rejects wider ranges) and exports
    SYNTH_BACKTESTER_OFFLINE_DATA_ROOT. Thin wrapper around build_bundle
    (synth_lib/backtester/scripts/build_offline_bundle.py — a module-level function, cleanly
    importable, not local to the CLI script, so no subprocess is needed here); same call
    shape as run_backtest.py's --no-auto-bundle path (l.394-402). Returns the
    bundle directory.
    """
    anchor = window_end.to_pydatetime()
    # Simulating registration pushes the frame back another competition.window_days before the
    # registration day, so the field has to be bundled for a period the candidate did not exist in.
    extra = max(c.window_days for c in COMPETITIONS) if simulate_registration is not None else 0
    days = window_days + extra
    out = out_root or Path("offline_data") / f"verdict_{anchor:%Y%m%d}_{days}d"
    for comp in COMPETITIONS:
        build_bundle(
            slug=slug_for(comp),
            days=days,
            eval_end=anchor,
            assets=list(comp.asset_list),
            chunk_days=2.0,
            out=out,
        )
    os.environ[_OFFLINE_ENV_VAR] = str(out)
    return out
