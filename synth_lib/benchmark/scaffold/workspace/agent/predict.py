"""Self-evaluation via the official synth-lib backtester.

Usage: uv run agent/predict.py --asset BTC --days 3 --eval-end 2026-07-15
Generates predictions from agent/modeling.py over a grid of past prompts (covered by the
snapshot), writes them in the standard format, then scores them with backtest() (validator
CRPS + rank vs the real field via the Synth API).

Works on every asset, including the ones whose minutes do not all trade. Prices load through the
same lenient path the scored evaluation uses, so a NaN close reaches simulate() rather than
stopping the run — your model must tolerate them (drop, never interpolate).

--num-simulations defaults to 1000, matching the validator's real serving size
(PromptConfig.num_simulations). Lowering it (e.g. for a quick local iteration loop) speeds up
generation at the cost of a slightly pessimistic CRPS: empirical CRPS is biased upward for
fewer sampled paths, and the real field's CRPS was computed at 1000.

The context a prompt gets ends BEFORE its start_time, and that is not an off-by-one. The store
labels a minute bar by its OPEN time, so the bar labelled t closes at t+60s — and that close is
the first point the prompt is scored against. A prompt is also issued before the start_time it is
scored from (ISSUANCE_LEAD_SECONDS). Ending the context at t would hand the model the answer to
its own first step, and a model tuned that way is optimising a problem it will not be scored on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from synth_lib.backtester.orchestration import backtest
from synth_lib.benchmark.generate_predictions import context_end, load_minute_prices
from synth_lib.preparation.config import STORE_SUBDIR

# All paths are resolved from the location of THIS file, never from the cwd: the
# script therefore works whether launched from the workspace root (`uv run agent/predict.py`)
# or from agent/. The price snapshot is mounted at the workspace root, not inside agent/.
AGENT_DIR = Path(__file__).resolve().parent
WORKSPACE = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

from modeling import simulate  # noqa: E402  (import after adding AGENT_DIR to sys.path)

CONTEXT_MINUTES = 7 * 24 * 60
PREDICTIONS_DIR = WORKSPACE / "miner_outputs" / "campaign" / "predictions"
# backtest() fetches prices up to 26 h past the last prompt's start (synth_lib/backtester/orchestration.py).
BACKTEST_PADDING = timedelta(hours=26)


def ensure_workspace_cwd() -> None:
    """Force the cwd to the workspace root.

    NECESSARY: synth-lib's `backtest()` resolves the realized-price store via
    `MinutePriceStore(asset)` WITHOUT an explicit root (see `_price_store`), so it resolves
    relative to the cwd. Launched from agent/, it would look for `agent/market_data/`, find
    nothing, and attempt an `ingest_range` (download + write) against a snapshot mounted
    READ-ONLY — a cryptic error. The snapshot is mounted at the workspace root: we move there."""
    os.chdir(WORKSPACE)


def store_root(asset: str) -> Path:
    """Where this asset's partitions live inside the snapshot.

    Resolved by synth-lib rather than hardcoded: the canonical directory is `market_data/prices/`,
    with a fallback to the legacy `market_data/pyth/` when that is what the snapshot contains. Do
    Anchored on WORKSPACE, not the cwd — synth-lib's default_store_root() resolves relative to the
    cwd, which would break the invariant this file relies on everywhere else."""
    return WORKSPACE / "market_data" / STORE_SUBDIR / asset / "1m"


def check_snapshot_coverage(asset: str, start: datetime, end: datetime) -> None:
    """Fails EARLY and clearly if the requested window falls outside the frozen snapshot.

    Without this guard, synth-lib would try to ingest the missing days (network + write) and
    the error would surface in an unreadable form."""
    root = store_root(asset)
    missing = []
    cursor = start.date()
    while cursor <= end.date():
        if not (root / f"date={cursor.isoformat()}.parquet").exists():
            missing.append(cursor.isoformat())
        cursor += timedelta(days=1)
    if missing:
        raise SystemExit(
            f"window outside snapshot for {asset}: missing partitions {missing[:5]}"
            f"{' …' if len(missing) > 5 else ''}\n"
            f"The snapshot is frozen (read-only) and stops at the campaign's data_cutoff. "
            f"Reduce --eval-end / --days to stay within the available coverage ({root})."
        )


def check_backtest_padding(asset: str, end: datetime, time_length: int) -> None:
    """Fails if backtest() would read past the snapshot.

    backtest() reads beyond --eval-end — scores to the end of that day, rewards a day further,
    prices 26 h past the last prompt's start — and fetches whatever the snapshot lacks from the live
    API and the venues: data from after the cutoff. --eval-end therefore stops a horizon and 26 h
    before the snapshot's last day."""
    days = sorted(p.name for p in store_root(asset).glob("date=*.parquet"))
    if not days:
        return  # check_snapshot_coverage reports the missing partitions
    last_day = datetime.fromisoformat(days[-1].removeprefix("date=").removesuffix(".parquet"))
    latest = last_day.replace(tzinfo=timezone.utc) - timedelta(seconds=time_length) - BACKTEST_PADDING
    if end > latest:
        raise SystemExit(
            f"--eval-end {end.date()} is too late for --time-length {time_length}: backtest() reads a horizon "
            f"and 26 h past it, beyond the snapshot's last day ({last_day.date()}), and would fetch that "
            f"from the live API. Use --eval-end {latest.date()} or earlier."
        )


def generate(
    asset: str,
    start: datetime,
    end: datetime,
    cadence_minutes: int = 60,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
) -> int:
    # The same loader the scored evaluation uses. NOT MinutePriceStore.load_range, which raises
    # on a single NaN close: an untraded minute is normal on the thin markets (42% of AAPLX
    # minutes never trade), so that loader cannot read 9 of the 13 assets at all. NaN reaches
    # simulate() untouched, which is the contract — never interpolate it away.
    prices = load_minute_prices(
        WORKSPACE / "market_data", asset, pd.Timestamp(start) - pd.Timedelta(minutes=CONTEXT_MINUTES), pd.Timestamp(end)
    )
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    end_ts = pd.Timestamp(end)
    n = 0
    for t in pd.date_range(start, end, freq=f"{cadence_minutes}min", tz="UTC"):
        if t >= end_ts:
            break
        # Never `prices.loc[:t]` — see the module docstring. Same cut the scored evaluation uses.
        context = prices.loc[: context_end(t, time_length)][-CONTEXT_MINUTES:]
        out = simulate(
            asset=asset,
            start_time=t.isoformat(),
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            context_prices=context,
        )
        paths = [list(map(float, p)) for p in out[2:]]
        record = {
            "start_timestamp": t.isoformat(),
            "asset": asset,
            "time_increment": time_increment,
            "time_length": time_length,
            "num_simulations": len(paths),
            "num_steps": len(paths[0]),
            "paths": paths,
        }
        name = t.strftime("%Y-%m-%d_%H:%M:%SZ") + f"_{asset}_{time_length}.json"
        (PREDICTIONS_DIR / name).write_text(json.dumps(record))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--eval-end", required=True, help="YYYY-MM-DD, at most data_cutoff minus the horizon and 26 h")
    ap.add_argument("--time-increment", type=int, default=300, help="seconds between points (60 for CRYPTO_1H)")
    ap.add_argument("--time-length", type=int, default=86_400, help="horizon in seconds (3600 for CRYPTO_1H)")
    ap.add_argument(
        "--num-simulations",
        type=int,
        default=1000,
        help="Paths per prompt. The validator uses 1000; lowering this speeds up iteration at "
        "the cost of a slightly pessimistic CRPS (empirical CRPS is biased upward for fewer "
        "sampled paths, and the real field's CRPS was computed at 1000).",
    )
    args = ap.parse_args()
    ensure_workspace_cwd()
    end = datetime.fromisoformat(args.eval_end).replace(tzinfo=timezone.utc)
    check_backtest_padding(args.asset, end, args.time_length)
    start = end - timedelta(days=args.days)
    # the context reaches back 7 days before the first prompt: coverage must extend that far
    check_snapshot_coverage(args.asset, start - timedelta(minutes=CONTEXT_MINUTES), end)
    n = generate(
        args.asset,
        start,
        end,
        time_increment=args.time_increment,
        time_length=args.time_length,
        num_simulations=args.num_simulations,
    )
    print(f"predictions generated: {n}")
    result = backtest(
        miner_name="campaign",
        asset=args.asset,
        time_length=args.time_length,
        time_increment=args.time_increment,
        n_backtest_days=args.days,
        predictions_dir=PREDICTIONS_DIR,
        eval_end=end,
    )
    print("summary:", result.summary)


if __name__ == "__main__":
    main()
