"""Build an offline data bundle for the backtester.

Long backtests need more scores/rewards history than the Synth API serves in
one request (the scores endpoints cap ranges at a few days). This script
downloads the required data in small chunks and writes the exact layout the
backtester's offline mode (SYNTH_BACKTESTER_OFFLINE_DATA_ROOT) expects:

    {out}/miner_scores_{asset}_{slug}.parquet
    {out}/rewards_history_{slug}.parquet
    {out}/miner_pool_usd.parquet

Prices are not bundled: the backtester reads them from the local
market_data/pyth/{asset}/1m parquets (see synth_lib/preparation/market_data.py
to pre-download them).

Already-written parquets are skipped, so an interrupted run can be resumed.

Usage (then run the backtest with SYNTH_BACKTESTER_OFFLINE_DATA_ROOT={out}):

    uv run synth_lib/backtester/scripts/build_offline_bundle.py \
        --competition crypto-24h --days 30 --eval-end 2026-07-19

    SYNTH_BACKTESTER_OFFLINE_DATA_ROOT=offline_data/crypto-24h \
        uv run synth_lib/backtester/scripts/run_backtest.py \
        --miner-name my_agent --competition crypto-24h --days 30
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Callable

import pandas as pd

from synth_lib.backtester.backtest import (
    SLUG_TO_COMPETITION,
    get_daily_miner_pool_usd,
    get_miner_scores,
    get_rewards_history,
)

UTC = timezone.utc

# The backtester queries scores over [eval_end - days - match tolerance,
# eval_end] and rewards over the scores' scored_time range +-24h; pad both.
SCORES_PAD = timedelta(hours=1)
REWARDS_PAD = timedelta(hours=25)
MAX_RETRIES = 3


def fetch_chunked(
    fetch: Callable[[datetime, datetime], pd.DataFrame],
    start: datetime,
    end: datetime,
    chunk_days: float,
    label: str,
) -> pd.DataFrame:
    """Fetch [start, end) in chunk_days slices with retry, concat the results."""
    frames = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        for attempt in range(MAX_RETRIES):
            try:
                df = fetch(cursor, chunk_end)
                break
            except Exception as e:  # noqa: BLE001 - transient API errors, retried
                print(f"  {label} [{cursor:%m-%d} -> {chunk_end:%m-%d}] attempt {attempt + 1} failed: {e}")
                sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"{label}: chunk [{cursor} -> {chunk_end}] failed {MAX_RETRIES} times")
        if not df.empty:
            frames.append(df)
        print(f"  {label} [{cursor:%m-%d} -> {chunk_end:%m-%d}]: {len(df)} rows", flush=True)
        cursor = chunk_end
        sleep(2)
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def build_bundle(
    slug: str,
    days: int,
    eval_end: datetime,
    assets: list[str],
    chunk_days: float,
    out: Path,
) -> None:
    competition = SLUG_TO_COMPETITION[slug]
    out.mkdir(parents=True, exist_ok=True)

    scores_start = eval_end - timedelta(days=days) - SCORES_PAD
    rewards_start = scores_start - REWARDS_PAD
    rewards_end = eval_end + REWARDS_PAD

    for asset in assets:
        path = out / f"miner_scores_{asset}_{slug}.parquet"
        if path.exists():
            print(f"skip {path} (exists)")
            continue
        df = fetch_chunked(
            lambda s, e, a=asset: get_miner_scores(s, e, a, competition.time_length, competition.time_increment),
            scores_start,
            eval_end,
            chunk_days,
            f"scores/{asset}",
        )
        df.to_parquet(path, index=False)
        prompts = df["scored_time"].nunique() if not df.empty else 0
        print(f"wrote {path}: {len(df)} rows, {prompts} prompts")

    path = out / f"rewards_history_{slug}.parquet"
    if path.exists():
        print(f"skip {path} (exists)")
    else:
        df = fetch_chunked(
            lambda s, e: get_rewards_history(s, e, prompt_name=slug),
            rewards_start,
            rewards_end,
            chunk_days,
            "rewards",
        )
        df.to_parquet(path, index=False)
        rounds = df["updated_at"].nunique() if not df.empty else 0
        print(f"wrote {path}: {len(df)} rows, {rounds} update rounds")

    path = out / "miner_pool_usd.parquet"
    if path.exists():
        print(f"skip {path} (exists)")
    else:
        pool = get_daily_miner_pool_usd(rewards_start, rewards_end)
        pd.DataFrame({"date": pool.index, "usd": pool.values}).to_parquet(path, index=False)
        print(f"wrote {path}: {len(pool)} days")

    print(f"bundle complete: {out.resolve()}")
    print(f"run backtests with SYNTH_BACKTESTER_OFFLINE_DATA_ROOT={out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Synth API data into an offline backtest bundle.")
    parser.add_argument(
        "--competition",
        required=True,
        choices=sorted(SLUG_TO_COMPETITION),
        help="competition slug to bundle",
    )
    parser.add_argument("--days", type=int, default=30, help="backtest window length in days (default: 30)")
    parser.add_argument(
        "--eval-end",
        default=None,
        metavar="YYYY-MM-DD",
        help="window end date; every prompt's horizon must have settled (default: today - 2 days)",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=None,
        metavar="ASSET",
        help="assets to bundle (default: the competition's full asset list)",
    )
    parser.add_argument(
        "--chunk-days",
        type=float,
        default=2.0,
        help="per-request range in days; the scores API rejects ranges over ~3 days (default: 2)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="bundle directory (default: offline_data/{competition})",
    )
    args = parser.parse_args()

    if args.eval_end is not None:
        eval_end = datetime.strptime(args.eval_end, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        eval_end = (datetime.now(UTC) - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)

    competition = SLUG_TO_COMPETITION[args.competition]
    assets = args.assets if args.assets is not None else list(competition.asset_list)
    out = Path(args.out) if args.out is not None else Path("offline_data") / args.competition

    build_bundle(args.competition, args.days, eval_end, assets, args.chunk_days, out)


if __name__ == "__main__":
    main()
