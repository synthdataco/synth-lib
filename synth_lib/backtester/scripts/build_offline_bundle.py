"""Build an offline data bundle for the backtester.

Long backtests need more scores/rewards history than the Synth API serves in
one request (the scores endpoints cap ranges at a few days). This script
downloads the required data one day at a time and writes the day partitions the
backtester's offline mode (SYNTH_BACKTESTER_OFFLINE_DATA_ROOT) reads
(synth_lib.backtester.field_store):

    {out}/miner_scores/{slug}/{asset}/date=YYYY-MM-DD.parquet
    {out}/rewards_history/{slug}/date=YYYY-MM-DD.parquet
    {out}/miner_pool_usd/date=YYYY-MM-DD.parquet

Prices are not bundled: the backtester reads them from the local
market_data/prices/{asset}/1m parquets (see synth_lib/preparation/market_data.py
to pre-download them). Hyperliquid-routed assets have no minute history beyond
~3.5 days, so each bundled prompt's realized path is cached under
market_data/realized/ instead.

A run fetches only the days of its window that have no complete partition, so an interrupted run
resumes and a wider window adds only its new days; --force-refresh fetches every day again.

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
import requests

from synth.validator.competition_config import CompetitionConfig
from synth_lib.backtester.config import SLUG_TO_COMPETITION
from synth_lib.backtester.field_store import pool_dir, read_days, refresh_days, rewards_dir, scores_dir
from synth_lib.backtester.loading import (
    get_daily_miner_pool_usd,
    get_miner_scores,
    get_rewards_history,
)
from synth_lib.preparation.realized_path_store import (
    RealizedPathStore,
    prefetch_realized_paths,
)

UTC = timezone.utc

# The backtester queries scores over [eval_end - days - match tolerance,
# eval_end] and rewards over the scores' scored_time range +-24h; pad both.
SCORES_PAD = timedelta(hours=1)
REWARDS_PAD = timedelta(hours=25)
MAX_RETRIES = 3

REQUEST_SPACING_SECONDS = 0.2


def retrying(
    fetch: Callable[[datetime, datetime], pd.DataFrame], label: str
) -> Callable[[datetime, datetime], pd.DataFrame]:
    """`fetch` with retries, a 404 as no rows, and numeric columns that parquet accepts."""

    def fetch_range(start: datetime, end: datetime) -> pd.DataFrame:
        for attempt in range(MAX_RETRIES):
            try:
                df = fetch(start, end)
                break
            except requests.HTTPError as e:
                # A 404 means the endpoint has no rows for this range — the asset was
                # not yet listed in the competition (assets are added mid-window).
                # Treat it as an empty range, not a transient error.
                if e.response is not None and e.response.status_code == 404:
                    df = pd.DataFrame()
                    break
                print(f"  {label} [{start:%m-%d %H:%M} -> {end:%m-%d %H:%M}] attempt {attempt + 1} failed: {e}")
                sleep(10 * (attempt + 1))
            except Exception as e:  # noqa: BLE001 - transient API errors, retried
                print(f"  {label} [{start:%m-%d %H:%M} -> {end:%m-%d %H:%M}] attempt {attempt + 1} failed: {e}")
                sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"{label}: [{start} -> {end}] failed {MAX_RETRIES} times")
        print(f"  {label} [{start:%m-%d %H:%M} -> {end:%m-%d %H:%M}]: {len(df)} rows", flush=True)
        sleep(REQUEST_SPACING_SECONDS)
        return coerce_numeric_columns(df.drop_duplicates()) if not df.empty else df

    return fetch_range


def coerce_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Give numeric-looking object columns a real numeric dtype before parquet.

    A single score above 2**53 makes pandas type the whole column `object`, and pyarrow then
    refuses the int -> double conversion as inexact and aborts the write — losing an otherwise
    complete asset, and with it every asset after it in the run. Scores that large are pathological
    to begin with, so the float64 they land on is no worse than the number itself.

    Columns that are genuinely textual (asset, timestamps) fail the parse and are left alone.
    """
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        try:
            numeric = pd.to_numeric(frame[column])
        except (TypeError, ValueError):
            continue  # a genuinely textual column
        # A value beyond int64 leaves to_numeric's result `object` rather than raising, and parquet
        # then refuses it just the same. float64 is lossy there, which is the right trade for a
        # number no honest score reaches.
        frame[column] = numeric.astype("float64") if numeric.dtype == object else numeric
    return frame


def bundle_realized_paths(asset: str, competition: CompetitionConfig, scores: pd.DataFrame) -> None:
    """Cache the validator's realized path for every bundled prompt, whatever the asset's venue.

    This used to run for Hyperliquid assets only, on the reasoning that they are the ones whose
    minute history the venue cannot serve. But it also decided where realized prices COME FROM: a
    Hyperliquid asset was scored against the validator's own arrays, a Binance one against whatever
    the operator's local store happened to hold. Two boxes then produce two different CRPS for the
    same champion on the same window, and only the crypto majors move.

    The bias has a direction. A 24h prompt's CRPS is a sum over its scoring points and NaN points
    are dropped, so a minute missing locally shrinks the sum: a thinner store scores BETTER. Caching
    every asset makes the ground truth the validator's for all of them, so the score stops depending
    on the box it was computed on.
    """
    if scores.empty:
        return
    store = RealizedPathStore(asset, competition.time_length, competition.time_increment)
    mapping = prefetch_realized_paths(store, sorted(scores["start_time"].unique()))
    print(f"realized paths/{asset}: {len(mapping)} prompts cached under {store.root}")


def build_bundle(
    slug: str,
    days: int,
    eval_end: datetime,
    assets: list[str],
    out: Path,
    realized_paths: bool = True,
    force_refresh: bool = False,
) -> None:
    competition = SLUG_TO_COMPETITION[slug]
    out.mkdir(parents=True, exist_ok=True)
    scores_start = eval_end - timedelta(days=days) - SCORES_PAD
    rewards_start = scores_start - REWARDS_PAD
    rewards_end = eval_end + REWARDS_PAD
    for asset in assets:
        directory = scores_dir(out, slug, asset)
        fetch = retrying(
            lambda s, e, a=asset: get_miner_scores(s, e, a, competition.time_length, competition.time_increment),
            f"scores/{asset}",
        )
        fetched = refresh_days(directory, scores_start, eval_end, fetch, "scored_time", force=force_refresh)
        print(f"{directory}: {fetched} days fetched")
        if realized_paths:
            scores = read_days(directory, scores_start, eval_end)
            bundle_realized_paths(asset, competition, scores)

    directory = rewards_dir(out, slug)
    fetch = retrying(lambda s, e: get_rewards_history(s, e, prompt_name=slug), "rewards")
    fetched = refresh_days(directory, rewards_start, rewards_end, fetch, "updated_at", force=force_refresh)
    print(f"{directory}: {fetched} days fetched")

    def pool(start: datetime, end: datetime) -> pd.DataFrame:
        series = get_daily_miner_pool_usd(start, end)
        return pd.DataFrame({"date": series.index, "usd": series.values})

    directory = pool_dir(out)
    fetched = refresh_days(directory, rewards_start, rewards_end, retrying(pool, "pool"), "date", force=force_refresh)
    print(f"{directory}: {fetched} days fetched")
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
        "--force-refresh",
        action="store_true",
        help="fetch every day of the window again, complete partitions included",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="bundle directory (default: offline_data/{competition})",
    )
    parser.add_argument(
        "--no-realized-paths",
        action="store_true",
        help="skip caching validator realized paths. Hyperliquid assets then have no prices "
        "beyond the venue's ~3.5 days and score as NaN CRPS; every asset falls back to the local "
        "minute store, where a missing minute silently lowers CRPS rather than failing",
    )
    args = parser.parse_args()

    if args.eval_end is not None:
        eval_end = datetime.strptime(args.eval_end, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        eval_end = (datetime.now(UTC) - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)

    competition = SLUG_TO_COMPETITION[args.competition]
    assets = args.assets if args.assets is not None else list(competition.asset_list)
    out = Path(args.out) if args.out is not None else Path("offline_data") / args.competition

    build_bundle(
        args.competition,
        args.days,
        eval_end,
        assets,
        out,
        realized_paths=not args.no_realized_paths,
        force_refresh=args.force_refresh,
    )


if __name__ == "__main__":
    main()
