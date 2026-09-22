"""Build an offline data bundle for the backtester.

Long backtests need more scores/rewards history than the Synth API serves in
one request (the scores endpoints cap ranges at a few days). This script
downloads the required data in small chunks and writes the exact layout the
backtester's offline mode (SYNTH_BACKTESTER_OFFLINE_DATA_ROOT) expects:

    {out}/miner_scores_{asset}_{slug}.parquet
    {out}/rewards_history_{slug}.parquet
    {out}/miner_pool_usd.parquet

Prices are not bundled: the backtester reads them from the local
market_data/prices/{asset}/1m parquets (see synth_lib/preparation/market_data.py
to pre-download them). Hyperliquid-routed assets have no minute history beyond
~3.5 days, so each bundled prompt's realized path is cached under
market_data/realized/ instead.

Already-written parquets are skipped, so an interrupted run can be resumed. The window each one
was fetched for is recorded in manifest.json and checked before that skip: the filenames carry
the asset and the competition but not the dates, so without it a bundle built for one window
satisfies any --days/--eval-end and the backtest runs over whatever days the two windows share.

Usage (then run the backtest with SYNTH_BACKTESTER_OFFLINE_DATA_ROOT={out}):

    uv run synth_lib/backtester/scripts/build_offline_bundle.py \
        --competition crypto-24h --days 30 --eval-end 2026-07-19

    SYNTH_BACKTESTER_OFFLINE_DATA_ROOT=offline_data/crypto-24h \
        uv run synth_lib/backtester/scripts/run_backtest.py \
        --miner-name my_agent --competition crypto-24h --days 30
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Callable

import pandas as pd
import requests

from synth.validator.competition_config import CompetitionConfig
from synth_lib.backtester.config import SLUG_TO_COMPETITION
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

MANIFEST_NAME = "manifest.json"


def _manifest(out: Path) -> dict[str, dict[str, str]]:
    path = out / MANIFEST_NAME
    return json.loads(path.read_text()) if path.exists() else {}


def _record_window(out: Path, filename: str, start: datetime, end: datetime) -> None:
    """Stamp the window a file was fetched for, as each file lands rather than at the end, so an
    interrupted run still describes everything it wrote."""
    manifest = _manifest(out)
    manifest[filename] = {"start": start.isoformat(), "end": end.isoformat()}
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _require_window(out: Path, filename: str, start: datetime, end: datetime) -> None:
    """Refuse a file on disk that was not fetched for a window containing [start, end].

    A bundle may be the archived record a published verdict was produced from, so a file that does
    not cover the requested window is never refetched over in place.
    """
    entry = _manifest(out).get(filename)
    if entry is not None:
        if datetime.fromisoformat(entry["start"]) <= start and datetime.fromisoformat(entry["end"]) >= end:
            return
        held = f"covers {entry['start']} .. {entry['end']}"
    else:
        held = f"predates window tracking, so what it covers is unknown ({MANIFEST_NAME} has no entry)"
    raise SystemExit(
        f"{out / filename} {held}, but this run needs {start.isoformat()} .. {end.isoformat()}.\n"
        f"Build into a fresh --out, or delete the file to refetch it."
    )


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
            except requests.HTTPError as e:
                # A 404 means the endpoint has no rows for this range — the asset was
                # not yet listed in the competition (assets are added mid-window).
                # Treat it as an empty chunk, not a transient error.
                if e.response is not None and e.response.status_code == 404:
                    df = pd.DataFrame()
                    break
                print(f"  {label} [{cursor:%m-%d} -> {chunk_end:%m-%d}] attempt {attempt + 1} failed: {e}")
                sleep(10 * (attempt + 1))
            except Exception as e:  # noqa: BLE001 - transient API errors, retried
                print(f"  {label} [{cursor:%m-%d} -> {chunk_end:%m-%d}] attempt {attempt + 1} failed: {e}")
                sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"{label}: chunk [{cursor} -> {chunk_end}] failed {MAX_RETRIES} times")
        if not df.empty:
            frames.append(df)
        print(f"  {label} [{cursor:%m-%d} -> {chunk_end:%m-%d}]: {len(df)} rows", flush=True)
        cursor = chunk_end
        sleep(REQUEST_SPACING_SECONDS)
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


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
    chunk_days: float,
    out: Path,
    realized_paths: bool = True,
) -> None:
    competition = SLUG_TO_COMPETITION[slug]
    out.mkdir(parents=True, exist_ok=True)

    scores_start = eval_end - timedelta(days=days) - SCORES_PAD
    rewards_start = scores_start - REWARDS_PAD
    rewards_end = eval_end + REWARDS_PAD

    for asset in assets:
        name = f"miner_scores_{asset}_{slug}.parquet"
        path = out / name
        if path.exists():
            _require_window(out, name, scores_start, eval_end)
            print(f"skip {path} (exists)")
            df = pd.read_parquet(path)
        else:
            df = fetch_chunked(
                lambda s, e, a=asset: get_miner_scores(s, e, a, competition.time_length, competition.time_increment),
                scores_start,
                eval_end,
                chunk_days,
                f"scores/{asset}",
            )
            df = coerce_numeric_columns(df)
            df.to_parquet(path, index=False)
            _record_window(out, name, scores_start, eval_end)
            prompts = df["scored_time"].nunique() if not df.empty else 0
            print(f"wrote {path}: {len(df)} rows, {prompts} prompts")
        if realized_paths:
            bundle_realized_paths(asset, competition, df)

    name = f"rewards_history_{slug}.parquet"
    path = out / name
    if path.exists():
        _require_window(out, name, rewards_start, rewards_end)
        print(f"skip {path} (exists)")
    else:
        df = fetch_chunked(
            lambda s, e: get_rewards_history(s, e, prompt_name=slug),
            rewards_start,
            rewards_end,
            chunk_days,
            "rewards",
        )
        df = coerce_numeric_columns(df)
        df.to_parquet(path, index=False)
        _record_window(out, name, rewards_start, rewards_end)
        rounds = df["updated_at"].nunique() if not df.empty else 0
        print(f"wrote {path}: {len(df)} rows, {rounds} update rounds")

    name = "miner_pool_usd.parquet"
    path = out / name
    if path.exists():
        _require_window(out, name, rewards_start, rewards_end)
        print(f"skip {path} (exists)")
    else:
        pool = get_daily_miner_pool_usd(rewards_start, rewards_end)
        pd.DataFrame({"date": pool.index, "usd": pool.values}).to_parquet(path, index=False)
        _record_window(out, name, rewards_start, rewards_end)
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
        args.chunk_days,
        out,
        realized_paths=not args.no_realized_paths,
    )


if __name__ == "__main__":
    main()
