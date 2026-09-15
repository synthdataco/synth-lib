"""Download entrypoint for Synth Subnet minute-price data.

Kept as the CLI's home (`uv run synth_lib/preparation/market_data.py`) because that
path is documented in the miner tutorial. The pieces live in sibling modules:
config, {hyperliquid,binance}_client, price_client, validator_api,
minute_price_store, realized_path_store.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from synth_lib.preparation.config import (
    ALL_SYMBOLS,
    DEFAULT_HELDOUT_MONTHS,
    DEFAULT_TOTAL_MONTHS,
    UTC,
)
from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.preparation.price_client import build_price_client


def _compute_date_range(total_months: int, heldout_months: int) -> tuple[date, date]:
    """Compute (start_day, end_day) for the download window."""
    utc_now = datetime.now(tz=UTC)
    anchored = pd.Timestamp(utc_now).floor("D") - pd.DateOffset(months=heldout_months) - pd.Timedelta(days=1)
    end_day = anchored.date()
    start_day = (pd.Timestamp(end_day, tz="UTC") + pd.Timedelta(days=1) - pd.DateOffset(months=total_months)).date()
    return start_day, end_day


def download_market_data(
    asset: str = "BTC",
    total_months: int = DEFAULT_TOTAL_MONTHS,
    heldout_months: int = DEFAULT_HELDOUT_MONTHS,
    force_refresh: bool = False,
) -> Path:
    """Download minute-level price data for the given asset.

    Downloads a ``total_months`` window ending at (today - heldout_months).
    Stores data as daily parquet partitions.
    Skips days that already exist unless force_refresh=True.

    Returns the root directory containing the parquet files.
    """
    start_day, end_day = _compute_date_range(total_months, heldout_months)
    print(f"Downloading {asset} data: {start_day} to {end_day} ({total_months} months, {heldout_months} held out)")
    client = build_price_client(asset)
    store = MinutePriceStore(asset, client=client)
    store.ingest_range(start_day, end_day, force_refresh=force_refresh)
    print(f"Done. Data stored in {store.root}")
    return store.root


def download_all_assets(
    total_months: int = DEFAULT_TOTAL_MONTHS,
    heldout_months: int = DEFAULT_HELDOUT_MONTHS,
    force_refresh: bool = False,
    assets: list[str] | None = None,
    days: int | None = None,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, Path]:
    """Download data for Synth Subnet assets, one day at a time across every asset.

    Parameters
    ----------
    assets : list of asset names to download. Defaults to all supported assets.
    days : if set, download this many days ending today (ignores total_months / heldout_months).
    start, end : explicit inclusive bounds, which is what makes a crashed run resumable — restart
        from the day it stopped instead of re-fetching the days already written.

    The outer loop is the day and the inner loop is the asset, so an interrupted run leaves every
    asset complete up to the same date. Iterating assets outermost instead would leave a ragged
    edge that only a per-asset survey could describe.
    """
    asset_list = assets if assets is not None else list(ALL_SYMBOLS.keys())

    if start is not None or end is not None:
        if start is None or end is None:
            raise ValueError("start and end must be given together")
        start_day, end_day = start, end
    elif days is not None:
        end_day = datetime.now(tz=UTC).date()
        start_day = end_day - timedelta(days=days)
    else:
        start_day, end_day = _compute_date_range(total_months, heldout_months)

    stores: dict[str, MinutePriceStore] = {}
    for asset in asset_list:
        try:
            client = build_price_client(asset)
        except ValueError:
            print(f"  Skipping unsupported asset: {asset}")
            continue
        stores[asset] = MinutePriceStore(asset, client=client)

    total_days = (end_day - start_day).days + 1
    print(f"Downloading {len(stores)} assets: {start_day} to {end_day} ({total_days} days)")
    cursor = start_day
    while cursor <= end_day:
        for asset, store in stores.items():
            store.ensure_root()
            store.ingest_day(cursor, force_refresh=force_refresh)
        print(f"  {cursor.isoformat()} done ({len(stores)} assets)", flush=True)
        cursor += timedelta(days=1)
    print("Done.")
    return {asset: store.root for asset, store in stores.items()}


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download minute-level price data for Synth Subnet assets.")
    parser.add_argument("--asset", default=None, help="Single asset to download (default: all)")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="If set, download this many days ending today (includes today; ignores --months/--heldout-months)",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Re-download existing partitions")
    parser.add_argument("--start", type=date.fromisoformat, default=None, metavar="YYYY-MM-DD")
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="with --start, an explicit inclusive window; restart a crashed run from where it stopped",
    )
    args = parser.parse_args()

    download_all_assets(
        assets=[args.asset] if args.asset else None,
        days=args.days,
        force_refresh=args.force_refresh,
        start=args.start,
        end=args.end,
    )
