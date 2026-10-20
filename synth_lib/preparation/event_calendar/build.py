"""Fetches every calendar source into `<root>/calendar/ledger.parquet` and writes the current view.

    FRED_API_KEY=... uv run python -m synth_lib.preparation.event_calendar.build --start 2025-08-01

Each source is fetched and merged on its own: one that fails, or that would drop more than one
upcoming event, leaves its rows as they were and the others still update. Without FRED_API_KEY the
US macro releases are skipped. Run it daily where the store lives: a snapshot freezes what the
ledger knew at its cutoff, and a new row is dated at the fetch that first lists it.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from synth_lib.preparation.config import MARKET_DATA_DIR
from synth_lib.preparation.event_calendar import earnings, eia, fomc, fred, nyse
from synth_lib.preparation.event_calendar.events import (
    CALENDAR_DIR,
    LEDGER_FILE,
    VIEW_FILE,
    TruncatedFetch,
    known_at,
    merge,
    read_ledger,
    write_atomic,
)

logger = logging.getLogger(__name__)

DAYS_AHEAD = 90


def _fred(start: date, end: date) -> dict[str, pd.DataFrame]:
    api_key = os.environ.get(fred.KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{fred.KEY_ENV} is not set")
    return fred.fetch(start, end, api_key)


SOURCES: dict[str, Callable[[date, date], dict[str, pd.DataFrame]]] = {
    "fred": _fred,
    "fomc": fomc.fetch,
    "nyse": nyse.fetch,
    "eia": eia.fetch,
    "earnings": earnings.fetch,
}


def update(
    root: Path,
    start: date,
    end: date,
    now: pd.Timestamp,
    sources: list[str] | None = None,
    allow_losses: bool = False,
) -> pd.DataFrame:
    """Merges each source's fetch of [start, end] into the ledger under `root`; returns the view at `now`.

    A source that fails, comes back empty or is refused is logged as a warning."""
    ledger_path = root / CALENDAR_DIR / LEDGER_FILE
    ledger = read_ledger(ledger_path)
    # Every event falls on the same day in New York and in UTC, so whole UTC days bound the fetch.
    span = (pd.Timestamp(start, tz="UTC"), pd.Timestamp(end + timedelta(days=1), tz="UTC"))
    for name in sources or list(SOURCES):
        try:
            fetched = SOURCES[name](start, end)
        except Exception as exc:  # a source down must not stop the others
            logger.warning("%s: fetch failed (%s); its rows are kept as they were", name, exc)
            continue
        for source, rows in fetched.items():
            try:
                ledger, added, superseded = merge(ledger, rows, source, now, span, allow_losses)
            except TruncatedFetch as exc:
                logger.warning("%s; its rows are kept as they were", exc)
                continue
            logger.info("%s: %d events, %d new, %d superseded", source, len(rows), added, superseded)
    write_atomic(ledger, ledger_path)
    view = known_at(ledger, now)
    write_atomic(view, root / CALENDAR_DIR / VIEW_FILE)
    return view


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=MARKET_DATA_DIR, help="market_data directory")
    ap.add_argument("--start", type=date.fromisoformat, required=True, help="first event day, YYYY-MM-DD")
    ap.add_argument("--days-ahead", type=int, default=DAYS_AHEAD, help="last event day, counted from today")
    ap.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=None, help="default: all")
    ap.add_argument(
        "--allow-losses",
        action="store_true",
        help="accept a fetch that drops several upcoming events, as a shutdown that moves releases does",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    now = pd.Timestamp.now(tz="UTC")
    view = update(
        args.root, args.start, now.date() + timedelta(days=args.days_ahead), now, args.sources, args.allow_losses
    )
    counts = view.groupby("event").size().to_dict()
    print(f"{len(view)} events known now in {args.root / CALENDAR_DIR / VIEW_FILE}: {counts}")


if __name__ == "__main__":
    main()
