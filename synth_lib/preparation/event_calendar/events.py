"""The event calendar: a ledger of every scheduled event a fetch has seen, and the view known at a time.

An event row says when it happens (`time_utc`), what it is (`event`), which assets it concerns
(`scope`: `all`, or asset names separated by commas), a free-text `detail` and the `source` that
listed it. The ledger adds `as_of`, the time by which the date was public, and `superseded_at`, the
time a later fetch stopped listing it (a moved or cancelled release). `known_at(ledger, t)` is the
calendar as it stood at `t`: what a snapshot freezes at its cutoff, so a champion never sees a date
published after it.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")

CALENDAR_DIR = "calendar"
LEDGER_FILE = "ledger.parquet"
VIEW_FILE = "events.parquet"
# The path of the view a champion reads. The agent sandbox, the verdict's generation sandbox and the
# serving miner each set it.
CALENDAR_ENV = "SYNTH_BENCHMARK_CALENDAR"
ALL_ASSETS = "all"

VIEW_COLUMNS = ["time_utc", "event", "scope", "detail", "source"]
LEDGER_COLUMNS = [*VIEW_COLUMNS, "as_of", "superseded_at"]
KEY = ["time_utc", "event", "scope"]

# Days before an event by which its date is always public. A source's first fetch dates its rows
# this far before the event; every later fetch dates a new row at the fetch.
PUBLISHED_DAYS_AHEAD = {
    "CPI": 21,
    "NFP": 21,
    "PPI": 21,
    "PCE": 21,
    "RETAIL_SALES": 21,
    "GDP": 21,
    "FOMC": 90,
    "NYSE_CLOSED": 365,
    "NYSE_EARLY_CLOSE": 365,
    "EIA_PETROLEUM": 21,
    "EARNINGS": 7,
}


class TruncatedFetch(ValueError):
    """A fetch that would drop more upcoming events than a schedule change does."""


def new_york(day: date, hour: int, minute: int) -> pd.Timestamp:
    """A New York wall-clock time on `day`, in UTC, daylight saving included."""
    return pd.Timestamp(datetime.combine(day, time(hour, minute), tzinfo=NEW_YORK)).tz_convert("UTC")


def event_frame(rows: list[dict]) -> pd.DataFrame:
    """Event rows as a view-shaped frame, sorted by time."""
    frame = pd.DataFrame(rows, columns=VIEW_COLUMNS)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    for column in VIEW_COLUMNS[1:]:
        frame[column] = frame[column].fillna("").astype(str)
    return frame.sort_values(KEY, kind="stable").reset_index(drop=True)


def empty_ledger() -> pd.DataFrame:
    frame = event_frame([])
    frame["as_of"] = pd.Series(dtype="datetime64[ns, UTC]")
    frame["superseded_at"] = pd.Series(dtype="datetime64[ns, UTC]")
    return frame


def read_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_ledger()
    frame = pd.read_parquet(path)
    for column in ("time_utc", "as_of", "superseded_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame[LEDGER_COLUMNS]


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Writes `frame` as parquet so that a reader sees the old file or the new one, never part of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _keys(frame: pd.DataFrame) -> list[tuple]:
    return list(zip(frame["time_utc"], frame["event"], frame["scope"]))


def merge(
    ledger: pd.DataFrame,
    fetched: pd.DataFrame,
    source: str,
    fetched_at: pd.Timestamp,
    span: tuple[pd.Timestamp, pd.Timestamp],
    allow_losses: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """Folds one source's fetch of the event times in `span` [start, end) into the ledger. Returns
    (ledger, rows added, rows superseded).

    Rows the ledger already lists stay as they are. A new row is dated at the fetch, or, on the
    source's first fetch, `PUBLISHED_DAYS_AHEAD` before the event. An upcoming row in `span` that the
    fetch no longer lists is superseded at the fetch; a past one, or one outside `span`, never is. A
    fetch that is empty, or that would supersede more than one upcoming row, raises `TruncatedFetch`
    unless `allow_losses`.
    """
    fetched = fetched.drop_duplicates(KEY).assign(source=source)
    if fetched.empty:
        raise TruncatedFetch(f"{source}: the fetch listed no events")
    own = ledger[ledger["source"] == source]
    current = own[own["superseded_at"].isna()]
    listed = set(_keys(fetched))
    start, end = span
    lost = current.index[
        [key not in listed and key[0] > fetched_at and start <= key[0] < end for key in _keys(current)]
    ]
    if len(lost) > 1 and not allow_losses:
        raise TruncatedFetch(f"{source}: the fetch drops {len(lost)} upcoming events; refusing it")
    held = set(_keys(current))
    added = fetched[[key not in held for key in _keys(fetched)]].copy()
    if own.empty:
        lead = pd.to_timedelta(added["event"].map(PUBLISHED_DAYS_AHEAD), unit="D")
        if lead.isna().any():
            raise KeyError(f"{source}: no publication lead for {sorted(added['event'][lead.isna()].unique())}")
        added["as_of"] = (added["time_utc"] - lead).clip(upper=fetched_at)
    else:
        added["as_of"] = fetched_at
    added["superseded_at"] = pd.Series(pd.NaT, index=added.index, dtype="datetime64[ns, UTC]")
    out = ledger.copy()
    out.loc[lost, "superseded_at"] = fetched_at
    if not added.empty:
        out = pd.concat([out, added[LEDGER_COLUMNS]], ignore_index=True)
    out = out.sort_values([*KEY, "as_of"], kind="stable").reset_index(drop=True)
    return out, len(added), len(lost)


def known_at(ledger: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """The calendar as it stood at `at`: rows public by then that no fetch had dropped yet."""
    public = ledger["as_of"] <= at
    standing = ledger["superseded_at"].isna() | (ledger["superseded_at"] > at)
    view = ledger.loc[public & standing, VIEW_COLUMNS].drop_duplicates(KEY)
    return view.sort_values(KEY, kind="stable").reset_index(drop=True)


def concerns(view: pd.DataFrame, asset: str) -> pd.Series:
    """Rows of the view whose scope includes `asset`."""
    return view["scope"].map(lambda scope: scope == ALL_ASSETS or asset in scope.split(","))
