"""Earnings dates of the companies behind the tokenised stocks, from Nasdaq's earnings calendar.

Nasdaq gives the day and the session, not the minute: a release before the open is anchored at
08:00 New York, one after the close, or with no session given, at 16:00.

It answers one day per request. A day read more than `SETTLE_DAYS` after it passed is recorded in
`SETTLED_FILE` beside the ledger, once the ledger holding its rows is written, and not read again.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from synth_lib.preparation.config import venue_session
from synth_lib.preparation.event_calendar.events import event_frame, new_york, write_atomic

logger = logging.getLogger(__name__)

NASDAQ_URL = "https://api.nasdaq.com/api/calendar/earnings"
# Nasdaq refuses a client without a browser user agent.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept": "application/json",
}
# ticker -> the asset it trades as
TICKERS = {"NVDA": "NVDAX", "TSLA": "TSLAX", "AAPL": "AAPLX", "GOOGL": "GOOGLX", "SPCX": "SPCX"}
SESSIONS = {"time-pre-market": ("before the open", 8, 0), "time-after-hours": ("after the close", 16, 0)}
UNKNOWN_SESSION = ("session not given", 16, 0)
SETTLED_FILE = "nasdaq_settled_days.parquet"
# How long after a day its listing is taken as final.
SETTLE_DAYS = 7


def parse_day(payload: dict, day: date) -> list[dict]:
    rows = []
    for row in (payload.get("data") or {}).get("rows") or []:
        asset = TICKERS.get(str(row.get("symbol", "")).strip())
        if asset is None:
            continue
        session, hour, minute = SESSIONS.get(row.get("time", ""), UNKNOWN_SESSION)
        quarter = str(row.get("fiscalQuarterEnding", "")).strip()
        rows.append(
            {
                "time_utc": new_york(day, hour, minute),
                "event": "EARNINGS",
                "scope": asset,
                "detail": f"{session}, fiscal quarter {quarter}" if quarter else session,
                "source": source_of(asset),
            }
        )
    return rows


def source_of(asset: str) -> str:
    """One source per company, so that a moved date is judged on its own."""
    ticker = next(t for t, a in TICKERS.items() if a == asset)
    return f"nasdaq:{ticker}"


def _read_settled(path: Path | None) -> set[date]:
    if path is None or not path.exists():
        return set()
    return {date.fromisoformat(day) for day in pd.read_parquet(path)["day"]}


def fetch(
    start: date,
    end: date,
    session: requests.Session | None = None,
    pause_seconds: float = 0.2,
    settled_file: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], Callable[[set[str]], None]]:
    """Every weekday in [start, end] but the settled ones, one request each. A company with no date
    in what was read is left out: its rows are kept as they were.

    Returns the frames and a commit to call once the ledger holding them is written, with the
    sources whose merge was refused: it records the days read that passed more than `SETTLE_DAYS`
    ago, except the days of a refused row."""
    session = session or venue_session()
    settled = _read_settled(settled_file)
    span = (start + timedelta(days=n) for n in range((end - start).days + 1))
    weekdays = [day for day in span if day.weekday() < 5]
    days = [day for day in weekdays if day not in settled]
    logger.info("nasdaq: %d of %d weekdays in %s .. %s to read, the rest settled", len(days), len(weekdays), start, end)
    rows = []
    for n, day in enumerate(days, 1):
        response = session.get(NASDAQ_URL, params={"date": day.isoformat()}, headers=_HEADERS, timeout=30)
        response.raise_for_status()
        rows += parse_day(response.json(), day)
        if n % 50 == 0:
            logger.info("nasdaq: %d/%d days read", n, len(days))
        time.sleep(pause_seconds)
    frame = event_frame(rows)
    out = {}
    for asset in TICKERS.values():
        own = frame[frame["scope"] == asset].reset_index(drop=True)
        if not own.empty:
            out[source_of(asset)] = own

    def commit(refused: set[str]) -> None:
        if settled_file is None:
            return
        today = datetime.now(timezone.utc).date()
        held = {t.tz_convert("America/New_York").date() for source in refused for t in out[source]["time_utc"]}
        done = {day for day in days if day < today - timedelta(days=SETTLE_DAYS) and day not in held}
        write_atomic(pd.DataFrame({"day": sorted(day.isoformat() for day in settled | done)}), settled_file)

    return out, commit
