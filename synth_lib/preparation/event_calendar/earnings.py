"""Earnings dates of the companies behind the tokenised stocks, from Nasdaq's earnings calendar.

Nasdaq gives the day and the session, not the minute: a release before the open is anchored at
08:00 New York, one after the close, or with no session given, at 16:00.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

from synth_lib.preparation.config import venue_session
from synth_lib.preparation.event_calendar.events import event_frame, new_york

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


def fetch(
    start: date, end: date, session: requests.Session | None = None, pause_seconds: float = 0.2
) -> dict[str, pd.DataFrame]:
    """Every weekday in [start, end], one request each."""
    session = session or venue_session()
    rows = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            response = session.get(NASDAQ_URL, params={"date": day.isoformat()}, headers=_HEADERS, timeout=30)
            response.raise_for_status()
            rows += parse_day(response.json(), day)
            time.sleep(pause_seconds)
        day += timedelta(days=1)
    frame = event_frame(rows)
    return {source_of(asset): frame[frame["scope"] == asset].reset_index(drop=True) for asset in TICKERS.values()}
