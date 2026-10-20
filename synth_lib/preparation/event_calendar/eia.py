"""The EIA Weekly Petroleum Status Report: Wednesdays at 10:30 New York, moved in the holiday weeks
EIA's schedule page lists. It concerns WTIOIL.

Reports are generated only in the years the page lists a release for: the holiday weeks of a later
year are unknown until EIA publishes that year.
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from synth_lib.preparation.config import venue_session
from synth_lib.preparation.event_calendar.events import event_frame, new_york

SOURCE = "eia:wpsr"
SCOPE = "WTIOIL"
SCHEDULE_URL = "https://www.eia.gov/petroleum/supply/weekly/schedule.php"
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.S)
_TIME = re.compile(r"(\d{1,2}):(\d{2})\s*([ap])\.?m\.?", re.I)
# The report on the week ending Friday comes out the Wednesday after.
REPORT_LAG = timedelta(days=5)


def _day(text: str) -> date:
    return datetime.strptime(text, "%B %d, %Y").date()


def parse_exceptions(page: str) -> dict[date, tuple[date, int, int, str]]:
    """Week-ending Friday -> (release day, hour, minute, holiday), from the schedule's exceptions table."""
    out = {}
    for row in _ROW.findall(page):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", cell)).strip() for cell in _CELL.findall(row)]
        if len(cells) != 5:
            continue
        try:
            week_ending, released = _day(cells[0]), _day(cells[1])
        except ValueError:
            continue  # the header row
        clock = _TIME.search(cells[3])
        if clock is None:
            raise ValueError(f"unreadable EIA release time {cells[3]!r}")
        hour = int(clock.group(1)) % 12 + (12 if clock.group(3).lower() == "p" else 0)
        out[week_ending] = (released, hour, int(clock.group(2)), cells[4])
    if not out:
        raise ValueError("no exception row on the EIA schedule page: has its layout changed?")
    return out


def releases(start: date, end: date, exceptions: dict[date, tuple[date, int, int, str]]) -> pd.DataFrame:
    """Every report released in [start, end] and in a year the exceptions list: the Wednesday after
    each week, or its listed exception."""
    years = sorted({released.year for released, *_ in exceptions.values()})
    start, end = max(start, date(years[0], 1, 1)), min(end, date(years[-1], 12, 31))
    rows = []
    friday = start - timedelta(days=(start.weekday() - 4) % 7 + 7)
    while friday <= end:
        day, hour, minute, holiday = exceptions.get(friday, (friday + REPORT_LAG, 10, 30, ""))
        if start <= day <= end:
            rows.append(
                {
                    "time_utc": new_york(day, hour, minute),
                    "event": "EIA_PETROLEUM",
                    "scope": SCOPE,
                    "detail": holiday,
                    "source": SOURCE,
                }
            )
        friday += timedelta(days=7)
    return event_frame(rows)


def fetch(start: date, end: date, session: requests.Session | None = None) -> dict[str, pd.DataFrame]:
    response = (session or venue_session()).get(SCHEDULE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    return {SOURCE: releases(start, end, parse_exceptions(response.text))}
