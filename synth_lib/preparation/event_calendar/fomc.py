"""FOMC statements, from the Fed's meeting calendar page: one per scheduled meeting, at 14:00 New York."""

from __future__ import annotations

import calendar
import re
from datetime import date

import pandas as pd
import requests

from synth_lib.preparation.config import venue_session
from synth_lib.preparation.event_calendar.events import ALL_ASSETS, event_frame, new_york

SOURCE = "fed:fomc"
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_MONTHS = {name[:3].lower(): number for number, name in enumerate(calendar.month_name) if name}
_YEAR = re.compile(r"(\d{4}) FOMC Meetings</a>(.*?)(?=\d{4} FOMC Meetings</a>|$)", re.S)
_MEETING = re.compile(
    r"fomc-meeting__month[^>]*><strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)</div>", re.S
)


def parse_meetings(html: str) -> list[tuple[date, bool]]:
    """(decision day, with projections) for each scheduled meeting the page lists.

    A meeting is `<strong>April/May</strong>` then `28-1*`: the decision is on its last day, in the
    last month named, and `*` marks a meeting with a Summary of Economic Projections. Unscheduled
    meetings, notation votes and cancelled meetings carry no 14:00 statement and are skipped.
    """
    meetings = set()
    for year_match in _YEAR.finditer(html):
        year = int(year_match.group(1))
        for month, days in _MEETING.findall(year_match.group(2)):
            if re.search(r"unscheduled|notation|cancel", days, re.I):
                continue
            month_number = _MONTHS[month.split("/")[-1].strip()[:3].lower()]
            meetings.add((date(year, month_number, int(re.findall(r"\d+", days)[-1])), "*" in days))
    if not meetings:
        raise ValueError("no FOMC meeting on the Fed calendar page: has its layout changed?")
    return sorted(meetings)


def fetch(start: date, end: date, session: requests.Session | None = None) -> dict[str, pd.DataFrame]:
    response = (session or venue_session()).get(FOMC_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    rows = [
        {
            "time_utc": new_york(day, 14, 0),
            "event": "FOMC",
            "scope": ALL_ASSETS,
            "detail": "with projections" if projections else "",
            "source": SOURCE,
        }
        for day, projections in parse_meetings(response.text)
        if start <= day <= end
    ]
    return {SOURCE: event_frame(rows)}
