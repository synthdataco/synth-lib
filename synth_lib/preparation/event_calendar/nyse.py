"""NYSE full closures and 1 p.m. early closes, from the exchange's holiday rules.

A closure is anchored at 09:30 New York, the open that does not happen; an early close at 13:00.
Both concern the tokenised US stocks and SP500, whose underlying market is shut.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from synth_lib.preparation.event_calendar.events import event_frame, new_york

SOURCE = "nyse:rules"
SCOPE = "NVDAX,TSLAX,AAPLX,GOOGLX,SPCX,SP500"
# Closures no rule produces: national days of mourning.
SPECIAL_CLOSURES = {
    date(2018, 12, 5): "Day of mourning for President George H. W. Bush",
    date(2025, 1, 9): "Day of mourning for President Jimmy Carter",
}
MONDAY, THURSDAY = 0, 3


def _observed(day: date) -> date:
    """A holiday on a Saturday is observed on the Friday before, on a Sunday on the Monday after."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    following = date(year + month // 12, month % 12 + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Western Easter Sunday (the Meeus/Jones/Butcher algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    weekday = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * weekday) // 451
    month, day = divmod(h + weekday - 7 * m + 114, 31)
    return date(year, month, day + 1)


def holidays(year: int) -> dict[date, str]:
    """Full-day closures in `year`."""
    out: dict[date, str] = {}
    new_year = date(year, 1, 1)
    if new_year.weekday() == 6:
        out[new_year + timedelta(days=1)] = "New Year's Day (observed)"
    elif new_year.weekday() != 5:  # on a Saturday it is not observed: the Friday before trades
        out[new_year] = "New Year's Day"
    out[_nth_weekday(year, 1, MONDAY, 3)] = "Martin Luther King Jr. Day"
    out[_nth_weekday(year, 2, MONDAY, 3)] = "Washington's Birthday"
    out[_easter(year) - timedelta(days=2)] = "Good Friday"
    out[_last_weekday(year, 5, MONDAY)] = "Memorial Day"
    if year >= 2022:
        out[_observed(date(year, 6, 19))] = "Juneteenth"
    out[_observed(date(year, 7, 4))] = "Independence Day"
    out[_nth_weekday(year, 9, MONDAY, 1)] = "Labor Day"
    out[_nth_weekday(year, 11, THURSDAY, 4)] = "Thanksgiving Day"
    out[_observed(date(year, 12, 25))] = "Christmas Day"
    out.update({day: name for day, name in SPECIAL_CLOSURES.items() if day.year == year})
    return out


def early_closes(year: int) -> dict[date, str]:
    """1 p.m. closes in `year`: July 3 and Christmas Eve when they fall Monday to Thursday, and the
    day after Thanksgiving."""
    out: dict[date, str] = {}
    for day, name in ((date(year, 7, 3), "Independence Day eve"), (date(year, 12, 24), "Christmas Eve")):
        if day.weekday() <= THURSDAY:
            out[day] = name
    out[_nth_weekday(year, 11, THURSDAY, 4) + timedelta(days=1)] = "Day after Thanksgiving"
    return out


def fetch(start: date, end: date) -> dict[str, pd.DataFrame]:
    rows = []
    for year in range(start.year, end.year + 1):
        for day, name in holidays(year).items():
            rows.append({"time_utc": new_york(day, 9, 30), "event": "NYSE_CLOSED", "scope": SCOPE, "detail": name})
        for day, name in early_closes(year).items():
            rows.append({"time_utc": new_york(day, 13, 0), "event": "NYSE_EARLY_CLOSE", "scope": SCOPE, "detail": name})
    frame = event_frame([dict(row, source=SOURCE) for row in rows])
    days = frame["time_utc"].dt.tz_convert("America/New_York").dt.date
    return {SOURCE: frame[(days >= start) & (days <= end)].reset_index(drop=True)}
