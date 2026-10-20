"""US macro release dates, from FRED's release calendar: each at 08:30 New York.

FRED lists every scheduled date of a release, past and published-ahead. It needs an API key
(`FRED_API_KEY`). Error messages never carry the request URL, which holds the key.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from synth_lib.preparation.config import venue_session
from synth_lib.preparation.event_calendar.events import ALL_ASSETS, event_frame, new_york

FRED_URL = "https://api.stlouisfed.org/fred"
KEY_ENV = "FRED_API_KEY"
# event -> (FRED release id, text its FRED name contains)
RELEASES = {
    "CPI": (10, "Consumer Price Index"),
    "NFP": (50, "Employment Situation"),
    "PPI": (46, "Producer Price Index"),
    "PCE": (54, "Personal Income and Outlays"),
    "RETAIL_SALES": (9, "Retail"),
    "GDP": (53, "Gross Domestic Product"),
}


def _get(session: requests.Session, path: str, params: dict) -> dict:
    try:
        response = session.get(f"{FRED_URL}/{path}", params={**params, "file_type": "json"}, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"FRED {path}: {type(exc).__name__}") from None
    if response.status_code != 200:
        raise RuntimeError(f"FRED {path} (release {params.get('release_id')}): HTTP {response.status_code}")
    return response.json()


def parse_release_dates(payload: dict) -> list[date]:
    return sorted({date.fromisoformat(row["date"]) for row in payload["release_dates"]})


def fetch(start: date, end: date, api_key: str, session: requests.Session | None = None) -> dict[str, pd.DataFrame]:
    """One source per release, `fred:<event>`, so that a change to one is judged on its own."""
    session = session or venue_session()
    out = {}
    for event, (release_id, expected) in RELEASES.items():
        params = {"release_id": release_id, "api_key": api_key}
        name = _get(session, "release", params)["releases"][0]["name"]
        if expected.lower() not in name.lower():
            raise ValueError(f"FRED release {release_id} is {name!r}, not {expected!r}")
        payload = _get(
            session,
            "release/dates",
            {
                **params,
                "realtime_start": start.isoformat(),
                "realtime_end": "9999-12-31",
                "include_release_dates_with_no_data": "true",
                "sort_order": "asc",
                "limit": 10000,
            },
        )
        source = f"fred:{event}"
        rows = [
            {"time_utc": new_york(day, 8, 30), "event": event, "scope": ALL_ASSETS, "detail": "", "source": source}
            for day in parse_release_dates(payload)
            if start <= day <= end
        ]
        out[source] = event_frame(rows)
    return out
