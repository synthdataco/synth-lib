"""Synth validator API: exact prompt start_times and the realized paths it scored."""

from __future__ import annotations

import bisect
import time
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from synth_lib.preparation.config import (
    PROMPT_START_MATCH_TOLERANCE_MINUTES,
    PROMPTS_PAGE_SIZE_DAYS,
    SYNTH_API_MAX_RETRIES,
    SYNTH_API_RETRY_STATUS_CODES,
    SYNTHDATA_API_BASE,
    utc_datetime,
)


def _synth_api_get(path: str, params: dict[str, Any], timeout: int = 30) -> requests.Response:
    """GET a Synth API endpoint, retrying 429/5xx with backoff.

    Callers handle other statuses; 404 means "no data" on these endpoints.
    """
    last_exc: Exception | None = None
    for attempt in range(SYNTH_API_MAX_RETRIES):
        try:
            response = requests.get(f"{SYNTHDATA_API_BASE}{path}", params=params, timeout=timeout)
            if response.status_code in SYNTH_API_RETRY_STATUS_CODES:
                raise requests.HTTPError(f"{response.status_code} from {path}", response=response)
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < SYNTH_API_MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Synth API failed after {SYNTH_API_MAX_RETRIES} attempts: {path}") from last_exc


def get_prompt_start_times(
    asset: str,
    time_length: int,
    time_increment: int,
    start_time: datetime,
    end_time: datetime,
) -> list[pd.Timestamp]:
    """GET /validation/prompts — the exact prompt start_times in a range.

    get_realized_path matches start_time exactly, so it needs these rather than
    the approximation derived from scored_time. Paginated; sorted and deduped.
    """
    start_time = utc_datetime(start_time)
    end_time = utc_datetime(end_time)
    starts: list[pd.Timestamp] = []
    cursor = start_time
    while cursor < end_time:
        chunk_end = min(cursor + timedelta(days=PROMPTS_PAGE_SIZE_DAYS), end_time)
        response = _synth_api_get(
            "/validation/prompts",
            params={
                "asset": asset,
                "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "time_length": time_length,
                "time_increment": time_increment,
            },
        )
        response.raise_for_status()
        payload = response.json() or {}
        starts.extend(pd.to_datetime(payload.get("start_times") or [], utc=True))
        cursor = chunk_end
    return sorted(set(starts))


def get_realized_path(
    asset: str,
    start_time: datetime,
    time_length: int,
    time_increment: int,
) -> pd.Series | None:
    """GET /validation/realized-path — the array the validator's CRPS consumed.

    `start_time` must be exact (see get_prompt_start_times); anything else 404s.
    Returns time_length // time_increment + 1 prices indexed by UTC timestamp,
    or None when unavailable. Venue gaps are stored as nulls and read as NaN.
    """
    start_time = utc_datetime(start_time)
    response = _synth_api_get(
        "/validation/realized-path",
        params={
            "asset": asset,
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_length": time_length,
            "time_increment": time_increment,
        },
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    prices = (response.json() or {}).get("real_prices")
    if not prices:
        return None
    index = pd.date_range(
        start=pd.Timestamp(start_time),
        periods=len(prices),
        freq=pd.Timedelta(seconds=time_increment),
    )
    return pd.Series(prices, index=index, dtype=float, name="close")


def snap_to_prompt_start(
    approx_start_time: datetime,
    prompt_start_times: list[pd.Timestamp],
    tolerance_minutes: int = PROMPT_START_MATCH_TOLERANCE_MINUTES,
) -> pd.Timestamp | None:
    """Map an approximate prompt start onto the exact one, or None if too far.

    `prompt_start_times` must be sorted. Prefers a candidate at or before the
    approximation: scored_time - time_length overshoots by the scoring delay.
    """
    if not prompt_start_times:
        return None
    target = pd.Timestamp(approx_start_time)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    position = bisect.bisect_right(prompt_start_times, target)
    if position > 0 and target - prompt_start_times[position - 1] <= tolerance:
        return prompt_start_times[position - 1]
    if position < len(prompt_start_times) and prompt_start_times[position] - target <= tolerance:
        return prompt_start_times[position]
    return None
