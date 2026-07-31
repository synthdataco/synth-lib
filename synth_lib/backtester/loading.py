"""Synth API clients and local price loading: scores, rewards, pool USD, predictions, prices."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from synth_lib.backtester.caveats import _is_rate_limit_or_server_error, _warn_on_middle_gap
from synth_lib.backtester.config import (
    API_POOL_PAGE_SIZE_DAYS,
    API_SCORES_PAGE_SIZE_DAYS,
    MINER_DASHBOARD_API_BASE,
    UTC,
    _filter_time_range,
    _offline_root,
    competition_for,
    slug_for,
)
from synth_lib.preparation.config import SYNTHDATA_API_BASE
from synth_lib.preparation.minute_price_store import MinutePriceStore
from synth_lib.preparation.price_client import build_price_client


@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=(
        retry_if_result(_is_rate_limit_or_server_error)
        | retry_if_exception_type((requests.Timeout, requests.ConnectionError))
    ),
)
def _http_get(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
) -> requests.Response:
    """GET with exponential backoff on 429/5xx and transient network errors.

    Needed because the backtest dispatches up to ~11 concurrent threads against
    the Synth API, which rate-limits; long-running multi-day pulls also hit
    occasional ReadTimeout / ConnectionError. Returns the last response; caller
    calls raise_for_status() to handle non-retryable errors (e.g. 404).
    """
    return requests.get(url, params=params, timeout=timeout)


def get_miner_scores(
    start_time: datetime,
    end_time: datetime,
    asset: str,
    time_length: int,
    time_increment: int,
) -> pd.DataFrame:
    """
    GET https://api.synthdata.co/validation/scores/historical
    API has a 7-day max range, so we paginate in 7-day chunks.
    Returns DataFrame with columns: miner_uid, asset, crps, scored_time, time_length,
    time_increment, start_time (derived).
    """
    start_time = start_time.replace(microsecond=0)
    end_time = end_time.replace(microsecond=0)

    offline = _offline_root()
    if offline is not None:
        path = (
            offline
            / f"miner_scores_{asset}_{slug_for(competition_for(asset, time_length))}.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Offline mode: expected {path}. Pre-fetch the bundled data snapshot first."
            )
        df = pd.read_parquet(path)
        df = _filter_time_range(df, "scored_time", start_time, end_time)
        if df.empty:
            return df
        df["scored_time"] = pd.to_datetime(df["scored_time"], utc=True)
        df["time_increment"] = time_increment
        df["start_time"] = df["scored_time"] - pd.Timedelta(seconds=time_length)
        return df.reset_index(drop=True)

    chunks = []
    chunk_log: list[tuple[datetime, datetime, int]] = []
    cursor = start_time
    while cursor < end_time:
        chunk_end = min(cursor + timedelta(days=API_SCORES_PAGE_SIZE_DAYS), end_time)
        if cursor >= chunk_end:
            break
        resp = _http_get(
            f"{SYNTHDATA_API_BASE}/validation/scores/historical",
            params={
                "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "asset": asset,
                "time_length": time_length,
                "time_increment": time_increment,
            },
        )
        if resp.status_code == 404:
            # 404 = no scores in this range, not a fault. Same as build_offline_bundle.
            chunk_log.append((cursor, chunk_end, 0))
            cursor = chunk_end
            continue
        resp.raise_for_status()
        data = resp.json()
        chunk_log.append((cursor, chunk_end, len(data) if data else 0))
        if data:
            chunks.append(pd.DataFrame(data))
        cursor = chunk_end

    _warn_on_middle_gap(
        chunk_log, label=f"miner_scores asset={asset} time_length={time_length}"
    )

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if df.empty:
        return df
    df["scored_time"] = pd.to_datetime(df["scored_time"], utc=True)
    df = df.drop_duplicates(subset=["miner_uid", "scored_time", "asset", "time_length"])
    df["time_increment"] = time_increment
    # API does not return start_time. Approximate as scored_time - time_length.
    # The real start_time is a few minutes earlier (scoring delay), but this is
    # close enough — _find_prediction_file does closest-match on the filename.
    df["start_time"] = df["scored_time"] - pd.Timedelta(seconds=time_length)
    return df


def get_rewards_history(
    start_time: datetime,
    end_time: datetime,
    prompt_name: str | None = None,
) -> pd.DataFrame:
    """GET https://api.synthdata.co/rewards/scores.

    API ranges are paginated in API_SCORES_PAGE_SIZE_DAYS-day chunks.
    Returns DataFrame with columns: miner_uid, smoothed_score, reward_weight,
    prompt_name, updated_at.
    """
    start_time = start_time.replace(microsecond=0)
    end_time = end_time.replace(microsecond=0)

    offline = _offline_root()
    if offline is not None:
        suffix = prompt_name or "all"
        path = offline / f"rewards_history_{suffix}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Offline mode: expected {path}. Pre-fetch the bundled data snapshot first."
            )
        df = pd.read_parquet(path)
        df = _filter_time_range(df, "updated_at", start_time, end_time)
        if df.empty:
            return df
        df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
        return df.drop_duplicates().reset_index(drop=True)

    chunks = []
    chunk_log: list[tuple[datetime, datetime, int]] = []
    cursor = start_time
    while cursor < end_time:
        chunk_end = min(cursor + timedelta(days=API_SCORES_PAGE_SIZE_DAYS), end_time)
        if cursor >= chunk_end:
            break
        params: dict[str, Any] = {
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if prompt_name is not None:
            params["prompt_name"] = prompt_name
        resp = _http_get(
            f"{SYNTHDATA_API_BASE}/rewards/scores", params=params, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        chunk_log.append((cursor, chunk_end, len(data) if data else 0))
        if data:
            chunks.append(pd.DataFrame(data))
        cursor = chunk_end

    _warn_on_middle_gap(chunk_log, label=f"rewards_history prompt_name={prompt_name}")

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if df.empty:
        return df
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df = df.drop_duplicates()
    return df


def get_daily_miner_pool_usd(
    start_date: datetime,
    end_date: datetime,
) -> pd.Series:
    """Fetch daily subnet miner pool USD from /v1/miners/rewards/pool.

    The endpoint computes the pool server-side (daily-mean SUB-50 USD rate ×
    burn-adjusted alpha emission) over an inclusive [from, to] date range,
    paginated here in 300-day chunks (API_POOL_PAGE_SIZE_DAYS).

    Returns Series indexed by date (UTC midnight, tz-aware) → USD amount.
    """
    start_date = start_date.replace(microsecond=0)
    end_date = end_date.replace(microsecond=0)

    offline = _offline_root()
    if offline is not None:
        path = offline / "miner_pool_usd.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Offline mode: expected {path}. Pre-fetch the bundled data snapshot first."
            )
        df = pd.read_parquet(path)
        if df.empty:
            return pd.Series(dtype=float)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        start_ts = (
            pd.Timestamp(start_date)
            if start_date.tzinfo
            else pd.Timestamp(start_date, tz="UTC")
        )
        end_ts = (
            pd.Timestamp(end_date)
            if end_date.tzinfo
            else pd.Timestamp(end_date, tz="UTC")
        )
        mask = (df["date"] >= start_ts) & (df["date"] < end_ts)
        sub = df.loc[mask].sort_values("date")
        return pd.Series(
            sub["usd"].values,
            index=pd.DatetimeIndex(sub["date"]),
            dtype=float,
        ).sort_index()

    merged: dict[pd.Timestamp, float] = {}
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=API_POOL_PAGE_SIZE_DAYS), end_date)
        resp = _http_get(
            f"{MINER_DASHBOARD_API_BASE}/miners/rewards/pool",
            params={
                "from": cursor.strftime("%Y-%m-%d"),
                "to": chunk_end.strftime("%Y-%m-%d"),
            },
        )
        resp.raise_for_status()
        for row in resp.json()["rows"]:
            merged[pd.to_datetime(row["date"], utc=True)] = float(row["usd"])
        cursor = chunk_end

    if not merged:
        return pd.Series(dtype=float)
    return pd.Series(merged).sort_index()


def load_prediction(path: Path) -> dict:
    """

    Handles:
      - ArtifactManager format: {"simulation_input": {...}, "prediction": [meta, meta, path, ...]}
      - Notebook flat format:   {"start_timestamp": int, "paths": [...], ...}

    Always returns dict with keys: paths, num_simulations, num_steps, asset,
    time_increment, time_length.
    """
    raw = json.loads(path.read_text())
    if "simulation_input" in raw:
        sim = raw["simulation_input"]
        return {
            "start_timestamp": int(
                datetime.fromisoformat(sim["start_time"]).timestamp()
            ),
            "asset": sim["asset"],
            "time_increment": sim["time_increment"],
            "time_length": sim["time_length"],
            "num_simulations": sim["num_simulations"],
            "num_steps": sim["time_length"] // sim["time_increment"],
            "paths": raw["prediction"][2:],  # skip two metadata entries
        }
    return raw


def _price_store(asset: str) -> MinutePriceStore:
    """Return a MinutePriceStore configured for the asset's upstream provider.

    Provider routing (Binance / Hyperliquid / Pyth) mirrors the validator via
    build_price_client.
    """
    return MinutePriceStore(asset, client=build_price_client(asset))


def download_price_data(
    start_time: datetime,
    end_time: datetime,
    asset: str,
    freq: str = "1min",
) -> pd.DataFrame:
    """Returns DataFrame with DatetimeIndex (UTC) and a 'close' column at freq resolution.

    Uses MinutePriceStore for routing (Pyth vs Hyperliquid) and on-demand
    ingestion. Reads parquets directly for NaN-tolerance — futures assets like
    WTIOIL have legitimate market-closure gaps, and downstream scoring already
    filters NaN per-prompt.
    """
    store = _price_store(asset)
    frames: list[pd.DataFrame] = []
    cursor = start_time.date()
    missing: list = []
    while cursor <= end_time.date():
        path = store.day_path(cursor)
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            missing.append(cursor)
        cursor += timedelta(days=1)

    if missing:
        store.ingest_range(missing[0], missing[-1], verbose=False)
        for day in missing:
            frames.append(pd.read_parquet(store.day_path(day)))

    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = (
        frame.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .set_index("timestamp")
    )
    return frame[["close"]].loc[start_time:end_time].resample(freq).last()
