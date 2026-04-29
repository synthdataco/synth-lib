"""Backtest a miner against historical Synth subnet data.

Translates llm_miners_backtest_test.ipynb into a Python module.
External dependencies (DB, GCS, synth_subnet_analyses) are replaced with:
  - Synth public API (validation/scores and rewards/scores endpoints)
  - Local parquet price cache with Pyth API fallback
  - Local prediction artifacts from miner_outputs/{miner_name}/predictions/
  - synth.validator.crps_calculation for CRPS scoring
"""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from time import perf_counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import warnings
import seaborn as sns

from sqlalchemy import create_engine
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from synth.validator.crps_calculation import calculate_crps_for_miner
from synth.validator.miner_data_handler import MinerDataHandler
from synth.validator.moving_average import (
    compute_smoothed_score,
    prepare_df_for_moving_average,
)
from synth.validator.prompt_config import LOW_FREQUENCY, PromptConfig
from synth.validator.reward import compute_prompt_scores

from app.lib.preparation.market_data import (
    HYPERLIQUID_SYMBOLS,
    PYTH_SYMBOLS,
    HyperliquidClient,
    MinutePriceStore,
)

SYNTHDATA_API_BASE = "https://api.synthdata.co"
SCORING_INTERVALS: dict[str, int] = {
    "5min": 300,
    "30min": 1_800,
    "3hour": 10_800,
    "24hour_abs": 86_400,
}
DEFAULT_MINER_OUTPUT_ROOT = Path("miner_outputs")
DEFAULT_MARKET_DATA_ROOT = Path("market_data/pyth/BTC/1m")
_LEGACY_FALLBACK_WINDOW_DAYS = 10  # only used when no prompt_config is in scope
_COMBINED_EMPTY_COLS = [
    "updated_at",
    "miner_uid",
    "new_smoothed_score",
    "reward_weight",
]

UTC = timezone.utc

# -- Pagination limits (empirical — not confirmed from API docs) --
# Synth API /validation/scores/historical has a 7-day max range.
# We use 6-day chunks to stay safely within the limit.
API_SCORES_PAGE_SIZE_DAYS = 6

# -- Prediction file matching --
# Scoring delay: the real prediction start_time is a few minutes before
# scored_time - time_length. We allow up to 30 minutes of tolerance when
# matching prediction filenames to scored prompts.
PREDICTION_MATCH_TOLERANCE_MINUTES = 30


@dataclass
class BacktestResult:
    miner_name: str
    prompt_df: pd.DataFrame
    smoothed_scores: pd.DataFrame
    summary: dict[str, Any]


def _is_rate_limit_or_server_error(resp: requests.Response) -> bool:
    return resp.status_code == 429 or 500 <= resp.status_code < 600


def _warn_on_middle_gap(
    chunk_log: list[tuple[datetime, datetime, int]],
    label: str,
) -> None:
    """Warn when a paginated API fetch has an empty chunk between two non-empty ones.

    A 0-row chunk surrounded by data indicates a silent mid-range data gap (transient
    API issue or partial outage). Without this warning the gap propagates downstream
    into smoothed_scores and rank charts as a phantom dead zone.
    """
    if len(chunk_log) < 3:
        return
    first_nonempty = next((i for i, (_, _, n) in enumerate(chunk_log) if n > 0), None)
    last_nonempty = next(
        (i for i in range(len(chunk_log) - 1, -1, -1) if chunk_log[i][2] > 0), None
    )
    if (
        first_nonempty is None
        or last_nonempty is None
        or last_nonempty - first_nonempty < 2
    ):
        return
    bad = [
        (s, e) for s, e, n in chunk_log[first_nonempty + 1 : last_nonempty] if n == 0
    ]
    if not bad:
        return
    ranges = ", ".join(f"{s.date()}→{e.date()}" for s, e in bad)
    warnings.warn(
        f"{label}: {len(bad)} empty chunk(s) between non-empty chunks: {ranges}. "
        f"Likely a silent API gap; downstream smoothed_scores and rank charts will "
        f"show a phantom dead zone over this range.",
        UserWarning,
        stacklevel=2,
    )


def _trim_warmup(
    smoothed_scores: pd.DataFrame,
    scored: pd.DataFrame,
    warmup_days: int = _LEGACY_FALLBACK_WINDOW_DAYS,
    warmup_anchor: datetime | None = None,
) -> pd.DataFrame:
    """Drop smoothed_scores rows whose `updated_at` is in the first `warmup_days`
    after `warmup_anchor` (or after `scored["scored_time"].min()` if no anchor).

    During that window the moving average is dominated by synth's new-miner
    worst-score backfill, so ranks are not directly comparable.
    No-op if trimming would leave the frame empty (short-window backtests / tests).
    """
    if smoothed_scores.empty or scored.empty:
        return smoothed_scores
    anchor = (
        pd.Timestamp(warmup_anchor)
        if warmup_anchor is not None
        else pd.Timestamp(scored["scored_time"].min())
    )
    warmup_end = anchor + pd.Timedelta(days=warmup_days)
    trimmed = smoothed_scores.loc[smoothed_scores["updated_at"] >= warmup_end]
    if trimmed.empty:
        return smoothed_scores
    return trimmed.reset_index(drop=True)


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
    """
    GET https://api.synthdata.co/rewards/scores
    API has a 7-day max range, so we paginate in 7-day chunks.
    Returns DataFrame with columns: miner_uid, smoothed_score, reward_weight,
    prompt_name, updated_at.
    """
    start_time = start_time.replace(microsecond=0)
    end_time = end_time.replace(microsecond=0)
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
    """Fetch daily subnet miner pool USD from /rewards/historical.

    Paginated in 6-day chunks (same as API_SCORES_PAGE_SIZE_DAYS). The endpoint's
    asset/time_increment/time_length query params are required by the API but the
    response is subnet-wide (verified identical across asset choices).

    Returns Series indexed by date (UTC midnight, tz-aware) → USD amount.
    """
    start_date = start_date.replace(microsecond=0)
    end_date = end_date.replace(microsecond=0)

    merged: dict[pd.Timestamp, float] = {}
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=API_SCORES_PAGE_SIZE_DAYS), end_date)
        if cursor >= chunk_end:
            break
        resp = _http_get(
            f"{SYNTHDATA_API_BASE}/rewards/historical",
            params={
                "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "asset": "BTC",
                "time_increment": 300,
                "time_length": 86400,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        for row in data:
            date = pd.to_datetime(row["date"], utc=True).floor("D")
            usd = next(
                (x["amount"] for x in row["rewards"] if x["asset"] == "USD"),
                None,
            )
            if usd is not None:
                merged[date] = float(usd)
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
    """Return a MinutePriceStore configured for the asset's upstream (Pyth or Hyperliquid)."""
    if asset in HYPERLIQUID_SYMBOLS:
        return MinutePriceStore(asset, client=HyperliquidClient())
    if asset in PYTH_SYMBOLS:
        return MinutePriceStore(asset)
    raise ValueError(
        f"Unsupported asset: {asset}. "
        f"Known Pyth: {sorted(PYTH_SYMBOLS)}; Hyperliquid: {sorted(HYPERLIQUID_SYMBOLS)}."
    )


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


def _compute_prompt_scores_for_group(crps: pd.Series) -> pd.Series:
    """Wrapper around synth's compute_prompt_scores for use in groupby.apply."""
    result = compute_prompt_scores(crps.values)
    if result[0] is None:
        return pd.Series(0.0, index=crps.index)
    return pd.Series(result[0], index=crps.index)


def _compute_prompt_score_stats_for_group(crps: pd.Series) -> pd.DataFrame:
    """Like _compute_prompt_scores_for_group but also returns percentile90 and
    lowest_score so synth.prepare_df_for_moving_average can apply its worst-score
    backfill rule to new miners.
    """
    capped, p90, low = compute_prompt_scores(crps.values)
    n = len(crps)
    if capped is None:
        return pd.DataFrame(
            {
                "new_prompt_scores": [0.0] * n,
                "percentile90": [0.0] * n,
                "lowest_score": [0.0] * n,
            },
            index=crps.index,
        )
    return pd.DataFrame(
        {
            "new_prompt_scores": capped,
            "percentile90": [float(p90)] * n,
            "lowest_score": [float(low)] * n,
        },
        index=crps.index,
    )


class _BacktestMinerDataHandler(MinerDataHandler):
    """MinerDataHandler that maps miner_uid == miner_id without a real DB."""

    def __init__(self) -> None:
        # Pass a dummy SQLite engine to avoid connecting to PostgreSQL
        self.engine = create_engine("sqlite://")

    def populate_miner_uid_in_miner_data(self, miner_data: list[dict]) -> list[dict]:
        for row in miner_data:
            row["miner_uid"] = row["miner_id"]
        return miner_data


_BACKTEST_MDH = _BacktestMinerDataHandler()


def calculate_smoothed_scores(
    all_scores: pd.DataFrame,
    rewards_history: pd.DataFrame,
    cutoff_days: int = 10,
    scores_column: str = "new_prompt_scores",
    prompt_config: PromptConfig = LOW_FREQUENCY,
) -> pd.DataFrame:
    """Compute smoothed scores and reward weights using synth's compute_smoothed_score.

    Delegates to synth.validator.moving_average.compute_smoothed_score for each
    rewards_history timestamp, using a fake MinerDataHandler (miner_uid == miner_id).

    Returns DataFrame with columns: updated_at, miner_uid, new_smoothed_score, reward_weight.
    """
    # Adapt column names to what synth expects: miner_id, prompt_score_v3
    input_df = all_scores.copy()
    # Use miner_uid as miner_id for synth; drop any existing miner_id to avoid duplication
    if "miner_id" in input_df.columns:
        input_df = input_df.drop(columns=["miner_id"])
    input_df = input_df.rename(
        columns={"miner_uid": "miner_id", scores_column: "prompt_score_v3"}
    )
    input_df["scored_time"] = pd.to_datetime(input_df["scored_time"])

    # Prepare the df (backfill new miners, etc.)
    prepared = prepare_df_for_moving_average(input_df)

    result_rows = []
    for updated_at in rewards_history["updated_at"].sort_values().unique():
        cutoff = updated_at - pd.Timedelta(days=cutoff_days)
        window_df = prepared.loc[
            (prepared["scored_time"] >= cutoff)
            & (prepared["scored_time"] <= updated_at)
        ]
        rewards = compute_smoothed_score(
            _BACKTEST_MDH, window_df, updated_at, prompt_config
        )
        if rewards is None:
            continue

        for r in rewards:
            result_rows.append(
                {
                    "updated_at": pd.Timestamp(r["updated_at"]),
                    "miner_uid": r["miner_uid"],
                    "new_smoothed_score": r["smoothed_score"],
                    "reward_weight": r["reward_weight"],
                }
            )

    return pd.DataFrame(result_rows)


def compute_combined_smoothed_scores(
    results: list[BacktestResult],
    prompt_config: PromptConfig = LOW_FREQUENCY,
    cutoff_days: int | None = None,
    simulate_registration: datetime | None = None,
) -> pd.DataFrame:
    """Real-validator-equivalent cross-asset smoothed scores.

    Concatenates per-asset CRPS frames into one multi-asset frame, then calls
    synth's compute_smoothed_score once per rewards round (union of updated_at
    timestamps across all per-asset smoothed_scores). That function applies
    ASSET_COEFFICIENTS, per-miner coefficient-sum normalization, and a single
    softmax across all miners — matching the real validator rather than our
    previous un-weighted hand-rolled aggregation.

    Returns DataFrame with columns: updated_at, miner_uid, new_smoothed_score,
    reward_weight. reward_weight sums to prompt_config.smoothed_score_coefficient
    across miners per timestamp.
    """
    if not results:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)

    # Default the leaderboard window to the prompt's own setting (HF=3, LF=10).
    if cutoff_days is None:
        cutoff_days = prompt_config.window_days

    # Concat per-asset prompt_df frames. percentile90 and lowest_score must be
    # carried through so synth's prepare_df_for_moving_average can backfill new
    # miners (it silently skips backfill when those columns are absent).
    cols = [
        "scored_time",
        "miner_uid",
        "asset",
        "new_prompt_scores",
        "percentile90",
        "lowest_score",
    ]
    frames = []
    for r in results:
        if r.prompt_df.empty:
            continue
        df = r.prompt_df[[c for c in cols if c in r.prompt_df.columns]].copy()
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)
    combined_crps = pd.concat(frames, ignore_index=True)

    # Adapt column names to what synth expects: miner_id, prompt_score_v3
    if "miner_id" in combined_crps.columns:
        combined_crps = combined_crps.drop(columns=["miner_id"])
    combined_crps = combined_crps.rename(
        columns={"miner_uid": "miner_id", "new_prompt_scores": "prompt_score_v3"}
    )
    combined_crps["scored_time"] = pd.to_datetime(combined_crps["scored_time"])

    prepared = prepare_df_for_moving_average(combined_crps)

    # Union of rewards-round timestamps across all per-asset smoothed_scores
    timestamps: set[pd.Timestamp] = set()
    for r in results:
        if r.smoothed_scores.empty:
            continue
        for t in pd.to_datetime(r.smoothed_scores["updated_at"]).unique():
            timestamps.add(pd.Timestamp(t))

    result_rows = []
    for updated_at in sorted(timestamps):
        cutoff = updated_at - pd.Timedelta(days=cutoff_days)
        window_df = prepared.loc[
            (prepared["scored_time"] >= cutoff)
            & (prepared["scored_time"] <= updated_at)
        ]
        rewards = compute_smoothed_score(
            _BACKTEST_MDH, window_df, updated_at, prompt_config
        )
        if rewards is None:
            continue
        for row in rewards:
            result_rows.append(
                {
                    "updated_at": pd.Timestamp(row["updated_at"]),
                    "miner_uid": row["miner_uid"],
                    "new_smoothed_score": row["smoothed_score"],
                    "reward_weight": row["reward_weight"],
                }
            )

    if not result_rows:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)
    if simulate_registration is not None:
        # Simulating registration → show the backfilled onboarding period as-is.
        return pd.DataFrame(result_rows)
    return _trim_warmup(
        pd.DataFrame(result_rows),
        combined_crps,
        warmup_days=cutoff_days,
    )


def _parse_prediction_filename_time(path: Path) -> datetime | None:
    """Extract start_time from a prediction filename like 2026-03-23_00:01:00Z_BTC_86400.json."""
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    time_str = f"{parts[0]}_{parts[1]}"
    return datetime.strptime(time_str, "%Y-%m-%d_%H:%M:%SZ").replace(tzinfo=UTC)


def _find_prediction_file(
    prediction_files: list[Path],
    start_time: datetime,
    asset: str,
    time_length: int,
    tolerance_minutes: int = PREDICTION_MATCH_TOLERANCE_MINUTES,
) -> Path | None:
    """Find the closest prediction file matching the given start_time, asset, and time_length.

    start_time is approximate (derived from scored_time - time_length). The real
    prediction file start_time is a few minutes earlier due to scoring delay, so
    we match the closest file within tolerance_minutes.
    """
    suffix = f"_{asset}_{time_length}.json"
    candidates = [p for p in prediction_files if p.name.endswith(suffix)]

    best_path = None
    best_delta = timedelta(minutes=tolerance_minutes)

    for path in candidates:
        file_time = _parse_prediction_filename_time(path)
        if file_time is None:
            continue
        delta = abs(start_time - file_time)
        if delta < best_delta:
            best_delta = delta
            best_path = path

    return best_path


def _score_single_prompt(
    file_path: Path | None,
    start_time: Any,
    asset_val: str,
    scored_time: Any,
    time_len: int,
    time_incr: int,
    real_prices: list[float],
    scoring_intervals: dict[str, int],
    miner_id: int,
) -> dict:
    """Score a single prompt's prediction against real prices. Runs in a worker process.

    Returns a dict with scoring results, or a dict with crps=-1 for missing predictions.
    """
    if file_path is None:
        return {
            "miner_uid": miner_id,
            "scored_time": scored_time,
            "crps": -1,
            "asset": asset_val,
            "start_time": start_time,
            "time_increment": time_incr,
            "time_length": time_len,
            "miner_id": miner_id,
        }

    llm_predictions_raw = load_prediction(file_path)
    simulation_runs = np.asarray(llm_predictions_raw["paths"], dtype=float)
    real_price_array = np.asarray(real_prices, dtype=float)
    total_crps, _ = calculate_crps_for_miner(
        simulation_runs, real_price_array, time_incr, scoring_intervals
    )

    return {
        "miner_uid": miner_id,
        "scored_time": scored_time,
        "crps": float(total_crps),
        "asset": asset_val,
        "start_time": start_time,
        "time_increment": time_incr,
        "time_length": time_len,
        "miner_id": miner_id,
    }


def backtest(
    miner_name: str = "initial_research",
    asset: str = "BTC",
    *,
    time_length: int,
    time_increment: int = None,
    n_backtest_days: int = 15,
    miner_id: int = 999,
    predictions_dir: Path | None = None,
    scoring_intervals: dict[str, int] | None = None,
    prompt_config: PromptConfig | None = None,
    scoring_executor: ProcessPoolExecutor | None = None,
    eval_end: datetime | None = None,
    simulate_registration: datetime | None = None,
    simulate_deregistration: datetime | None = None,
) -> BacktestResult:
    """Backtest a local miner against Synth subnet scoring data.
    Loads predictions from predictions_dir (or miner_outputs/{miner_name}/predictions/).
    """
    if scoring_intervals is None:
        scoring_intervals = SCORING_INTERVALS
    if prompt_config is None:
        prompt_config = LOW_FREQUENCY

    # Discover prediction files and their date range
    predictions_root = predictions_dir or (
        DEFAULT_MINER_OUTPUT_ROOT / miner_name / "predictions"
    )
    prediction_files = sorted(
        p for p in predictions_root.glob("**/*.json") if not p.name.startswith("_")
    )
    pred_times = [_parse_prediction_filename_time(p) for p in prediction_files]
    pred_times = [t for t in pred_times if t is not None]
    if not pred_times:
        raise FileNotFoundError(
            f"No valid prediction files found in {predictions_root}. "
            "Expected filenames like 2026-03-28_00:00:00Z_BTC_86400.json"
        )
    pred_start = min(pred_times)
    pred_end = max(pred_times)

    # Constrain backtest window to prediction coverage. The buffer before pred_start matches the
    # file-matching tolerance, so every returned prompt can resolve to a prediction file.
    # Anchor priority: simulate_deregistration > eval_end > now. When the user passes
    # simulate_registration, the window's lower bound shifts to it (minus 10d for the
    # smoothed-score moving-average lookback) instead of `anchor − n_backtest_days`,
    # so the API actually fetches scored prompts in the simulated mining period.
    if simulate_deregistration is not None:
        anchor = pd.Timestamp(simulate_deregistration).to_pydatetime()
    elif eval_end is not None:
        anchor = eval_end
    else:
        anchor = datetime.now(UTC)
    if simulate_registration is not None:
        window_start = pd.Timestamp(simulate_registration).to_pydatetime() - timedelta(
            days=prompt_config.window_days
        )
    else:
        window_start = anchor - timedelta(days=n_backtest_days)
    query_start = max(
        pred_start - timedelta(minutes=PREDICTION_MATCH_TOLERANCE_MINUTES),
        window_start,
    )
    query_end = pred_end + timedelta(seconds=time_length) + timedelta(hours=1)

    # Step 2: fetch existing miner scores
    asset_backtest = asset
    time_length_backtest = time_length
    prompt_label = "low" if time_length_backtest == 86400 else "high"
    log_prefix = f"  [{prompt_label}/{asset_backtest}]"
    t_start = perf_counter()

    print(
        f"{log_prefix} fetching scored prompts ({query_start.date()} → {min(query_end, anchor).date()})...",
        flush=True,
    )
    asset_scores = get_miner_scores(
        start_time=query_start,
        end_time=min(query_end, anchor),
        asset=asset_backtest,
        time_length=time_length_backtest,
        time_increment=time_increment,
    )
    if asset_scores.empty:
        raise RuntimeError(
            f"Synth API returned no miner scores for asset={asset_backtest}, "
            f"time_length={time_length_backtest}, range=[{query_start}, {query_end}]"
        )
    print(
        f"{log_prefix} {len(asset_scores)} score rows ({asset_scores['scored_time'].nunique()} prompts) "
        f"in {perf_counter()-t_start:.1f}s",
        flush=True,
    )

    # Step 3: fetch rewards history
    prompt_name = None if not time_length_backtest else prompt_label
    t_rh = perf_counter()
    print(
        f"{log_prefix} fetching rewards history (prompt_name={prompt_name})...",
        flush=True,
    )
    rewards_history = get_rewards_history(
        asset_scores["scored_time"].min() - pd.Timedelta(hours=24),
        asset_scores["scored_time"].max() + pd.Timedelta(hours=24),
        prompt_name=prompt_name,
    )
    if rewards_history.empty:
        raise RuntimeError(
            f"Synth API returned no rewards history for prompt_name={prompt_name}, "
            f"range=[{asset_scores['scored_time'].min()}, {asset_scores['scored_time'].max()}]"
        )
    print(
        f"{log_prefix} {len(rewards_history)} rewards rows "
        f"({rewards_history['updated_at'].nunique()} update rounds) in {perf_counter()-t_rh:.1f}s",
        flush=True,
    )

    # Step 4: align scores and rewards history time ranges
    if asset_backtest is not None:
        asset_scores = asset_scores.sort_values("scored_time")
        rewards_history = rewards_history.sort_values("updated_at")

        result = pd.merge_asof(
            asset_scores[["scored_time"]].drop_duplicates(),
            rewards_history[["updated_at"]].drop_duplicates(),
            left_on="scored_time",
            right_on="updated_at",
            direction="forward",
        )
        result = result[~result["updated_at"].isna()]
        rewards_history = rewards_history.loc[
            rewards_history["updated_at"].isin(result["updated_at"])
        ].copy()
        asset_scores = asset_scores.loc[
            asset_scores["scored_time"].isin(result["scored_time"])
        ].copy()
    else:
        if rewards_history["updated_at"].max() < asset_scores["scored_time"].max():
            max_scored_time = asset_scores.loc[
                asset_scores["scored_time"] <= rewards_history["updated_at"].max(),
                "scored_time",
            ].max()
            asset_scores = asset_scores.loc[
                asset_scores["scored_time"] <= max_scored_time
            ].copy()
        elif rewards_history["updated_at"].max() > asset_scores["scored_time"].max():
            max_upd_at = rewards_history.loc[
                rewards_history["updated_at"] >= asset_scores["scored_time"].max(),
                "updated_at",
            ].min()
            rewards_history = rewards_history.loc[
                rewards_history["updated_at"] <= max_upd_at
            ].copy()

    # Step 5: download asset price data
    asset_prices: dict[str, pd.DataFrame] = {}
    for asset_ in asset_scores["asset"].unique():
        st_ = asset_scores["start_time"].min() - pd.Timedelta(hours=2)
        en_ = asset_scores["start_time"].max() + pd.Timedelta(hours=26)
        asset_prices[asset_] = download_price_data(st_, en_, asset_, freq="1min")

    # Step 6: score LLM miner predictions
    start_times = asset_scores["start_time"].sort_values().unique()
    t_score = perf_counter()
    print(
        f"{log_prefix} scoring {len(start_times)} prompts against local predictions...",
        flush=True,
    )

    # Pre-build work items: resolve file paths and slice prices upfront
    work_items = []

    for start_time in start_times:
        scores_subset = asset_scores.loc[
            asset_scores["start_time"] == start_time
        ].copy()

        for _, asset_val, scored_time, time_len, time_incr in (
            scores_subset[["asset", "scored_time", "time_length", "time_increment"]]
            .drop_duplicates()
            .itertuples()
        ):
            file_path = _find_prediction_file(
                prediction_files, start_time, asset_val, time_len
            )

            # Pre-slice real prices in the main process (avoids pickling DataFrames)
            real_prices: list[float] = []
            if file_path is not None:
                step_minutes = time_incr // 60
                real_prices = (
                    asset_prices[asset_val]
                    .loc[
                        start_time : start_time
                        + pd.Timedelta(seconds=time_len) : step_minutes
                    ]
                    .iloc[:, 0]
                    .tolist()
                )
                expected_steps = (time_len // time_incr) + 1
                if len(real_prices) != expected_steps:
                    raise ValueError(
                        f"Price data length mismatch for {asset_val} at {start_time}: "
                        f"got {len(real_prices)} prices, expected {expected_steps}"
                    )

            work_items.append(
                (
                    file_path,
                    start_time,
                    asset_val,
                    scored_time,
                    time_len,
                    time_incr,
                    real_prices,
                    scoring_intervals,
                    miner_id,
                )
            )

    # Dispatch scoring — parallel if executor provided, else sequential
    if scoring_executor is not None:
        futures = [
            scoring_executor.submit(_score_single_prompt, *item) for item in work_items
        ]
        raw_results = [f.result() for f in futures]
    else:
        raw_results = [_score_single_prompt(*item) for item in work_items]

    scores_extensions = [pd.DataFrame(r, index=[0]) for r in raw_results]

    # Step 7: concatenate LLM miner scores
    if not scores_extensions:
        raise RuntimeError(
            "No prediction files matched any scored prompt. "
            "Check that prediction filenames align with the scored time range."
        )
    new_scores = pd.concat(scores_extensions, ignore_index=True)
    print(
        f"{log_prefix} scored {len(new_scores)} prompts in {perf_counter()-t_score:.1f}s",
        flush=True,
    )

    # Step 8: trim to valid range
    new_scores = new_scores.sort_values("scored_time")
    valid_scores = new_scores.loc[new_scores["crps"] != -1]
    if valid_scores.empty:
        raise RuntimeError(
            "All matched predictions resulted in crps=-1 (scoring failures). "
            "No valid backtest results to report."
        )
    end_backtest_time = valid_scores["scored_time"].max()
    new_scores = new_scores.loc[new_scores["scored_time"] <= end_backtest_time]
    asset_scores = asset_scores.loc[asset_scores["scored_time"] <= end_backtest_time]

    # Optional: simulate the miner registering on a specific date by dropping our
    # CRPS rows before that date. synth.validator.prepare_df_for_moving_average
    # then sees our miner's first scored_time > global_min and applies the same
    # worst-score backfill it gives real late-joining miners — making rank
    # comparisons symmetric instead of giving us an unfair "from-the-start" boost.
    if simulate_registration is not None:
        sim_reg = pd.Timestamp(simulate_registration)
        new_scores = new_scores.loc[new_scores["scored_time"] >= sim_reg]
        if new_scores.empty:
            raise RuntimeError(
                f"simulate_registration={sim_reg} drops all of our miner's CRPS rows; "
                f"no valid backtest results."
            )
    if simulate_deregistration is not None:
        sim_dereg = pd.Timestamp(simulate_deregistration)
        new_scores = new_scores.loc[new_scores["scored_time"] <= sim_dereg]
        if new_scores.empty:
            raise RuntimeError(
                f"simulate_deregistration={sim_dereg} drops all of our miner's CRPS rows; "
                f"no valid backtest results."
            )

    # Step 9: merge LLM scores with all miner scores
    all_scores = pd.concat([asset_scores, new_scores], ignore_index=True)
    all_scores = all_scores.sort_values(["scored_time", "miner_uid"])

    if rewards_history["updated_at"].max() < all_scores["scored_time"].max():
        max_scored_time = all_scores.loc[
            all_scores["scored_time"] <= rewards_history["updated_at"].max(),
            "scored_time",
        ].max()
        all_scores = all_scores.loc[all_scores["scored_time"] <= max_scored_time].copy()
    elif rewards_history["updated_at"].max() > all_scores["scored_time"].max():
        max_upd_at = rewards_history.loc[
            rewards_history["updated_at"] >= all_scores["scored_time"].max(),
            "updated_at",
        ].min()
        rewards_history = rewards_history.loc[
            rewards_history["updated_at"] <= max_upd_at
        ].copy()

    # Step 12: apply prompt score calculation across all scores. We also persist
    # percentile90 and lowest_score per group because prepare_df_for_moving_average
    # uses them to compute the worst-score backfill for new miners (silently
    # skips backfill when those columns are absent).
    _stats = all_scores.groupby(
        ["scored_time", "asset", "time_length", "time_increment"], group_keys=False
    )["crps"].apply(_compute_prompt_score_stats_for_group)
    all_scores["new_prompt_scores"] = _stats["new_prompt_scores"]
    all_scores["percentile90"] = _stats["percentile90"]
    all_scores["lowest_score"] = _stats["lowest_score"]

    # Step 13: build rewards history entries for LLM miner
    new_miner_rewards_list = []
    for prompt_name_item in rewards_history["prompt_name"].unique():
        new_miner_rewards_prompt = pd.DataFrame(
            {
                "updated_at": rewards_history.loc[
                    rewards_history["prompt_name"] == prompt_name_item, "updated_at"
                ]
                .sort_values()
                .unique(),
                "miner_uid": miner_id,
                "prompt_name": prompt_name_item,
            }
        )
        new_miner_rewards_list.append(new_miner_rewards_prompt)
    new_miner_rewards = pd.concat(
        new_miner_rewards_list, ignore_index=True
    ).sort_values("updated_at")

    # Step 14: merge LLM miner rewards into full rewards history
    rewards_history = pd.concat(
        [
            rewards_history,
            new_miner_rewards,
        ]
    ).sort_values(["updated_at", "prompt_name", "miner_uid"])

    # Step 15: recalculate smoothed scores including LLM miner
    n_rounds = rewards_history["updated_at"].nunique()
    t_smooth = perf_counter()
    print(
        f"{log_prefix} computing smoothed scores over {n_rounds} reward rounds...",
        flush=True,
    )
    recalculated_smoothed_scores = calculate_smoothed_scores(
        all_scores,
        rewards_history,
        cutoff_days=prompt_config.window_days,
        scores_column="new_prompt_scores",
        prompt_config=prompt_config,
    )
    print(
        f"{log_prefix} smoothed scores done in {perf_counter()-t_smooth:.1f}s "
        f"(total backtest: {perf_counter()-t_start:.1f}s)",
        flush=True,
    )

    # Trim the first `prompt_config.window_days` of smoothed_scores when no
    # simulate_registration is set. In that case the backfill artifact applies to
    # real late-joining miners and inflates our rank — we want to hide it.
    # When simulate_registration IS set, the backfill applies to OUR miner and
    # the warmup ramp-up is the whole point of the simulation — don't trim it.
    if simulate_registration is None:
        recalculated_smoothed_scores = _trim_warmup(
            recalculated_smoothed_scores,
            all_scores,
            warmup_days=prompt_config.window_days,
        )

    miner_smoothed = recalculated_smoothed_scores.loc[
        recalculated_smoothed_scores["miner_uid"] == miner_id, "new_smoothed_score"
    ]
    summary: dict[str, Any] = {
        "miner_name": miner_name,
        "miner_id": miner_id,
        "num_prompts": int((new_scores["crps"] != -1).sum()),
        "mean_crps": float(new_scores.loc[new_scores["crps"] != -1, "crps"].mean()),
        "final_smoothed_score": (
            float(miner_smoothed.iloc[-1]) if not miner_smoothed.empty else None
        ),
    }

    return BacktestResult(
        miner_name=miner_name,
        prompt_df=all_scores,
        smoothed_scores=recalculated_smoothed_scores,
        summary=summary,
    )


def plot_rank_evolution(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot the miner's rank over time based on reward_weight and save the chart.

    Rank 1 = highest reward_weight at that timestamp.
    Returns the path to the saved PNG.
    """
    miner_id = result.summary["miner_id"]
    df = result.smoothed_scores.copy()
    df["updated_at"] = pd.to_datetime(df["updated_at"])

    # Derive asset and time_length from the prompt data for labelling
    assets = result.prompt_df["asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    time_lengths = result.prompt_df["time_length"].unique()
    tl_label = str(int(time_lengths[0])) if len(time_lengths) == 1 else "mixed"

    # Compute rank per timestamp (highest reward_weight = rank 1)
    df["rank"] = (
        df.groupby("updated_at")["reward_weight"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    miner_df = df.loc[df["miner_uid"] == miner_id].sort_values("updated_at")

    if miner_df.empty:
        raise RuntimeError(f"Miner {miner_id} not found in smoothed scores.")

    total_miners = df.groupby("updated_at")["miner_uid"].nunique()

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    sns.lineplot(
        data=miner_df,
        x="updated_at",
        y="rank",
        marker="o",
        markersize=5,
        linewidth=2,
        ax=ax,
    )

    # Shade the total miners range
    ax.fill_between(
        total_miners.index,
        1,
        total_miners.values,
        alpha=0.08,
        color="grey",
        label="Total miners",
    )

    ax.invert_yaxis()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(f"Rank evolution — {result.miner_name} — {asset_label} ({tl_label}s)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    # Save
    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"rank_evolution_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return chart_path


def plot_total_rank_evolution(
    results: list[BacktestResult],
    combined: pd.DataFrame,
    profile_label: str,
    output_dir: Path | None = None,
) -> Path:
    """Plot aggregate rank evolution across all assets.

    Ranks miners per `updated_at` timestamp by `reward_weight` (descending, 1 = best).
    Expects `combined` as the output of `compute_combined_smoothed_scores` — a single
    real-validator-equivalent smoothed-score DataFrame covering all assets.
    """
    if not results:
        raise RuntimeError("No backtest results to aggregate.")
    if combined.empty:
        raise RuntimeError(
            "No combined smoothed scores available for total rank chart."
        )

    miner_name = results[0].miner_name
    miner_id = results[0].summary["miner_id"]

    df = combined.copy()
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df["rank"] = (
        df.groupby("updated_at")["reward_weight"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    miner_df = df.loc[df["miner_uid"] == miner_id].sort_values("updated_at")
    if miner_df.empty:
        raise RuntimeError(f"Miner {miner_id} not found in combined smoothed scores.")

    total_miners = df.groupby("updated_at")["miner_uid"].nunique()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=miner_df,
        x="updated_at",
        y="rank",
        marker="o",
        markersize=5,
        linewidth=2,
        ax=ax,
    )
    ax.fill_between(
        total_miners.index,
        1,
        total_miners.values,
        alpha=0.08,
        color="grey",
        label="Total miners",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(f"Total rank evolution — {miner_name} — {profile_label} (all assets)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"rank_evolution_TOTAL_{profile_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    # Print final total rank
    last_ts = miner_df["updated_at"].max()
    last_slice = df.loc[df["updated_at"] == last_ts]
    miner_row = last_slice.loc[last_slice["miner_uid"] == miner_id]
    if not miner_row.empty:
        rank = int(miner_row["rank"].iloc[0])
        n = len(last_slice)
        rw = float(miner_row["reward_weight"].iloc[0])
        print(f"\n  TOTAL [{profile_label}] rank: {rank}/{n}  reward_weight: {rw:.6f}")

    return chart_path


def plot_grand_total_rank_evolution(
    results_by_profile: dict[str, list[BacktestResult]],
    combined_by_profile: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> Path:
    """Rank evolution across ALL assets from ALL profiles.

    Merges per-profile `combined` frames via `_compute_grand_total_weights`
    (forward-fill each profile onto the union of timestamps, then sum per
    miner). Ranks miners descending by grand-total reward_weight at each
    timestamp.
    """
    grand = _compute_grand_total_weights(combined_by_profile)
    if grand.empty:
        raise RuntimeError("No grand-total rows available for rank chart.")

    # miner_name / miner_id from any profile's first result
    first_result = next(
        (rs[0] for rs in results_by_profile.values() if rs),
        None,
    )
    if first_result is None:
        raise RuntimeError("No backtest results to aggregate.")
    miner_name = first_result.miner_name
    miner_id = first_result.summary["miner_id"]

    df = grand.copy()
    df["rank"] = (
        df.groupby("updated_at")["reward_weight"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    miner_df = df.loc[df["miner_uid"] == miner_id].sort_values("updated_at")
    if miner_df.empty:
        raise RuntimeError(f"Miner {miner_id} not found in grand-total weights.")

    total_miners = df.groupby("updated_at")["miner_uid"].nunique()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=miner_df,
        x="updated_at",
        y="rank",
        marker="o",
        markersize=5,
        linewidth=2,
        ax=ax,
    )
    ax.fill_between(
        total_miners.index,
        1,
        total_miners.values,
        alpha=0.08,
        color="grey",
        label="Total miners",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(
        f"Grand-total rank evolution — {miner_name} — all profiles, all assets"
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "rank_evolution_GRAND_TOTAL.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    # Print final grand-total rank
    last_ts = miner_df["updated_at"].max()
    last_slice = df.loc[df["updated_at"] == last_ts]
    miner_row = last_slice.loc[last_slice["miner_uid"] == miner_id]
    if not miner_row.empty:
        rank = int(miner_row["rank"].iloc[0])
        n = len(last_slice)
        rw = float(miner_row["reward_weight"].iloc[0])
        print(f"\n  GRAND TOTAL rank: {rank}/{n}  reward_weight: {rw:.6f}")

    return chart_path


def _compute_relative_crps(result: BacktestResult) -> pd.DataFrame:
    """Compute per-prompt CRPS ratio and percentile rank vs other miners.

    Returns DataFrame with one row per scored prompt: scored_time, our_crps,
    others_median, crps_ratio, percentile_rank, hour, day_of_week, iso_week.
    """
    miner_id = result.summary["miner_id"]
    df = result.prompt_df.copy()
    df["scored_time"] = pd.to_datetime(df["scored_time"])
    df = df.loc[df["crps"] != -1]
    if df.empty:
        raise RuntimeError("No valid CRPS data for relative analysis.")

    # Median CRPS of other miners at each scored_time
    others = df.loc[df["miner_uid"] != miner_id]
    others_median = (
        others.groupby("scored_time")["crps"].median().rename("others_median")
    )

    # Our miner's CRPS
    ours = df.loc[df["miner_uid"] == miner_id, ["scored_time", "crps"]].rename(
        columns={"crps": "our_crps"}
    )
    if ours.empty:
        raise RuntimeError(f"Miner {miner_id} has no valid CRPS data.")

    merged = ours.merge(others_median, on="scored_time", how="inner")
    merged["crps_ratio"] = merged["our_crps"] / merged["others_median"]

    # Percentile rank: what fraction of miners we beat at each scored_time
    # Lower CRPS = better, so rank ascending. percentile = (rank-1)/(n-1)*100
    ranks = []
    for st in merged["scored_time"]:
        all_at_st = df.loc[df["scored_time"] == st, "crps"]
        our_val = merged.loc[merged["scored_time"] == st, "our_crps"].iloc[0]
        n = len(all_at_st)
        rank = (all_at_st < our_val).sum() + 1  # 1-based rank, ascending
        pct = (rank - 1) / (n - 1) * 100 if n > 1 else 0.0
        ranks.append(pct)
    merged["percentile_rank"] = ranks

    # Time-derived columns
    merged["hour"] = merged["scored_time"].dt.hour
    merged["day_of_week"] = merged["scored_time"].dt.dayofweek
    merged["iso_week"] = merged["scored_time"].dt.isocalendar().week.astype(int)

    return merged


def plot_crps_over_time(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot our miner's CRPS over time vs the distribution of other miners.

    Shows: our miner as a line, others as median + shaded IQR and 10th-90th bands.
    Returns the path to the saved PNG.
    """
    miner_id = result.summary["miner_id"]
    df = result.prompt_df.copy()
    df["scored_time"] = pd.to_datetime(df["scored_time"])

    # Filter out missing predictions
    df = df.loc[df["crps"] != -1]
    if df.empty:
        raise RuntimeError("No valid CRPS data to plot.")

    # Derive labels
    assets = df["asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    time_lengths = df["time_length"].unique()
    tl_label = str(int(time_lengths[0])) if len(time_lengths) == 1 else "mixed"

    # Split our miner vs others
    miner_df = df.loc[df["miner_uid"] == miner_id].sort_values("scored_time")
    others_df = df.loc[df["miner_uid"] != miner_id]

    if miner_df.empty:
        raise RuntimeError(f"Miner {miner_id} not found in CRPS data.")

    # Compute percentile stats for other miners at each scored_time
    others_stats = (
        others_df.groupby("scored_time")["crps"]
        .agg(
            median="median",
            p10=lambda x: np.percentile(x, 10),
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
            p90=lambda x: np.percentile(x, 90),
        )
        .sort_index()
    )

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    # 10th-90th band (light)
    ax.fill_between(
        others_stats.index,
        others_stats["p10"],
        others_stats["p90"],
        alpha=0.08,
        color="steelblue",
        label="Range (10th\u201390th)",
    )
    # IQR band (darker)
    ax.fill_between(
        others_stats.index,
        others_stats["p25"],
        others_stats["p75"],
        alpha=0.2,
        color="steelblue",
        label="IQR (25th\u201375th)",
    )
    # Median line
    ax.plot(
        others_stats.index,
        others_stats["median"],
        linestyle="--",
        color="grey",
        linewidth=1.5,
        label="Median (others)",
    )
    # Our miner
    ax.plot(
        miner_df["scored_time"].values,
        miner_df["crps"].values,
        marker="o",
        markersize=5,
        linewidth=2,
        label=result.miner_name,
    )

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("CRPS")
    ax.set_title(
        f"CRPS over time \u2014 {result.miner_name} \u2014 {asset_label} ({tl_label}s)"
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    # Save
    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"crps_over_time_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return chart_path


def plot_crps_by_hour(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot mean/median CRPS ratio by hour of day. Returns path to saved PNG."""
    rel = _compute_relative_crps(result)

    assets = result.prompt_df.loc[result.prompt_df["crps"] != -1, "asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    tls = result.prompt_df.loc[result.prompt_df["crps"] != -1, "time_length"].unique()
    tl_label = str(int(tls[0])) if len(tls) == 1 else "mixed"

    hourly = (
        rel.groupby("hour")["crps_ratio"].agg(["mean", "median"]).reindex(range(24))
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hourly.index, hourly["mean"], color="salmon", label="Mean", zorder=2)
    ax.plot(
        hourly.index,
        hourly["median"],
        marker="o",
        markersize=5,
        linewidth=2,
        color="steelblue",
        label="Median",
        zorder=3,
    )
    ax.axhline(
        y=1.0, linestyle="--", color="grey", linewidth=1, label="Even with median miner"
    )
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("CRPS Ratio (ours / others median)")
    ax.set_title(
        f"Performance by Hour of Day \u2014 {result.miner_name} \u2014 {asset_label} ({tl_label}s)"
    )
    ax.set_xticks(range(24))
    ax.legend()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"crps_by_hour_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plot_crps_by_day(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot mean/median CRPS ratio by day of week. Returns path to saved PNG."""
    rel = _compute_relative_crps(result)

    assets = result.prompt_df.loc[result.prompt_df["crps"] != -1, "asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    tls = result.prompt_df.loc[result.prompt_df["crps"] != -1, "time_length"].unique()
    tl_label = str(int(tls[0])) if len(tls) == 1 else "mixed"

    daily = (
        rel.groupby("day_of_week")["crps_ratio"]
        .agg(["mean", "median"])
        .reindex(range(7))
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(7), daily["mean"], color="salmon", label="Mean", zorder=2)
    ax.plot(
        range(7),
        daily["median"],
        marker="o",
        markersize=5,
        linewidth=2,
        color="steelblue",
        label="Median",
        zorder=3,
    )
    ax.axhline(
        y=1.0, linestyle="--", color="grey", linewidth=1, label="Even with median miner"
    )
    ax.set_xlabel("Day of Week (UTC)")
    ax.set_ylabel("CRPS Ratio (ours / others median)")
    ax.set_title(
        f"Performance by Day of Week \u2014 {result.miner_name} \u2014 {asset_label} ({tl_label}s)"
    )
    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_LABELS)
    ax.legend()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"crps_by_day_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def plot_crps_ratio_distribution(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot histogram of CRPS ratio distribution. Returns path to saved PNG."""
    rel = _compute_relative_crps(result)

    assets = result.prompt_df.loc[result.prompt_df["crps"] != -1, "asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    tls = result.prompt_df.loc[result.prompt_df["crps"] != -1, "time_length"].unique()
    tl_label = str(int(tls[0])) if len(tls) == 1 else "mixed"

    our_median = float(rel["crps_ratio"].median())

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(rel["crps_ratio"], bins=50, color="salmon", edgecolor="white", zorder=2)
    ax.axvline(
        x=1.0,
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Even with median miner",
    )
    ax.axvline(
        x=our_median,
        linestyle="--",
        color="steelblue",
        linewidth=1.5,
        label=f"Our median: {our_median:.2f}",
    )
    ax.set_xlabel("CRPS Ratio (ours / others median)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Distribution of Relative Performance \u2014 {result.miner_name} \u2014 {asset_label} ({tl_label}s)"
    )
    ax.legend()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"crps_ratio_dist_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def plot_weekly_percentile(
    result: BacktestResult,
    output_dir: Path | None = None,
) -> Path:
    """Plot mean percentile rank by ISO week. Returns path to saved PNG."""
    rel = _compute_relative_crps(result)

    assets = result.prompt_df.loc[result.prompt_df["crps"] != -1, "asset"].unique()
    asset_label = assets[0] if len(assets) == 1 else "_".join(sorted(assets))
    tls = result.prompt_df.loc[result.prompt_df["crps"] != -1, "time_length"].unique()
    tl_label = str(int(tls[0])) if len(tls) == 1 else "mixed"

    weekly = rel.groupby("iso_week")["percentile_rank"].mean()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(weekly)), weekly.values, color="salmon", zorder=2)
    ax.axhline(y=50, linestyle="--", color="grey", linewidth=1, label="50th pctile")
    ax.set_xlabel("ISO Week")
    ax.set_ylabel("Mean Percentile Rank")
    ax.set_title(
        f"Weekly Percentile Trend \u2014 {result.miner_name} \u2014 {asset_label} ({tl_label}s)"
    )
    ax.set_xticks(range(len(weekly)))
    ax.set_xticklabels([str(w) for w in weekly.index])
    ax.legend()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / result.miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"weekly_percentile_{asset_label}_{tl_label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def _compute_grand_total_weights(
    combined_by_profile: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge per-profile combined frames into one grand-total reward_weight frame.

    At each union timestamp, forward-fill each profile's per-miner reward_weight
    from its most recent round and sum across profiles. Drops timestamps before
    all profiles have produced their first round (otherwise the grand-total
    weights wouldn't sum to 1.0 across miners).

    Returns DataFrame with columns: updated_at, miner_uid, reward_weight,
    new_smoothed_score. Empty if any profile frame is empty or the input dict is
    empty (single-profile input also returns empty, since grand-total requires
    all profiles active).
    """
    if not combined_by_profile or any(df.empty for df in combined_by_profile.values()):
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)
    if len(combined_by_profile) < 2:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)

    # Union of timestamps and miners across profiles
    all_miners: set[int] = set()
    all_timestamps: set[pd.Timestamp] = set()
    profile_starts: dict[str, pd.Timestamp] = {}
    for label, df in combined_by_profile.items():
        df_ts = pd.to_datetime(df["updated_at"])
        all_timestamps.update(df_ts.unique())
        all_miners.update(df["miner_uid"].unique())
        profile_starts[label] = df_ts.min()

    # Drop timestamps before ALL profiles have started
    latest_start = max(profile_starts.values())
    union_ts = sorted(t for t in all_timestamps if t >= latest_start)
    if not union_ts:
        return pd.DataFrame(columns=_COMBINED_EMPTY_COLS)

    # Forward-fill each profile's reward_weight onto the union grid per miner.
    # Reindex onto the union of ALL profile timestamps first (including those
    # before latest_start), so ffill can carry values forward from earlier rounds
    # of a profile whose first round pre-dates the other profile's first round.
    # Then restrict to union_ts.
    miners = sorted(all_miners)
    full_index = sorted(all_timestamps)
    ff_frames: list[pd.DataFrame] = []
    for _label, df in combined_by_profile.items():
        df_local = df.copy()
        df_local["updated_at"] = pd.to_datetime(df_local["updated_at"])
        pivot = (
            df_local.pivot_table(
                index="updated_at",
                columns="miner_uid",
                values="reward_weight",
                aggfunc="last",
            )
            .reindex(full_index)
            .ffill()
            .reindex(index=union_ts)
            .reindex(columns=miners, fill_value=0.0)
        )
        pivot = pivot.fillna(0.0)
        ff_frames.append(pivot)

    # Sum forward-filled weights across profiles
    total = sum(ff_frames)
    long = total.reset_index().melt(
        id_vars="updated_at", var_name="miner_uid", value_name="reward_weight"
    )
    long["new_smoothed_score"] = (
        0.0  # grand-total doesn't have a meaningful smoothed score
    )
    return (
        long[_COMBINED_EMPTY_COLS]
        .sort_values(["updated_at", "miner_uid"])
        .reset_index(drop=True)
    )


def _compute_earnings_df(
    combined: pd.DataFrame,
    miner_id: int,
    prompt_config: PromptConfig,
    daily_pool_usd: pd.Series,
) -> pd.DataFrame:
    """Compute per-round estimated USD earnings for a single miner.

    Formula per round:
      share_of_round = reward_weight / prompt_config.smoothed_score_coefficient
      usd_per_round  = reward_weight * daily_pool_usd[date] / rounds_per_day[date]

    `reward_weight` is already the miner's share of the FULL on-chain weight
    attributable to this profile: synth.validator.moving_average.combine_moving_averages
    sums per-profile reward_weights, and each profile's softmax contributes up
    to `smoothed_score_coefficient` (0.5 for both LOW and HIGH today, summing
    to 1.0 on-chain). So multiplying reward_weight by the subnet-wide daily
    pool yields the profile's realistic USD contribution — summing the LOW
    and HIGH charts gives the miner's total expected daily earnings.

    `share_of_round` is kept as a display-friendly intermediate ("share of this
    profile's pool") used by the chart's top panel.

    Rounds on dates missing from `daily_pool_usd` (API gap / recent day not yet
    populated) are dropped with a stdout warning. The returned totals reflect
    only days with pool data.

    Returns DataFrame (sorted by updated_at) with columns:
      updated_at, miner_uid, reward_weight, share_of_round, date,
      usd_per_round, usd_cumulative.
    Empty if the miner is absent from `combined` or every round lacks pool data.
    """
    our = (
        combined.loc[combined["miner_uid"] == miner_id].sort_values("updated_at").copy()
    )
    if our.empty:
        return our

    our["updated_at"] = pd.to_datetime(our["updated_at"])
    our["share_of_round"] = (
        our["reward_weight"] / prompt_config.smoothed_score_coefficient
    )
    our["date"] = our["updated_at"].dt.floor("D")

    missing_mask = ~our["date"].isin(daily_pool_usd.index)
    if missing_mask.any():
        missing_dates = sorted(
            {d.date().isoformat() for d in our.loc[missing_mask, "date"]}
        )
        print(
            f"  Warning: dropping {int(missing_mask.sum())} round(s) on "
            f"{len(missing_dates)} day(s) with no /rewards/historical pool data: "
            f"{missing_dates}"
        )
        our = our.loc[~missing_mask].copy()
        if our.empty:
            return our

    rounds_per_day = our.groupby("date").size()
    our["usd_per_round"] = (
        our["reward_weight"]
        * our["date"].map(daily_pool_usd)
        / our["date"].map(rounds_per_day)
    )
    our["usd_cumulative"] = our["usd_per_round"].cumsum()
    return our


def plot_estimated_earnings(
    results: list[BacktestResult],
    prompt_config: PromptConfig,
    combined: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Three-panel estimated USD earnings chart for our miner across all assets.

    Panel 1: Share of round (%) line, window-mean reference.
    Panel 2: Daily earnings ($) bars, mean-daily reference.
    Panel 3: Cumulative earnings ($) line, total annotated.

    Uses live subnet daily miner-pool USD via /rewards/historical. Attributes the
    full daily pool to this profile; if the subnet splits the pool across high/low
    frequency prompts, the $ here is an upper bound for a single-profile run.
    """
    if not results:
        raise RuntimeError("No backtest results to plot.")
    if combined.empty:
        raise RuntimeError("No combined smoothed scores — cannot compute earnings.")

    miner_name = results[0].miner_name
    miner_id = results[0].summary["miner_id"]

    first_ts = pd.to_datetime(combined["updated_at"].min())
    last_ts = pd.to_datetime(combined["updated_at"].max())
    daily_pool = get_daily_miner_pool_usd(
        first_ts.floor("D").to_pydatetime(),
        (last_ts.floor("D") + pd.Timedelta(days=1)).to_pydatetime(),
    )
    if daily_pool.empty:
        raise RuntimeError("No daily pool data available from /rewards/historical.")

    earnings = _compute_earnings_df(combined, miner_id, prompt_config, daily_pool)
    if earnings.empty:
        raise RuntimeError(f"Miner {miner_id} not found in combined smoothed scores.")

    daily = (
        earnings.set_index("updated_at")
        .resample("1D")
        .agg(usd=("usd_per_round", "sum"), share_mean=("share_of_round", "mean"))
        .dropna()
    )

    total_usd = float(earnings["usd_cumulative"].iloc[-1])
    mean_share_pct = float(earnings["share_of_round"].mean() * 100)
    mean_daily_usd = float(daily["usd"].mean()) if not daily.empty else 0.0
    dur_days = (last_ts - first_ts).total_seconds() / 86400
    n_assets = len(results)
    total_assets = len(prompt_config.asset_list)
    partial_coverage = n_assets < total_assets

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.1, 1.3]},
    )

    # Panel 1: share of round (%)
    ax = axes[0]
    ax.plot(
        earnings["updated_at"],
        earnings["share_of_round"] * 100,
        color="tab:blue",
        lw=0.9,
    )
    ax.fill_between(
        earnings["updated_at"],
        0,
        earnings["share_of_round"] * 100,
        color="tab:blue",
        alpha=0.15,
    )
    ax.axhline(
        mean_share_pct,
        color="tab:orange",
        ls="--",
        lw=0.8,
        label=f"window mean {mean_share_pct:.2f}%",
    )
    ax.set_ylabel("Our share of round (%)")
    ax.legend(loc="upper right", fontsize=9)

    # Panel 2: daily earnings ($)
    ax = axes[1]
    ax.bar(daily.index, daily["usd"], color="tab:green", width=0.85, alpha=0.8)
    ax.axhline(
        mean_daily_usd,
        color="black",
        ls="--",
        lw=0.8,
        label=f"mean daily ${mean_daily_usd:,.0f}",
    )
    ax.set_ylabel("Daily earnings ($)")
    ax.legend(loc="upper right", fontsize=9)

    # Panel 3: cumulative earnings ($)
    ax = axes[2]
    ax.plot(
        earnings["updated_at"], earnings["usd_cumulative"], color="tab:purple", lw=1.8
    )
    ax.fill_between(
        earnings["updated_at"],
        0,
        earnings["usd_cumulative"],
        color="tab:purple",
        alpha=0.15,
    )
    ax.set_ylabel("Cumulative earnings ($)")
    ax.set_xlabel("Date (UTC)")
    ax.annotate(
        f"${total_usd:,.0f}",
        xy=(earnings["updated_at"].iloc[-1], total_usd),
        xytext=(-80, -10),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="tab:purple",
    )

    partial_line = (
        f"PARTIAL COVERAGE ({n_assets}/{total_assets} assets) — "
        "earnings overestimate real deployment. "
        if partial_coverage
        else ""
    )
    profile_pct = prompt_config.smoothed_score_coefficient * 100
    fig.suptitle(
        f"Estimated earnings — {miner_name} — {prompt_config.label} "
        f"({n_assets}/{total_assets} assets, {dur_days:.1f}d)\n"
        f"{partial_line}"
        f"Uses live daily miner-pool USD × {profile_pct:.0f}% "
        f"(this profile's on-chain weight share; sum {prompt_config.label.upper()} "
        "+ other profile charts for total).",
        fontsize=11,
        y=0.995,
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"estimated_earnings_{prompt_config.label}.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    partial_tag = (
        f"  PARTIAL COVERAGE ({n_assets}/{total_assets} assets)"
        if partial_coverage
        else ""
    )
    print(
        f"\n  EARNINGS [{prompt_config.label}] total: ${total_usd:,.0f} "
        f"over {dur_days:.1f}d (mean share {mean_share_pct:.2f}%){partial_tag}"
    )
    return chart_path


def plot_grand_total_earnings(
    results_by_profile: dict[str, list[BacktestResult]],
    combined_by_profile: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> Path:
    """Three-panel estimated earnings chart using grand-total reward_weight.

    Uses the full daily miner-pool USD (no × smoothed_score_coefficient factor),
    applied to each miner's grand-total share (reward_weight sums to 1.0 across
    miners per timestamp once both profiles are active).
    """
    grand = _compute_grand_total_weights(combined_by_profile)
    if grand.empty:
        raise RuntimeError("No grand-total rows — cannot compute earnings.")

    first_result = next(
        (rs[0] for rs in results_by_profile.values() if rs),
        None,
    )
    if first_result is None:
        raise RuntimeError("No backtest results to plot.")
    miner_name = first_result.miner_name
    miner_id = first_result.summary["miner_id"]

    first_ts = pd.to_datetime(grand["updated_at"].min())
    last_ts = pd.to_datetime(grand["updated_at"].max())
    daily_pool = get_daily_miner_pool_usd(
        first_ts.floor("D").to_pydatetime(),
        (last_ts.floor("D") + pd.Timedelta(days=1)).to_pydatetime(),
    )
    if daily_pool.empty:
        raise RuntimeError("No daily pool data available from /rewards/historical.")

    # Use a synthetic PromptConfig with smoothed_score_coefficient=1.0 so
    # _compute_earnings_df treats grand-total reward_weight as the full share.
    grand_config = dataclasses.replace(
        LOW_FREQUENCY,
        smoothed_score_coefficient=1.0,
        label="GRAND_TOTAL",
    )
    earnings = _compute_earnings_df(grand, miner_id, grand_config, daily_pool)
    if earnings.empty:
        raise RuntimeError(f"Miner {miner_id} not found in grand-total weights.")

    daily = (
        earnings.set_index("updated_at")
        .resample("1D")
        .agg(usd=("usd_per_round", "sum"), share_mean=("share_of_round", "mean"))
        .dropna()
    )

    total_usd = float(earnings["usd_cumulative"].iloc[-1])
    mean_share_pct = float(earnings["share_of_round"].mean() * 100)
    mean_daily_usd = float(daily["usd"].mean()) if not daily.empty else 0.0
    dur_days = (last_ts - first_ts).total_seconds() / 86400

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.1, 1.3]},
    )

    ax = axes[0]
    ax.plot(
        earnings["updated_at"],
        earnings["share_of_round"] * 100,
        color="tab:blue",
        lw=0.9,
    )
    ax.fill_between(
        earnings["updated_at"],
        0,
        earnings["share_of_round"] * 100,
        color="tab:blue",
        alpha=0.15,
    )
    ax.axhline(
        mean_share_pct,
        color="tab:orange",
        ls="--",
        lw=0.8,
        label=f"window mean {mean_share_pct:.2f}%",
    )
    ax.set_ylabel("Our share of subnet (%)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.bar(daily.index, daily["usd"], color="tab:green", width=0.85, alpha=0.8)
    ax.axhline(
        mean_daily_usd,
        color="black",
        ls="--",
        lw=0.8,
        label=f"mean daily ${mean_daily_usd:,.0f}",
    )
    ax.set_ylabel("Daily earnings ($)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    ax.plot(
        earnings["updated_at"], earnings["usd_cumulative"], color="tab:purple", lw=1.8
    )
    ax.fill_between(
        earnings["updated_at"],
        0,
        earnings["usd_cumulative"],
        color="tab:purple",
        alpha=0.15,
    )
    ax.set_ylabel("Cumulative earnings ($)")
    ax.set_xlabel("Date (UTC)")
    ax.annotate(
        f"${total_usd:,.0f}",
        xy=(earnings["updated_at"].iloc[-1], total_usd),
        xytext=(-80, -10),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color="tab:purple",
    )

    total_assets = sum(len(rs) for rs in results_by_profile.values())
    fig.suptitle(
        f"Estimated grand-total earnings — {miner_name} — "
        f"{len(results_by_profile)} profiles, {total_assets} assets, {dur_days:.1f}d\n"
        f"Uses full daily miner-pool USD × grand-total share "
        f"(sums both profile coefficients, matches real-validator weights).",
        fontsize=11,
        y=0.995,
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    if output_dir is None:
        output_dir = DEFAULT_MINER_OUTPUT_ROOT / miner_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "estimated_earnings_GRAND_TOTAL.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(
        f"\n  GRAND TOTAL EARNINGS total: ${total_usd:,.0f} "
        f"over {dur_days:.1f}d (mean share {mean_share_pct:.2f}%)"
    )
    return chart_path


def run_backtest(
    miner_name: str,
    prompt_config: PromptConfig,
    n_backtest_days: int,
    predictions_dir: Path | None = None,
    miner_id: int = 999,
    scoring_executor: ProcessPoolExecutor | None = None,
    eval_end: datetime | None = None,
    simulate_registration: datetime | None = None,
    simulate_deregistration: datetime | None = None,
) -> tuple[list[BacktestResult], pd.DataFrame]:
    """Run the full backtest for all assets in a prompt config.

    For each asset: runs backtest, prints rank summary, saves per-asset rank chart.
    After all assets: saves total rank evolution chart + estimated earnings chart
    when ≥2 assets succeed.

    Returns (results, combined) where:
      - results: list of successful BacktestResult (one per asset)
      - combined: per-profile combined smoothed-scores DataFrame (empty when <2 assets)
    """
    assets = prompt_config.asset_list
    max_workers = min(len(assets), 6)
    print(f"  Dispatching {len(assets)} assets to {max_workers} backtest workers...")

    # Phase 1: run backtest() for each asset in parallel (I/O-bound: API calls + disk)
    asset_results: dict[str, BacktestResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                backtest,
                miner_name=miner_name,
                asset=asset,
                time_length=prompt_config.time_length,
                time_increment=prompt_config.time_increment,
                n_backtest_days=n_backtest_days,
                miner_id=miner_id,
                predictions_dir=predictions_dir,
                scoring_intervals=prompt_config.scoring_intervals,
                prompt_config=prompt_config,
                scoring_executor=scoring_executor,
                eval_end=eval_end,
                simulate_registration=simulate_registration,
                simulate_deregistration=simulate_deregistration,
            ): asset
            for asset in assets
        }
        for future in as_completed(futures):
            asset = futures[future]
            try:
                asset_results[asset] = future.result()
            except Exception as e:
                print(f"\n--- BACKTEST [{prompt_config.label}/{asset}] ---")
                print(f"  FAILED: {e}")

    # Phase 2: print summaries + plot charts sequentially (matplotlib is not thread-safe)
    results: list[BacktestResult] = []
    for asset in assets:
        if asset not in asset_results:
            continue
        result = asset_results[asset]
        results.append(result)
        summary = result.summary

        print(f"\n--- BACKTEST [{prompt_config.label}/{asset}] ---")
        ss = result.smoothed_scores.copy()
        if not ss.empty:
            last_ts = ss["updated_at"].max()
            last_slice = ss.loc[ss["updated_at"] == last_ts]
            total_miners = len(last_slice)
            miner_rw = last_slice.loc[last_slice["miner_uid"] == miner_id]
            if not miner_rw.empty:
                rank = int(
                    last_slice["reward_weight"]
                    .rank(ascending=False, method="min")
                    .loc[miner_rw.index[0]]
                )
                rw = float(miner_rw["reward_weight"].iloc[0])
                sm = float(summary.get("final_smoothed_score", 0) or 0)
                print(
                    f"  rank: {rank}/{total_miners}  reward_weight: {rw:.6f}  smoothed_score: {sm:.2f}"
                )
            else:
                print(f"  Miner {miner_id} not found in final smoothed scores.")
        else:
            print(f"  No smoothed scores computed.")

        print(
            f"  Prompts scored: {summary['num_prompts']}  Mean CRPS: {summary['mean_crps']:.6f}"
        )

        try:
            chart_path = plot_rank_evolution(result)
            print(f"  Rank chart saved to: {chart_path}")
        except (RuntimeError, Exception) as e:
            print(f"  Chart generation failed: {e}")

        try:
            crps_chart = plot_crps_over_time(result)
            print(f"  CRPS chart saved to: {crps_chart}")
        except (RuntimeError, Exception) as e:
            print(f"  CRPS chart generation failed: {e}")

        for plot_fn, chart_name in [
            (plot_crps_by_hour, "CRPS-by-hour"),
            (plot_crps_by_day, "CRPS-by-day"),
            (plot_crps_ratio_distribution, "CRPS-ratio-dist"),
            (plot_weekly_percentile, "Weekly-percentile"),
        ]:
            try:
                path = plot_fn(result)
                print(f"  {chart_name} chart saved to: {path}")
            except (RuntimeError, Exception) as e:
                print(f"  {chart_name} chart generation failed: {e}")

    # Cross-asset charts: compute combined smoothed scores once, share across both charts.
    combined = pd.DataFrame(columns=_COMBINED_EMPTY_COLS)
    if len(results) >= 2:
        try:
            combined = compute_combined_smoothed_scores(
                results,
                prompt_config,
                simulate_registration=simulate_registration,
            )
        except Exception as e:
            print(f"  Combined smoothed scores failed: {e}")

        if not combined.empty:
            try:
                total_chart = plot_total_rank_evolution(
                    results,
                    combined=combined,
                    profile_label=prompt_config.label,
                )
                print(f"  Total rank chart saved to: {total_chart}")
            except Exception as e:
                print(f"  Total rank chart failed: {e}")

            try:
                earnings_chart = plot_estimated_earnings(
                    results,
                    prompt_config=prompt_config,
                    combined=combined,
                )
                print(f"  Earnings chart saved to: {earnings_chart}")
            except Exception as e:
                print(f"  Earnings chart failed: {e}")

    return results, combined
