"""Backtester constants, competition slug maps and offline-mode helpers."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from synth.validator.competition_config import (
    ALL_COMPETITIONS,
    COM_EQU_24H,
    CRYPTO_1H,
    CRYPTO_24H,
    CompetitionConfig,
)



# Miner-dashboard read API (shared host with the monitoring API, /v1 prefix).
MINER_DASHBOARD_API_BASE = "https://monitoring.synthdata.co/v1"
# Default 24h scoring intervals (crypto-24h / com-equ-24h); only used when no
# competition is passed to backtest().
SCORING_INTERVALS: dict[str, int] = {
    "5min": 300,
    "30min": 1_800,
    "3hour": 10_800,
    "24hour_abs": 86_400,
}
DEFAULT_MINER_OUTPUT_ROOT = Path("miner_outputs")
DEFAULT_MARKET_DATA_ROOT = Path("market_data/pyth/BTC/1m")
_LEGACY_FALLBACK_WINDOW_DAYS = 10  # only used when no competition is in scope
_COMBINED_EMPTY_COLS = [
    "updated_at",
    "miner_uid",
    "new_smoothed_score",
    "reward_weight",
]

UTC = timezone.utc

# -- HF CRPS formula change --
# On 2026-03-11 the Synth validator changed how it computes CRPS for the
# high-frequency competition (now crypto-1h). CRPS values stored in the API for
# prompts before that date were produced by the old formula and aren't directly
# comparable to ranks computed under the current formula. See README "Known
# caveats".
HF_CRPS_FORMULA_CHANGE_DATE = datetime(2026, 3, 11, tzinfo=UTC)

# On-chain emission normalization for the USD earnings estimate. A miner's
# realized emission is not exactly proportional to the reward_weight the
# validator sets: Yuma consensus (trust/clipping) and the presence of other
# neurons flatten realized emissions relative to set weight, so
# `reward_weight * pool` OVERESTIMATES the USD a competitive miner actually
# earns. This factor rescales the raw formula to realized emissions.
#
# Empirically calibrated against live on-chain data (validate_earnings_formula.py,
# window 2026-07-04→2026-07-06, all 3 competitions, top-15 miners by actual USD):
# raw bt/actual SUM = 1414.19/980.97 = 1.442 (a consistent +44-47% overestimate
# across the top miners), so factor = actual/bt = 0.6935. It is an approximation
# (per-miner spread ~±a few %; the tail of small miners is under-corrected since
# realized emission is flatter than set weight) and may drift over time — treat
# absolute USD as an estimate. 1.0 disables the correction (raw reward-weight USD).
EMISSION_NORMALIZATION_FACTOR = 0.6935

# The subnet split into 3 competitions (crypto-24h, com-equ-24h, crypto-1h) on
# this date (first com-equ-24h appearance in /rewards/scores, verified live).
# Before it the reward structure was 2 profiles (each /2) and com-equ-24h did
# not exist, so a backtest window predating it mixes incompatible structures.
COMPETITION_SPLIT_DATE = datetime(2026, 6, 23, tzinfo=UTC)

# -- Pagination limits (empirical — not confirmed from API docs) --
# Synth API /validation/scores/historical has a 7-day max range.
# We use 6-day chunks to stay safely within the limit.
API_SCORES_PAGE_SIZE_DAYS = 1  # /validation/scores/historical: "to" = inclusive whole day; >2d spans 400. Endpoint thins crypto-1h prompts to ~10-min density (~144/day, full 256-miner field each) at ANY range size. 1-day pages + drop_duplicates = full accessible coverage.

# /v1/miners/rewards/pool caps ranges at 366 days; 300-day chunks stay under it.
API_POOL_PAGE_SIZE_DAYS = 300

# -- Prediction file matching --
# Scoring delay: the real prediction start_time is a few minutes before
# scored_time - time_length. We allow up to 30 minutes of tolerance when
# matching prediction filenames to scored prompts.
PREDICTION_MATCH_TOLERANCE_MINUTES = 30


# -- Offline mode --
# When SYNTH_BACKTESTER_OFFLINE_DATA_ROOT is set, read bundled parquets from that
# root instead of calling api.synthdata.co / Pyth. Required for reproducible /
# network-restricted use cases (e.g. concurrent benchmark rollouts that would
# otherwise rate-limit the Synth API, or sandboxes with internet disabled).
#
# Expected layout under the root (slug ∈ crypto-1h / crypto-24h / com-equ-24h):
#   miner_scores_{asset}_{slug}.parquet          # slug = competition slug
#   rewards_history_{slug}.parquet               # slug = competition slug
#   miner_pool_usd.parquet                       # columns: date, usd
#   market_data/pyth/{asset}/1m/date=*.parquet   # daily price partitions
_OFFLINE_ENV_VAR = "SYNTH_BACKTESTER_OFFLINE_DATA_ROOT"

# comp.label -> API slug used by /rewards/scores?prompt_name=. synth does not
# define these slugs; they are the backtester's canonical competition keys and
# double as offline-parquet filename suffixes.
COMPETITION_SLUGS: dict[str, str] = {
    "Crypto 1h": "crypto-1h",
    "Crypto 24h": "crypto-24h",
    "Commodities/Equities 24h": "com-equ-24h",
}
# Fail loudly at import if synth adds/renames a competition we haven't mapped,
# instead of a bare KeyError surfacing later deep in a call site.
_UNMAPPED_LABELS = [c.label for c in ALL_COMPETITIONS if c.label not in COMPETITION_SLUGS]
if _UNMAPPED_LABELS:
    raise RuntimeError(
        f"COMPETITION_SLUGS has no slug for competition label(s) {_UNMAPPED_LABELS}; "
        "update it to match synth.validator.competition_config.ALL_COMPETITIONS."
    )
SLUG_TO_COMPETITION: dict[str, CompetitionConfig] = {
    COMPETITION_SLUGS[c.label]: c for c in ALL_COMPETITIONS
}


def slug_for(comp: CompetitionConfig) -> str:
    """API slug / filename suffix for a competition."""
    try:
        return COMPETITION_SLUGS[comp.label]
    except KeyError:
        raise ValueError(
            f"No slug mapped for competition label {comp.label!r}; "
            "add it to COMPETITION_SLUGS."
        ) from None


def competition_for(asset: str, time_length: int) -> CompetitionConfig:
    """Resolve the competition owning (asset, time_length).

    (time_length, time_increment) alone is ambiguous for the two 24h
    competitions (crypto-24h and com-equ-24h share 86400/300); the asset
    disambiguates because their asset_lists are disjoint.
    """
    for comp in ALL_COMPETITIONS:
        if comp.time_length == time_length and asset in comp.asset_list:
            return comp
    raise ValueError(
        f"No competition for asset={asset!r} time_length={time_length}. "
        f"Known: {[(c.label, c.asset_list) for c in ALL_COMPETITIONS]}"
    )


def _offline_root() -> Path | None:
    val = os.environ.get(_OFFLINE_ENV_VAR)
    return Path(val) if val else None


def _filter_time_range(
    df: pd.DataFrame, col: str, start: datetime, end: datetime
) -> pd.DataFrame:
    if df.empty:
        return df
    s = pd.to_datetime(df[col], utc=True)
    start_ts = pd.Timestamp(start) if start.tzinfo else pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end) if end.tzinfo else pd.Timestamp(end, tz="UTC")
    return df.loc[(s >= start_ts) & (s < end_ts)].copy()
