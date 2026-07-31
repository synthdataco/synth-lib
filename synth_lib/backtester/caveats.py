"""Warnings for scoring-regime changes and silent API gaps that would skew a backtest."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import pandas as pd
import requests

from synth.validator.competition_config import CRYPTO_1H, CompetitionConfig

from synth_lib.backtester.config import (
    COMPETITION_SPLIT_DATE,
    HF_CRPS_FORMULA_CHANGE_DATE,
    UTC,
    slug_for,
)



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


def _hf_crps_window_start(
    n_backtest_days: int,
    competition: CompetitionConfig,
    eval_end: datetime | None,
    simulate_registration: datetime | None,
    simulate_deregistration: datetime | None,
) -> datetime:
    """Compute the effective backtest window start, mirroring backtest()."""
    if simulate_deregistration is not None:
        anchor = simulate_deregistration
    elif eval_end is not None:
        anchor = eval_end
    else:
        anchor = datetime.now(UTC)
    if simulate_registration is not None:
        return simulate_registration - timedelta(days=competition.window_days)
    return anchor - timedelta(days=n_backtest_days)


def _maybe_warn_hf_crps_formula_change(
    competition: CompetitionConfig,
    n_backtest_days: int,
    eval_end: datetime | None,
    simulate_registration: datetime | None,
    simulate_deregistration: datetime | None,
) -> None:
    """Warn when an HF backtest window starts before the validator's CRPS
    formula change on 2026-03-11. Pre-cutoff CRPS values stored in the API
    were computed with the old formula, so ranks derived from them aren't
    comparable to current live ranks. The smoothing window also pulls in
    pre-cutoff data for `competition.window_days` after the cutoff, so
    we recommend resuming evaluation only after `cutoff + window_days`.
    """
    if slug_for(competition) != "crypto-1h":
        return
    window_start = _hf_crps_window_start(
        n_backtest_days,
        competition,
        eval_end,
        simulate_registration,
        simulate_deregistration,
    )
    if window_start >= HF_CRPS_FORMULA_CHANGE_DATE:
        return
    safe_sim_reg = (
        HF_CRPS_FORMULA_CHANGE_DATE + timedelta(days=competition.window_days)
    ).date()
    cutoff = HF_CRPS_FORMULA_CHANGE_DATE.date()
    warnings.warn(
        f"crypto-1h backtest window starts {window_start.date()}, before "
        f"{cutoff} when the validator's CRPS formula changed. API-stored CRPS "
        f"for prompts before that date does not reflect how CRPS is computed "
        f"today, so ranks before {safe_sim_reg} may not be representative. "
        f"To get current-formula ranks: restrict eval to >= {cutoff}, or use "
        f"simulate_registration={safe_sim_reg}. See README 'Known caveats'.",
        UserWarning,
        stacklevel=2,
    )


def _maybe_warn_competition_split(
    competition: CompetitionConfig,
    n_backtest_days: int,
    eval_end: datetime | None,
    simulate_registration: datetime | None,
    simulate_deregistration: datetime | None,
) -> None:
    """Warn when a backtest window starts before the 3-competition split.

    The backtester models only the current 3-competition era; pre-split windows
    are scored under a 2-profile structure that no longer exists.
    """
    window_start = _hf_crps_window_start(
        n_backtest_days, competition, eval_end,
        simulate_registration, simulate_deregistration,
    )
    if window_start >= COMPETITION_SPLIT_DATE:
        return
    warnings.warn(
        f"{slug_for(competition)} backtest window starts {window_start.date()}, "
        f"before the 3-competition split on {COMPETITION_SPLIT_DATE.date()}. "
        f"The backtester models only the current 3-competition era; results over "
        f"pre-split prompts mix an incompatible 2-profile reward structure. "
        f"Restrict the window to >= {COMPETITION_SPLIT_DATE.date()}.",
        UserWarning,
        stacklevel=2,
    )
