"""Backtest result type and the skip signal for assets with no scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class NoScoresAvailable(RuntimeError):
    """An asset has no scored prompts in the window — a skip, not a failure.

    Distinct type so run_backtest can tell it apart from a genuine break.
    """


@dataclass
class BacktestResult:
    miner_name: str
    prompt_df: pd.DataFrame
    smoothed_scores: pd.DataFrame
    summary: dict[str, Any]
