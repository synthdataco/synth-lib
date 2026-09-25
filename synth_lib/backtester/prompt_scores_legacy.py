"""The validator's prompt scoring as it stood before the outlier cap.

Recovered verbatim from synth-subnet `0d014cb`, the rev pinned before v1.12.0. The current
`synth.validator.reward.compute_prompt_scores` clips raw CRPS above a median multiple and takes the
p95 that fills missed responses over the unclipped scores; the validator has done that since
`OUTLIER_CAP_DATE`. Prompts scored before it were not clipped, and a backtest window that spans the
cutover has to score each side the way the validator did.

Frozen: this is retired behaviour and will never change again. It is not a reimplementation to be
kept in step with upstream — the version upstream no longer exists.
"""

from __future__ import annotations

import numpy as np


def compute_prompt_scores_pre_cap(score_values: np.ndarray):
    """Returns the current function's 4-tuple shape; nothing was capped, so was_capped is None."""
    if np.all(score_values == -1):
        return None, 0, 0, None
    score_values_valid = score_values[score_values != -1]
    percentile95 = np.percentile(score_values_valid, 95)
    # Valid scores are not capped; only missed responses (-1) are filled.
    filled_scores = np.where(score_values == -1, percentile95, score_values)
    lowest_score = np.min(filled_scores)
    return filled_scores - lowest_score, percentile95, lowest_score, None
