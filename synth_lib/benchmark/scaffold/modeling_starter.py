"""Starter model written by the harness: log-normal martingale calibrated on the context.

This is a deliberately simple STARTING POINT — replace it with your own model.
Contract: simulate(...) -> (start_timestamp, time_increment, *paths), each path has
time_length//time_increment + 1 points starting at the context's last price.
"""

from __future__ import annotations

import numpy as np


def simulate(
    asset: str,
    start_time,
    time_increment: int = 300,
    time_length: int = 86_400,
    num_simulations: int = 1000,
    context_prices=None,
):
    steps = time_length // time_increment + 1
    last = float(context_prices.iloc[-1])
    rets = np.diff(np.log(np.asarray(context_prices, dtype=float)))
    rets = rets[np.isfinite(rets)]
    per_step_sigma = float(np.std(rets)) * (time_increment / 60.0) ** 0.5 if rets.size > 10 else 1e-3
    rng = np.random.default_rng()
    shocks = rng.normal(0.0, per_step_sigma, size=(num_simulations, steps))
    shocks[:, 0] = 0.0
    paths = last * np.exp(np.cumsum(shocks, axis=1))
    return (start_time, time_increment, *[p.tolist() for p in paths])
