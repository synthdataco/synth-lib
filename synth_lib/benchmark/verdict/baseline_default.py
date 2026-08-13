"""Baseline 'synth_default': the subnet's default generator, adapted to the simulate() contract."""

from __future__ import annotations

from synth.miner.simulations import generate_simulations  # type: ignore[import-untyped]


def simulate(asset, start_time, time_increment=300, time_length=86_400, num_simulations=1000, context_prices=None):
    # generate_simulations() already returns (start_timestamp, time_increment, *paths) via
    # convert_prices_to_time_format — same shape as the simulate() contract, so no
    # repacking here (verified at runtime: repacking would have duplicated the metadata).
    # context_prices is ignored: the subnet's default generator fetches the current price
    # itself (network, Binance/Hyperliquid) instead of using the supplied context.
    return generate_simulations(
        asset=asset,
        start_time=start_time,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
    )
