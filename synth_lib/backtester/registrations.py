"""When each SN50 uid's current occupant registered, read from the chain.

The smoothing replay needs it to score a re-registered uid as the two miners live sees
(`scoring._split_reregistered`). The public scores carry the uid only, so this is the one input that
has to come from the chain rather than the Synth API.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import bittensor as bt

SN50 = 50
# Block time is about 12 s; used only to skip the exact timestamp lookup for registrations far older
# than `since`, never as the time returned.
_APPROX_BLOCK_SECONDS = 12
_APPROX_MARGIN = timedelta(days=2)


def _block_at(subtensor: bt.Subtensor, when: datetime) -> int:
    """The block produced at `when`, to within a block or two: estimate, then correct twice."""
    block = subtensor.get_current_block()
    for _ in range(3):
        drift = (when - subtensor.get_timestamp(block)).total_seconds()
        block = min(subtensor.get_current_block(), block + round(drift / _APPROX_BLOCK_SECONDS))
    return block


def fetch_registrations(
    since: datetime,
    until: datetime | None = None,
    netuid: int = SN50,
    network: str = "archive",
) -> dict[int, datetime]:
    """uid -> exact chain time its occupant at `until` registered, for registrations in [since, until].

    Pass `since` = the start of the scored data (window start minus the competition's horizon) and
    `until` = its end (default: now). The metagraph is read as of `until`, not today: uids churn
    (11 of the 31 re-registered between 2026-09-12 and 09-24 had changed hands again by 10-02), and
    a later registration would hide the one inside the span. Old blocks are only served by an
    archive node, hence the default network. One registration per uid: a uid re-registered twice
    inside the span is split at the later one only.
    """
    subtensor = bt.Subtensor(network=network)
    end_block = subtensor.get_current_block() if until is None else _block_at(subtensor, until)
    end = subtensor.get_timestamp(end_block)
    blocks = subtensor.metagraph(netuid, lite=True, block=end_block).block_at_registration
    out = {}
    for uid, block in enumerate(blocks):
        approx = end - timedelta(seconds=(end_block - int(block)) * _APPROX_BLOCK_SECONDS)
        if approx < since - _APPROX_MARGIN:
            continue
        registered_at = subtensor.get_timestamp(int(block))
        if registered_at >= since:
            out[uid] = registered_at
    return out
