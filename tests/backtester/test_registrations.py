"""fetch_registrations against a fake chain of 12 s blocks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import synth_lib.backtester.registrations as reg

T0 = datetime(2026, 9, 1, tzinfo=UTC)
HEAD = 31 * 7_200  # 2026-10-02
UNTIL = datetime(2026, 9, 25, tzinfo=UTC)
SINCE = datetime(2026, 9, 14, tzinfo=UTC)


def _at(block: int) -> datetime:
    return T0 + timedelta(seconds=12 * block)


class _FakeSubtensor:
    networks: list[str] = []
    metagraph_blocks: list[int | None] = []

    def __init__(self, network: str) -> None:
        self.networks.append(network)

    def get_current_block(self) -> int:
        return HEAD

    def get_timestamp(self, block: int | None = None) -> datetime:
        return _at(HEAD if block is None else block)

    def metagraph(self, netuid: int, lite: bool = True, block: int | None = None) -> SimpleNamespace:
        self.metagraph_blocks.append(block)
        # uid 0 long before the span, uid 1 and 2 inside it, uid 3 one block before it.
        return SimpleNamespace(block_at_registration=[100, 100_000, 170_000, 13 * 7_200 - 1])


def test_registrations_inside_the_span_with_their_exact_time(monkeypatch):
    monkeypatch.setattr(reg.bt, "Subtensor", _FakeSubtensor)

    out = reg.fetch_registrations(SINCE, UNTIL)

    assert out == {1: _at(100_000), 2: _at(170_000)}
    assert _FakeSubtensor.networks == ["archive"]
    assert _FakeSubtensor.metagraph_blocks == [24 * 7_200]  # the metagraph as of UNTIL, not today
