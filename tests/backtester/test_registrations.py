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


class _Response:
    def __init__(self, status_code: int, rows: list[dict] | None = None) -> None:
        self.status_code = status_code
        self._rows = rows

    def json(self) -> list[dict] | None:
        return self._rows

    def raise_for_status(self) -> None:
        assert self.status_code == 200


def test_get_registrations_pages_the_api_and_keeps_the_span(monkeypatch):
    """45 days take two pages; overlapping rows count once and rows past `until` are dropped."""
    import synth_lib.backtester.loading as loading

    since, until = datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 9, 15, tzinfo=UTC)
    calls: list[dict] = []
    shared = {"miner_uid": 7, "hotkey": "5Hb", "created_at": "2026-08-31T03:00:00Z"}

    def fake_get(url, params=None, timeout=30):
        calls.append(params)
        if params["from"].startswith("2026-08-01"):
            return _Response(200, [{"miner_uid": 12, "hotkey": "5Fa", "created_at": "2026-08-02T08:00:00Z"}, shared])
        return _Response(200, [shared, {"miner_uid": 3, "hotkey": "5Gc", "created_at": "2026-09-15T12:00:00Z"}])

    monkeypatch.setattr(loading, "_http_get", fake_get)
    out = loading.get_registrations(since, until)

    assert [(c["from"], c["to"]) for c in calls] == [
        ("2026-08-01T00:00:00Z", "2026-08-31T00:00:00Z"),
        ("2026-08-31T00:00:00Z", "2026-09-15T00:00:00Z"),
    ]
    assert list(out["miner_uid"]) == [12, 7]
    assert list(out["hotkey"]) == ["5Fa", "5Hb"]
    assert out["created_at"].iloc[0] == datetime(2026, 8, 2, 8, tzinfo=UTC)


def test_get_registrations_reads_a_404_as_no_registration(monkeypatch):
    import synth_lib.backtester.loading as loading

    monkeypatch.setattr(loading, "_http_get", lambda url, params=None, timeout=30: _Response(404))
    out = loading.get_registrations(datetime(2026, 9, 14, tzinfo=UTC), datetime(2026, 9, 25, tzinfo=UTC))
    assert out.empty and list(out.columns) == ["miner_uid", "hotkey", "created_at"]
