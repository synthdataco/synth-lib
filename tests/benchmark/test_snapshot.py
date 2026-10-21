import json
from datetime import date
from pathlib import Path

import pandas as pd

from synth_lib.benchmark.snapshot import build_snapshot, render_data_md, verify_snapshot
from synth_lib.preparation.event_calendar.events import (
    CALENDAR_DIR,
    LEDGER_COLUMNS,
    LEDGER_FILE,
    event_frame,
    write_atomic,
)

ASSETS = ["BTC", "WTIOIL"]  # WTIOIL deliberately present: it must NEVER be dropped


def _make_source(tmp_path: Path) -> Path:
    src = tmp_path / "market_data" / "pyth"
    for asset in ASSETS:
        d = src / asset / "1m"
        d.mkdir(parents=True)
        for day in ["2026-07-10", "2026-07-15", "2026-07-20"]:
            (d / f"date={day}.parquet").write_bytes(day.encode())
    return tmp_path / "market_data"


def test_snapshot_stops_at_cutoff_and_writes_manifest(tmp_path):
    src = _make_source(tmp_path)
    dest = tmp_path / "snapshot"
    manifest = build_snapshot(src, dest, cutoff=date(2026, 7, 15))
    for asset in ASSETS:
        files = sorted(p.name for p in (dest / "pyth" / asset / "1m").iterdir())
        assert files == ["date=2026-07-10.parquet", "date=2026-07-15.parquet"]  # cutoff included, after excluded
    entries = json.loads((dest / "manifest.json").read_text())
    assert len(entries) == 4 and all(len(h) == 64 for h in entries.values())
    assert manifest == entries


def test_snapshot_start_bound_excludes_earlier_partitions(tmp_path):
    """The disk may hold months; a campaign gets only its window (e.g. post-competition-split)."""
    src = _make_source(tmp_path)
    dest = tmp_path / "snapshot"
    build_snapshot(src, dest, cutoff=date(2026, 7, 20), start=date(2026, 7, 15))
    for asset in ASSETS:
        files = sorted(p.name for p in (dest / "pyth" / asset / "1m").iterdir())
        assert files == ["date=2026-07-15.parquet", "date=2026-07-20.parquet"]  # start inclusive


def test_snapshot_empty_window_names_both_bounds(tmp_path):
    src = _make_source(tmp_path)
    import pytest

    with pytest.raises(ValueError, match=r"\[2026-08-01, 2026-08-10\]"):
        build_snapshot(src, tmp_path / "snapshot", cutoff=date(2026, 8, 10), start=date(2026, 8, 1))


def test_verify_detects_tampering(tmp_path):
    src = _make_source(tmp_path)
    dest = tmp_path / "snapshot"
    build_snapshot(src, dest, cutoff=date(2026, 7, 15))
    assert verify_snapshot(dest) is True
    next(iter((dest / "pyth" / "BTC" / "1m").iterdir())).write_bytes(b"tampered")
    assert verify_snapshot(dest) is False


def test_render_data_md_measures_the_snapshot(tmp_path):
    import numpy as np
    import pandas as pd

    from synth_lib.benchmark.snapshot import render_data_md

    root = tmp_path / "snapshot"
    btc = root / "prices" / "BTC" / "1m"
    btc.mkdir(parents=True)
    idx1 = pd.date_range("2026-07-01", periods=1440, freq="1min", tz="UTC")
    pd.DataFrame({"timestamp": idx1, "close": 100.0}).to_parquet(btc / "date=2026-07-01.parquet")
    closes = np.full(1440, 101.0)
    closes[:144] = np.nan  # 10% NaN day
    idx2 = pd.date_range("2026-07-02", periods=1440, freq="1min", tz="UTC")
    pd.DataFrame({"timestamp": idx2, "close": closes}).to_parquet(btc / "date=2026-07-02.parquet")
    # an all-NaN skeleton day must NOT count as a real day
    idx0 = pd.date_range("2026-06-30", periods=1440, freq="1min", tz="UTC")
    pd.DataFrame({"timestamp": idx0, "close": np.full(1440, np.nan)}).to_parquet(btc / "date=2026-06-30.parquet")

    offline = root / "offline_data"
    offline.mkdir()
    scored = pd.date_range("2026-07-01", "2026-07-10", freq="1D", tz="UTC")
    pd.DataFrame({"scored_time": scored}).to_parquet(offline / "miner_scores_BTC_crypto-24h.parquet")

    md = render_data_md(root)
    assert "| BTC | 2026-07-01 | 2026-07-02 | 2 | 5.0% |" in md  # skeleton day excluded, NaN measured
    assert "2026-07-01 → 2026-07-10" in md  # offline bundle coverage
    assert "NaN close" in md and "Missing file" in md  # semantics block present


def _realized(root: Path, asset: str, days: list[str]) -> None:
    d = root / "realized" / asset / "86400_300"
    d.mkdir(parents=True, exist_ok=True)
    for day in days:
        (d / f"date={day}.parquet").write_bytes(b"")


def test_data_md_states_the_realized_path_span(tmp_path):
    """The agent is told which prompts can fall back on validator truth — measured, not asserted."""
    from synth_lib.benchmark.snapshot import render_data_md

    _realized(tmp_path, "BTC", ["2026-07-02", "2026-07-20"])
    _realized(tmp_path, "ETH", ["2026-07-11"])
    md = render_data_md(tmp_path)
    assert "**Cached realized paths** (2 assets): **2026-07-02 → 2026-07-20**" in md


def test_data_md_measures_the_flat_steps_of_the_scored_paths(tmp_path):
    """A thin market's scored path often does not move between two points; agents had to find it."""
    import pandas as pd

    from synth_lib.benchmark.snapshot import render_data_md

    for asset, path in (("XAU", [10.0, 10.0, 10.0, 11.0, 11.0]), ("BTC", [1.0, 2.0, 3.0, 4.0, 4.0])):
        d = tmp_path / "realized" / asset / "86400_300"
        d.mkdir(parents=True)
        pd.DataFrame({"start_time": [pd.Timestamp("2026-07-02", tz="UTC")], "prices": [path]}).to_parquet(
            d / "date=2026-07-02.parquet"
        )
    md = render_data_md(tmp_path)
    assert "computed against: XAU (5 min) 75.0%, BTC (5 min) 25.0%." in md


def test_data_md_states_the_timestamp_unit(tmp_path):
    import pandas as pd

    from synth_lib.benchmark.snapshot import render_data_md

    btc = tmp_path / "prices" / "BTC" / "1m"
    btc.mkdir(parents=True)
    idx = pd.date_range("2026-07-01", periods=1440, freq="1min", tz="UTC").as_unit("us")
    pd.DataFrame({"timestamp": idx, "close": 100.0}).to_parquet(btc / "date=2026-07-01.parquet")
    assert "`timestamp` loads as `datetime64[us, UTC]`" in render_data_md(tmp_path)


def test_data_md_omits_the_span_when_nothing_is_cached(tmp_path):
    """No paths cached is not a span of zero; the line must not appear at all."""
    from synth_lib.benchmark.snapshot import render_data_md

    assert "Cached realized paths" not in render_data_md(tmp_path)


def _write_ledger(market_data: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    for column in ("time_utc", "as_of", "superseded_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    write_atomic(frame, market_data / CALENDAR_DIR / LEDGER_FILE)


def _event(time_utc: str, event: str, as_of: str, superseded_at: str | None = None) -> dict:
    return {
        "time_utc": time_utc,
        "event": event,
        "scope": "all",
        "detail": "",
        "source": "test",
        "as_of": as_of,
        "superseded_at": superseded_at,
    }


def test_snapshot_freezes_the_calendar_known_at_the_cutoff(tmp_path):
    """Upcoming dates published by the cutoff are in; one published, or moved away, by then is not."""
    src = _make_source(tmp_path)
    _write_ledger(
        src,
        [
            _event("2026-07-10 12:30", "NFP", "2026-06-19"),
            _event("2026-07-22 12:30", "CPI", "2026-07-01"),  # upcoming at the cutoff, published before it
            _event("2026-07-29 18:00", "FOMC", "2026-07-16"),  # published the day after the cutoff
            _event("2026-07-24 12:30", "PPI", "2026-07-01", superseded_at="2026-07-12"),  # moved before it
            _event("2026-07-30 12:30", "GDP", "2026-07-01", superseded_at="2026-07-20"),  # moved after it
        ],
    )
    dest = tmp_path / "snapshot"
    manifest = build_snapshot(src, dest, cutoff=date(2026, 7, 15))
    view = pd.read_parquet(dest / "calendar" / "events.parquet")
    assert sorted(view["event"]) == ["CPI", "GDP", "NFP"]
    assert "calendar/events.parquet" in manifest and verify_snapshot(dest)


def test_a_rebuilt_snapshot_keeps_its_frozen_calendar(tmp_path):
    """A setup run again must not change a campaign's inputs, even when the ledger has since gained
    a backdated row the cutoff would now see."""
    src = _make_source(tmp_path)
    _write_ledger(src, [_event("2026-07-10 12:30", "NFP", "2026-06-19")])
    dest = tmp_path / "snapshot"
    build_snapshot(src, dest, cutoff=date(2026, 7, 15))
    frozen = (dest / "calendar" / "events.parquet").read_bytes()
    _write_ledger(
        src, [_event("2026-07-10 12:30", "NFP", "2026-06-19"), _event("2026-07-22 12:30", "CPI", "2026-07-01")]
    )
    build_snapshot(src, dest, cutoff=date(2026, 7, 15))
    assert (dest / "calendar" / "events.parquet").read_bytes() == frozen and verify_snapshot(dest)


def test_snapshot_without_a_ledger_has_no_calendar(tmp_path):
    dest = tmp_path / "snapshot"
    build_snapshot(_make_source(tmp_path), dest, cutoff=date(2026, 7, 15))
    assert not (dest / "calendar").exists()


def test_data_md_counts_each_event_type_and_the_ones_released_with_it(tmp_path):
    root = tmp_path / "snapshot"
    btc = root / "prices" / "BTC" / "1m"
    btc.mkdir(parents=True)
    for day in ("2026-07-10", "2026-07-15"):
        minutes = pd.date_range(day, periods=60, freq="1min", tz="UTC")
        pd.DataFrame({"timestamp": minutes, "close": 100.0}).to_parquet(btc / f"date={day}.parquet")
    events = [
        ("2026-07-10 12:30", "GDP"),
        ("2026-07-10 12:30", "PCE"),
        ("2026-07-14 12:30", "CPI"),
        ("2026-08-27 12:30", "GDP"),  # after the snapshot's last day
        ("2026-08-27 12:30", "PCE"),
    ]
    write_atomic(
        event_frame([{"time_utc": pd.Timestamp(t, tz="UTC"), "event": e, "scope": "all"} for t, e in events]),
        root / "calendar" / "events.parquet",
    )

    md = render_data_md(root)
    assert "**Scheduled events**" in md and "$SYNTH_BENCHMARK_CALENDAR" in md and "Volatility" not in md
    assert "| CPI | 1 | 0 | — |" in md
    assert "| GDP | 1 | 1 | PCE 2/2 |" in md and "| PCE | 1 | 1 | GDP 2/2 |" in md
