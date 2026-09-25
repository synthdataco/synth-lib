import json
from datetime import date
from pathlib import Path

from synth_lib.benchmark.snapshot import build_snapshot, verify_snapshot

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


def test_data_md_omits_the_span_when_nothing_is_cached(tmp_path):
    """No paths cached is not a span of zero; the line must not appear at all."""
    from synth_lib.benchmark.snapshot import render_data_md

    assert "Cached realized paths" not in render_data_md(tmp_path)
